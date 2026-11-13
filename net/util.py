from pathlib import Path
import onnx
from onnx import numpy_helper
from onnx2torch import convert
import torch
import torch.nn as nn
from experiments import ( mnist, acas )
import sytorch as st


def fold_normalization_into_conv(W, b, mean, std):
    """
    W: (out_c, in_c, kh, kw)
    b: (out_c,)
    mean, std: (in_c,)
    """
    if b is None:
        b = torch.zeros(W.shape[0], dtype=W.dtype)

    mean = mean.view(1, -1, 1, 1)
    std  = std.view(1, -1, 1, 1)

    # new weight
    W_new = W / std

    # bias correction
    bias_corr = (W * (mean / std)).sum(dim=(1, 2, 3))
    b_new = b - bias_corr

    return W_new, b_new



def onnx_to_sequential(net, device, dtype, debug=False):
    modules = []
    nodes = list(net.graph.node)

    # initializer dict
    weights = {
        init.name: torch.tensor(numpy_helper.to_array(init), device=device, dtype=dtype)
        for init in net.graph.initializer
    }

    const_values = {}
    for node in nodes:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_values[node.output[0]] = torch.tensor(
                        numpy_helper.to_array(attr.t),
                        device=device, dtype=dtype
                    )

    producer = {}
    for node in nodes:
        for out in node.output:
            producer[out] = node

    def get_value(x):
        if x in weights:
            return weights[x]
        if x in const_values:
            return const_values[x]
        raise KeyError(f"{x} not found in weights or constants")

    def attr_map(node):
        return {a.name: a for a in node.attribute}

    need_flatten = False

    if debug:
        for node_i in range(len(nodes)):
            node = nodes[node_i]
            op = node.op_type
            print(f"node {node_i}/{len(nodes)}: {op}")

    i = 0
    while i < len(nodes):
        node = nodes[i]
        op = node.op_type
        nd_inps = list(node.input)

        # ==================================================
        # Detect Constant → Sub → Constant → Div → Conv pattern
        # ==================================================
        if op == "Conv":
            conv_node = node
            folded = False

            div_node = None
            for inp in conv_node.input:
                candidate = producer.get(inp, None)
                if candidate is not None and candidate.op_type == "Div":
                    div_node = candidate
                    break

            if div_node is not None:

                sub_node = None
                for inp in div_node.input:
                    candidate = producer.get(inp, None)
                    if candidate is not None and candidate.op_type == "Sub":
                        sub_node = candidate
                        break

                if sub_node is not None:

                    # --- mean ---
                    data_input = sub_node.input[0]
                    mean = next(get_value(x) for x in sub_node.input if x != data_input)

                    # --- std ---
                    sub_output = sub_node.output[0]
                    std = next(get_value(x) for x in div_node.input if x != sub_output)

                    mean = mean.view(-1)
                    std  = std.view(-1)

                    # --- conv params ---
                    conv_inputs = conv_node.input
                    W = get_value(conv_inputs[1]).clone()

                    b = (
                        get_value(conv_inputs[2]).clone()
                        if len(conv_inputs) > 2 and (conv_inputs[2] in weights or conv_inputs[2] in const_values)
                        else torch.zeros(W.shape[0], dtype=W.dtype, device=W.device)
                    )

                    W_new, b_new = fold_normalization_into_conv(W, b, mean, std)

                    am = attr_map(conv_node)
                    stride = tuple(am["strides"].ints) if "strides" in am else (1, 1)
                    pads = tuple(am["pads"].ints) if "pads" in am else (0, 0, 0, 0)
                    padding = (pads[0], pads[1]) if len(pads) == 4 else (0, 0)

                    out_c, in_c, kh, kw = W.shape

                    conv = nn.Conv2d(in_c, out_c, (kh, kw), stride=stride, padding=padding)
                    conv.weight.data = W_new
                    conv.bias.data = b_new

                    modules.append(conv)
                    print("[INFO] Folded normalization into Conv")

                    folded = True

            if folded:
                i += 1
                continue
            else:
            # -------------------------
            # Conv normal
            # -------------------------
                W = get_value(nd_inps[1]).clone()
                b = get_value(nd_inps[2]).clone() if len(nd_inps) > 2 and (nd_inps[2] in weights or nd_inps[2] in const_values) else torch.zeros(W.shape[0], dtype=W.dtype, device=W.device)

                am = attr_map(node)
                stride = tuple(am["strides"].ints) if "strides" in am else (1, 1)
                pads = tuple(am["pads"].ints) if "pads" in am else (0, 0, 0, 0)
                padding = (pads[0], pads[1]) if len(pads) == 4 else (0, 0)

                out_c, in_c, kh, kw = W.shape

                conv = nn.Conv2d(in_c, out_c, (kh, kw), stride=stride, padding=padding)
                conv.weight.data = W
                conv.bias.data = b

                modules.append(conv)

        # -------------------------
        # ReLU
        # -------------------------
        elif op == "Relu":
            modules.append(nn.ReLU())

        # -------------------------
        # Gemm (Linear)
        # -------------------------
        elif op == "Gemm":
            A, B = nd_inps[0], nd_inps[1]
            C = nd_inps[2] if len(nd_inps) > 2 else None

            am = attr_map(node)
            transB = am.get("transB").i if "transB" in am else 0

            W = get_value(B).clone()

            # convert to (out, in)
            if transB == 0:
                W = W.t()
            
            b = (
                get_value(C).clone()
                if C is not None and (C in weights or C in const_values)
                else torch.zeros(W.shape[0], dtype=W.dtype, device=W.device)
            )

            # insert flatten BEFORE first linear
            if not need_flatten:
                modules.append(nn.Flatten())
                need_flatten = True

            linear = nn.Linear(W.shape[1], W.shape[0])
            linear.weight.data = W
            linear.bias.data = b

            modules.append(linear)

        # -------------------------
        # MatMul (+ Add)
        # -------------------------
        elif op == "MatMul":
            W = next(get_value(x) for x in nd_inps if x in weights or x in const_values).clone()

            # assume X @ W → transpose
            W = W.t()

            if not need_flatten:
                modules.append(nn.Flatten())
                need_flatten = True

            linear = nn.Linear(W.shape[1], W.shape[0])
            linear.weight.data = W
            linear.bias.data.zero_()

            modules.append(linear)

        elif op in ["Add", "Shape", "Gather", "Unsqueeze", "Concat", "Reshape", "Constant", "Flatten"]:
            # skip ONNX graph ops
            pass

        else:
            print(f"[WARNING] Unsupported op: {op}")

        i += 1

    return nn.Sequential(*modules)


def get_net(args, debug=False):
    if args.model_name == "mnist 9x100":
        net_structure = '9x100'
        return mnist.model(net_structure).to(dtype=args.dtype).to(device=args.device), None, None
    
    if args.model_name == "acasxu":
        aca_net, _norm, _denorm = acas.models.acas(args.acasxu_net_key[0], args.acasxu_net_key[1]).to(dtype=args.dtype).to(device=args.device)
        return aca_net, _norm, _denorm


    base_dir = Path(__file__).resolve().parent
    onnx_path = base_dir / "onnx" / f"{args.model_name}.onnx"

    onnx_model = onnx.load(onnx_path)

    sequential_model = onnx_to_sequential(onnx_model, args.device, args.dtype, debug=debug).to(dtype=args.dtype).to(device=args.device)
    return sequential_model, None, None


def get_net_aprnn(args):
    if args.model_name == "mnist 9x100":
        net_structure = '9x100'
        return mnist.model(net_structure).to(dtype=args.dtype).to(device=args.device), None, None
    
    base_dir = Path(__file__).resolve().parent
    onnx_path = base_dir / "onnx" / f"{args.model_name}.onnx"

    return st.nn.from_file(onnx_path.as_posix()).to(dtype=args.dtype).to(device=args.device), None, None