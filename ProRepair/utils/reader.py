import torch
import ast
import numpy as np
import torch.nn as nn

from torch.utils.data import TensorDataset, Subset, DataLoader


def _parse_np_array_as_tensor(serialized):
    """Given a string, returns a Numpy array of its contents.

    Used when parsing the ERAN model definition files.
    """
    if isinstance(serialized, str):
        return torch.from_numpy(np.array(ast.literal_eval(serialized)))
    # Helper to read directly from a file.
    return _parse_np_array_as_tensor(serialized.readline()[:-1].strip())


def from_eran(path):
    """
    Modified from https://github.com/95616ARG/indra/blob/5cfbd139745d720dac31854b87efc
    d221f5e620b/SyReNN/pysyrenn/frontend/network.py

    Helper method to read an ERAN net_file into a Network.

    Currently only supports a subset of those supported by the original
    read_net_file.py. See an example of the type of network file we're
    reading here:

    https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf

    This code has been adapted (with heavy modifications) from the ERAN
    source code. Each layer has a header line that describes the type of
    layer, which is then followed by the weights (if applicable). Note that
    some layers are rolled together in ERAN but we do separately (eg.
    "ReLU" in the ERAN format corresponds to Affine + ReLU in our
    representation).
    """
    layers = []
    net_file = open(path, "r")
    with torch.no_grad():
        curr_line = None
        while True:
            prev_line = curr_line
            curr_line = net_file.readline()[:-1]
            if curr_line in {"Affine", "ReLU", "HardTanh"}:

                if prev_line == "MaxPooling2D":
                    # Make sure to add a flattening operation, so the dimensions match
                    layer = nn.Flatten(start_dim=1)
                    layers.append(layer)

                weight = _parse_np_array_as_tensor(net_file)
                bias   = _parse_np_array_as_tensor(net_file)

                # Add the fully-connected layer.
                layer = nn.Linear(weight.shape[1], weight.shape[0])
                layer.weight[:] = weight
                layer.bias[:] = bias
                layers.append(layer)

                # Maybe add a non-linearity.
                if curr_line == "ReLU":
                    layers.append(nn.ReLU())
                else:
                    raise NotImplementedError(
                        f"unimplemented {curr_line}"
                    )

            elif curr_line.strip() == "":
                break

            elif curr_line.startswith("Conv2D"):

                info_line = net_file.readline()[:-1].strip()
                activation = info_line.split(",")[0]

                input_shape = info_line.split("input_shape=")[1].split("],")[0]
                input_shape = _parse_np_array_as_tensor(input_shape)

                if "stride=" in info_line:
                    stride = _parse_np_array_as_tensor(
                        info_line.split("stride=")[1].split("],")[0] + "]")
                else:
                    stride = 1 # Default.

                pad = (0, 0)
                if "padding=" in info_line:
                    pad = int(info_line.split("padding=")[1])
                    pad = (pad, pad)

                # (f_h, f_w, i_c, o_c)
                filter_weights = _parse_np_array_as_tensor(net_file)
                filter_weights = filter_weights.permute(3, 2, 0, 1).float() # (o_c, i_c, f_h, f_w) (torch style)
                # (o_c,)
                biases = _parse_np_array_as_tensor(net_file).float()

                in_channels = filter_weights.shape[1]
                out_channels = filter_weights.shape[0]
                kernel_size = filter_weights.shape[2:]

                layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
                layer.weight.data = filter_weights
                layer.bias.data = biases
                layers.append(layer)

                if activation == 'ReLU':
                    layers.append(nn.ReLU())
                else:
                    raise NotImplementedError

            elif curr_line.startswith('MaxPooling2D'):
                info_line = net_file.readline()[:-1].strip()

                if "stride=" in info_line:
                    stride = _parse_np_array_as_tensor(info_line.split("stride=")[1].split("],")[0] + "]")
                else:
                    stride = None # default.

                if "padding=" in info_line:
                    pad = int(info_line.split("padding=")[1])
                    pad = (pad, pad)
                else:
                    pad = 0

                # tuple(_parse_np_array_as_tensor(info_line.split("pool_size=")[1].split("],")[0] + "]"))
                kernel_size = tuple(ast.literal_eval(info_line.split("pool_size=")[1].split("],")[0] + "]"))

                layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pad)
                layers.append(layer)

            else:
                raise NotImplementedError(
                    f"unimplemented {curr_line.split(' ')[0]}"
                )

    return nn.Sequential(*layers)


def from_npy(data_path, label_path, index):
    data = torch.from_numpy(np.load(data_path)).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.from_numpy(np.load(label_path))  
    dataset = TensorDataset(data[index], labels[index])
    return dataset
