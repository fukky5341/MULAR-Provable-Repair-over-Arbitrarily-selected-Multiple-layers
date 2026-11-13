import torch
import torch.nn.functional as F

from input_space.region import RegionStatus
from repair.bound import get_concrete_bounds

from .util import FLmode
from repair.logging import logging
from FL.arachne import arachne_selection

'''
Fault Localization (FL) 

loss:
    - distance between the output of the repaired region and the safe subspace
    - change of the output of the base (positive) region

loss sampling:
    - center point
    - corner points
'''


# loss
def output_loss(out, C):
    '''violation --> activate loss
    out: 1d output
    C: (num_constraints, output_dim)
    C*out: positive value means safe, negative value means violation
    '''
    v = F.relu(-(C * out))  # 1D vector of constraint violations
    return (v**2).sum()

def output_loss_corner(outlb, outub, C):
    pos_C = torch.clamp(C, min=0)
    neg_C = torch.clamp(C, max=0)
    v = F.relu(-(pos_C * outlb + neg_C * outub))  # worst-case corner violation
    return (v**2).sum()

def subspace_loss(y, subspace):
    '''violation --> activate loss'''
    lb = subspace.lb
    ub = subspace.ub

    v_low = F.relu(lb - y)
    v_up  = F.relu(y - ub)

    return (v_low**2 + v_up**2).sum()

def corner_loss(outlb, outub, subspace):
    lb = subspace.lb
    ub = subspace.ub

    v_low = F.relu(lb - outlb)
    v_up  = F.relu(outub - ub)

    return (v_low**2 + v_up**2).sum()

def corner_bounds(repaired_net, region):
    fixed_net = repaired_net[:len(repaired_net)-1]  # exclude the last (repaired) layer
    # get bounds
    lbs, ubs, bounder = get_concrete_bounds(
        net = fixed_net,
        in_lb = region.lb,
        in_ub = region.ub,
        save_coeffs=False
    )

    out_shape = bounder.shapes[-1]
    if isinstance(out_shape, int):
        lb = lbs[-1]
        ub = ubs[-1]
    else:
        C, H, W = out_shape
        lb = lbs[-1].view(C, H, W)
        ub = ubs[-1].view(C, H, W)
    
    return lb, ub

def compute_importance(args, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, last_layer_repair=False):
    def batch_input(x):
        if x.dim() == 1:
            return x.unsqueeze(0)
        elif x.dim() == 3:
            return x.unsqueeze(0)
        return x

    def add_loss(current, value):
        return value if current is None else current + value
    
    # prepare bounds with no grad
    precomputed = []

    for pair in repaired_region_pairs:
        precomputed_pair = []
        for _nid, Nregion in enumerate(pair.repaired_regions):
            if Nregion.status == RegionStatus.positive:
                precomputed_pair.append((None, None))
                continue
            Ncornlb, Ncornub = corner_bounds(repaired_net, Nregion)

            precomputed_pair.append((Ncornlb.detach(), Ncornub.detach()))
        
        precomputed.append(precomputed_pair)

    repaired_layer.weight.requires_grad_(True)
    if repaired_layer.bias is not None:
        repaired_layer.bias.requires_grad_(True)
    
    W = repaired_layer.weight
    bias = repaired_layer.bias
    importance = torch.zeros_like(W)
    importance_b = torch.zeros_like(bias) if bias is not None else None

    repaired_net.zero_grad()

    total_loss = None

    for i, pair in enumerate(repaired_region_pairs):
        Pregion = pair.base_region
        Nregions = pair.repaired_regions

        for _nid, Nregion in enumerate(Nregions):

            if Nregion.status == RegionStatus.positive:
                continue

            # lb/ub bounds of the repaired region (X) or (C, H, W)
            Nreglb = batch_input(Nregion.lb.to(W.device))
            Nregub = batch_input(Nregion.ub.to(W.device))
            Nreglbout = repaired_net(Nreglb).view(-1)
            Nregubout = repaired_net(Nregub).view(-1)
            # corner points
            Ncornlb, Ncornub = precomputed[i][_nid]

            z_lb, z_ub = Ncornlb.to(W.device), Ncornub.to(W.device)  # 1D

            if isinstance(repaired_layer, torch.nn.Linear):
                y_lb = []
                y_ub = []
                for i_out in range(W.shape[0]):
                    w = W[i_out]

                    pos = torch.clamp(w, min=0)
                    neg = torch.clamp(w, max=0)

                    y_lb_i = (pos * z_lb + neg * z_ub).sum() + bias[i_out]
                    y_ub_i = (pos * z_ub + neg * z_lb).sum() + bias[i_out]

                    y_lb.append(y_lb_i)
                    y_ub.append(y_ub_i)
                Ncornlb = torch.stack(y_lb)
                Ncornub = torch.stack(y_ub)
            elif isinstance(repaired_layer, torch.nn.Conv2d):
                z_lb_b = batch_input(z_lb)
                z_ub_b = batch_input(z_ub)

                stride = repaired_layer.stride
                padding = repaired_layer.padding
                dilation = repaired_layer.dilation
                groups = repaired_layer.groups

                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)

                y_lb_b = F.conv2d(z_lb_b, W_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups) + \
                    F.conv2d(z_ub_b, W_neg, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
                y_ub_b = F.conv2d(z_ub_b, W_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups) + \
                    F.conv2d(z_lb_b, W_neg, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
                if bias is not None:
                    y_lb_b += bias.view(1, -1, 1, 1)
                    y_ub_b += bias.view(1, -1, 1, 1)
                # flatten
                Ncornlb = y_lb_b.view(-1)
                Ncornub = y_ub_b.view(-1)
            else:
                raise NotImplementedError("Unsupported layer type")

            if not last_layer_repair:
                Nsubspace = Pregion.subspace
                # lb/ub points of the repaired region (all 1D)
                total_loss = add_loss(total_loss, subspace_loss(Nreglbout, Nsubspace))
                total_loss = add_loss(total_loss, subspace_loss(Nregubout, Nsubspace))
                # lb/ub corner points (all 1D)
                total_loss = add_loss(total_loss, corner_loss(Ncornlb, Ncornub, Nsubspace))

            else:
                # for last layer repair, consider the satisfaction of output constraints
                outC = Nregion.spec.C  # (num_constraints, output_dim)
                total_loss = add_loss(total_loss, output_loss(Nreglbout, outC))
                total_loss = add_loss(total_loss, output_loss(Nregubout, outC))
                total_loss = add_loss(total_loss, output_loss_corner(Ncornlb, Ncornub, outC))

    # gradient
    if total_loss is not None:
        total_loss.backward()
        importance += W.grad.abs()
        if bias is not None and bias.grad is not None:
            importance_b += bias.grad.abs()
    
    repaired_net.zero_grad()

    return importance, importance_b

def parameter_selection(args, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, last_layer_repair=False):
    k_ratio = args.fl_k_ratio
    importance_w, importance_b = compute_importance(args, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, last_layer_repair=last_layer_repair)
    
    # --- weight selection ---
    num_params = importance_w.numel()
    nonzero_num_w = (importance_w > 0).sum().item()
    tok_num_w = min(int(k_ratio * num_params), nonzero_num_w)
    k_w = max(1, tok_num_w)  # ensure at least 1 parameter is selected

    # debug length of importance whose value is greater than 0
    logging(f"Number of parameters with nonzero importance: {nonzero_num_w} / {num_params}", args.logfile)

    flat = importance_w.view(-1)
    topk_idx = torch.topk(flat, k_w).indices

    weight_mask = torch.zeros_like(flat, dtype=torch.bool)
    weight_mask[topk_idx] = True
    weight_mask = weight_mask.view_as(importance_w)

    # --- bias selection ---
    if importance_b is not None:
        # use bias importance directly
        neuron_score = importance_b.abs()

        nonzero_num_b = (neuron_score > 0).sum().item()
        tok_num_b = min(int(k_ratio * neuron_score.numel()), nonzero_num_b)
        k_b = max(1, tok_num_b)

        topk = torch.topk(neuron_score, k_b).indices

        bias_mask = torch.zeros_like(neuron_score, dtype=torch.bool)
        bias_mask[topk] = True
    else:
        bias_mask = None

    return weight_mask, bias_mask

def neuron_selection(args, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, last_layer_repair=False):
    k_ratio = args.fl_k_ratio
    importance_w, importance_b = compute_importance(
        args, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, last_layer_repair=last_layer_repair
    )

    # aggregate weight importance per neuron (sum over incoming edges)
    neuron_score = importance_w.abs().view(importance_w.shape[0], -1).sum(dim=1)

    # add bias importance if exists
    if importance_b is not None:
        neuron_score = neuron_score + importance_b.abs()

    # topk selection
    nonzero_num = (neuron_score > 0).sum().item()
    topk_num = min(int(k_ratio * len(neuron_score)), nonzero_num)
    k = max(1, topk_num)

    topk = torch.topk(neuron_score, k).indices

    mask = torch.zeros_like(neuron_score, dtype=torch.bool)
    mask[topk] = True

    # return:
    # - weight mask (per neuron)
    # - bias mask (same neurons)
    return mask, mask



def parameter_selection_arachne(args, full_net, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, tr_dataloader, last_layer_repair=False):
    grad_neg_w, grad_neg_b = compute_importance(args, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, last_layer_repair=last_layer_repair)
    weight_mask, bias_mask = arachne_selection(args, full_net, repaired_layer, grad_neg_w, grad_neg_b, tr_dataloader, repaired_region_pairs, last_layer_repair=last_layer_repair)
    return weight_mask, bias_mask


def FL_selection(args, full_net, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, tr_dataloader, last_layer_repair=False):
    if args.flmode == FLmode.PARAMETER:
        weight_mask, bias_mask = parameter_selection(args, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, last_layer_repair=last_layer_repair)
    elif args.flmode == FLmode.NEURON:
        weight_mask, bias_mask = neuron_selection(args, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, last_layer_repair=last_layer_repair)
    elif args.flmode == FLmode.ARACHNE:
        weight_mask, bias_mask = parameter_selection_arachne(args, full_net, repaired_net, repaired_layer_idx, repaired_layer, repaired_region_pairs, tr_dataloader, last_layer_repair=last_layer_repair)
    else:
        raise ValueError("Invalid FL mode")

    return weight_mask, bias_mask
