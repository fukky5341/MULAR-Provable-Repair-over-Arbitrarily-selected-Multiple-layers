import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from input_space.region import RegionStatus
from repair.bound import get_concrete_bounds
from .pareto.pareto import is_pareto_front, nondominated_rank

'''
Arachne
-------
arachne takes two components:
    - gradient loss
    - forward impact
--> pareto selection to get the score

outline
-------
goal: 
    1. get parameters that are less likely to cause deterioration of positive behavior, while improving the negative behavior
    2. select layer to repair

based on the score on positive and negative behavior individually, we select parameters and layers to repair
at least, we need to select parameters that aren't likely to deteriorate the positive behavior, which enables us to focus on parameters execpt them

given
-----
- repaired_net
- positive inputs
- negative region set
    - negative points are derived as corner, representative (likely to produce lower and upper bounds) points
- repaired layer index (optional)
- specification
- k_ratio (optional)
'''


'''helper functions'''
def _enable_target_layer_grads(target_layers):
    prev = []
    for layer in target_layers:
        w_prev = layer.weight.requires_grad
        b_prev = layer.bias.requires_grad if layer.bias is not None else None
        prev.append((layer, w_prev, b_prev))

        layer.weight.requires_grad_(True)
        if layer.bias is not None:
            layer.bias.requires_grad_(True)
    return prev

def _restore_target_layer_grads(prev):
    for layer, w_prev, b_prev in prev:
        layer.weight.requires_grad_(w_prev)
        if layer.bias is not None and b_prev is not None:
            layer.bias.requires_grad_(b_prev)


def collect_boundary_dataloader(model, dataloader, device, k=200, batch_size=32):
    model.eval()

    margins = []
    xs_all = []
    ys_all = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).long().view(-1)

            logits = model(x)

            true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            max_other = torch.topk(logits, 2, dim=1).values[:, 1]

            margin = true_logit - max_other

            margins.extend(margin.cpu().tolist())
            xs_all.append(x.cpu())
            ys_all.append(y.cpu())

    xs_all = torch.cat(xs_all)
    ys_all = torch.cat(ys_all)

    # sort by smallest margin
    idx = sorted(range(len(margins)), key=lambda i: margins[i])
    selected_idx = idx[:k]

    xs = xs_all[selected_idx]
    ys = ys_all[selected_idx]

    # create dataset + dataloader
    dataset = TensorDataset(xs, ys)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

def collect_neg_dataloader(args, full_net, repaired_region_pairs, device, batch_size=32):
    xs = []
    ys = []

    for pair in repaired_region_pairs:
        for Nregion in pair.repaired_regions:

            if Nregion.status == RegionStatus.positive:
                continue

            label = Nregion.target_label
            out_C = Nregion.spec.C

            Nlb = Nregion.lb.unsqueeze(0).to(device)
            Nub = Nregion.ub.unsqueeze(0).to(device)

            xs.append(Nlb.cpu())
            ys.append(torch.tensor([label]))

            xs.append(Nub.cpu())
            ys.append(torch.tensor([label]))

            # ---- bounds + coeff ----
            lbs, ubs, bounder = get_concrete_bounds(full_net, Nlb.squeeze(0), Nub.squeeze(0), save_coeffs=True)
            output_coeff = bounder.saved_coeffs[(len(full_net)-1, 0)]

            rlb = Nlb.view(-1)
            rub = Nub.view(-1)

            # ---- target dim and violating dims ----
            out_ub = ubs[-1].view(-1)
            out_lb = lbs[-1].view(-1)
            # violating dim
            pos_C = torch.clamp(out_C, min=0)
            target_dim_lb = (pos_C[0] * out_lb).sum()
            dims = (target_dim_lb - out_ub) < 0
            dims = dims.nonzero(as_tuple=True)[0]

            for vdim in dims:
                coeff_lb_mv = output_coeff.n_C_lb[vdim].view(-1)
                coeff_ub_mv = output_coeff.n_C_ub[vdim].view(-1)

                corner_lb_mv = torch.where(coeff_lb_mv >= 0, rlb, rub).view_as(Nlb)
                corner_ub_mv = torch.where(coeff_ub_mv >= 0, rlb, rub).view_as(Nub)

                xs.append(corner_lb_mv.cpu())
                ys.append(torch.tensor([label]))

                xs.append(corner_ub_mv.cpu())
                ys.append(torch.tensor([label]))

    if len(xs) == 0:
        raise ValueError("No negative points were collected.")

    xs_tensor = torch.cat(xs, dim=0)
    ys_tensor = torch.cat(ys, dim=0).long()

    dataset = TensorDataset(xs_tensor, ys_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def collect_test_negative_dataloader(
    args, full_net, max_samples=200, batch_size=32
):
    """Collect concrete test examples currently misclassified by ``full_net``."""
    if max_samples < 1:
        raise ValueError("Arachne negative sample count must be at least one.")

    # Import locally to avoid making the generic Arachne utilities depend on
    # dataset construction unless this source is explicitly requested.
    import copy
    from input_space.generate_input import clean_points

    dataset_args = copy.copy(args)
    dataset_args.device = "cpu"
    test_points = clean_points(dataset_args)
    test_loader = test_points.dataloader(batch_size=100, shuffle=False)
    negative_inputs = []
    negative_labels = []
    collected = 0
    was_training = full_net.training
    full_net.eval()
    try:
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device).long().view(-1)
                predictions = full_net(inputs).argmax(dim=1)
                negative_mask = predictions != labels
                if args.target_label is not None:
                    negative_mask &= labels == args.target_label
                if not negative_mask.any():
                    continue

                remaining = max_samples - collected
                selected_inputs = inputs[negative_mask][:remaining]
                selected_labels = labels[negative_mask][:remaining]
                negative_inputs.append(selected_inputs.cpu())
                negative_labels.append(selected_labels.cpu())
                collected += selected_inputs.shape[0]
                if collected >= max_samples:
                    break
    finally:
        full_net.train(was_training)

    if not negative_inputs:
        target_text = (
            f" for target label {args.target_label}"
            if args.target_label is not None else ""
        )
        raise ValueError(
            "No misclassified test samples were found"
            f"{target_text}; use --arachne-negative-source region or choose "
            "another target label."
        )

    dataset = TensorDataset(
        torch.cat(negative_inputs, dim=0),
        torch.cat(negative_labels, dim=0),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

'''
gradient loss
-------------
dL/dw
'''

def compute_positive_grad_full(
    repaired_net,
    target_layers,
    pos_loader,
    device,
    loss_type="logit",
):
    repaired_net.eval()
    prev_grad_flags = _enable_target_layer_grads(target_layers)

    # initialize storage
    grad_w = {}
    grad_b = {}
    for layer in target_layers:
        grad_w[layer] = torch.zeros_like(layer.weight)
        if layer.bias is not None:
            grad_b[layer] = torch.zeros_like(layer.bias)

    # cache original outputs (no grad)
    orig_outputs = []
    with torch.no_grad():
        for x, _ in pos_loader:
            x = x.to(device)
            orig_outputs.append(repaired_net(x))

    # iterate again for gradient computation
    for (x, y), y_orig in zip(pos_loader, orig_outputs):
        x = x.to(device)
        y = y.to(device).long().view(-1)
        y_orig = y_orig.to(device)

        repaired_net.zero_grad()
        out = repaired_net(x)

        if loss_type == "logit":
            loss = ((out - y_orig) ** 2).mean()
        elif loss_type == "ce":
            loss = F.cross_entropy(out, y)
        else:
            raise ValueError("Unknown loss_type")

        if not loss.requires_grad:
            raise RuntimeError("Loss does not require grad. Target layers may be frozen.")
        loss.backward()

        # ---- collect gradients for each target layer ----
        for layer in target_layers:
            if layer.weight.grad is not None:
                grad_w[layer] += layer.weight.grad.detach().abs()
            if layer.bias is not None and layer.bias.grad is not None:
                grad_b[layer] += layer.bias.grad.detach().abs()
    
    num_batches = len(pos_loader)
    for layer in grad_w:
        grad_w[layer] /= num_batches
        if layer in grad_b:
            grad_b[layer] /= num_batches

    _restore_target_layer_grads(prev_grad_flags)
    return grad_w, grad_b


'''
forward impact
--------------
(weight)
example of the connection between oi -- wij --> o'j -- relu --> ...
1. the importance of wij in incoming connection relating to the o'j
    --> oi * wij normalized by the sum of incoming connection (sum_k ok * wkj)
2. the gradient of o'j with respect to output loss
    --> dO/do'j
3. total importance
    --> normalized(oi * wij) * dO/do'j

(bias)
bias is simpler since it doesn't have the importance of incoming connection
--> dO/do'j
'''

def arachne_forward_importance_layer_(layer, activation, grad_out):
    """
    layer: nn.Linear or Conv2d
    activation: input to layer (batch, in_features, ...)
    grad_out: gradient wrt pre-activation (batch, out_features, ...)
    """

    W = layer.weight

    # flatten spatial if needed
    if activation.dim() > 2:
        activation = activation.view(activation.shape[0], activation.shape[1], -1).mean(dim=2)
        grad_out = grad_out.view(grad_out.shape[0], grad_out.shape[1], -1).mean(dim=2)

    # average over batch
    a = activation.mean(dim=0)        # (in,)
    g = grad_out.mean(dim=0)          # (out,)

    # contribution
    contrib = a.unsqueeze(0) * W      # (out, in)

    denom = contrib.abs().sum(dim=1, keepdim=True) + 1e-8
    norm_contrib = contrib / denom

    importance = norm_contrib * g.unsqueeze(1)
    importance_b = g

    return importance.abs(), importance_b.abs()


def arachne_forward_importance_layer(layer, activation, grad_out):
    W = layer.weight  # (out, in, kh, kw)

    if isinstance(layer, torch.nn.Linear):
        # ---- same as before ----
        a = activation.mean(dim=0)   # (in,)
        g = grad_out.mean(dim=0)     # (out,)

        contrib = a.unsqueeze(0) * W
        denom = contrib.abs().sum(dim=1, keepdim=True) + 1e-8
        norm_contrib = contrib / denom

        importance = norm_contrib * g.unsqueeze(1)
        importance_b = g

        return importance.abs(), importance_b.abs()

    elif isinstance(layer, torch.nn.Conv2d):
        B, Cin, Hin, Win = activation.shape
        Cout, _, Kh, Kw = W.shape

        # ---- unfold input into patches ----
        patches = F.unfold(
            activation,
            kernel_size=(Kh, Kw),
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation
        )  # (B, Cin*Kh*Kw, L)

        # reshape weight
        W_flat = W.view(Cout, -1)  # (Cout, Cin*Kh*Kw)

        # flatten grad_out spatially
        grad_flat = grad_out.view(B, Cout, -1)  # (B, Cout, L)

        # ---- compute contribution ----
        # patches: (B, K, L)
        # W_flat: (Cout, K)
        # grad_flat: (B, Cout, L)

        # contribution per weight
        weighted_patches = patches.unsqueeze(1) * W_flat.unsqueeze(0).unsqueeze(-1)
        # (B, Cout, K, L)

        contrib = (weighted_patches * grad_flat.unsqueeze(2)).sum(dim=(0, 3))
        # (Cout, K)

        # normalize per output channel
        denom = contrib.abs().sum(dim=1, keepdim=True) + 1e-8
        norm_contrib = contrib / denom

        importance = norm_contrib.view_as(W)

        # bias importance
        importance_b = grad_out.sum(dim=(0, 2, 3))  # (Cout,)

        return importance.abs(), importance_b.abs()

    else:
        raise NotImplementedError("Unsupported layer type")


def compute_forward_impact_positive(
    repaired_net,
    target_layers,
    pos_loader,
    device,
    loss_type="logit"
):
    fwd_w = {layer: torch.zeros_like(layer.weight) for layer in target_layers}
    fwd_b = {layer: torch.zeros_like(layer.bias) for layer in target_layers if layer.bias is not None}

    repaired_net.eval()
    prev_grad_flags = _enable_target_layer_grads(target_layers)

    # register hooks (as above)
    activations = {}
    grad_outputs = {}
    hooks = []

    for layer in target_layers:
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, layer=layer: activations.update({layer: inp[0]})
        ))
        hooks.append(layer.register_full_backward_hook(
            lambda m, gin, gout, layer=layer: grad_outputs.update({layer: gout[0]})
        ))

    # precompute outside loop
    orig_outputs = []
    with torch.no_grad():
        for x, _ in pos_loader:
            orig_outputs.append(repaired_net(x))

    total_samples = 0
    for (x, y), y_orig in zip(pos_loader, orig_outputs):
        x = x.to(device)
        y = y.to(device).long().view(-1)
        y_orig = y_orig.to(device)

        total_samples += x.size(0)

        repaired_net.zero_grad()
        out = repaired_net(x)

        if loss_type == "logit":
            loss = ((out - y_orig) ** 2).mean()
        elif loss_type == "ce":
            loss = F.cross_entropy(out, y)
        else:
            raise ValueError("Unknown loss_type")

        if not loss.requires_grad:
            raise RuntimeError("Loss does not require grad. Target layers may be frozen.")
        loss.backward()

        for layer in target_layers:
            if layer in activations and layer in grad_outputs:
                importance_w, importance_b = arachne_forward_importance_layer(
                    layer,
                    activations[layer].detach(),
                    grad_outputs[layer].detach()
                )
                fwd_w[layer] += importance_w
                if layer.bias is not None:
                    fwd_b[layer] += importance_b

        activations.clear()
        grad_outputs.clear()

    for h in hooks:
        h.remove()

    for layer in fwd_w:
        fwd_w[layer] /= total_samples
        if layer in fwd_b:
            fwd_b[layer] /= total_samples

    _restore_target_layer_grads(prev_grad_flags)
    return fwd_w, fwd_b

def compute_forward_impact_negative(
    repaired_net,
    target_layers,
    neg_loader,
    device,
    loss_type="margin"
):
    fwd_w = {layer: torch.zeros_like(layer.weight) for layer in target_layers}
    fwd_b = {layer: torch.zeros_like(layer.bias) for layer in target_layers if layer.bias is not None}

    repaired_net.eval()
    prev_grad_flags = _enable_target_layer_grads(target_layers)

    activations = {}
    grad_outputs = {}
    hooks = []

    for layer in target_layers:
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out, layer=layer: activations.update({layer: inp[0]})
        ))
        hooks.append(layer.register_full_backward_hook(
            lambda m, gin, gout, layer=layer: grad_outputs.update({layer: gout[0]})
        ))

    total_samples = 0

    for x, y in neg_loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)

        repaired_net.zero_grad()
        out = repaired_net(x)

        if loss_type == "margin":
            true_logit = out.gather(1, y.unsqueeze(1)).squeeze(1)
            tmp = out.clone()
            tmp[torch.arange(out.size(0), device=out.device), y] = -1e9
            max_other = tmp.max(dim=1).values
            loss = (max_other - true_logit).mean()
        else:
            raise ValueError("Unknown loss_type")

        if not loss.requires_grad:
            raise RuntimeError("Loss does not require grad. Target layers may be frozen.")
        loss.backward()

        B = x.size(0)
        total_samples += B

        for layer in target_layers:
            if layer in activations and layer in grad_outputs:
                importance_w, importance_b = arachne_forward_importance_layer(
                    layer,
                    activations[layer].detach(),
                    grad_outputs[layer].detach()
                )
                fwd_w[layer] += importance_w * B
                if layer.bias is not None:
                    fwd_b[layer] += importance_b * B

        activations.clear()
        grad_outputs.clear()

    for h in hooks:
        h.remove()

    for layer in fwd_w:
        fwd_w[layer] /= total_samples
        if layer in fwd_b:
            fwd_b[layer] /= total_samples

    _restore_target_layer_grads(prev_grad_flags)
    return fwd_w, fwd_b


'''
paleto selection
-----------------
we have:
    - gradient importance for positive behavior (grad_pos)
        --> minimize to avoid touching risky parameters
    - gradient importance for negative behavior (grad_neg)
        --> maximize to fix negative behavior
    - forward impact for positive behavior (fwd_pos)
        --> minimize to avoid deteriorating positive behavior
    - forward impact for negative behavior (fwd_neg)
        --> maximize to fix negative behavior

1. normalize each importance
2. add sign for each importance based on whether we want to maximize or minimize
3. pareto selection based on the 4 scores
'''

def build_score_matrix4d(grad_pos, grad_neg, fwd_pos, fwd_neg):
    # flatten
    g_pos = grad_pos.view(-1)
    g_neg = grad_neg.view(-1)
    f_pos = fwd_pos.view(-1)
    f_neg = fwd_neg.view(-1)

    # normalize (important!)
    def norm(x):
        return x / (x.mean() + 1e-8)

    g_pos = norm(g_pos)
    g_neg = norm(g_neg)
    f_pos = norm(f_pos)
    f_neg = norm(f_neg)

    # maximize all
    scores = torch.stack([
        -g_pos,    # minimize positive gradient importance
        g_neg,     # maximize negative gradient importance
        -f_pos,    # minimize positive forward impact
        f_neg      # maximize negative forward impact
    ], dim=1)

    return scores

def build_score_matrix2d(grad_pos, grad_neg, fwd_pos, fwd_neg):
    # flatten
    g_pos = grad_pos.view(-1)
    g_neg = grad_neg.view(-1)
    f_pos = fwd_pos.view(-1)
    f_neg = fwd_neg.view(-1)

    # normalize (important!)
    def norm(x):
        return x / (x.mean() + 1e-8)

    g_pos = norm(g_pos)
    g_neg = norm(g_neg)
    f_pos = norm(f_pos)
    f_neg = norm(f_neg)

    # combine into 2D score (for visualization and simpler selection)
    score_pos = - g_pos - f_pos
    score_neg = g_neg + f_neg

    scores = torch.stack([score_pos, score_neg], dim=1)

    return scores

def pareto_front(scores, eps=1e-2):
    """
    scores: (N, K) tensor
    returns: boolean mask (N,)
    """
    N = scores.shape[0]
    keep = torch.ones(N, dtype=torch.bool, device=scores.device)

    for i in range(N):
        if not keep[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            if torch.all(scores[j] >= scores[i] - eps) and torch.any(scores[j] > scores[i] + eps):
                keep[i] = False
                break
    return keep

def pareto_selection_with_fallback(grad_neg, grad_pos, fwd_pos, fwd_neg, k_ratio=0.1):
    if k_ratio is None:
        pre_ratio = 1.0
    else:
        pre_ratio = min(1.0, k_ratio + 0.3)

    flat_neg = grad_neg.abs().view(-1)
    # pre-filtering
    k = max(1, int(pre_ratio * flat_neg.numel()))
    topk = torch.topk(flat_neg, k).indices

    # score matrix
    scores = build_score_matrix2d(grad_pos, grad_neg, fwd_pos, fwd_neg)
    scores_filtered = scores[topk]

    # pareto selection
    mask = pareto_front(scores_filtered)

    # map back
    mask_flat = torch.zeros_like(flat_neg, dtype=torch.bool)
    mask_flat[topk[mask]] = True

    return mask_flat.view_as(grad_neg)

def multi_pareto_selection(
    grad_neg, grad_pos, fwd_pos, fwd_neg,
    k_ratio=0.01,
    max_rounds=3,
    prefilter=False
):
    flat_size = grad_neg.numel()
    k_total = max(1, int(k_ratio * flat_size))

    # ---- scores (full space) ----
    scores = build_score_matrix2d(
        grad_pos, grad_neg, fwd_pos, fwd_neg
    ).view(flat_size, -1)

    # =====================
    # Pre-filter (reduce space)
    # =====================
    if prefilter:
        prefilter_ratio = min(1.0, k_ratio)
        k_pref = max(1, int(prefilter_ratio * flat_size))
        tmp_scores = (grad_neg + fwd_neg) - (grad_pos + fwd_pos)

        topk_pref = torch.topk(tmp_scores, k_pref).indices

        scores_pref = scores[topk_pref]
    else:
        k_pref = flat_size
        topk_pref = torch.arange(flat_size, device=grad_neg.device)
        scores_pref = scores

    # =====================
    # Work in reduced space
    # =====================
    selected_small = torch.zeros(k_pref, dtype=torch.bool, device=grad_neg.device)
    remaining_small = torch.ones(k_pref, dtype=torch.bool, device=grad_neg.device)

    for _ in range(max_rounds):

        if selected_small.sum() >= k_total:
            break

        idx = torch.where(remaining_small)[0]
        if len(idx) == 0:
            break

        scores_subset = scores_pref[idx]

        pareto_mask = pareto_front(scores_subset)

        selected_idx = idx[pareto_mask]

        selected_small[selected_idx] = True
        remaining_small[selected_idx] = False

    # =====================
    # Map back to full space
    # =====================
    selected_mask = torch.zeros(flat_size, dtype=torch.bool, device=grad_neg.device)
    selected_mask[topk_pref[selected_small]] = True

    return selected_mask.view_as(grad_neg)

def fast_pareto_selection(
    grad_neg, grad_pos, fwd_pos, fwd_neg,
    k_ratio=0.01,
    prefilter_ratio=None
):
    device = grad_neg.device
    flat_size = grad_neg.numel()
    k_total = max(1, int(k_ratio * flat_size))

    def norm(x):
        return x / (x.mean() + 1e-8)

    # ---- flatten ----
    g_neg = grad_neg.view(-1)
    g_pos = grad_pos.view(-1)
    f_pos = fwd_pos.view(-1)
    f_neg = fwd_neg.view(-1)
    # ---- normalize ----
    g_neg = norm(g_neg)
    g_pos = norm(g_pos)
    f_pos = norm(f_pos)
    f_neg = norm(f_neg)

    # =====================
    # Optional prefilter (strongly recommended)
    # =====================
    if prefilter_ratio is None:
        prefilter_ratio = min(1.0, k_ratio + 0.1)

    k_pref = max(1, int(prefilter_ratio * flat_size))

    # fast scalar score for filtering
    score_fast = (g_neg + f_neg) - (g_pos + f_pos)  # higher is better 
    topk_idx = torch.topk(score_fast, k_pref).indices

    # restrict
    g_neg = g_neg[topk_idx]
    g_pos = g_pos[topk_idx]
    f_pos = f_pos[topk_idx]
    f_neg = f_neg[topk_idx]

    # =====================
    # Build 2D score (Pareto)
    # =====================
    # NOTE: in nondominated_rank, scores are designed so that smaller is better
    # convert to minimization form
    score_pos = g_pos + f_pos          # minimize
    score_neg = -(g_neg + f_neg)       # maximize → minimize

    scores = torch.stack([score_pos, score_neg], dim=1)

    # =====================
    # Fast Pareto ranking
    # =====================
    ranks = nondominated_rank(scores.detach().cpu().numpy())
    ranks = torch.from_numpy(ranks).to(device)

    # =====================
    # Select top-k by rank
    # =====================
    # smaller rank = better
    sorted_idx = torch.argsort(ranks)

    selected_small = sorted_idx[:k_total]

    # =====================
    # map back to full space
    # =====================
    selected_mask = torch.zeros(flat_size, dtype=torch.bool, device=device)
    selected_mask[topk_idx[selected_small]] = True

    return selected_mask.view_as(grad_neg)

def two_stage_pareto(
    grad_neg, grad_pos, fwd_pos, fwd_neg,
    k_ratio=0.01,
    prefilter_ratio=None,
    pareto_round=3
):
    """
    1. fast pareto
    2. multi pareto
    """
    device = grad_neg.device
    flat_size = grad_neg.numel()

    # flatten
    g_neg = grad_neg.view(-1)
    g_pos = grad_pos.view(-1)
    f_pos = fwd_pos.view(-1)
    f_neg = fwd_neg.view(-1)
    
    # stage 1: fast pareto
    mask_fast = fast_pareto_selection(
        grad_neg, grad_pos, fwd_pos, fwd_neg,
        k_ratio=k_ratio,
        prefilter_ratio=prefilter_ratio
    ).view(-1)
    idx_fast = torch.where(mask_fast)[0]

    # stage 2: multi pareto
    # ---- select only the fast pareto ones ----
    grad_neg_sel = g_neg[idx_fast]
    grad_pos_sel = g_pos[idx_fast]
    fwd_pos_sel = f_pos[idx_fast]
    fwd_neg_sel = f_neg[idx_fast]
    # ---- multi pareto selection ----
    mask_multi_sel = multi_pareto_selection(
        grad_neg_sel, grad_pos_sel, fwd_pos_sel, fwd_neg_sel,
        k_ratio=1.0,  # use all since we already reduced by fast pareto
        max_rounds=pareto_round,
        prefilter=False
    ).view(-1)

    # =====================
    # map back
    # =====================
    selected_mask = torch.zeros(flat_size, dtype=torch.bool, device=device)
    selected_mask[idx_fast[mask_multi_sel]] = True

    return selected_mask.view_as(grad_neg)


def arachne_selection(args, full_net, repaired_layer, grad_neg_w, grad_neg_b, tr_loader, repaired_region_pairs, last_layer_repair=False):
    '''Note: currently only supports one layer (repaired_layer)'''
    # prepare positive loader (boundary samples) with 200 samples (hyperparameter)
    pos_loader = collect_boundary_dataloader(full_net, tr_loader, device=args.device, k=200)
    # compute grad_pos, fwd_pos
    grad_pos_w, grad_pos_b = compute_positive_grad_full(full_net, [repaired_layer], pos_loader, device=grad_neg_w.device)
    fwd_pos_w, fwd_pos_b = compute_forward_impact_positive(full_net, [repaired_layer], pos_loader, device=grad_neg_w.device)
    # prepare negative set
    neg_loader = collect_neg_dataloader(args, full_net, repaired_region_pairs, device=args.device)
    fwd_neg_w, fwd_neg_b = compute_forward_impact_negative(full_net, [repaired_layer], neg_loader, device=grad_neg_w.device)

    if last_layer_repair:
        pareto_round = 3  # todo
    else:
        pareto_round = 3
    # pareto selection
    # mask_w = multi_pareto_selection(grad_neg_w, grad_pos_w[repaired_layer], fwd_pos_w[repaired_layer], fwd_neg_w[repaired_layer], k_ratio=args.fl_k_ratio, max_rounds=pareto_round)
    mask_w = two_stage_pareto(grad_neg_w, grad_pos_w[repaired_layer], fwd_pos_w[repaired_layer], fwd_neg_w[repaired_layer], k_ratio=args.fl_k_ratio, pareto_round=args.pareto_round)
    if repaired_layer.bias is not None:
        # mask_b = multi_pareto_selection(grad_neg_b, grad_pos_b[repaired_layer], fwd_pos_b[repaired_layer], fwd_neg_b[repaired_layer], k_ratio=args.fl_k_ratio, max_rounds=pareto_round)
        mask_b = two_stage_pareto(grad_neg_b, grad_pos_b[repaired_layer], fwd_pos_b[repaired_layer], fwd_neg_b[repaired_layer], k_ratio=args.fl_k_ratio, pareto_round=args.pareto_round)
    else:
        mask_b = None

    # # two-stage selection
    # mask_w = two_stage_selection(grad_neg_w, grad_pos_w[repaired_layer], fwd_pos_w[repaired_layer], fwd_neg_w[repaired_layer], k_ratio=args.fl_k_ratio)
    # if repaired_layer.bias is not None:
    #     mask_b = two_stage_selection(grad_neg_b, grad_pos_b[repaired_layer], fwd_pos_b[repaired_layer], fwd_neg_b[repaired_layer], k_ratio=args.fl_k_ratio)
    # else:
    #     mask_b = None

    return mask_w, mask_b


def _original_forward_impact(layer, activation):
    """Return summed |activation * weight| and its normalization count."""
    weight = layer.weight.detach().abs()
    if isinstance(layer, torch.nn.Linear):
        activation = activation.detach().reshape(activation.shape[0], -1)
        impact = weight * activation.abs().sum(dim=0).unsqueeze(0)
        return impact, activation.shape[0]

    if isinstance(layer, torch.nn.Conv2d):
        patches = F.unfold(
            activation.detach(),
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )
        # Each convolutional weight is used at every output position.
        input_impact = patches.abs().sum(dim=(0, 2))
        impact = weight.view(weight.shape[0], -1) * input_impact.unsqueeze(0)
        return impact.view_as(weight), activation.shape[0] * patches.shape[-1]

    raise NotImplementedError(
        f"Original Arachne does not support layer type {type(layer).__name__}."
    )


def compute_original_arachne_scores(
    full_net, target_layers, neg_loader, device
):
    """Compute Arachne's two negative-input objectives for all target weights.

    The original localization method uses the magnitude of the loss gradient
    and the forward impact ``|activation * weight|``. Biases are deliberately
    excluded, matching the original Arachne localization algorithm.
    """
    full_net.eval()
    previous_grad_flags = _enable_target_layer_grads(target_layers)
    signed_gradients = {
        layer: torch.zeros_like(layer.weight) for layer in target_layers
    }
    forward_impacts = {
        layer: torch.zeros_like(layer.weight) for layer in target_layers
    }
    forward_counts = {layer: 0 for layer in target_layers}
    activations = {}
    hooks = [
        layer.register_forward_hook(
            lambda _module, inputs, _output, layer=layer:
                activations.__setitem__(layer, inputs[0])
        )
        for layer in target_layers
    ]
    num_negative = 0

    try:
        for inputs, labels in neg_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long().view(-1)
            full_net.zero_grad()
            outputs = full_net(inputs)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            loss.backward()
            num_negative += inputs.shape[0]

            for layer in target_layers:
                if layer.weight.grad is not None:
                    signed_gradients[layer] += layer.weight.grad.detach()
                if layer in activations:
                    impact, count = _original_forward_impact(
                        layer, activations[layer]
                    )
                    forward_impacts[layer] += impact
                    forward_counts[layer] += count
            activations.clear()
    finally:
        for hook in hooks:
            hook.remove()
        _restore_target_layer_grads(previous_grad_flags)
        full_net.zero_grad()

    if num_negative == 0:
        raise ValueError("Original Arachne requires at least one negative input.")

    gradient_scores = {
        layer: gradient.abs() / num_negative
        for layer, gradient in signed_gradients.items()
    }
    for layer in target_layers:
        count = forward_counts[layer]
        if count == 0:
            raise RuntimeError(
                f"No activation was captured for {type(layer).__name__}."
            )
        forward_impacts[layer] /= count
    return gradient_scores, forward_impacts, num_negative


def _pareto_max_mask(scores, max_rounds=1):
    """Return the first ``max_rounds`` fronts for maximization objectives."""
    if max_rounds < 1:
        raise ValueError("Pareto rounds must be at least one.")
    score_values = scores.detach().cpu().numpy()
    keep = torch.zeros(scores.shape[0], dtype=torch.bool).numpy()
    remaining = torch.arange(scores.shape[0]).numpy()
    for _ in range(max_rounds):
        if remaining.size == 0:
            break
        on_front = is_pareto_front(
            score_values[remaining],
            larger_is_better_objectives=[0, 1],
        )
        keep[remaining[on_front]] = True
        remaining = remaining[~on_front]
    return torch.from_numpy(keep).to(device=scores.device)


def original_arachne_localize(
    args, full_net, layer_indices, repaired_region_pairs, random_seed=None
):
    """Localize weights once, then order their containing layers.

    Candidate weights are the largest ``20 * number_of_negative_inputs`` loss
    gradients, as in the original implementation. The first
    ``args.pareto_round`` fronts over loss gradient and forward impact become
    the fixed repair mask. Layers are ordered either by the strongest selected
    output-neuron score or by the number of Arachne-selected weights contained
    in the layer. With ``random_matched`` selection, Arachne is still run to
    determine the selected count, then exactly that many weights are sampled
    uniformly from all configured layer weights. With
    ``random_layer_matched``, random sampling instead preserves Arachne's
    selected weight count separately within every layer.
    """
    target_layers = [full_net[index] for index in layer_indices]
    if args.arachne_negative_source == "region":
        neg_loader = collect_neg_dataloader(
            args, full_net, repaired_region_pairs, device=args.device
        )
    elif args.arachne_negative_source == "test":
        neg_loader = collect_test_negative_dataloader(
            args,
            full_net,
            max_samples=args.arachne_negative_sample_count,
        )
    else:
        raise ValueError(
            "Unsupported Original-Arachne negative sample source: "
            f"{args.arachne_negative_source}"
        )
    gradient_scores, impact_scores, num_negative = \
        compute_original_arachne_scores(
            full_net, target_layers, neg_loader, device=args.device
        )

    flat_gradients = torch.cat([
        gradient_scores[layer].reshape(-1) for layer in target_layers
    ])
    flat_impacts = torch.cat([
        impact_scores[layer].reshape(-1) for layer in target_layers
    ])
    candidate_count = min(flat_gradients.numel(), 20 * num_negative)
    candidate_indices = torch.topk(
        flat_gradients, candidate_count
    ).indices
    candidate_scores = torch.stack(
        [flat_gradients[candidate_indices], flat_impacts[candidate_indices]],
        dim=1,
    )
    selected_candidates = _pareto_max_mask(
        candidate_scores, max_rounds=args.pareto_round
    )
    selected_global_indices = candidate_indices[selected_candidates]

    # A scalar score is only needed to order neurons/layers. Pareto selection
    # itself continues to use the original two independent objectives.
    selected_scores = candidate_scores[selected_candidates]
    normalized_scores = selected_scores / selected_scores.amax(
        dim=0, keepdim=True
    ).clamp_min(1e-12)
    scalar_scores = normalized_scores.mean(dim=1)

    if args.arachne_parameter_selection in (
        "random_matched", "random_layer_matched"
    ):
        matched_count = selected_global_indices.numel()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            args.arachne_random_seed if random_seed is None else random_seed
        )
        if args.arachne_parameter_selection == "random_matched":
            selected_global_indices = torch.randperm(
                flat_gradients.numel(), generator=generator, device="cpu"
            )[:matched_count].to(flat_gradients.device)
        else:
            sampled_by_layer = []
            offset = 0
            for layer in target_layers:
                num_weights = layer.weight.numel()
                selected_in_layer = int((
                    (selected_global_indices >= offset)
                    & (selected_global_indices < offset + num_weights)
                ).sum().item())
                if selected_in_layer:
                    local_indices = torch.randperm(
                        num_weights, generator=generator, device="cpu"
                    )[:selected_in_layer]
                    sampled_by_layer.append(local_indices + offset)
                offset += num_weights
            selected_global_indices = torch.cat(sampled_by_layer).to(
                flat_gradients.device
            )
        # Random weights have no Arachne score. Unit scores let the existing
        # neuron aggregation construct masks, while ordering below uses counts.
        scalar_scores = torch.ones(
            matched_count,
            device=flat_gradients.device,
            dtype=flat_gradients.dtype,
        )
    elif args.arachne_parameter_selection != "original":
        raise ValueError(
            "Unsupported Arachne parameter selection method: "
            f"{args.arachne_parameter_selection}"
        )

    masks = {}
    layer_scores = {}
    offset = 0
    for layer_index, layer in zip(layer_indices, target_layers):
        num_weights = layer.weight.numel()
        in_layer = (
            (selected_global_indices >= offset)
            & (selected_global_indices < offset + num_weights)
        )
        local_indices = selected_global_indices[in_layer] - offset
        weight_mask = torch.zeros(
            num_weights, dtype=torch.bool, device=layer.weight.device
        )
        weight_mask[local_indices] = True
        weight_mask = weight_mask.view_as(layer.weight)
        bias_mask = (
            torch.zeros_like(layer.bias, dtype=torch.bool)
            if layer.bias is not None else None
        )
        masks[layer_index] = (weight_mask, bias_mask)

        if local_indices.numel() > 0:
            local_scalar_scores = scalar_scores[in_layer]
            output_indices = torch.div(
                local_indices,
                layer.weight[0].numel(),
                rounding_mode="floor",
            )
            neuron_scores = torch.zeros(
                layer.weight.shape[0],
                device=local_scalar_scores.device,
                dtype=local_scalar_scores.dtype,
            )
            neuron_scores.scatter_reduce_(
                0, output_indices, local_scalar_scores,
                reduce="amax", include_self=True,
            )
            layer_scores[layer_index] = neuron_scores.max().item()
        offset += num_weights

    if not layer_scores:
        raise RuntimeError("Original Arachne did not select any repair weight.")

    selected_counts = {
        layer_index: int(masks[layer_index][0].sum().item())
        for layer_index in layer_scores
    }
    if args.arachne_parameter_selection in (
        "random_matched", "random_layer_matched"
    ):
        order_values = selected_counts
        ordered_layers = sorted(
            layer_scores,
            key=lambda index: (
                -selected_counts[index], layer_indices.index(index)
            ),
        )
    elif args.arachne_layer_order == "max_neuron_score":
        order_values = layer_scores
        ordered_layers = sorted(
            layer_scores,
            key=lambda index: (
                -layer_scores[index], layer_indices.index(index)
            ),
        )
    elif args.arachne_layer_order == "selected_parameter_count":
        order_values = selected_counts
        ordered_layers = sorted(
            layer_scores,
            key=lambda index: (
                -selected_counts[index],
                -layer_scores[index],
                layer_indices.index(index),
            ),
        )
    else:
        raise ValueError(
            "Unsupported Original-Arachne layer ordering method: "
            f"{args.arachne_layer_order}"
        )
    return ordered_layers, masks, order_values, candidate_count, num_negative

