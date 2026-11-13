from dataclasses import dataclass
from typing import Optional
import torch
import gc
import time

from repair import args
from repair.bound import (
    check_violation_lp,
    get_concrete_bounds,
    check_violation,
    check_violation_batch,
    batched_monotone_search,
    exponential_monotone_search,
)
from input_space.region import RegionStatus, Region, RegionPair
from .logging import logging
from network_bound.bounder import IndividualBounds
from network_bound.utill import filter_important_spec


'''
Build subspace that under-approximates the feasible region.
If the output of the repaired layer is in the subspace, 
then the repaired DNN is guaranteed to satisfy the specification.

network: N
subspace for k-th layer: S^(k)
S^(k) is a subspace of N[k:], i.e., N[:k](x) in S^(k) => N(x) in OutSpec(x)

Assumption:
    given:
        - positive regions Ps (e.g., v-polytopes)
        - network N with alternating linear and ReLU layers
        - repaired layer index k and its type is linear

From here on, we take one P to make description simpler.

Steps:
- compute concrete bounds N[:k+1](P) --> lb, ub
- initialize the subspace S^(k) with the bounds lb and ub
- get gradient to enlarge the subspace S^(k)
    - gradient can be the direction towards the violated region
- itereatively enlarge the subspace S^(k) until it meet violating points (e.g., lb' -= subsp_lr*gradient, ub' += subsp_lr*gradient)
    - compute the output of N[k:](x) for x in [lb', ub'] to check whether it meets violating points
'''

@dataclass
class Subspace:
    repaired_layer_idx: int
    lb: torch.Tensor
    ub: torch.Tensor
    data_id: int
    
    # optional
    bounder: Optional[object] = None


# ==== Debug ====
def debug_subspace(stage, lb, ub, netS, spec, ini_lb=None, ini_ub=None, step=None, logfile=None):
    """
    Print useful information about the current subspace.
    """

    center = (lb + ub) / 2
    width = ub - lb

    is_violated = check_violation(netS, lb, ub, spec)

    print("\n[Subspace Debug]")
    print(f"stage: {stage}")
    if step is not None:
        print(f"iter: {step}")

    print(f"center norm: {center.norm().item():.6f}")
    print(f"avg width: {width.mean().item():.6f}")
    print(f"max width: {width.max().item():.6f}")
    print(f"min width: {width.min().item():.6f}")

    print(f"violation: {is_violated}")

    # show first few dimensions
    k = min(5, lb.shape[0])
    print("lb[:5]:", lb[:k].detach().cpu().numpy())
    print("ub[:5]:", ub[:k].detach().cpu().numpy())

    # show the number of dimensions whose bounds are better than the initial bounds
    if ini_lb is not None and ini_ub is not None:
        better_lb = (lb < ini_lb).sum().item()
        better_ub = (ub > ini_ub).sum().item()
        print(f"dimensions with better lb: {better_lb}")
        print(f"dimensions with better ub: {better_ub}")

    if logfile is not None:
        with open(logfile, "a") as f:
            f.write("\n[Subspace Debug]\n")
            f.write(f"stage: {stage}\n")
            if step is not None:
                f.write(f"iter: {step}\n")

            f.write(f"center norm: {center.norm().item():.6f}\n")
            f.write(f"avg width: {width.mean().item():.6f}\n")
            f.write(f"max width: {width.max().item():.6f}\n")
            f.write(f"min width: {width.min().item():.6f}\n")

            f.write(f"violation: {is_violated}\n")

            k = min(5, lb.shape[0])
            f.write("lb[:5]: " + str(lb[:k].detach().cpu().numpy()) + "\n")
            f.write("ub[:5]: " + str(ub[:k].detach().cpu().numpy()) + "\n")

            if ini_lb is not None and ini_ub is not None:
                f.write(f"dimensions with better lb: {better_lb}\n")
                f.write(f"dimensions with better ub: {better_ub}\n")


# ==== Update subspace ====
# --- helper ---
def _box_center_and_widths(lb, ub, center=None):
    if center is None:
        center = (lb + ub) / 2
    if center.shape != lb.shape:
        raise ValueError(
            f"Subspace center shape {tuple(center.shape)} does not match "
            f"bound shape {tuple(lb.shape)}."
        )
    width_lb = center - lb
    width_ub = ub - center
    tolerance = 1e-7
    if (width_lb < -tolerance).any() or (width_ub < -tolerance).any():
        raise ValueError("Subspace center must lie inside its bounds.")
    # Suppress insignificant negative widths caused by floating-point bounds.
    return center, width_lb.clamp_min(0), width_ub.clamp_min(0)


def scale_box(lb, ub, alpha, center=None):
    center, width_lb, width_ub = _box_center_and_widths(
        lb, ub, center=center
    )
    new_lb = center - alpha * width_lb
    new_ub = center + alpha * width_ub
    return new_lb, new_ub


def _search_monotone_candidates(
        check_candidates, low, high, *, search_method, iters, batch_size,
        tolerance, growth_factor):
    common_options = {
        "iters": iters,
        "batch_size": batch_size,
        "tolerance": tolerance,
    }
    if search_method == "binary":
        if high is None:
            raise ValueError("Binary search requires a finite upper bound")
        return batched_monotone_search(
            check_candidates, low, high, **common_options
        )
    if search_method == "exponential":
        return exponential_monotone_search(
            check_candidates,
            low,
            high,
            growth_factor=growth_factor,
            **common_options,
        )
    raise ValueError(f"Unsupported subspace search method: {search_method}")


def maximal_uniform_scale(
        netS, lb, ub, spec, alpha_max=10.0, iters=20, batch_size=5,
        search_method="binary", growth_factor=2.0, center=None):
    center, width_lb, width_ub = _box_center_and_widths(
        lb, ub, center=center
    )

    def check_candidates(candidates):
        alphas = lb.new_tensor(candidates).view(-1, *([1] * lb.ndim))
        test_lbs = center.unsqueeze(0) - alphas * width_lb.unsqueeze(0)
        test_ubs = center.unsqueeze(0) + alphas * width_ub.unsqueeze(0)
        # Tighter alpha-optimized bounds reduce unresolved candidates and can
        # avoid more expensive downstream checks.
        return check_violation_batch(
            netS, test_lbs, test_ubs, spec, optimize_alpha=True
        )

    search_high = None if search_method == "exponential" else alpha_max
    return _search_monotone_candidates(
        check_candidates,
        1.0,
        search_high,
        search_method=search_method,
        iters=iters,
        batch_size=batch_size,
        tolerance=1e-5,
        growth_factor=growth_factor,
    )


def maximal_safe_shrink(
        netS, lb, ub, spec, iters=20, batch_size=5, center=None):
    center, width_lb, width_ub = _box_center_and_widths(
        lb, ub, center=center
    )

    low = 0.0      # always safe (point)
    high = 1.0     # current box (unsafe)

    def check_candidates(candidates):
        alphas = lb.new_tensor(candidates).view(-1, *([1] * lb.ndim))
        test_lbs = center.unsqueeze(0) - alphas * width_lb.unsqueeze(0)
        test_ubs = center.unsqueeze(0) + alphas * width_ub.unsqueeze(0)
        # Tighter alpha-optimized bounds reduce unresolved candidates and can
        # avoid more expensive downstream checks.
        return check_violation_batch(
            netS, test_lbs, test_ubs, spec, optimize_alpha=True
        )

    # Shrinking always has the known safe/unsafe bracket [0, 1], so binary
    # refinement is the appropriate search regardless of expansion mode.
    best = batched_monotone_search(
        check_candidates, low, high,
        iters=iters,
        batch_size=batch_size,
        tolerance=1e-5,
    )

    new_lb = center - best * width_lb
    new_ub = center + best * width_ub

    return new_lb, new_ub, best

def per_dimension_scale(netS, lb, ub, spec, alpha_max=10.0, iters=15):
    ''' Binary search for each dimension to find the maximal scaling factor '''

    center = (lb + ub) / 2
    width = (ub - lb) / 2

    dim = lb.shape[0]
    alpha = torch.ones_like(lb)

    for i in range(dim):

        low = 1.0
        high = alpha_max
        best = 1.0

        for _ in range(iters):
            mid = (low + high) / 2

            test_lb = center.clone()
            test_ub = center.clone()

            test_lb[i] = center[i] - mid * width[i]
            test_ub[i] = center[i] + mid * width[i]

            if check_violation(netS, test_lb, test_ub, spec):
                high = mid
            else:
                best = mid
                low = mid

        alpha[i] = best

    new_lb = center - alpha * width
    new_ub = center + alpha * width

    return new_lb, new_ub

def find_worst_point(netS, lb, ub, spec, n_samples=10):

    points = []

    # center
    center = (lb + ub) / 2
    points.append(center)

    # bounds
    points.append(lb)
    points.append(ub)

    # random samples
    for _ in range(n_samples):
        r = torch.rand_like(lb)
        points.append(lb + r * (ub - lb))

    points = torch.stack(points)

    ys = netS(points)

    vals = (spec.C @ ys.T).min(dim=0).values  # per point loss

    worst_idx = torch.argmin(vals)

    return points[worst_idx]

def violation_and_grad(netS, x, spec):

    x = x.clone().detach().requires_grad_(True)
    y = netS(x.unsqueeze(0)).squeeze(0)

    loss = spec.violation_loss(y)
    loss.backward()
    grad = x.grad.detach()

    return loss.detach(), grad

def gradient_expand(netS, lb, ub, spec,
                    subsp_lr=1e-2,
                    max_iters=20,
                    subsp_lr_decay=0.5):

    subsp_lb = lb.clone()
    subsp_ub = ub.clone()
    grad_lb, grad_ub = None, None

    for _ in range(max_iters):
        if grad_lb is None or grad_ub is None:
            # lb
            loss_lb, grad_lb = violation_and_grad(netS, subsp_lb, spec)
            # ub
            loss_ub, grad_ub = violation_and_grad(netS, subsp_ub, spec)

        # normalize direction
        # lb: pick the direction that can expand the subspace (i.e., negative components of the negative gradient)
        direction_lb = -grad_lb
        neg_direction_lb = torch.clamp(direction_lb, max=0)
        new_lb = subsp_lb + subsp_lr * (neg_direction_lb / (neg_direction_lb.abs().max().item() + 1e-8))
        # ub: pick the direction that can expand the subspace (i.e., positive components of the positive gradient)
        direction_ub = -grad_ub
        pos_direction_ub = torch.clamp(direction_ub, min=0)
        new_ub = subsp_ub + subsp_lr * (pos_direction_ub / (pos_direction_ub.abs().max().item() + 1e-8))

        if check_violation(netS, new_lb, new_ub, spec):
            subsp_lr *= subsp_lr_decay
            if subsp_lr < 1e-5:
                break
            continue

        subsp_lb = new_lb
        subsp_ub = new_ub
        grad_lb = None
        grad_ub = None

    return subsp_lb, subsp_ub


def _original_input_output(netF, region, reference):
    """Evaluate N_F at the region's original center input."""
    if region.center_point is None:
        raise ValueError(
            "Cannot derive a safe subspace center because the region has no "
            "original center_point."
        )
    point = region.center_point
    if point.shape == region.lb.shape:
        point_batch = point.unsqueeze(0)
    elif (
        point.ndim == region.lb.ndim + 1
        and point.shape[0] == 1
        and point.shape[1:] == region.lb.shape
    ):
        point_batch = point
    elif point.numel() == region.lb.numel():
        point_batch = point.reshape_as(region.lb).unsqueeze(0)
    else:
        raise ValueError(
            f"Original center shape {tuple(point.shape)} is incompatible "
            f"with region shape {tuple(region.lb.shape)}."
        )

    with torch.no_grad():
        center_output = netF(point_batch).reshape(-1)
    if center_output.numel() != reference.numel():
        raise ValueError(
            "N_F(original input) has "
            f"{center_output.numel()} values, but intermediate bounds have "
            f"{reference.numel()}."
        )
    return center_output.reshape_as(reference)

def update_subspace(netF, netS, P, 
                    use_gradient=False, use_uniform_scale=True, use_per_dim_scale=False,
                    # use_gradient=False, use_uniform_scale=False, use_per_dim_scale=True,
                    search_batch_size=5, search_method="binary",
                    search_growth_factor=2.0, debug=False, logfile=None):
    # input bounds for netF
    inF_lb, inF_ub = P.lb.clone(), P.ub.clone()
    # compute concrete bounds for the positive region
    lb, ub, bounder = get_concrete_bounds(netF, inF_lb, inF_ub)
    # check violation for initial subspace
    inS_lb, inS_ub = lb[-1].clone(), ub[-1].clone()  # input bounds for netS
    scale_center = (inS_lb + inS_ub) / 2

    if debug:
        debug_subspace("initial", inS_lb, inS_ub, netS, P.spec, ini_lb=None, ini_ub=None, logfile=logfile)


    # Use tighter alpha-optimized bounds to reduce unnecessary LP fallbacks.
    ini_violated = check_violation_lp(
        netS, inS_lb, inS_ub, P.spec, optimize_alpha=True
    )
    # initial subspace is already violated → need to shrink first
    if ini_violated:
        if debug:
            logging("Initial subspace violated → shrinking...", logfile=logfile)

        midpoint_violated = check_violation(
            netS,
            scale_center,
            scale_center,
            P.spec,
            optimize_alpha=True,
        )
        if midpoint_violated:
            original_center = _original_input_output(
                netF, P, reference=inS_lb
            )
            # Concrete bounds should contain N_F(original input). Permit only
            # tiny numerical discrepancies before projecting back into them.
            lower_error = (inS_lb - original_center).clamp_min(0).max()
            upper_error = (original_center - inS_ub).clamp_min(0).max()
            if max(lower_error.item(), upper_error.item()) > 1e-7:
                raise ValueError(
                    "N_F(original input) lies outside its propagated "
                    "intermediate bounds."
                )
            original_center = torch.maximum(
                torch.minimum(original_center, inS_ub), inS_lb
            )
            if check_violation(
                netS,
                original_center,
                original_center,
                P.spec,
                optimize_alpha=True,
            ):
                raise ValueError(
                    "N_F(original input) is not certifiably safe and cannot "
                    "anchor subspace shrinking."
                )
            scale_center = original_center
            logging(
                "Intermediate-bound midpoint is not certifiably safe; "
                "shrinking around N_F(original input) with asymmetric "
                "lower/upper widths.",
                logfile=logfile,
            )

        sub_lb, sub_ub, alpha = maximal_safe_shrink(
            netS,
            inS_lb,
            inS_ub,
            P.spec,
            batch_size=search_batch_size,
            center=scale_center,
        )

        if debug:
            debug_subspace("after shrink", sub_lb, sub_ub, netS, P.spec, step=alpha, logfile=logfile)

        # return sub_lb, sub_ub, bounder
    else:
        # initialize the subspace with the bounds
        sub_lb, sub_ub = inS_lb.clone(), inS_ub.clone()

    # ---- Stage 2: uniform scaling ----
    if use_uniform_scale:
        keep_lb, keep_ub = sub_lb.clone(), sub_ub.clone()
        alpha = maximal_uniform_scale(
            netS,
            sub_lb,
            sub_ub,
            P.spec,
            batch_size=search_batch_size,
            search_method=search_method,
            growth_factor=search_growth_factor,
            center=scale_center,
        )
        sub_lb, sub_ub = scale_box(
            sub_lb, sub_ub, alpha, center=scale_center
        )

        if debug:
            debug_subspace("after uniform scaling", sub_lb, sub_ub, netS, P.spec, keep_lb, keep_ub, step=alpha, logfile=logfile)

    # ---- Stage 1: gradient expansion ----
    if use_gradient:
        keep_lb, keep_ub = sub_lb.clone(), sub_ub.clone()
        sub_lb, sub_ub = gradient_expand(netS, sub_lb, sub_ub, P.spec)

        if debug:
            debug_subspace("after gradient expansion", sub_lb, sub_ub, netS, P.spec, keep_lb, keep_ub, logfile=logfile)

    # ---- Stage 3: per-dimension scaling ----
    if use_per_dim_scale:
        sub_lb, sub_ub = per_dimension_scale(netS, keep_lb, keep_ub, P.spec)

        if debug:
            debug_subspace("after per-dimension scaling", sub_lb, sub_ub, netS, P.spec, keep_lb, keep_ub, logfile=logfile)

    return sub_lb, sub_ub, bounder


# ==== BaB for input space division ====
def divide_input_by_bab(net, region, max_depth=10):
    boxes = [(region.lb, region.ub)] if region.lb2 is None else [(region.lb, region.ub), (region.lb2, region.ub2)]
    pair_list = []

    def can_terminate(split_counter, temp_positive_regions):
        if len(temp_positive_regions) > 0:
            return True
        elif split_counter >= max_depth:
            return True
        else:
            return False

    for box in boxes:
        split_counter = 0
        temp_negative_regions, temp_positive_regions = [], []
        split_queue = [box]
        next_queue = []
        while not can_terminate(split_counter, temp_positive_regions):
            for queue_id, box in enumerate(split_queue):
                box_lb, box_ub = box
                bounder = IndividualBounds(net, box_lb, box_ub, device=box_lb.device)
                lbs, ubs = bounder.run()
                filtered_C_list = filter_important_spec(region.spec, lbs[-1], ubs[-1])  # list shape: (spec_num, box_num)
                if filtered_C_list is None:  # just in case: we assume safe refion has already been handled
                    temp_positive_regions.append(box)

                # divide the box
                split_dim = bounder.estimate_input_importance(filtered_C_list)
                box_lb_flat = lbs[0]
                box_ub_flat = ubs[0]
                mid = (box_lb_flat[split_dim] + box_ub_flat[split_dim]) * 0.5
                lb_1, ub_1 = box_lb_flat.clone(), box_ub_flat.clone()
                lb_2, ub_2 = box_lb_flat.clone(), box_ub_flat.clone()
                ub_1[split_dim] = mid
                lb_2[split_dim] = mid

                # check the two sub-boxes
                box1_is_violated = check_violation_lp(net, lb_1, ub_1, region.spec)  # is violated
                box2_is_violated = check_violation_lp(net, lb_2, ub_2, region.spec)  # is violated

                if box1_is_violated and box2_is_violated:
                    next_queue.append((lb_1, ub_1))
                    next_queue.append((lb_2, ub_2))
                elif box1_is_violated and not box2_is_violated:
                    temp_positive_regions.append((lb_2, ub_2))
                elif not box1_is_violated and box2_is_violated:
                    temp_positive_regions.append((lb_1, ub_1))
                else:
                    raise ValueError("Unexpected case: both boxes are not violated after splitting violated box")
                
                if not box1_is_violated or not box2_is_violated:
                    temp_negative_regions.extend(next_queue)
                    temp_negative_regions.extend(split_queue[queue_id+1:])
                    break
            
            split_counter += 1
            split_queue = next_queue
            next_queue = []
        
        pair_list.append((temp_positive_regions, temp_negative_regions))
    
    return pair_list


# ==== Pairing negative and positive regions ====
def pair_negative_positive_regions(negative_regions, positive_regions, netF, netS):
    raise NotImplementedError("pair_negative_positive_regions is not implemented yet.")


# ==== Build subspace ====
def update_region_with_subspace(
        args, region, netF, netS, repaired_layer, logfile=None,
        region_role="region"):
    if region.lb.is_cuda:
        torch.cuda.synchronize(region.lb.device)
    construction_start = time.perf_counter()

    subsp_lb, subsp_ub, bounder = update_subspace(
        netF,
        netS,
        region,
        search_batch_size=args.subspace_search_batch_size,
        search_method=args.subspace_search_method,
        search_growth_factor=args.subspace_search_growth_factor,
        debug=args.debug,
        logfile=logfile,
    )
    subspace = Subspace(repaired_layer, subsp_lb, subsp_ub, region.data_id)
    subspace.bounder = bounder
    region.subspace = subspace

    if region.lb.is_cuda:
        torch.cuda.synchronize(region.lb.device)
    construction_time = time.perf_counter() - construction_start
    logging(
        f"Subspace construction time: layer={repaired_layer}, "
        f"data_id={region.data_id}, region={region_role}, "
        f"time={construction_time:.2f} seconds",
        logfile=logfile,
    )
    return construction_time

def build_safe_subspaces_pairwise(args, net, repaired_layer: int, repair_region_pairs, logfile=None):
    '''
    repaired_layer (k): index of the repaired affine layer (Li)
    net: [L0, R1, L2, R3, ..., Rk-1, Lk, Rk+1, Lk+2, Rk+3, ...]
    netF = net[:k+1]: [L0, R1, L2, R3, ..., Rk-1, Lk]
    netS = net[k+1:]: [Rk+1, ...]
    '''
    netF = net[:repaired_layer+1]
    netS = net[repaired_layer+1:]

    # For each regions, compute subspace
    for pair in repair_region_pairs:
        logging(f"Building subspace for data_id {pair.base_region.data_id}...", logfile=logfile)

        # positive (base) region
        P = pair.base_region
        if args.debug:
            logging("Positive", logfile=logfile)
        update_region_with_subspace(
            args, P, netF, netS, repaired_layer,
            logfile=logfile,
            region_role="base",
        )
        
        # negative (repaired) region
        for repaired_region_idx, Nregion in enumerate(pair.repaired_regions):
            if Nregion.status == RegionStatus.positive:  # build subspace
                if args.debug:
                    logging("Negative", logfile=logfile)
                update_region_with_subspace(
                    args, Nregion, netF, netS, repaired_layer,
                    logfile=logfile,
                    region_role=f"repaired[{repaired_region_idx}]",
                )
    
    return repair_region_pairs

def build_safe_subspaces_localregion(args, net, repair_regions, logfile=None):
    '''
    Procedure
        1. divide the input space into v-polytopes
        2. for buggy v-polytopes, assign the safe subspace
    repaired_layer (k): index of the repaired affine layer (Li)
    net: [L0, R1, L2, R3, ..., Rk-1, Lk, Rk+1, Lk+2, Rk+3, ...]
    netF = net[:k+1]: [L0, R1, L2, R3, ..., Rk-1, Lk]
    netS = net[k+1:]: [Rk+1, ...]
    '''

    repair_region_pairs = []  # list of RegionPair

    while len(repair_regions) > 0:
        input_region = repair_regions.pop()
        # step 1: divide the input space into v-polytopes
        negative_regions, positive_regions = divide_input_by_bab(net, input_region, max_depth=args.max_bab_depth)
        # step 2: for buggy v-polytopes, assign the safe subspace
        for neg_regions, pos_region in zip(negative_regions, positive_regions):
            pair_neg_regions = []
            for neg_region in neg_regions:
                Nregion = Region(
                    lb=neg_region[0],
                    ub=neg_region[1],
                    data_id=input_region.data_id,
                    status=RegionStatus.negative
                )
                pair_neg_regions.append(Nregion)
            Pregion = Region(
                lb=pos_region[0],
                ub=pos_region[1],
                data_id=input_region.data_id,
                status=RegionStatus.positive
            )
            repair_region_pairs.append(RegionPair(repaired_regions=pair_neg_regions, base_region=Pregion))
    
    return repair_region_pairs


def build_safe_subspaces(args, net, repaired_layer: int, repair_region_pairs, repair_regions_set=[], logfile=None):

    if len(repair_regions_set) > 0:
        repair_region_pairs = build_safe_subspaces_localregion(args, net, repair_regions_set, logfile=logfile)
        assert len(repair_regions_set) == 0, "repair_regions_set should be empty after building subspaces for local regions"

    return build_safe_subspaces_pairwise(args, net, repaired_layer, repair_region_pairs, logfile=logfile)
