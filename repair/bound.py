
import torch

from network_bound.bounder import BatchBounds, IndividualBounds
from network_bound.utill import filter_violated_spec, update_violated_mask
from LPsolver.solver import LPSolver
from .logging import logging


def get_concrete_bounds(
        net, in_lb, in_ub,
        optimize_alpha=True,
        alpha_steps=20,
        alpha_lr=1e-1,
        save_coeffs=False,
        verbose=False,
        ):
    """ returns concrete lower and upper bounds """

    bounder = IndividualBounds(
        net = net,
        lb_inp = in_lb,
        ub_inp = in_ub,
        device = in_lb.device,
    )

    lbs, ubs = bounder.run(
        optimize_alpha=optimize_alpha,
        alpha_steps=alpha_steps,
        alpha_lr=alpha_lr,
        save_coeffs=save_coeffs,
        verbose=verbose
    )

    return lbs, ubs, bounder


def get_concrete_bounds_batch(
        net, in_lbs, in_ubs,
        optimize_alpha=True,
        alpha_steps=20,
        alpha_lr=1e-1,
        verbose=False,
        ):
    """Return bounds for boxes stacked along the leading candidate axis."""
    bounder = BatchBounds(net, in_lbs, in_ubs, device=in_lbs.device)
    lbs, ubs = bounder.run(
        optimize_alpha=optimize_alpha,
        alpha_steps=alpha_steps,
        alpha_lr=alpha_lr,
        verbose=verbose,
    )
    return lbs, ubs, bounder


# === bound lp ===
def bounds_lp(net, lb, ub, spec, logfile=None):
    # get bounds from backsubstitution
    
    lbs, ubs, bounder = get_concrete_bounds(net, lb, ub)
    out_lb, out_ub = lbs[-1], ubs[-1]
    violated_mask, spec_lb = filter_violated_spec(
                C=spec.C,
                out_lb=out_lb,
                out_ub=out_ub,
            )
    if not violated_mask.any():
        logging.info("Skip LP check since no violation detected by backsubstitution.")
        del bounder
        return

    # lp
    C_lp = spec.C[violated_mask]
    solver = LPSolver(enable_repair=False)
    model, vars = solver.build_net(net, lb, ub, bounder)
    for i in range(C_lp.shape[0]):
        c = C_lp[i]
        c_np = c.detach().cpu().numpy()
        out_var = vars[-1]
        obj_expr = c_np @ out_var
        c_lb = solver.minimize(obj_expr)
        print(f"spec dimension {i}: c_lb = {c_lb}")
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"    spec dimension {i}: c_lb = {c_lb}\n")
    solver.dispose()
    del solver, bounder


# === violation check ===

def check_violation(net, lb, ub, spec, lp_check=False, optimize_alpha=True):
    '''
    returns:
        is_violated: bool
    '''
    # approximate checking by backsubstitution and dual optimization

    # --- check violation by backsubstitution bounds ---
    lbs, ubs, bounder = get_concrete_bounds(
        net, lb, ub, optimize_alpha=optimize_alpha
    )
    out_lb, out_ub = lbs[-1], ubs[-1]
    ini_C = spec.C
    violated_mask_back, spec_lb = filter_violated_spec(
                C=ini_C,
                out_lb=out_lb,
                out_ub=out_ub,
            )
    
    if not violated_mask_back.any():
        # no violation
        if lp_check:
            return False, None, bounder
        else:
            del bounder
            return False
    else:
        is_violated = violated_mask_back.any()
        if lp_check:
            return is_violated, violated_mask_back, bounder
        else:
            del bounder
            return is_violated


def check_violation_batch(net, lbs, ubs, spec, optimize_alpha=True):
    """Return a Boolean violation result for every input box in a batch."""
    all_lbs, all_ubs, bounder = get_concrete_bounds_batch(
        net, lbs, ubs, optimize_alpha=optimize_alpha
    )
    out_lb, out_ub = all_lbs[-1], all_ubs[-1]
    C = spec.C
    if C.dim() == 1:
        C = C.unsqueeze(0)
    C_pos = torch.clamp(C, min=0)
    C_neg = torch.clamp(C, max=0)
    spec_lb = (
        out_lb.unsqueeze(1) * C_pos.unsqueeze(0)
        + out_ub.unsqueeze(1) * C_neg.unsqueeze(0)
    ).sum(dim=-1)
    del bounder
    return (spec_lb < 0).any(dim=1)


def batched_monotone_search(
        check_candidates, low, high, *, iters=20, batch_size=5, tolerance=1e-5):
    """Find the largest safe value while testing ordered candidates in batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")

    best = low
    for _ in range(iters):
        candidates = torch.linspace(
            low, high, batch_size + 2, dtype=torch.float64
        )[1:-1].tolist()
        violated = torch.as_tensor(check_candidates(candidates), dtype=torch.bool).cpu()
        unsafe_indices = violated.nonzero(as_tuple=True)[0]

        if unsafe_indices.numel() == 0:
            best = candidates[-1]
            low = best
        else:
            first_unsafe = unsafe_indices[0].item()
            high = candidates[first_unsafe]
            if first_unsafe > 0:
                best = candidates[first_unsafe - 1]
                low = best

        if high - low < tolerance:
            break
    return best


def exponential_monotone_search(
        check_candidates, low, high, *, iters=20, batch_size=5,
        tolerance=1e-5, growth_factor=2.0):
    """Bracket the safe boundary exponentially, then refine the bracket.

    For a positive safe lower bound, candidates grow multiplicatively from
    that bound (for example, 1, 2, 4, 8). When the lower bound is zero, the
    first positive candidate is chosen so one batch geometrically spans the
    interval up to ``high``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if iters < 1:
        raise ValueError("iters must be at least one")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be greater than one")
    if low < 0 or (high is not None and high < low):
        raise ValueError(
            f"Exponential search requires low >= 0 and high >= low; got "
            f"low={low}, high={high}."
        )
    if high is None and low <= 0:
        raise ValueError(
            "Unbounded exponential search requires a positive safe lower bound"
        )
    if high is not None and high - low < tolerance:
        return low

    best = low
    current = low
    expansion_rounds = 0
    bracket_low = None
    bracket_high = None

    while expansion_rounds < iters:
        if current > 0:
            raw_candidates = [
                current * growth_factor ** exponent
                for exponent in range(1, batch_size + 1)
            ]
        else:
            # A finite upper bound is required only for zero-based searches.
            first = high / growth_factor ** max(batch_size - 1, 1)
            raw_candidates = [
                first * growth_factor ** exponent
                for exponent in range(batch_size)
            ]

        candidates = []
        for candidate in raw_candidates:
            candidate = float(candidate)
            if high is not None:
                candidate = min(candidate, float(high))
            if candidate > current and (
                    not candidates or candidate > candidates[-1]):
                candidates.append(candidate)
            if high is not None and candidate >= high:
                break

        if not candidates:
            return best

        violated = torch.as_tensor(
            check_candidates(candidates), dtype=torch.bool
        ).cpu()
        if violated.numel() != len(candidates):
            raise ValueError(
                "check_candidates must return one result per candidate"
            )
        expansion_rounds += 1
        unsafe_indices = violated.nonzero(as_tuple=True)[0]

        if unsafe_indices.numel() > 0:
            first_unsafe = unsafe_indices[0].item()
            bracket_high = candidates[first_unsafe]
            bracket_low = (
                current if first_unsafe == 0
                else candidates[first_unsafe - 1]
            )
            best = bracket_low
            break

        best = candidates[-1]
        current = best
        if high is not None and current >= high:
            return best

    if bracket_low is None or bracket_high is None:
        return best

    remaining_iters = iters - expansion_rounds
    if remaining_iters <= 0 or bracket_high - bracket_low < tolerance:
        return best
    return batched_monotone_search(
        check_candidates,
        bracket_low,
        bracket_high,
        iters=remaining_iters,
        batch_size=batch_size,
        tolerance=tolerance,
    )

def check_violation_lp(
        net, lb, ub, spec, logfile=None, debug=True, optimize_alpha=True):
    '''
    return is_violated: bool
    '''
    is_violated, violated_mask, bounder = check_violation(
        net, lb, ub, spec, lp_check=True, optimize_alpha=optimize_alpha
    )
    if not is_violated:
        print("No violation detected by backsubstitution or dual optimization.")
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write("  No violation detected by backsubstitution or dual optimization.\n")
        del bounder
        return False
    
    violated_dim = violated_mask.nonzero(as_tuple=True)[0]

    # Check the closest output competitor first.  For classification specs,
    # identify each row's competitor from its negative coefficient and rank it
    # by y_predicted - y_competitor at the midpoint of the input box.
    with torch.no_grad():
        midpoint = (lb + ub) / 2
        midpoint_output = net(midpoint.unsqueeze(0)).squeeze(0)
        predicted_class = midpoint_output.argmax().item()
        if spec.target_label is not None:
            competitor_classes = spec.C[violated_dim].argmin(dim=1)
            candidate_margins = (
                midpoint_output[predicted_class]
                - midpoint_output[competitor_classes]
            )
        else:
            # Generic linear specifications (for example, ACAS Xu) have no
            # competitor classes, so rank their constraint margins directly.
            competitor_classes = None
            candidate_margins = (spec.C @ midpoint_output)[violated_dim]
        order = torch.argsort(candidate_margins)
        violated_dim = violated_dim[order]
        if competitor_classes is not None:
            competitor_classes = competitor_classes[order]

    order_description = f"spec dimensions {violated_dim.tolist()}"
    if competitor_classes is not None:
        order_description += (
            f" (predicted class {predicted_class}, competitor classes "
            f"{competitor_classes.tolist()})"
        )
    print(f"Potential violation detected by backsubstitution/dual optimization. Checking {order_description} with LP solver in ascending margin order...")
    if logfile is not None:
        with open(logfile, "a") as f:
            f.write(f"  Potential violation detected by backsubstitution/dual optimization. Checking {order_description} with LP solver in ascending margin order...\n")
    
    # lp check
    solver = LPSolver(enable_repair=False)
    _, vars = solver.build_net(net, lb, ub, bounder)
    for dim in violated_dim:
        c = spec.C[dim.item()]
        c_np = c.detach().cpu().numpy()
        out_var = vars[-1]
        obj_expr = c_np @ out_var
        c_lb = solver.minimize(obj_expr)
        if c_lb < 0:
            if debug:
                print(f"dimension {dim.item()} is violated with LP check: c_lb = {c_lb}")
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"    spec dimension {dim.item()}: c_lb = {c_lb}\n")
            solver.dispose()
            del solver, bounder
            return True
        else:
            if debug:
                print(f"dimension {dim.item()} is NOT violated with LP check: c_lb = {c_lb}")
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"    spec dimension {dim.item()}: c_lb = {c_lb}\n")
    solver.dispose()
    del solver, bounder
    return False
