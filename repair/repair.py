import torch
import torch.nn as nn
import copy
import time
import os
import gc
import random

from net.util import get_net
from input_space.generate_input import repair_regions, train_dataloader, evaluation_data
from input_space.region import RegionStatus
from LPsolver.solver import LPSolver
from FL.fl import FL_selection
from FL.arachne import original_arachne_localize
from FL.util import FLmode
from .subspace import build_safe_subspaces
from .bound import ( get_concrete_bounds, check_violation_lp, bounds_lp )
from .args import RepairArgs, RepairMode, RepairTask
from .logging import completed_result_exists, logging
from .util import RepairStatus


# ==============
# helper functions
# ==============
def accuracy(dnn, dataloader):
    dnn.eval()
    correct = 0
    total = 0
    device = next(dnn.parameters()).device

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1)  # fix here

            outputs = dnn(inputs)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc, correct, total



def check_subspace_violation(net, lb, ub, subspace, logfile=None):
    ''' return True if the subspace is violated, False if the subspace is satisfied '''
    out_lb, out_ub, _ = get_concrete_bounds(net, lb, ub)
    subspace_lb, subspace_ub = subspace.lb, subspace.ub
    lb_violation_dim = (out_lb[-1] < subspace_lb).nonzero(as_tuple=True)[0]
    ub_violation_dim = (out_ub[-1] > subspace_ub).nonzero(as_tuple=True)[0]
    if len(lb_violation_dim) > 0 or len(ub_violation_dim) > 0:
        logging(f'  The number of violating dimensions: {len(lb_violation_dim)} (lb), {len(ub_violation_dim)} (ub)', line_break=0, logfile=logfile)
        return True
    else:
        logging('  No violation detected.', line_break=0, logfile=logfile)
        return False


def is_dnn_repaired(args: RepairArgs, repaired_dnn, repaired_layer_idx, repair_region_pairs):
    '''
    Evaluate every base and repaired region against the repaired DNN.

    return:
        - is_repaired:
            True if the repaired DNN satisfies all specifications
            False if the repaired DNN violates any specification
        - Pcenter_violation:
            True if the center point of any base region is misclassified after repair
            False otherwise

    '''
    any_region_violated = False
    base_violation_count = 0
    repaired_violation_count = 0
    Pcenter_violation = False
    has_intermediate_subspace = repaired_layer_idx < len(repaired_dnn) - 1
    sub_net = (
        repaired_dnn[:repaired_layer_idx + 1]
        if has_intermediate_subspace else None
    )

    for pair in repair_region_pairs:
        Nregions = pair.repaired_regions
        Pregion = pair.base_region
        Plb, Pub = Pregion.lb, Pregion.ub

        # The base region must remain safe after repair.
        logging("Checking base region on the whole repaired DNN", logfile=args.logfile)
        logging(f"  base region ...", logfile=args.logfile)
        p_is_violated = check_violation_lp(repaired_dnn, Plb, Pub, Pregion.spec, logfile=args.logfile, debug=args.debug)

        logging(f"Status check after repair for data id {Pregion.data_id}:", logfile=args.logfile)
        if p_is_violated:
            Pregion.status = RegionStatus.negative
            any_region_violated = True
            base_violation_count += 1
            logging(f"  Warning: Base region with data id {Pregion.data_id} is violating the specification after repair.", logfile=args.logfile)
            p_x = Pregion.center_point
            p_label = Pregion.spec.target_label
            if repaired_dnn(p_x).argmax().item() == p_label:
                logging(f"  However, the center point of the base region is still correctly classified.", logfile=args.logfile)
            else:
                logging(f"  The center point of the base region is misclassified after repair", logfile=args.logfile)
                Pcenter_violation = True
        else:
            Pregion.status = RegionStatus.positive
            logging(f"  Base region is still satisfying.", line_break=0, logfile=args.logfile)

        for nidx, Nregion in enumerate(Nregions):
            Nlb, Nub = Nregion.lb, Nregion.ub
            n_is_violated = None

            # Containment in a certified subspace is already a complete safety
            # proof for the unchanged suffix, so no whole-network LP is needed.
            if has_intermediate_subspace:
                Nsubspace = (
                    Pregion.subspace
                    if Nregion.status == RegionStatus.negative
                    else Nregion.subspace
                )
                if Nsubspace is not None:
                    logging(
                        f"Checking repaired region {nidx + 1}/{len(Nregions)} "
                        f"at repaired layer {repaired_layer_idx}...",
                        logfile=args.logfile,
                    )
                    outside_subspace = check_subspace_violation(
                        sub_net, Nlb, Nub, Nsubspace, logfile=args.logfile
                    )
                    if not outside_subspace:
                        Nregion.status = RegionStatus.positive
                        n_is_violated = False
                        logging(
                            "  Repaired-layer output is inside the certified "
                            "subspace; marking this repaired region safe and "
                            "skipping whole-network LP verification.",
                            logfile=args.logfile,
                        )

            if n_is_violated is None:
                logging(
                    f"Checking repaired region {nidx + 1}/{len(Nregions)} "
                    "on the whole repaired DNN...",
                    logfile=args.logfile,
                )
                n_is_violated = check_violation_lp(
                    repaired_dnn, Nlb, Nub, Nregion.spec,
                    logfile=args.logfile, debug=args.debug,
                )
                if n_is_violated:
                    Nregion.status = RegionStatus.negative
                    logging(
                        "  Repaired region is still violating.",
                        line_break=0, logfile=args.logfile,
                    )
                else:
                    Nregion.status = RegionStatus.positive
                    logging(
                        "  Repaired region is no longer violating.",
                        line_break=0, logfile=args.logfile,
                    )

            if n_is_violated:
                any_region_violated = True
                repaired_violation_count += 1
    
    if any_region_violated:
        logging(
            "Repaired DNN still has violations: "
            f"{base_violation_count} base region(s), "
            f"{repaired_violation_count} repaired region(s).",
            logfile=args.logfile,
        )
        return False, Pcenter_violation
    else:
        logging(
            "Repaired DNN satisfies the specification for all base and "
            "repaired regions.",
            logfile=args.logfile,
        )
        return True, Pcenter_violation

def update_repaired_layer(dnn, layer_idx, new_params):
    new_W, new_B = new_params
    layer = dnn[layer_idx]

    layer.weight.data = new_W.to(layer.weight.device)
    layer.bias.data = new_B.to(layer.bias.device)


def _snapshot_layer_parameters(dnn, layer_idx):
    """Clone only the parameters that a single repair attempt may modify."""
    layer = dnn[layer_idx]
    weight = layer.weight.detach().clone()
    bias = layer.bias.detach().clone() if layer.bias is not None else None
    return weight, bias


@torch.no_grad()
def _restore_layer_parameters(dnn, layer_idx, parameter_snapshot):
    """Roll back one layer without discarding accepted changes elsewhere."""
    layer = dnn[layer_idx]
    weight, bias = parameter_snapshot
    layer.weight.copy_(weight)
    if bias is not None:
        layer.bias.copy_(bias)


# ==============
# LP repair
# ==============

def _objective_bounder(
        args, repaired_net, region, repaired_layer_idx,
        reuse_subspace_bounder=False):
    """Reuse a same-layer subspace bounder only before any parameter update."""
    subspace = region.subspace
    if (
        reuse_subspace_bounder
        and subspace is not None
        and subspace.repaired_layer_idx == repaired_layer_idx
        and subspace.bounder is not None
    ):
        logging(
            f"Reusing subspace bounds for data_id {region.data_id}, "
            f"layer {repaired_layer_idx} (parameters unchanged).",
            logfile=args.logfile,
        )
        return subspace.bounder

    _, _, bounder = get_concrete_bounds(
        repaired_net, region.lb, region.ub, save_coeffs=False
    )
    return bounder


def repair_params_by_lp(
        args: RepairArgs, dnn, repaired_layer_idx, repair_region_pairs,
        tr_dataloader, last_layer_repair=False,
        reuse_subspace_bounders=False, parameter_masks=None):
    if not last_layer_repair:
        repaired_net = dnn[:repaired_layer_idx+1]
    else:
        repaired_net = dnn
    layer = dnn[repaired_layer_idx]
    solver = LPSolver(modifiable_range=args.modifiable_range, weight=layer.weight, bias=layer.bias)

    # fault localization
    if parameter_masks is not None:
        weight_mask, bias_mask = parameter_masks[repaired_layer_idx]
        if bias_mask is None:
            raise ValueError(
                "The LP repair solver requires a bias mask for each layer."
            )
        solver.set_parameter_mask(args, weight_mask, bias_mask)
        logging(
            f"Using fixed Original-Arachne mask for layer "
            f"{repaired_layer_idx}: {weight_mask.sum().item()}/"
            f"{weight_mask.numel()} weights and "
            f"{bias_mask.sum().item()}/{bias_mask.numel()} biases selected.",
            logfile=args.logfile,
        )
    elif args.perform_FL and not last_layer_repair:
        time_fl_start = time.time()
        logging(f"Fault localization for layer {repaired_layer_idx}...", border="bottom", logfile=args.logfile)
        weight_mask, bias_mask = FL_selection(args, dnn, repaired_net, repaired_layer_idx, layer, repair_region_pairs, tr_dataloader, last_layer_repair=last_layer_repair)
        solver.set_parameter_mask(args, weight_mask, bias_mask)
        time_fl_end = time.time()
        logging(f"  Selected {weight_mask.sum().item()}/{weight_mask.numel()} parameters for repair (weight), {bias_mask.sum().item()}/{bias_mask.numel()} parameters for repair (bias)", logfile=args.logfile)
        logging(f"  Fault localization time: {time_fl_end - time_fl_start:.2f} seconds", logfile=args.logfile)
    else:
        solver.set_repairable_parameters_all()


    logging(f'Solving the LP model to get repaired parameters for layer {repaired_layer_idx}...', border="bottom", logfile=args.logfile)

    # add constraints or objective
    time_build_start = time.time()
    if not last_layer_repair:
        for pair in repair_region_pairs:
            Nregion_set = pair.repaired_regions
            Pregion = pair.base_region

            Plb, Pub = Pregion.lb, Pregion.ub
            Psubspace = Pregion.subspace
            # build net structure in the solver
            if args.repair_mode == RepairMode.LP_hard:
                # add constraints to ensure the output of the base region is in the safe subspace
                _, _ = solver.build_net_with_subspace(repaired_net, Plb, Pub, Psubspace)
            elif args.repair_mode == RepairMode.LP_ibp:
                # add ibp constraints for the base region
                Pbounder = _objective_bounder(
                    args, repaired_net, Pregion, repaired_layer_idx,
                    reuse_subspace_bounder=reuse_subspace_bounders,
                )
                solver.set_objective_ibp(args, repaired_net, Pbounder, Pregion, Psubspace, obj_weight=args.obj_weight_positive)
                del Pbounder

            for Nregion in Nregion_set:
                if Nregion.status == RegionStatus.positive:
                    Nlb, Nub = Nregion.lb, Nregion.ub
                    Nsubspace = Nregion.subspace
                    if args.repair_mode == RepairMode.LP_hard:
                        # add constraints to ensure the output of the repaired region is in the safe subspace
                        _, _ = solver.build_net_with_subspace(repaired_net, Nlb, Nub, Nsubspace)
                    elif args.repair_mode == RepairMode.LP_ibp:
                        # add ibp constraints for the repaired region
                        Nbounder = _objective_bounder(
                            args, repaired_net, Nregion, repaired_layer_idx,
                            reuse_subspace_bounder=reuse_subspace_bounders,
                        )
                        solver.set_objective_ibp(args, repaired_net, Nbounder, Nregion, Nsubspace)
                        del Nbounder
                elif Nregion.status == RegionStatus.negative:
                    # set the objective to minimize the distance between the repaired region and the safe subspace
                    Nlb, Nub = Nregion.lb, Nregion.ub
                    _, _, Nbounder = get_concrete_bounds(repaired_net, Nlb, Nub, save_coeffs=False)
                    if args.repair_mode == RepairMode.LP_hard:
                        solver.set_objective(args, repaired_net, Nbounder, Nregion, Psubspace)
                    elif args.repair_mode == RepairMode.LP_ibp:
                        solver.set_objective_ibp(args, repaired_net, Nbounder, Nregion, Psubspace)
                    del Nbounder
                else:
                    raise ValueError(f"Unsupported region status: {Nregion.status}")
    else:
        for pair in repair_region_pairs:
            Nregion_set = pair.repaired_regions
            Pregion = pair.base_region

            # -- Pregion --
            Plb, Pub = Pregion.lb, Pregion.ub
            _, _, Pbounder = get_concrete_bounds(
                repaired_net, Plb, Pub, save_coeffs=False
            )
            solver.set_objective_lastlayer(
                args, repaired_net, Pbounder, Pregion
            )
            del Pbounder

            # -- Nregion --
            for Nregion in Nregion_set:
                Nlb, Nub = Nregion.lb, Nregion.ub
                _, _, Nbounder = get_concrete_bounds(repaired_net, Nlb, Nub, save_coeffs=False)
                solver.set_objective_lastlayer(args, repaired_net, Nbounder, Nregion)
                del Nbounder
    
    if args.obj_regularization == "l1":
        solver.build_objective_l1(
            strength_dist=args.lambda_strength_dist,
            lambda_reg=args.lambda_reg
        )
    elif args.obj_regularization == "l2":
        solver.build_objective_l2(
            strength_dist=args.lambda_strength_dist,
            lambda_reg=args.lambda_reg
        )
    else:
        raise ValueError(f"Unsupported regularization type: {args.obj_regularization}")
    
    time_build_end = time.time()
    logging(f"LP model built ({time_build_end - time_build_start:.2f} seconds). Starting optimization...", logfile=args.logfile)

    time_opt_start = time.time()
    repaired_params = solver.repair_solve()
    time_opt_end = time.time()
    logging(f"LP optimization completed.", logfile=args.logfile)
    logging(f"  time: {time_opt_end - time_opt_start:.2f} seconds", line_break=0, logfile=args.logfile)

    solver.dispose()
    del solver

    # update the repaired layer with repaired parameters
    if repaired_params is not None:
        logging(f'LP solver found repaired parameters for layer {repaired_layer_idx}. Updating the layer with repaired parameters.', logfile=args.logfile)
        update_repaired_layer(dnn, repaired_layer_idx, repaired_params)
        return RepairStatus.PROCEEDING
    else:
        logging(f"LP solver failed to find a solution for layer {repaired_layer_idx}. Skipping this layer.", logfile=args.logfile)
        return RepairStatus.FAILED


# ===================================
# repair layer
# ===================================

def repair_layer(
        args: RepairArgs, dnn, repaired_layer_idx, repair_region_pairs,
        tr_dataloader, last_layer_repair=False,
        reuse_subspace_bounders=False, parameter_masks=None):

    if not last_layer_repair:
        # --- collect safe subspaces ---
        logging(f'Collecting safe subspaces for the repaired layer {repaired_layer_idx}...', border="bottom", logfile=args.logfile)
        repair_region_pairs = build_safe_subspaces(args, dnn, repaired_layer_idx, repair_region_pairs, logfile=args.logfile)

        # debug
        if args.debug:
            logging(f'Checking subspace validation (by base region)', logfile=args.logfile)
            for pair in repair_region_pairs:
                Pregion = pair.base_region
                logging(f"data id {Pregion.data_id}...", logfile=args.logfile)
                Plb, Pub = Pregion.lb, Pregion.ub
                psubspace = Pregion.subspace
                check_subspace_violation(dnn[:repaired_layer_idx+1], Plb, Pub, psubspace, logfile=args.logfile)

            logging(f'Satisfication of subspace constraints on the repaired region before repair', logfile=args.logfile)
            for pair in repair_region_pairs:
                Pregion = pair.base_region
                for _nid, Nregion in enumerate(pair.repaired_regions):
                    logging(f"data id {Nregion.data_id}, process {(_nid + 1)}/{len(pair.repaired_regions)}", logfile=args.logfile)
                    Nlb, Nub = Nregion.lb, Nregion.ub
                    psubspace = Pregion.subspace
                    check_subspace_violation(dnn[:repaired_layer_idx+1], Nlb, Nub, psubspace, logfile=args.logfile)
    else:
        logging(f"Last layer repair: skipping safe subspace construction.", border="bottom", logfile=args.logfile)

    # --- repair operation ---
    if args.repair_mode in (RepairMode.LP_hard, RepairMode.LP_ibp):
        # --- get repaired parameters by solving the lp model ---
        repair_status = repair_params_by_lp(
            args, dnn, repaired_layer_idx, repair_region_pairs, tr_dataloader,
            last_layer_repair=last_layer_repair,
            reuse_subspace_bounders=(
                reuse_subspace_bounders and not last_layer_repair
            ),
            parameter_masks=parameter_masks,
        )
    else:
        raise ValueError(f"Unsupported repair mode: {args.repair_mode}")

    return repair_status



# ===============
# Repair test
# ===============
# layer set to repair
def repair_layer_set(
        dnn, start_layer_idx, last_layer_idx, explicit_layer_indices=None):
    if explicit_layer_indices is None:
        if start_layer_idx < 0 or last_layer_idx >= len(dnn):
            raise ValueError(
                f"Repair layer range [{start_layer_idx}, {last_layer_idx}] "
                f"is outside network indices [0, {len(dnn) - 1}]."
            )
        if start_layer_idx > last_layer_idx:
            raise ValueError(
                "repair_start_layer_idx must not exceed "
                "repair_last_layer_idx."
            )
        candidate_indices = range(start_layer_idx, last_layer_idx + 1)
    else:
        candidate_indices = list(explicit_layer_indices)
        if not candidate_indices:
            raise ValueError("Explicit repair layer indices cannot be empty.")
        if len(candidate_indices) != len(set(candidate_indices)):
            raise ValueError(
                f"Explicit repair layer indices contain duplicates: "
                f"{candidate_indices}"
            )
        invalid_indices = [
            index for index in candidate_indices
            if index < 0 or index >= len(dnn)
        ]
        if invalid_indices:
            raise ValueError(
                f"Repair layer indices {invalid_indices} are outside network "
                f"indices [0, {len(dnn) - 1}]."
            )

    layer_set = [
        index for index in candidate_indices
        if isinstance(dnn[index], (nn.Linear, nn.Conv2d))
    ]
    if explicit_layer_indices is not None and len(layer_set) != len(candidate_indices):
        non_affine_indices = [
            index for index in candidate_indices if index not in layer_set
        ]
        layer_types = [type(dnn[index]).__name__ for index in non_affine_indices]
        raise ValueError(
            f"Explicit repair layers must be Linear or Conv2d; got indices "
            f"{non_affine_indices} with types {layer_types}."
        )
    return layer_set


def _repair_with_layer_set(
        args, dnn, repair_region_pairs, tr_dataloader, layer_indices,
        parameter_masks=None):
    """Attempt one candidate layer set on the supplied isolated state."""

    repair_status = RepairStatus.PROCEEDING
    parameters_modified = False
    last_layer_idx = len(dnn) - 1
    step = 0
    completed_subrepair_loops = 0
    for subrepair_loop_idx in range(args.max_subrepair_loops):
        if repair_status == RepairStatus.REPAIRED:
            break
        completed_subrepair_loops = subrepair_loop_idx + 1
        logging(
            f"Subrepair loop {completed_subrepair_loops}/"
            f"{args.max_subrepair_loops}: modifiable layers {layer_indices}",
            border_type="=", line_break=2, border="both",
            logfile=args.logfile,
        )
        for repaired_layer_idx in layer_indices:
            logging(
                f"Repairing layer {repaired_layer_idx} with strength "
                f"{args.lambda_strength_dist}",
                line_break=2, border="both", logfile=args.logfile,
            )
            parameter_snapshot = _snapshot_layer_parameters(
                dnn, repaired_layer_idx
            )
            last_layer_repair = repaired_layer_idx == last_layer_idx
            repair_status = repair_layer(
                args, dnn, repaired_layer_idx, repair_region_pairs,
                tr_dataloader,
                last_layer_repair=last_layer_repair,
                reuse_subspace_bounders=not parameters_modified,
                parameter_masks=parameter_masks,
            )
            step += 1

            if repair_status == RepairStatus.PROCEEDING:
                logging(
                    "Evaluating the repaired DNN after repairing "
                    f"layer {repaired_layer_idx}...",
                    border="bottom", logfile=args.logfile,
                )
                is_repaired, Pcenter_violation = is_dnn_repaired(
                    args, dnn, repaired_layer_idx, repair_region_pairs,
                )
                if is_repaired:
                    logging(
                        f"Step {step}: Successfully repaired the DNN!",
                        logfile=args.logfile,
                    )
                    repair_status = RepairStatus.REPAIRED
                    break

                if Pcenter_violation:
                    logging(
                        f"Step {step}: Center point violation. "
                        "Rolling back this layer...",
                        logfile=args.logfile,
                    )
                    _restore_layer_parameters(
                        dnn, repaired_layer_idx, parameter_snapshot,
                    )
                else:
                    parameters_modified = True
                logging(
                    f"Step {step}: Layer {repaired_layer_idx} failed. "
                    "Trying the next configured layer...",
                    logfile=args.logfile,
                )

        if (
            repair_status != RepairStatus.REPAIRED
            and completed_subrepair_loops < args.max_subrepair_loops
        ):
            args.lambda_strength_dist *= args.lambda_strength_rate

    return repair_status, completed_subrepair_loops, step


def _fresh_candidate_state(original_dnn, original_repair_region_pairs):
    """Return independent state for one sequential layer-set attempt."""

    return (
        copy.deepcopy(original_dnn),
        copy.deepcopy(original_repair_region_pairs),
    )

def repair_test(args: RepairArgs):

    if args.max_subrepair_loops < 1:
        raise ValueError("max_subrepair_loops must be at least one")
    if args.subspace_search_batch_size < 1:
        raise ValueError("subspace_search_batch_size must be at least one")
    if args.subspace_search_method not in ("binary", "exponential"):
        raise ValueError(
            f"Unsupported subspace search method: "
            f"{args.subspace_search_method}"
        )
    if (
        args.subspace_search_method == "exponential"
        and args.subspace_search_growth_factor <= 1.0
    ):
        raise ValueError(
            "subspace_search_growth_factor must be greater than one for "
            "exponential search"
        )
    if args.arachne_layer_order not in (
        "max_neuron_score",
        "selected_parameter_count",
    ):
        raise ValueError(
            "Unsupported Original-Arachne layer ordering method: "
            f"{args.arachne_layer_order}"
        )
    if args.pareto_round < 1:
        raise ValueError("pareto_round must be at least one")
    if args.arachne_negative_source not in ("region", "test"):
        raise ValueError(
            "Unsupported Original-Arachne negative sample source: "
            f"{args.arachne_negative_source}"
        )
    if args.arachne_negative_sample_count < 1:
        raise ValueError(
            "arachne_negative_sample_count must be at least one"
        )
    if args.arachne_parameter_selection not in (
        "original",
        "random_matched",
        "random_layer_matched",
    ):
        raise ValueError(
            "Unsupported Arachne parameter selection method: "
            f"{args.arachne_parameter_selection}"
        )
    effective_arachne_layer_order = (
        "selected_parameter_count"
        if args.arachne_parameter_selection in (
            "random_matched", "random_layer_matched"
        )
        else args.arachne_layer_order
    )

    # --- Load network ---
    repaired_dnn, _norm, _denorm = get_net(args)
    original_dnn = copy.deepcopy(repaired_dnn)

    # --- repaired layer index ---
    repair_start_layer_idx = args.repair_start_layer_idx
    repair_last_layer_idx = args.repair_last_layer_idx
    if args.repair_layer_candidates is not None:
        if not args.repair_layer_candidates:
            raise ValueError("Candidate repair layer sets cannot be empty.")
        if any(not candidate for candidate in args.repair_layer_candidates):
            raise ValueError("Each candidate repair layer set must be non-empty.")
        candidate_layer_sets = [
            repair_layer_set(
                repaired_dnn,
                min(candidate),
                max(candidate),
                explicit_layer_indices=candidate,
            )
            for candidate in args.repair_layer_candidates
        ]
        for previous, current in zip(
                candidate_layer_sets, candidate_layer_sets[1:]):
            if not set(previous) < set(current):
                raise ValueError(
                    "Sequential candidate repair layer sets must be strict "
                    f"supersets in order; got {previous} then {current}."
                )
        modifiable_layers = candidate_layer_sets[-1]
        layer_selection_mode = "sequential_reset"
    else:
        modifiable_layers = repair_layer_set(
            repaired_dnn,
            repair_start_layer_idx,
            repair_last_layer_idx,
            explicit_layer_indices=args.repair_layer_indices,
        )
        candidate_layer_sets = [modifiable_layers]
        layer_selection_mode = (
            "explicit" if args.repair_layer_indices is not None else "range"
        )
    if args.repair_layer_candidate_max_loops is None:
        candidate_max_subrepair_loops = [
            args.max_subrepair_loops for _candidate in candidate_layer_sets
        ]
    else:
        candidate_max_subrepair_loops = list(
            args.repair_layer_candidate_max_loops
        )
        if len(candidate_max_subrepair_loops) != len(candidate_layer_sets):
            raise ValueError(
                "Candidate max-subrepair-loop counts must align with the "
                "candidate repair layer sets."
            )
        if any(limit < 1 for limit in candidate_max_subrepair_loops):
            raise ValueError(
                "Every candidate max-subrepair-loop count must be at least one."
            )
    if not modifiable_layers:
        raise ValueError("No affine layers selected for repair.")
    if len(candidate_layer_sets) > 1 and args.perform_FL:
        raise ValueError(
            "Sequential-reset layer candidates currently support repair "
            "without fault localization only."
        )

    # inp eps
    int_eps = int(args.inp_eps)
    args.inp_eps = int_eps / 255.0

    if args.perturbation_pick == 'nonzero':
        log_name = (
            f"eps{int_eps}_N{args.num_v_polys}_ndims"
            f"{args.perturbation_ndim}"
        )
    elif args.perturbation_pick == 'all':
        log_name = f"eps{int_eps}_N{args.num_v_polys}"
    else:
        raise ValueError(
            f"Unsupported perturbation pick method: {args.perturbation_pick}"
        )
    if args.perform_FL:
        if args.flmode == FLmode.ORIGINAL_ARACHNE:
            log_name += (
                f"_{args.flmode.name}_pr{args.pareto_round}"
                f"_params_{args.arachne_parameter_selection}"
                f"_order_{effective_arachne_layer_order}"
                f"_neg_{args.arachne_negative_source}"
            )
            if args.arachne_parameter_selection in (
                "random_matched", "random_layer_matched"
            ):
                log_name += f"_seed{args.arachne_random_seed}"
            if args.arachne_negative_source == "test":
                log_name += f"{args.arachne_negative_sample_count}"
        else:
            log_name += (
                f"_{args.flmode.name}{args.fl_k_ratio}_pr{args.pareto_round}"
            )
        log_dir = (
            f"result/FL/{args.model_name}/{args.repair_task.name}/"
            f"{args.perturbation_pick}"
        )
    elif layer_selection_mode == "sequential_reset":
        candidates_dir = "__".join(
            "layers_" + "_".join(str(index) for index in candidate)
            for candidate in candidate_layer_sets
        )
        log_dir = (
            f"result/ours/sequential_reset/{candidates_dir}/"
            f"{args.model_name}/{args.repair_task.name}/"
            f"{args.perturbation_pick}"
        )
    else:
        modifiable_layers_dir = "layers_" + "_".join(
            str(layer_idx) for layer_idx in modifiable_layers
        )
        log_dir = (
            f"result/ours_target/"
            f"{modifiable_layers_dir}/{args.model_name}/"
            f"{args.repair_task.name}/{args.perturbation_pick}"
        )
    result_dir = os.path.join(log_dir, log_name)
    os.makedirs(result_dir, exist_ok=True)
    run_logfiles = [
        os.path.join(result_dir, f"run{run_idx + 1}.txt")
        for run_idx in range(args.num_runs)
    ]
    completed_runs = [
        completed_result_exists(run_logfile)
        for run_logfile in run_logfiles
    ]
    for run_idx, (run_logfile, is_complete) in enumerate(
            zip(run_logfiles, completed_runs)):
        if is_complete:
            print(f"Skipping completed result: {run_logfile}", flush=True)
            continue

        # Create and close each pending run log before dataset/region
        # construction so remote jobs expose their planned outputs immediately.
        with open(run_logfile, "w") as f:
            f.write("=== Repair Test Log ===\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
            f.write(f"Run: {run_idx + 1}/{args.num_runs}\n")
            f.write(f"Repair task: {args.repair_task.value}\n")
            f.write(f"Repair mode: {args.repair_mode.value}\n")
            f.write(f"Network: {args.model_name}\n\n")
            f.write(f"Target label: {args.target_label}\n")
            f.write(f"Number of v-polytopes: {args.num_v_polys}\n")
            f.write(f"Perturbation distance (eps): {args.inp_eps}\n")
            f.write(f"Perturbation pick: {args.perturbation_pick}\n")
            if args.perturbation_pick == 'nonzero':
                f.write(f"Perturbation ndim: {args.perturbation_ndim}\n")
            f.write("\n")
            f.write(f"Repair layer selection: {layer_selection_mode}\n")
            if layer_selection_mode == "range":
                f.write(
                    f"Repair layer range: {repair_start_layer_idx} to "
                    f"{repair_last_layer_idx}\n"
                )
            elif layer_selection_mode == "sequential_reset":
                f.write(
                    "Candidate repair layer sets: "
                    f"{candidate_layer_sets}\n"
                )
                f.write(
                    "Candidate max subrepair loops: "
                    f"{candidate_max_subrepair_loops}\n"
                )
                f.write(
                    "Candidate transition policy: advance only after failure; "
                    "reset network and repair-region state before each "
                    "candidate\n"
                )
            f.write(f"Modifiable layers: {modifiable_layers}\n\n")
            f.write(
                "Subspace shrink search method: binary\n"
            )
            f.write(
                f"Subspace expansion search method: "
                f"{args.subspace_search_method}\n"
            )
            f.write(
                f"Subspace search batch size: "
                f"{args.subspace_search_batch_size}\n"
            )
            if args.subspace_search_method == "exponential":
                f.write(
                    f"Subspace search growth factor: "
                    f"{args.subspace_search_growth_factor}\n"
                )
            f.write("\n")
            f.write(f"Modifiable range: {args.modifiable_range}\n")
            f.write(
                f"Objective parameter regularization: "
                f"{args.obj_regularization}\n"
            )
            f.write(f"Objective type: {args.obj_type}\n")
            f.write(
                f"Strength of distance term in objective: "
                f"{args.lambda_strength_dist}\n"
            )
            f.write(
                f"Strength increase rate for distance term: "
                f"{args.lambda_strength_rate}\n"
            )
            f.write(
                f"Strength of regularization term: {args.lambda_reg}\n"
            )
            f.write(f"Use priority weight: {args.use_priority_weight}\n")
            f.write(
                "Objective weight for positive region "
                f"(if not using priority weight): "
                f"{args.obj_weight_positive}\n"
            )
            f.write(
                f"Maximum subrepair loops: {args.max_subrepair_loops}\n\n"
            )
            f.write(f"FL performed: {args.perform_FL}\n")
            if args.perform_FL:
                f.write(f"FL mode: {args.flmode}\n")
                if args.flmode == FLmode.ORIGINAL_ARACHNE:
                    f.write(
                        "FL layer ordering: "
                        f"{effective_arachne_layer_order}\n"
                    )
                    f.write(
                        "FL parameter selection: "
                        f"{args.arachne_parameter_selection}\n"
                    )
                    if args.arachne_parameter_selection in (
                        "random_matched", "random_layer_matched"
                    ):
                        f.write(
                            "FL random base seed: "
                            f"{args.arachne_random_seed}\n"
                        )
                    f.write(
                        "FL negative sample source: "
                        f"{args.arachne_negative_source}\n"
                    )
                    if args.arachne_negative_source == "test":
                        f.write(
                            "FL maximum negative test samples: "
                            f"{args.arachne_negative_sample_count}\n"
                        )
                    f.write(
                        "FL candidates: top 20 * negative-input count by "
                        "loss gradient\n"
                    )
                    f.write(
                        "FL selection: first "
                        f"{args.pareto_round} gradient/forward-impact "
                        "Pareto fronts\n"
                    )
                else:
                    f.write(f"FL k ratio: {args.fl_k_ratio}\n")
                    f.write(f"Pareto rounds: {args.pareto_round}\n")
            f.write("\n")
        print(f"Initialized log file: {run_logfile}", flush=True)

    if all(completed_runs):
        return
    args.logfile = next(
        run_logfile
        for run_logfile, is_complete in zip(run_logfiles, completed_runs)
        if not is_complete
    )

    # Training data is only needed by fault localization.  Avoid loading and
    # moving the full training set to the device for the normal repair path.
    tr_dataloader = train_dataloader(args) if args.perform_FL else None

    total_num = args.num_runs * args.num_v_polys

    repair_region_pairs_all, _candidate_pool_additional_time = repair_regions(
        args=args,
        dnn=repaired_dnn,
        total_num=total_num,
        normalize_input=_norm
    )  # list, list
    # Store only candidate-pool indices here.  RegionPair objects are mutable
    # during repair, so each run receives a fresh deep copy from this untouched
    # pool immediately before that run starts.
    repair_region_pair_indices_list = []
    if len(repair_region_pairs_all) < total_num:
        # randomly sample pairs if the total number of pairs is smaller than num_runs * num_v_polys
        random.seed(0)
        candidate_indices = list(range(len(repair_region_pairs_all)))
        # num_runs lists, each containing num_v_polys candidate indices
        for _ in range(args.num_runs):
            repair_region_pair_indices_list.append(
                random.sample(candidate_indices, args.num_v_polys)
            )
    else:
        for i in range(args.num_runs):
            start_idx = i * args.num_v_polys
            end_idx = (i + 1) * args.num_v_polys
            repair_region_pair_indices_list.append(
                list(range(start_idx, end_idx))
            )

    selected_additional_times = [
        sum(
            repair_region_pairs_all[pair_idx].input_construction_time
            for pair_idx in run_indices
        )
        for run_indices in repair_region_pair_indices_list
    ]

    acc_dataloader = None
    original_accuracy = None
    configured_lambda_strength_dist = args.lambda_strength_dist
    configured_max_subrepair_loops = args.max_subrepair_loops
    # --- perform repair for each set of repair regions ---
    for run_idx, run_indices in enumerate(repair_region_pair_indices_list):
        logfile = run_logfiles[run_idx]
        args.logfile = logfile
        if completed_runs[run_idx]:
            continue

        # Strength increases apply only within one repair run. Each independent
        # run must start from the value configured by the caller.
        args.lambda_strength_dist = configured_lambda_strength_dist
        repair_region_pairs = copy.deepcopy([
            repair_region_pairs_all[pair_idx]
            for pair_idx in run_indices
        ])
        logging(f"=== Repair run {run_idx + 1}/{args.num_runs} ===", border="both", logfile=logfile)

        data_ids = [pair.base_region.data_id for pair in repair_region_pairs]
        logging(f"Data ids for this repair run: {data_ids}", logfile=logfile)
        run_additional_time = selected_additional_times[run_idx]
        logging(
            f"Input construction time for this run's {len(repair_region_pairs)} "
            f"selected target instances: {run_additional_time:.2f} seconds",
            logfile=logfile,
        )
        logging(
            "Average time per selected repair region for this run: "
            f"{run_additional_time / len(repair_region_pairs):.2f} seconds",
            logfile=logfile,
        )

        # repaired network
        dnn = copy.deepcopy(repaired_dnn)
        run_modifiable_layers = modifiable_layers
        original_arachne_masks = None
        if args.perform_FL and args.flmode == FLmode.ORIGINAL_ARACHNE:
            localization_start = time.time()
            logging(
                "Running Original-Arachne fault localization once across "
                f"candidate layers {modifiable_layers}...",
                border="bottom", logfile=logfile,
            )
            (
                run_modifiable_layers,
                original_arachne_masks,
                original_arachne_layer_scores,
                original_arachne_candidate_count,
                original_arachne_negative_count,
            ) = original_arachne_localize(
                args, dnn, modifiable_layers, repair_region_pairs,
                random_seed=args.arachne_random_seed + run_idx,
            )
            selected_weight_count = sum(
                original_arachne_masks[layer_idx][0].sum().item()
                for layer_idx in run_modifiable_layers
            )
            logging(
                "Original-Arachne negative samples: "
                f"{original_arachne_negative_count} "
                f"(source={args.arachne_negative_source}).",
                logfile=logfile,
            )
            if args.arachne_parameter_selection == "random_matched":
                logging(
                    "Original-Arachne selected "
                    f"{selected_weight_count} weights from "
                    f"{original_arachne_candidate_count} high-gradient "
                    "candidates; the repair mask was replaced by the same "
                    "number of uniformly sampled weights using seed "
                    f"{args.arachne_random_seed + run_idx}.",
                    logfile=logfile,
                )
            elif args.arachne_parameter_selection == "random_layer_matched":
                logging(
                    "Original-Arachne selected "
                    f"{selected_weight_count} weights from "
                    f"{original_arachne_candidate_count} high-gradient "
                    "candidates; within every selected layer, the repair "
                    "mask was replaced by the same number of uniformly "
                    "sampled weights using seed "
                    f"{args.arachne_random_seed + run_idx}.",
                    logfile=logfile,
                )
            else:
                logging(
                    "Original-Arachne localization considered "
                    f"{original_arachne_candidate_count} high-gradient "
                    f"weights and selected {selected_weight_count} weights "
                    f"from the first {args.pareto_round} Pareto fronts.",
                    logfile=logfile,
                )
            logging(
                "Original-Arachne layer ordering values "
                f"({effective_arachne_layer_order}): "
                + ", ".join(
                    f"{layer_idx}="
                    f"{original_arachne_layer_scores[layer_idx]:.6g}"
                    for layer_idx in run_modifiable_layers
                ),
                logfile=logfile,
            )
            logging(
                f"Original-Arachne repair order: {run_modifiable_layers}",
                logfile=logfile,
            )
            logging(
                "Original-Arachne localization time: "
                f"{time.time() - localization_start:.2f} seconds",
                logfile=logfile,
            )

        if args.debug:
            # logging how much original network is violating
            logging(f"Spec lower bound of original network", border="bottom", logfile=logfile)
            for pair in repair_region_pairs:
                for _nid, Nregion in enumerate(pair.repaired_regions):
                    logging(f"  data id {Nregion.data_id} ({_nid + 1}/{len(pair.repaired_regions)})", logfile=logfile)
                    Nlb, Nub = Nregion.lb, Nregion.ub
                    _ = bounds_lp(original_dnn, Nlb, Nub, Nregion.spec, logfile=logfile)

        time_repair_start = time.time()
        
        # --- Repair simultaneously ---
        if args.repair_mode in (RepairMode.LP_hard, RepairMode.LP_ibp):
            repair_status = RepairStatus.PROCEEDING
            step = 0
            completed_subrepair_loops = 0
            attempted_candidate_count = 0
            successful_candidate_layers = None
            baseline_repair_region_pairs = repair_region_pairs
            for candidate_idx, (candidate_layers, candidate_max_loops) in enumerate(
                    zip(candidate_layer_sets, candidate_max_subrepair_loops)):
                attempted_candidate_count = candidate_idx + 1
                if candidate_idx > 0:
                    del dnn
                    logging(
                        "Previous candidate failed. Restoring the original "
                        "network and repair-region state before trying the "
                        "next candidate.",
                        border_type="=", line_break=2, border="both",
                        logfile=logfile,
                    )
                dnn, repair_region_pairs = _fresh_candidate_state(
                    repaired_dnn, baseline_repair_region_pairs
                )
                args.lambda_strength_dist = configured_lambda_strength_dist
                args.max_subrepair_loops = candidate_max_loops
                logging(
                    f"Candidate layer set {attempted_candidate_count}/"
                    f"{len(candidate_layer_sets)}: {candidate_layers} "
                    f"(max subrepair loops: {candidate_max_loops})",
                    border_type="=", line_break=2, border="both",
                    logfile=logfile,
                )
                (
                    repair_status,
                    candidate_completed_loops,
                    candidate_steps,
                ) = _repair_with_layer_set(
                    args,
                    dnn,
                    repair_region_pairs,
                    tr_dataloader,
                    candidate_layers,
                    parameter_masks=original_arachne_masks,
                )
                completed_subrepair_loops += candidate_completed_loops
                step += candidate_steps
                logging(
                    f"Candidate {candidate_layers} result: "
                    f"{'repaired' if repair_status == RepairStatus.REPAIRED else 'failed'}",
                    logfile=logfile,
                )
                if repair_status == RepairStatus.REPAIRED:
                    successful_candidate_layers = list(candidate_layers)
                    break

        time_repair_end = time.time()
        logging(f"Total repair time: {time_repair_end - time_repair_start:.2f} seconds", logfile=logfile)
        logging(
            f"Candidate layer sets attempted: {attempted_candidate_count}/"
            f"{len(candidate_layer_sets)}",
            logfile=logfile,
        )
        logging(
            f"Successful candidate layer set: {successful_candidate_layers}",
            logfile=logfile,
        )
        logging(f"Subrepair loops completed: {completed_subrepair_loops}", logfile=logfile)
        logging(f"Total steps: {step}", logfile=logfile)

        # --- evaluate the repaired DNN ---
        if repair_status == RepairStatus.REPAIRED:
            logging('Evaluating the repaired DNN after repair...', border="bottom", logfile=logfile)
            # Load the clean evaluation dataset lazily and retain its dataloader
            # across successful runs.  Generalization data remains run-specific.
            acc_dataloader, gen_dataloader_all = evaluation_data(
                args,
                repair_region_pairs,
                acc_dataloader=acc_dataloader,
            )
            # original network accuracy
            if original_accuracy is None:
                original_accuracy = accuracy(original_dnn, acc_dataloader)
            original_acc, original_correct, original_total = original_accuracy
            logging(f"Original DNN accuracy on the evaluation dataset: {original_acc:.4f} ({original_correct}/{original_total})", logfile=logfile)
            # repaired network accuracy
            repaired_acc, repaired_correct, repaired_total = accuracy(dnn, acc_dataloader)
            logging(f"Repaired DNN accuracy on the evaluation dataset: {repaired_acc:.4f} ({repaired_correct}/{repaired_total})", logfile=logfile)

            # generalization check
            for gen_id, gen_dataloaders in gen_dataloader_all:
                logging(f"Generalization check for data id {gen_id}...", logfile=logfile)
                
                for gen_dist, gen_dataloader in gen_dataloaders:
                    gen_acc0, gen_correct0, gen_total0 = accuracy(original_dnn, gen_dataloader)
                    gen_acc, gen_correct, gen_total = accuracy(dnn, gen_dataloader)
                    logging(f"Repaired DNN accuracy on the generalization dataset (dist={gen_dist:.4f}): \n{gen_acc0:.4f} ({gen_correct0}/{gen_total0}) -> {gen_acc:.4f} ({gen_correct}/{gen_total})", logfile=logfile)
        else:
            logging("Repair failed. No repaired DNN to evaluate.", logfile=logfile)
            original_acc, repaired_acc = None, None

        logging("=== Summary of Repair Results ===", border="both", logfile=logfile)
        if original_acc is not None and repaired_acc is not None:
            logging(
                f"Run {run_idx + 1}: Original accuracy = {original_acc}, "
                f"Repaired accuracy = {repaired_acc}",
                logfile=logfile,
            )
        else:
            logging(
                f"Run {run_idx + 1}: Repair failed. No accuracy results.",
                logfile=logfile,
            )

        del dnn, repair_region_pairs
        if repair_status == RepairStatus.REPAIRED:
            del gen_dataloader_all
        gc.collect()

    # Do not expose a strength increased by the final run to the caller.
    args.lambda_strength_dist = configured_lambda_strength_dist
    args.max_subrepair_loops = configured_max_subrepair_loops
