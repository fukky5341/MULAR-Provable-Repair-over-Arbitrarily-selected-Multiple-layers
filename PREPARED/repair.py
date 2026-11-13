import torch
import torch.nn as nn
import copy
import time
import os
import gc
import random

from net.util import get_net
from input_space.generate_input import repair_regions
from repair.bound import ( get_concrete_bounds, check_violation_lp, bounds_lp )
from repair.args import RepairArgs
from repair.logging import completed_result_exists, logging
from .prepared_lp import PreparedLP
from repair.repair import evaluation_data, accuracy


def is_dnn_repaired(args: RepairArgs, dnn: nn.Module, repair_region_pairs):
    logging("Checking violation on the whole repaired DNN", logfile=args.logfile)
    is_violated = False
    for pair in repair_region_pairs:
        # pregion
        Pregion = pair.base_region
        p_is_violated = False
        # nregion
        n_is_violated = False
        for _nid, Nregion in enumerate(pair.repaired_regions):
            logging(f"  repaired region: data id {Nregion.data_id} ({_nid + 1}/{len(pair.repaired_regions)})", logfile=args.logfile)
            Nlb, Nub = Nregion.lb, Nregion.ub
            n_is_violated = n_is_violated or check_violation_lp(dnn, Nlb, Nub, Nregion.spec, args.logfile, debug=args.debug)
        
        logging(f"Status check after repair for data id {Pregion.data_id}: ")
        if p_is_violated:
            logging(f"  Warning: Base region with data id {Pregion.data_id} is violating the specification after repair.", logfile=args.logfile)
            raise ValueError(f"Base region with data id {Pregion.data_id} is violating the specification after repair.")
        else:
            logging(f"  Base region is still satisfying.", line_break=0, logfile=args.logfile)
        if n_is_violated:
            logging(f"  Repaired region is still violating.",line_break=0, logfile=args.logfile)
        else:
            logging(f"  Repaired region is no longer violating.", line_break=0, logfile=args.logfile)
        pair_is_violated = p_is_violated or n_is_violated

        is_violated = is_violated or pair_is_violated
    
    if is_violated:
        return False
    else:
        return True


def repair_layer(args: RepairArgs, dnn: nn.Module, layer_idx: int, repair_region_pairs):
    fixed_net = dnn[:layer_idx]
    repaired_net = dnn[layer_idx:]
    solver = PreparedLP(
        repaired_net,
        args.modifiable_range,
        enable_repair=True,
        soft_mem_limit_gb=args.gurobi_soft_mem_limit_gb,
    )
    if solver.soft_mem_limit_gb is None or solver.soft_mem_limit_gb <= 0:
        memory_limit_message = "Gurobi soft memory limit: disabled"
    else:
        memory_limit_message = (
            f"Gurobi soft memory limit: {solver.soft_mem_limit_gb:.2f} GB"
        )
    logging(memory_limit_message, logfile=args.logfile)

    # set modifiable variables
    solver.init_repair_vars(repaired_net)

    time_build_start = time.time()

    for pair in repair_region_pairs:
        Nregion_set = pair.repaired_regions
        Pregion = pair.base_region

        # -- Nregion --
        for Nregion in Nregion_set:
            Nlb, Nub = Nregion.lb, Nregion.ub
            # concrete input bounds for repaired net
            Nlbs_f, Nubs_f, _ = get_concrete_bounds(fixed_net, Nlb, Nub)
            Ninlb_r, Ninub_r = Nlbs_f[-1], Nubs_f[-1]
            # bounds for repaired net
            _, _, Nbounder = get_concrete_bounds(repaired_net, Ninlb_r, Ninub_r)
            # build the LP model
            model, vars = solver.build_net(repaired_net, Ninlb_r, Ninub_r, Nbounder)
            # add repair constraints for Nregion
            solver.add_repair_constraints(vars, Nregion.spec.C)
            del Nbounder
        
    # set objective
    solver.set_objective()

    logging(
        f"Gurobi model size: {solver.model.NumVars} variables, "
        f"{solver.model.NumConstrs} constraints, "
        f"{solver.model.NumNZs} nonzeros",
        logfile=args.logfile,
    )

    time_build = time.time() - time_build_start
    logging(f"LP model built ({time_build:.2f} seconds). Starting optimization...", logfile=args.logfile)

    # solve and update the repaired net
    time_solve_start = time.time()
    new_repaired_net = solver.repair_solve()
    time_solve = time.time() - time_solve_start
    failure_reason = solver.last_failure_reason

    solver.dispose()
    del solver

    if new_repaired_net is None:
        failure_detail = f": {failure_reason}" if failure_reason else ""
        logging(
            f"LP solver failed to find a solution{failure_detail} "
            f"({time_solve:.2f} seconds).",
            logfile=args.logfile,
        )
        return None
    logging(f"LP solver found a solution ({time_solve:.2f} seconds).", logfile=args.logfile)
    return fixed_net + new_repaired_net


def repair_prepared(args: RepairArgs):
    
    if args.perturbation_pick == 'nonzero':
        log_name = f"eps{int(args.inp_eps)}_N{args.num_v_polys}_ndims{args.perturbation_ndim}"
    elif args.perturbation_pick == 'all':
        log_name = f"eps{int(args.inp_eps)}_N{args.num_v_polys}"
    else:
        raise ValueError(f"Unsupported perturbation pick method: {args.perturbation_pick}")
    log_dir = f"result/prepared/{args.model_name}/{args.repair_task.name}/{args.perturbation_pick}"
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

    # inp eps
    args.inp_eps = args.inp_eps / 255.0

    repair_start_layer_idx = args.repair_start_layer_idx

    for run_idx, (run_logfile, is_complete) in enumerate(
            zip(run_logfiles, completed_runs)):
        if is_complete:
            print(f"Skipping completed result: {run_logfile}", flush=True)
            continue
        # Create every pending run file before expensive input construction.
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
            f.write(f"Modifiable range: {args.modifiable_range}\n\n")
            f.write(f"First layer to repair: {repair_start_layer_idx}\n\n")
        print(f"Initialized log file: {run_logfile}", flush=True)

    if all(completed_runs):
        return
    args.logfile = next(
        run_logfile
        for run_logfile, is_complete in zip(run_logfiles, completed_runs)
        if not is_complete
    )

    # Load network
    repaired_dnn, _norm, _denorm = get_net(args)
    original_dnn = copy.deepcopy(repaired_dnn)

    # Repaired regions and corresponding Base (positive) regions
    total_num = args.num_runs * args.num_v_polys
    repair_region_pairs_all, _ = repair_regions(
        args=args,
        dnn=repaired_dnn,
        total_num=total_num,
        normalize_input=_norm
    )  # list, list
    repair_region_pairs_list = []
    if len(repair_region_pairs_all) < total_num:
        # randomly sample pairs if the total number of pairs is smaller than num_runs * num_v_polys
        random.seed(0)
        # num_runs list, each containing num_v_polys pairs
        for _ in range(args.num_runs):
            tmp_sampled_list = random.sample(repair_region_pairs_all, args.num_v_polys)
            sampled_list = copy.deepcopy(tmp_sampled_list)
            repair_region_pairs_list.append(sampled_list)
    else:
        for i in range(args.num_runs):
            start_idx = i * args.num_v_polys
            end_idx = (i + 1) * args.num_v_polys
            repair_region_pairs_list.append(repair_region_pairs_all[start_idx:end_idx])

    for run_idx, repair_region_pairs in enumerate(repair_region_pairs_list):
        logfile = run_logfiles[run_idx]
        args.logfile = logfile
        if completed_runs[run_idx]:
            continue

        logging(f"=== Repair run {run_idx + 1}/{args.num_runs} ===", border="both", logfile=logfile)
        
        data_ids = [pair.base_region.data_id for pair in repair_region_pairs]
        logging(f"Data ids for repair regions in this run: {data_ids}", logfile=logfile)

        # repaired network
        dnn = copy.deepcopy(repaired_dnn)

        # evaluation data
        acc_dataloader, gen_dataloader_all = evaluation_data(args, repair_region_pairs)

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
        logging(f"Repairing Net[{args.repair_start_layer_idx}:]", line_break=2, border="both", logfile=logfile)

        dnn = repair_layer(args, dnn, repair_start_layer_idx, repair_region_pairs)

        # --- evaluate the repaired DNN ---
        if dnn is not None:
            logging('Evaluating the repaired DNN after repair...', border="bottom", logfile=logfile)
            is_repaired = is_dnn_repaired(args, dnn, repair_region_pairs)
            if is_repaired:
                logging("Successfully repaired the DNN!", logfile=logfile)
            else:
                logging("Repair failed.", logfile=logfile)

            # # debug
            # for pair in repair_region_pairs:
            #     Nregion_set = pair.repaired_regions
            #     for Nregion in Nregion_set:
            #         Nx = Nregion.center_point
            #         l = Nregion.target_label
            #         with torch.no_grad():
            #             original_output = original_dnn(Nx).argmax(dim=1).item()
            #             repaired_output = dnn(Nx).argmax(dim=1).item()
            #             logging(f"Debug check for data id {Nregion.data_id} (label {l}): {original_output} -> {repaired_output}", logfile=logfile)

        time_repair_end = time.time()
        logging(f"Total repair time: {time_repair_end - time_repair_start:.2f} seconds", logfile=logfile)

        # --- evaluate the repaired DNN ---
        if dnn is not None:
            logging('Evaluating the repaired DNN after repair...', border="bottom", logfile=logfile)
            # original network accuracy
            original_acc, original_correct, original_total = accuracy(original_dnn, acc_dataloader)
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

        del dnn, gen_dataloader_all, acc_dataloader
        if '_' in locals():
            del _
        gc.collect()
