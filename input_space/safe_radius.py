import argparse
import time
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # for imports

from repair.bound import check_violation
from input_space.dataset import Dataset
from repair.args import RepairArgs, RepairTask
from net.util import get_net
from input_space.advex import adv_dataset
from input_space.generate_input import (
    clean_points,
    maximal_safe_shrink,
)


class DummySpec:
    def __init__(self, C):
        self.C = C


def collect_safe_radii(target_ids=None, optimize_alpha=True):
    model_name = "cifar10_cnn1"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = f"log/safe_radius_log_{timestamp}.txt"

    args = RepairArgs(
        repair_task=RepairTask.LocalRobustness,
        logfile=logfile,
        model_name=model_name,
        dataset=Dataset.CIFAR10,
    )
    print(f"Using device: {args.device}")
    print(f"Alpha optimization: {'enabled' if optimize_alpha else 'disabled'}")

    net, _norm, _denorm = get_net(args)

    collect_num = 20
    args.inp_eps = 1./255.

    save_dir = "safe_radii"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"adv_{model_name}.pt")

    if os.path.exists(save_path):
        safe_input_spaces = torch.load(save_path, map_location="cpu")
        if not isinstance(safe_input_spaces, dict):
            raise ValueError(f"Expected a dictionary in {save_path}")
        print(
            f"Loaded {len(safe_input_spaces)} existing safe-radius records "
            f"from {save_path}."
        )
    else:
        safe_input_spaces = {}

    def checkpoint():
        temp_path = f"{save_path}.tmp"
        torch.save(safe_input_spaces, temp_path)
        os.replace(temp_path, save_path)

    clean_data = clean_points(args)
    samples_num = collect_num
    if target_ids:
        samples_num = max(samples_num, max(target_ids) + 1)
    adv_data = adv_dataset(
        args,
        clean_data,
        net,
        eps=args.inp_eps,
        samples_num=samples_num,
        target_label=args.target_label,
    )

    if target_ids is None:
        selected_positions = list(range(min(collect_num, len(adv_data.images))))
    else:
        selected_positions = list(target_ids)

    invalid_positions = [
        i for i in selected_positions if i < 0 or i >= len(adv_data.images)
    ]
    if invalid_positions:
        raise IndexError(
            f"Adversarial-data positions out of range: {invalid_positions}; "
            f"only {len(adv_data.images)} successful attacks were generated."
        )

    missed = []
    skipped = 0
    added = 0
    for position, i in enumerate(selected_positions, start=1):
        data_id = adv_data.indices[i].item()
        if data_id in safe_input_spaces:
            skipped += 1
            print(
                f"Skipping {position}/{len(selected_positions)}: "
                f"data index {data_id} already exists."
            )
            continue

        print(
            f"Processing {position}/{len(selected_positions)} "
            f"(data index {data_id})..."
        )
        time_start = time.time()
        x = adv_data.ori_images[i].unsqueeze(0).to(device=args.device, dtype=args.dtype)
        label = adv_data.labels[i].item()
        num_classes = net(x).shape[1]
        pre_lb = torch.clamp(x - args.inp_eps, 0.0, 1.0).squeeze(0)
        pre_ub = torch.clamp(x + args.inp_eps, 0.0, 1.0).squeeze(0)
        C = torch.eye(num_classes, device=args.device, dtype=args.dtype)
        C = C[label:label+1] - torch.cat([C[:label], C[label+1:]], dim=0)
        spec = DummySpec(C=C)
        if check_violation(
                net, pre_lb, pre_ub, spec,
                optimize_alpha=optimize_alpha):
            new_lb, new_ub, _ = maximal_safe_shrink(
                net,
                x,
                pre_lb,
                pre_ub,
                spec,
                iters=20,
                batch_size=args.subspace_search_batch_size,
                optimize_alpha=optimize_alpha,
            )
        else:
            new_lb, new_ub = pre_lb, pre_ub
        time_end = time.time()

        if new_lb is None or new_ub is None:
            missed.append((i, data_id))
            print(f"  Missed case for data index {data_id}.")
            safe_input_spaces[data_id] = {
                "label": label,
                "center": adv_data.ori_images[i].cpu(),
                "lb": None,
                "ub": None,
                "time": time_end - time_start,
            }
        else:
            print(f"  Safe radius found. Time taken: {time_end - time_start:.2f} seconds.")
            safe_input_spaces[data_id] = {
                "label": label,
                "center": adv_data.ori_images[i].cpu(),
                "lb": new_lb.cpu(),
                "ub": new_ub.cpu(),
                "time": time_end - time_start,
            }
        added += 1
        checkpoint()

    checkpoint()
    print(
        f"Updated {save_path}: added {added}, skipped {skipped}, "
        f"total {len(safe_input_spaces)} records."
    )

    # log missed cases
    if len(missed) > 0:
        print("Missed cases (order index, data index):")
        for idx, data_id in missed:
            print(f"  {idx}: {data_id}")
    else:
        print("No missed cases.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha-optimization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable alpha optimization during safe-radius checks.",
    )
    return parser.parse_args()


def main(optimize_alpha=True):
    time_start = time.time()
    collect_safe_radii(optimize_alpha=optimize_alpha)
    time_end = time.time()
    print(f"Total time taken: {time_end - time_start:.2f} seconds.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(optimize_alpha=cli_args.alpha_optimization)
    print("Safe radius collection completed.")
