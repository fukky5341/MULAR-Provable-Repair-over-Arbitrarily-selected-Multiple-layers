import torch
import numpy as np
import copy
import time

from experiments import mnist, cifar10, gtsrb

from .region import Region, RegionStatus, RegionPair
from .dataset import Dataset
from repair.bound import (
    batched_monotone_search,
    check_violation,
    check_violation_batch,
)
from repair.args import RepairTask, RepairMode
from repair.logging import logging


def _normalize_positions(keep_pos):
    if isinstance(keep_pos, torch.Tensor):
        keep_pos = keep_pos.detach().cpu().tolist()
    elif isinstance(keep_pos, np.ndarray):
        keep_pos = keep_pos.tolist()
    return [int(i) for i in keep_pos]


def _subset_field(field, keep_pos):
    if isinstance(field, torch.Tensor):
        idx = torch.tensor(keep_pos, device=field.device, dtype=torch.long)
        return field.index_select(0, idx).clone()
    elif isinstance(field, np.ndarray):
        return field[np.array(keep_pos, dtype=np.int64)].copy()
    else:
        return [field[i] for i in keep_pos]


def subset_by_positions(points, keep_pos):
    """
    Keep entries by local positions inside the current dataset object.
    """
    keep_pos = _normalize_positions(keep_pos)

    # IMPORTANT: use deepcopy, not points.copy()
    subset = copy.deepcopy(points)

    subset.images = _subset_field(points.images, keep_pos)
    subset.labels = _subset_field(points.labels, keep_pos)

    if hasattr(points, "indices") and points.indices is not None:
        subset.indices = _subset_field(points.indices, keep_pos)

    return subset


@torch.no_grad()
def split_by_misclassification(points, dnn):
    logits = dnn(points.images)
    preds = logits.argmax(dim=1)

    if isinstance(points.labels, torch.Tensor):
        labels = points.labels.to(device=preds.device, dtype=preds.dtype)
    elif isinstance(points.labels, np.ndarray):
        labels = torch.from_numpy(points.labels).to(device=preds.device, dtype=preds.dtype)
    else:
        labels = torch.tensor(points.labels, device=preds.device, dtype=preds.dtype)

    labels = labels.reshape(-1).to(dtype=preds.dtype)

    correct_mask = (preds == labels)
    wrong_mask = ~correct_mask

    pos_idx = correct_mask.nonzero(as_tuple=True)[0]
    neg_idx = wrong_mask.nonzero(as_tuple=True)[0]

    pos_points = subset_by_positions(points, pos_idx)
    neg_points = subset_by_positions(points, neg_idx)

    return pos_points, neg_points


def train_dataset(args):
    if args.dataset == Dataset.MNIST:
        data = mnist.datasets.Dataset('identity', 'train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)
        
    elif args.dataset == Dataset.CIFAR10:
        data = cifar10.datasets.Dataset('identity', 'train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)
    
    elif args.dataset == Dataset.GTSRB:
        data = gtsrb.gtsrb_datasets.Dataset('train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)


def train_dataloader(args):
    if args.dataset == Dataset.MNIST:
        data = mnist.datasets.Dataset('identity', 'train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype).dataloader(batch_size=100, shuffle=False)
        
    elif args.dataset == Dataset.CIFAR10:
        data = cifar10.datasets.Dataset('identity', 'train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype).dataloader(batch_size=100, shuffle=False)
    
    elif args.dataset == Dataset.GTSRB:
        data = gtsrb.gtsrb_datasets.Dataset('train')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype).dataloader(batch_size=100, shuffle=False)


def clean_points(args):
    if args.dataset == Dataset.MNIST:
        data = mnist.datasets.Dataset('identity', 'test')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)
        
    elif args.dataset == Dataset.CIFAR10:
        data = cifar10.datasets.Dataset('identity', 'test')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)
    
    elif args.dataset == Dataset.GTSRB:
        data = gtsrb.gtsrb_datasets.Dataset('test')
        return data.reshape(args.input_shape).to(device=args.device, dtype=args.dtype)


def maximal_safe_shrink(
        net, center, lb, ub, spec, iters=20, batch_size=5,
        optimize_alpha=True):
    if center.shape == lb.shape:
        search_center = center
    elif center.shape[0] == 1 and center.shape[1:] == lb.shape:
        search_center = center.squeeze(0)
    else:
        raise ValueError(
            f"center shape {tuple(center.shape)} is incompatible with "
            f"bound shape {tuple(lb.shape)}"
        )

    width_lb = search_center - lb
    width_ub = ub - search_center

    if (width_lb < 0).any() or (width_ub < 0).any():
        raise ValueError("center must lie inside the supplied lower and upper bounds")

    low = 0.0      # always safe (point)
    high = 1.0     # current box (unsafe)

    def check_candidates(candidates):
        scales = lb.new_tensor(candidates).view(-1, *([1] * lb.ndim))
        test_lbs = search_center.unsqueeze(0) - scales * width_lb.unsqueeze(0)
        test_ubs = search_center.unsqueeze(0) + scales * width_ub.unsqueeze(0)
        return check_violation_batch(
            net, test_lbs, test_ubs, spec,
            optimize_alpha=optimize_alpha,
        )

    best = batched_monotone_search(
        check_candidates,
        low,
        high,
        iters=iters,
        batch_size=batch_size,
        tolerance=1e-4,
    )
    
    if best == 0.0:
        return search_center.clone(), search_center.clone(), best

    new_lb = search_center - best * width_lb
    new_ub = search_center + best * width_ub

    return new_lb, new_ub, best


# ===== construct safe region with the largest radius =====
def find_largest_safe_radius(
        dnn, center_point, spec, max_radius, iters=20, batch_size=5,
        optimize_alpha=True):
    low = 0.0
    high = max_radius
    center = center_point.squeeze(0)
    batch_step = 0

    def check_candidates(candidates):
        nonlocal batch_step
        time_step_start = time.time()
        radii = center.new_tensor(candidates).view(-1, *([1] * center.ndim))
        lbs = center.unsqueeze(0) - radii
        ubs = center.unsqueeze(0) + radii
        violated = check_violation_batch(
            dnn, lbs, ubs, spec,
            optimize_alpha=optimize_alpha,
        )
        time_step_end = time.time()
        batch_step += 1
        radii_text = ", ".join(f"{radius:.4f}" for radius in candidates)
        print(
            f"Batched search step {batch_step}/{iters}, radii=[{radii_text}], "
            f"time={time_step_end - time_step_start:.2f}s"
        )
        return violated

    return batched_monotone_search(
        check_candidates,
        low,
        high,
        iters=iters,
        batch_size=batch_size,
        tolerance=1e-4,
    )


# ===== prepare repair box based on the perturbation_pick setting =====
def prepare_repair_box(center_point, inp_eps, perturbation_pick, perturbation_ndim):
    if perturbation_pick == 'nonzero':
        flat_image = center_point.flatten()
        non_zero_indices = torch.nonzero(flat_image, as_tuple=True)[0]
        num_elements = min(perturbation_ndim, len(non_zero_indices))
        indices_to_perturb = non_zero_indices[:num_elements]
        mask = torch.zeros_like(flat_image, dtype=torch.bool)
        mask[indices_to_perturb] = True
        mask = mask.view_as(center_point)
        lb = torch.where(mask, center_point - inp_eps, center_point)
        ub = torch.where(mask, center_point + inp_eps, center_point)
    elif perturbation_pick == 'all':  # 'all'
        lb = center_point - inp_eps
        ub = center_point + inp_eps
    else:
        raise ValueError(f"Invalid perturbation_pick value: {perturbation_pick}")

    lb = torch.clamp(lb, min=0, max=1).squeeze(0)
    ub = torch.clamp(ub, min=0, max=1).squeeze(0)

    return lb, ub


# ===== construct repair regions based on the experiment setting =====

def repair_regions_adv_and_perturbation(args, dnn, total_num):
    '''
    1. Generate adversarial examples from clean points based on the given:
        - target_label
        - inp_eps (strength of perturbation)
    2. Construct pairs of repaired regions and base regions
        - repaired region: centered at the generated adversarial example, with radius inp_eps
        - base region: centered at the corresponding clean point
            - but, the input region with a radius of inp_eps includes buggy region
            - so we search for the largest radius that does not violate the specification
    '''
    device = args.device
    dtype = args.dtype

    # safe radii
    data = torch.load(f"safe_radii/adv_{args.model_name}.pt")
    collect_num = total_num if len(data) >= total_num else len(data)

    additional_time = 0.0

    pair_list = []
    for i, (data_id, item) in enumerate(data.items()):
        if i >= collect_num:
            break
        # Local robustness repairs a box around the original input.
        center_poi_neg = item['center'].unsqueeze(0).to(
            device=device, dtype=dtype
        )
        repaired_region = Region(
            center_point = center_poi_neg,
            lb = torch.clamp(center_poi_neg - args.inp_eps, min=0, max=1).squeeze(0),
            ub = torch.clamp(center_poi_neg + args.inp_eps, min=0, max=1).squeeze(0),
            target_label = item['label'],
            data_id=data_id,
            status=RegionStatus.negative
        )
        # add spec
        num_classes = dnn(center_poi_neg).shape[1]  # assuming output shape is (1, num_classes)
        repaired_region.add_spec(num_classes, repaired_region.target_label, device=device, dtype=dtype)

        # base region
        center_poi_pos = item['center'].unsqueeze(0).to(device=device, dtype=dtype)
        pair_construction_time = 0.0
        if args.repair_mode not in (RepairMode.PREPARED, RepairMode.LastLayer_ibp):
            pair_construction_time = float(item['time'])
            additional_time += pair_construction_time
        lb = item['lb'].to(device=device, dtype=dtype)
        ub = item['ub'].to(device=device, dtype=dtype)
        base_region = Region(
            center_point = center_poi_pos,
            lb = torch.clamp(lb, min=0, max=1),
            ub = torch.clamp(ub, min=0, max=1),
            target_label = item['label'],
            data_id=data_id,
            status=RegionStatus.positive
        )
        # add spec
        base_region.add_spec(num_classes, base_region.target_label, device=device, dtype=dtype)

        pair_list.append(RegionPair(
            repaired_regions=[repaired_region],
            base_region=base_region,
            input_construction_time=pair_construction_time,
        ))

    return pair_list, additional_time


def repair_regions_local_counterexample(args, dnn, total_num):
    '''
    1. find counterexample from the test set
    2. find the safe region with the center point
    3. construct repaired region based on the counterexample and the safe region
    '''
    device = args.device
    dtype = args.dtype
    clean_data = clean_points(args)
    pos_xs, neg_xs = split_by_misclassification(clean_data, dnn)

    # safe centers
    data = torch.load(f"safe_centers/{args.model_name}_num300.pt")
    if total_num > len(data):
        total_num = len(data)

    additional_time = 0.0

    pair_list = []
    for i in range(min(total_num, len(neg_xs.images))):
        counterexample = neg_xs.images[i].unsqueeze(0).to(device=device, dtype=dtype)
        target_label = neg_xs.labels[i].item()

        # add perturbation based on the perturbation_pick setting
        Nlb, Nub = prepare_repair_box(counterexample, args.inp_eps, args.perturbation_pick, args.perturbation_ndim)

        repaired_region = Region(
            center_point=counterexample,
            lb=Nlb,
            ub=Nub,
            target_label=target_label,
            data_id=neg_xs.indices[i].item(),
            status=RegionStatus.negative
        )
        repaired_region.add_spec(num_classes=dnn(counterexample).shape[1], target_label=target_label, device=device, dtype=dtype)

        pair_construction_time = 0.0
        if args.repair_mode not in (RepairMode.PREPARED, RepairMode.LastLayer_ibp):
            x_id = neg_xs.indices[i].item()
            safe_center = data[x_id]['safe_center']
            radius = data[x_id]['radius']
            timecost = data[x_id]['time']
            additional_time += timecost
            pair_construction_time = float(timecost)
        else:  # for PREPARED and LastLayer_ibp, we don't use base_region
            safe_center = counterexample
            radius = args.inp_eps

        base_region = Region(
            center_point=safe_center,
            lb=torch.clamp(safe_center - radius, min=0, max=1).squeeze(0),
            ub=torch.clamp(safe_center + radius, min=0, max=1).squeeze(0),
            target_label=target_label,
            data_id=neg_xs.indices[i].item(),
            status=RegionStatus.positive
        )
        base_region.add_spec(num_classes=dnn(safe_center).shape[1], target_label=target_label, device=device, dtype=dtype)

        pair_list.append(RegionPair(
            repaired_regions=[repaired_region],
            base_region=base_region,
            input_construction_time=pair_construction_time,
        ))

    return pair_list, additional_time


def repair_regions(args, dnn, total_num, normalize_input=None):
    '''
    returns
    - pair_list: list of RegionPair, where each RegionPair contains a repaired_region and a base_region
    - region_list: list of Region, where each Region is a repaired_region without a corresponding base_region
    '''
    if args.repair_task != RepairTask.LocalRobustness:
        raise NotImplementedError(f"Repair experiment {args.repair_task} is not implemented.")
    return repair_regions_adv_and_perturbation(args, dnn, total_num)
    
def sample_annulus_points(center, inner_eps, outer_eps, num_points):
    """
    Sample points:
        inner_eps < ||x - center||_inf <= outer_eps
    """
    torch.manual_seed(0)  # for reproducibility
    device = center.device
    inp_shape = center.shape[1:]
    dim = center.numel()

    # flatten center
    center_flat = center.view(1, -1)

    # 1. random direction in [-1, 1]
    noise = torch.rand((num_points, dim), device=device) * 2 - 1  # [-1,1]

    # 2. normalize to L_inf = 1
    noise = noise / noise.abs().max(dim=1, keepdim=True)[0]

    # 3. sample radius
    r = torch.rand((num_points, 1), device=device) * (outer_eps - inner_eps) + inner_eps

    # 4. perturbation
    perturb = noise * r

    # 5. reshape back
    perturb = perturb.view(num_points, *inp_shape)

    sampled_points = center + perturb

    # 6. clamp
    sampled_points = torch.clamp(sampled_points, 0.0, 1.0)

    return sampled_points


def evaluation_data(args, repair_region_pairs, acc_dataloader=None):
    '''
    returns
    - acc_dataloader: dataloader for accuracy evaluation, containing clean test data
    - gen_dataloader: dataloader for generalization evaluation
    '''
    if args.repair_task != RepairTask.LocalRobustness:
        raise NotImplementedError(
            f"Repair experiment {args.repair_task} is not implemented for "
            "generalization evaluation."
        )

    if acc_dataloader is None:
        c_points = clean_points(args)
        acc_dataloader = c_points.dataloader(batch_size=100, shuffle=False)

    denom = 255.
    pixel_vals = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    gen_distance_list = [args.inp_eps + (v / denom) for v in pixel_vals]

    gen_dataloader_all = []
    data_num = 50  # todo: if you change this, also change the number in APRNN/eval_7_aprnn/gen_data()
    
    for pair in repair_region_pairs:
        Nregions = pair.repaired_regions
        for region in Nregions:
            gen_dataloader_list = []
            counter_x = region.center_point  # (1, input_dim)
            label = region.target_label
            data_id = region.data_id

            # Collect random points whose distance from the center point is
            # args.inp_eps < distance <= gen_distance.
            for gen_distance in gen_distance_list:
                if gen_distance <= args.inp_eps:
                    continue

                random_points = sample_annulus_points(
                    counter_x, args.inp_eps, gen_distance, data_num
                )
                gen_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        random_points,
                        torch.full(
                            (data_num,),
                            label,
                            device=counter_x.device,
                            dtype=torch.long,
                        ),
                    ),
                    batch_size=100,
                    shuffle=False,
                )
                gen_dataloader_list.append((gen_distance, gen_dataloader))
            
            gen_dataloader_all.append((data_id, gen_dataloader_list))

    return acc_dataloader, gen_dataloader_all
