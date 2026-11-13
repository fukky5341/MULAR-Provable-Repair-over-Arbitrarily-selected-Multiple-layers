import warnings; warnings.filterwarnings("ignore")
import gc
from experiments import mnist
from experiments.base import *
import sytorch as st
import numpy as np
import sys, argparse
from timeit import default_timer as timer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from net.util import get_net_aprnn
from input_space.dataset import Dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, dest='dataset', action='store',
                    default='MNIST',
                    choices=['MNIST', 'CIFAR10'],
                    help='Dataset to use in experiments.')
parser.add_argument('--net', '-n', type=str, dest='net', action='store',
                    default='mnist 9x100',
                    choices=['mnist 9x100', 'mnist_256x4', 'mnist_256x6', 'mnist_conv', 'cifar10_S', 'cifar10_M', 'cifar10_L'],
                    help='Networks to repair in experiments (if applicable).')
parser.add_argument('--repair_task', type=str, dest='repair_task', action='store',
                    default='Local Counterexample',
                    choices=['Corruption + Perturbation', 'Adversarial + Perturbation', 'Local Robustness', 'Local Counterexample'],
                    help='Repair task to perform in experiments.')
parser.add_argument('--num_runs', type=int, dest='num_runs', action='store',
                    default=1,
                    help='Number of runs to perform for each experiment.')
parser.add_argument('--ndims', type=int, dest='ndims', action='store', required=True,
                    help='Number of pixels or groups.')
parser.add_argument('--k', type=int, dest='k', action='store',
                    default=0,
                    help='k.')
parser.add_argument('--seed', type=int, dest='seed', action='store',
                    default=-1,
                    help='seed.')
parser.add_argument('--img_seed', type=int, dest='img_seed', action='store',
                    default=-1,
                    help='img_seed.')
parser.add_argument('--pick', type=str, dest='pick', action='store',
                    default='nonzero',
                    choices=['leading', 'center', 'nonzero', 'random', 'grouped', 'grouped_block', 'grouped_row'],
                    help='pick.')
parser.add_argument('--eps', type=float, dest='eps', action='store', required=True,
                    help='epsilon for L^\infty norm.')
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu',
                    help='device to use, e.g., cuda, cuda:0, cpu. (default=cpu).')

args = parser.parse_args()  # normalize eps for pixel values in [0,1]

path = f"result/aprnn/{args.net}/{args.repair_task}"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
log_path = f"{path}/eps{int(args.eps)}_{args.pick}_N1_ndims{args.ndims}.log"
log_file = open(log_path, 'w')
from datetime import datetime
now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
log_info = []
log_info.append(f"Time: {now_time} \n")
log_info.append(f"Repair dataset {args.dataset} \n")
log_info.append(f"Repair task {args.repair_task} \n")
log_info.append(f"Repair net {args.net} \n")
log_info.append(f"Repair num runs {args.num_runs} \n")
log_info.append(f"Repair num (misclassified) 1 \n")
log_info.append(f"Repair ndims {args.ndims} \n")
if args.dataset in ['MNIST', 'CIFAR10', 'GTSRB']:
    log_info.append(f'For {args.dataset} dataset, eps div 255, e.g., {args.eps} ---> {args.eps}/255 \n')
    args.eps = args.eps / 255.
log_info.append(f"Repair eps {args.eps} \n")

device = get_device(args.device)
dtype = st.float64

input_flatten_shape_map = {
    "mnist 9x100": (True, (784,)),
    "mnist_256x4": (True, (784,)),
    "mnist_256x6": (True, (784,)),
    "mnist_conv": (False, (1, 28, 28)),
    "cifar10_S": (False, (3, 32, 32)),
    "cifar10_M": (False, (3, 32, 32)),
    "cifar10_L": (False, (3, 32, 32)),
}

class dummyArgs:
    def __init__(self, ori_args, dataset, model_name, device, dtype):
        if dataset == 'MNIST':
            self.dataset = Dataset.MNIST
        elif dataset == 'CIFAR10':
            self.dataset = Dataset.CIFAR10
        else:
            raise NotImplementedError(f"Dataset {dataset} not supported yet.")
        self.model_name = model_name
        self.repair_task = ori_args.repair_task
        self.device = device
        self.dtype = dtype
        self.input_flatten = input_flatten_shape_map.get(model_name, (True, (784,)))[0]
        self.input_shape = input_flatten_shape_map.get(model_name, (784,))[1]
        self.eps = ori_args.eps
        self.target_label = None

# dnn = mnist.model(args.net).to(device,dtype)
dum_args = dummyArgs(args, args.dataset, args.net, device, dtype)
dnn, _norm, _denorm = get_net_aprnn(dum_args)


def sample_annulus_points(center, inner_eps, outer_eps, num_points):
    """
    Sample points:
        inner_eps < ||x - center||_inf <= outer_eps
    """
    torch.manual_seed(0)  # for reproducibility
    device = center.device
    inp_shape = center.shape[1:]
    dim = center.numel()

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

def gen_data(eps, counterx, label):
    denom = 255.
    pixel_vals = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    gen_distance_list = [eps + (v / denom) for v in pixel_vals]

    gen_dataloader_list = []
    data_num = 50

    for gen_distance in gen_distance_list:
        random_points = sample_annulus_points(counterx, eps, gen_distance, data_num)
        gen_dataloader = torch.utils.data.DataLoader(
                                torch.utils.data.TensorDataset(random_points, torch.full((data_num,), label, device=counterx.device, dtype=torch.long)),
                                batch_size=100, shuffle=False)
        gen_dataloader_list.append((gen_distance, gen_dataloader))
    
    return gen_dataloader_list

def gen_accuracy(dnn, dataloader):
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

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def facets(ndim):
    # assert vbox.shape[0] == 2**ndim
    vind = np.arange(2**ndim, dtype=int).reshape((2,)*ndim)
    for dim in range(ndim):
        facet_idx = [st.as_slice[:],]*ndim
        facet_idx[dim] = st.as_slice[0]
        yield vind[tuple(facet_idx)].flatten()
        facet_idx[dim] = st.as_slice[1]
        yield vind[tuple(facet_idx)].flatten()

def partition_by_facets(dnn, ndim, vboxes, centers, flatten=True):
    assert vboxes.shape[0] == 1
    # centers = ...
    facet_indices = facets(ndim)
    vboxes_aps = np.empty(vboxes.shape[:2], dtype=object)
    # vboxes_og_aps = dnn.activation_pattern(vboxes.flatten(0,1)).reshape(vboxes.shape[:2])
    for vbox, center, aps in zip(vboxes, centers, vboxes_aps):
        for facet_idx in facet_indices:
            facet = vbox[facet_idx]
            # facet_og_aps = og_aps[facet_idx]
            # facet_aps = aps[facet_idx]

            facet_ref = (facet.mean(0) + center) / 2.
            facet_ref_ap = dnn.activation_pattern(facet_ref[None])

            for idx in facet_idx:
                if aps[idx] is None:
                    aps[idx] = facet_ref_ap
                else:
                    aps[idx] = st.meet_patterns(aps[idx], facet_ref_ap)


    out_ap = []
    example_ap = vboxes_aps.item(0)
    for i in range(len(example_ap)):
        if example_ap[i] == []:
            out_ap.append([])
        else:
            out_ap.append(
                np.concatenate([
                    ap[i]
                    for ap in vboxes_aps.reshape(-1, *vboxes_aps.shape[2:])
                ], 0)
            )
            if not flatten:
                out_ap[-1] = out_ap[-1].reshape(
                    *vboxes_aps.shape[:2], *out_ap[-1].shape[1:]
                )

    return out_ap

def foo(N, vpolytopes, lb=-3., ub=3., centers=None):
    N.requires_symbolic_(False).to(None).repair(False)
    solver = st.GurobiSolver()
    solver.solver.Params.BarConvTol = 1e-2
    N.to(solver).repair().requires_symbolic_weight_and_bias(lb=lb, ub=ub)
    if centers is None:
        sy_vpolytopes = N.v(vpolytopes)
    else:
        print('partition facets')
        vertices = vpolytopes.flatten(0, 1)
        vertices_ap = partition_by_facets(N, args.ndims, vpolytopes, centers)
        # import pdb; pdb.set_trace()
        sy_vertices = N(vertices, pattern=vertices_ap)
        sy_vpolytopes = sy_vertices.reshape(*vpolytopes.shape[:2], *sy_vertices.shape[1:])
    return sy_vpolytopes, solver

def vpoly_repair(dnn, vpolytopes, labels, s, k, lb=-3., ub=3., Method=-1, centers=None):
    N = dnn.deepcopy()
    for l0, l1 in s:
        if l0 == l1:
            continue
        N.requires_symbolic_(False).to(None).repair(False)
        if centers is not None:
            centers_upto_here = N[:l0](centers)
        else:
            centers_upto_here = None
        sy_vpolytopes, solver = foo(N[l0:l1], N[:l0].v(vpolytopes), lb=lb, ub=ub, centers=centers_upto_here)
        param_delta = N[l0:l1].parameter_deltas(concat=True)
        output_delta = (sy_vpolytopes - dnn[:l1].v(vpolytopes)).reshape(-1)
        delta = param_delta.norm_ub('linf+l1_normalized') + output_delta.norm_ub('linf+l1_normalized')
        # assert solver.solve(minimize=delta, Method=Method)
        solver_result = solver.solve(minimize=delta, Method=Method)
        if not solver_result:
            print(f"Solver failed at layers {l0} to {l1}.")
            # free up memory
            solver.solver.dispose()
            del solver
            del sy_vpolytopes
            del param_delta
            del output_delta
            del delta

            return None
        N[l0:l1].update_().requires_symbolic_(False).to(None).repair(False)

    N.requires_symbolic_(False).to(None).repair(False)
    if centers is not None:
        centers_upto_here = N[:k](centers)
    else:
        centers_upto_here = None
    sy_vpolytopes, solver = foo(N[k:], N[:k].v(vpolytopes), lb=lb, ub=ub, centers=centers_upto_here)
    param_delta = N[k:].parameter_deltas(concat=True)
    output_delta = (sy_vpolytopes - dnn.v(vpolytopes)).reshape(-1)
    delta = param_delta.norm_ub('linf+l1_normalized') + output_delta.norm_ub('linf+l1_normalized')
    # assert solver.solve(sy_vpolytopes.argmax(-1) == labels, minimize=delta, Method=Method)
    solver_result = solver.solve(sy_vpolytopes.argmax(-1) == labels, minimize=delta, Method=Method)
    if not solver_result:
        print(f"Solver failed at layers {k} to {len(N)}.")
        # free up memory
        solver.solver.dispose()
        del solver
        del sy_vpolytopes
        del param_delta
        del output_delta
        del delta

        return None
    N[k:].update_().requires_symbolic_(False).to(None).repair(False)

    return N

if args.repair_task == 'Local Robustness':
    data = torch.load(f"safe_radii/adv_{args.net}.pt")
    data_keys = list(data.keys())
else:
    raise NotImplementedError(f"Repair task {args.repair_task} not supported yet.")

testset = mnist.datasets.Dataset('identity', 'test').reshape(784).to(device,dtype)
correctset, buggyset = testset.filter_misclassified(dnn)
# # debug
# for i in range(100):
#     bug_id = buggyset.indices[i]
#     image = buggyset.images[bug_id]
#     label = buggyset.labels[bug_id]
#     pred = dnn(image.unsqueeze(0)).argmax(dim=1).item()
#     print(f"Buggy data id: {bug_id}, label: {label}, pred: {pred}")

log_file.write(''.join(log_info))

for run in range(args.num_runs):
    log_info = []
    log_info.append(f"****** Run {run+1}/{args.num_runs} ******\n")

    if args.repair_task == 'Local Robustness':
        bug_id = data_keys[run]
        item = data[bug_id]
        images = item['center'].unsqueeze(0).to(device, dtype)
        labels = torch.tensor([item['label']])
    else:
        raise NotImplementedError(f"Repair task {args.repair_task} not supported yet.")

    # bug_id = buggyset.indices[run]
    # images, labels = buggyset.images[bug_id].unsqueeze(0), buggyset.labels[bug_id]  # single data point, (1, C), (1,)
    # print(images, labels)
    print(f"original prediction: {dnn(images).argmax(dim=1).item()}")
    print(f"buggy data id: {bug_id}")
    log_info.append(f"buggy data id: {bug_id} \n")


    gen_dataloader_list = gen_data(args.eps, images, labels.item())


    if args.pick == 'grouped':
        pixel_indices = np.arange(784, dtype=int)
        # np.random.default_rng(args.seed).shuffle(pixel_indices)
        if args.seed >= 0:
            print(f"shuffle with seed {args.seed}")
            np.random.default_rng(args.seed).shuffle(pixel_indices)
        dim_groups = np.array_split(pixel_indices, args.ndims)
        print("dim groups:", [
            np.sort(arr)
            for arr in dim_groups
        ])
        vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups)[None]
        inner_vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps/2., groups=dim_groups)[None]

    elif args.pick == 'grouped_row':
        a = np.arange(28*28,dtype=int).reshape(28,28)
        b = list(np.ndindex(4,4))

        pixel_indices = np.arange(28, dtype=int)
        if args.seed >= 0:
            print(f"shuffle with seed {args.seed}")
            np.random.default_rng(args.seed).shuffle(pixel_indices)

        dim_groups = np.array_split(pixel_indices, args.ndims)

        dim_groups = [
            np.concatenate(
                [
                    # a[b[idx][0]*7:b[idx][0]*7+7, b[idx][1]*7:b[idx][1]*7+7]
                    a[idx]
                    for idx in arr
                ]
            ).reshape(-1)
            for arr in dim_groups
        ]

        print("dim groups:", [
            np.sort(arr)
            for arr in dim_groups
        ])
        assert np.unique(np.concatenate(dim_groups)).size == 784
        vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups)[None]
        inner_vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps/2., groups=dim_groups)[None]

    elif args.pick == 'grouped_block':
        a = np.arange(28*28,dtype=int).reshape(28,28)
        b = list(np.ndindex(4,4))

        pixel_indices = np.arange(4*4, dtype=int)
        if args.seed >= 0:
            print(f"shuffle with seed {args.seed}")
            np.random.default_rng(args.seed).shuffle(pixel_indices)

        dim_groups = np.array_split(pixel_indices, args.ndims)

        dim_groups = [
            np.concatenate(
                [
                    a[b[idx][0]*7:b[idx][0]*7+7, b[idx][1]*7:b[idx][1]*7+7]
                    for idx in arr
                ]
            ).reshape(-1)
            for arr in dim_groups
        ]

        print("dim groups:", [
            np.sort(arr)
            for arr in dim_groups
        ])
        vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, groups=dim_groups)[None]
        inner_vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps/2., groups=dim_groups)[None]

    else:
        if args.pick == 'leading':
            pixel_indices = list(range(args.ndims))

        elif args.pick == 'center':
            pixel_indices = crop_center(np.arange(784, dtype=int).reshape(28,28), 4, 4).reshape(-1)

        elif args.pick == 'nonzero':
            pixel_indices = st.where(images[0] != 0.)[0]

        elif args.pick == 'random':
            pixel_indices = np.random.default_rng(args.seed if args.seed != -1 else 0).choice(784, args.ndims, replace=False)

        else:
            raise NotImplementedError(f"{args.pick}")

        dims = pixel_indices[:args.ndims]
        print("dims:", dims)

        vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps, dims=dims)[None]
        inner_vpolytopes = st.points_to_vboxes(images.reshape(-1, 784), size=args.eps/2., dims=dims)[None]

    print(args)

    images = images.to(device=device,dtype=dtype)

    # vpolytopes = vpolytopes.reshape(*vpolytopes.shape[:2], 1, 28, 28)
    vpolytopes = vpolytopes.to(device,dtype)
    print(vpolytopes.shape)

    vpolytope_labels = torch.broadcast_to(labels, (*vpolytopes.shape[:-1], 1))
    print(vpolytope_labels.shape)

    s = [(0, args.k)]
    k = args.k

    start = timer()

    N = vpoly_repair(dnn, vpolytopes, vpolytope_labels, s=s, k=k, lb=-10., ub=10., Method=2,
                    # centers=images
        )

    time = timer() - start

    # result_path = (get_results_root() / 'eval_7' / f'aprnn_{args.net}_ndims={args.ndims}_eps={args.eps}_pick={args.pick}_k={k}_seed={args.seed}').as_posix()
    # N.save(
    #     (get_results_root() / 'eval_7' / f'aprnn_{args.net}_ndims={args.ndims}_eps={args.eps}_pick={args.pick}_k={k}_seed={args.seed}.pth').as_posix()
    # )
    print(f"APRNN Time: {time:.2f} seconds.")
    log_info.append(f"APRNN Time: {time:.2f} seconds.\n")


    if N is None:
        print(f"RESULT: APRNN failed to repair.")
        log_info.append(f"RESULT: APRNN failed to repair.\n")
        log_info.append('#' * 100 + '\n')
        log_info.append('\n')
        log_file.write(''.join(log_info))
        log_file.flush()
        continue

    log_info.append(f"RESULT: APRNN successfully repaired.\n")

    # evaluate repaired model
    acc0 = testset.accuracy(dnn)
    acc1 = testset.accuracy(N)
    D = acc0 - acc1
    print(f"APRNN Drawdown: {D:.2%} ({acc0:.2%} -> {acc1:.2%}).")
    log_info.append(f"APRNN Drawdown: {D:.2%} ({acc0:.2%} -> {acc1:.2%}).\n")
    # Dstr = f"{D:.2%} ({acc0:.2%} -> {acc1:.2%})"

    # generalization check
    for gen_dist, gen_dataloader in gen_dataloader_list:
        gen_acc0, gen_correct0, gen_total0 = gen_accuracy(dnn, gen_dataloader)
        gen_acc1, gen_correct1, gen_total1 = gen_accuracy(N, gen_dataloader)
        gen_D = gen_acc0 - gen_acc1
        print(f"Gen distance ({gen_dist:.4f}) Drawdown:")
        print(f"  {gen_acc0:.2%} ({gen_correct0}/{gen_total0}) -> {gen_acc1:.2%} ({gen_correct1}/{gen_total1}), D={gen_D:.2%}")
        log_info.append(f"Gen distance ({gen_dist:.4f}) Drawdown:\n")
        log_info.append(f"  {gen_acc0:.2%} ({gen_correct0}/{gen_total0}) -> {gen_acc1:.2%} ({gen_correct1}/{gen_total1}), D={gen_D:.2%}\n")
    log_info.append('#' * 100 + '\n')
    log_info.append('\n')
    log_file.write(''.join(log_info))
    log_file.flush()

    del N, vpolytopes, vpolytope_labels
    gc.collect()

    # result = {
    #     'APRNN': {
    #         'args': vars(args),
    #         'D' : Dstr,
    #         'T': 'N/A' if time is None else f'{time:.2f}',
    #     }
    # }

    # print(result)

    # np.save(result_path+".npy", result, allow_pickle=True)

    # print_msg_box(
    #     f"Experiment 7 using APRNN SUCCEED.\n"
    #     f"Saved result to {result_path}.npy"
    # )
