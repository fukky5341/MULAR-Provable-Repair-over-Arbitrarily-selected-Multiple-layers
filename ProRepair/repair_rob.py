import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from pr import ProvableRepiar
import time
import copy
from experiment.robustness import Robustness, robustness_cifar_model, robustness_mnist_model, robustness_gtsrb_model
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from net.util import get_net
from input_space.dataset import Dataset

parse = argparse.ArgumentParser(description='Mnist repair')    
parse.add_argument('--dataset', type=str, help='repair dataset', default='MNIST')
parse.add_argument('--model', type=str, help='repair model', default='3x100')
parse.add_argument('--repair_task', type=str, help='repair task', default='LocalCounterexample')
parse.add_argument('--num_runs', type=int, help='number of runs', default=1)
parse.add_argument('--N', type=int, help='mis_input_num for repair', default=1)
parse.add_argument('--exp', type=bool, default=False)
parse.add_argument('--eps', type=float, default=1)
parse.add_argument('--ndims', type=int, default=1)
parse.add_argument('--seed', type=int, default=0)
parse.add_argument('--refine_method', type=str, default="refine_score")
parse.add_argument('--pick', type=str, default='all', help='pick.')
parse.add_argument('--loc', type=str, default='', choices=['bottom-right', 'bottom-left', 'top-right', 'top-left'],)
parse.add_argument('--device', type=str, default="cuda:0")
parse.add_argument('--dtype', type=str, default="float32")
parse.add_argument('--target_label', type=int, default=None)
args = parse.parse_args()

if args.dtype == "float64":
    args.dtype = torch.float64
elif args.dtype == "float32":
    args.dtype = torch.float32

N = args.N
torch.set_printoptions(precision=3)
device = torch.device(args.device)
approximate_method = 'CROWN-Optimized' 
approximate_method = 'backward'

input_flatten_shape_map = {
    "mnist 9x100": (True, (784,)),
    "mnist_256x4": (True, (784,)),
    "mnist_256x6": (True, (784,)),
    "mnist_conv": (False, (1, 28, 28)),
    "cifar10_S": (False, (3, 32, 32)),
    "cifar10_M": (False, (3, 32, 32)),
    "cifar10_L": (False, (3, 32, 32)),
}

split_ind_map = {
    "mnist 9x100": 14,
    "mnist_256x4": 7,
    "mnist_256x6": 11,
    "mnist_conv": 5,  # todo: find a better split index for conv net
    # "cifar10_S": 15,
    # "cifar10_M": 15,
    # "cifar10_L": 15,
}

path = f"result/prorepair/{args.model}/{args.repair_task}"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
log_path = f"{path}/eps{int(args.eps)}_{args.pick}_N{args.N}_ndims{args.ndims}.log" if args.loc == '' else f"{path}/eps{int(args.eps)}_{args.pick}_N{args.N}_ndims{args.ndims}-{args.loc}.log"
log_file = open(log_path, 'w')
from datetime import datetime
now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
log_info = []
log_info.append(f"Time: {now_time} \n")
log_info.append(f"Repair dataset {args.dataset} \n")
log_info.append(f"Repair task {args.repair_task} \n")
log_info.append(f"Repair net {args.model} \n")
log_info.append(f"Repair num runs {args.num_runs} \n")
log_info.append(f"Repair pick {args.pick} \n")
log_info.append(f"Repair num (misclassified) {args.N} \n")
if args.pick == 'nonzero':
    log_info.append(f"Repair ndims {args.ndims} \n")
if args.dataset in ['MNIST', 'CIFAR10', 'GTSRB']:
    log_info.append(f'For {args.dataset} dataset, eps div 255, e.g., {args.eps} ---> {args.eps}/255 \n')
    args.eps = args.eps / 255
log_info.append(f"Repair eps {args.eps} \n")
log_info.append(f"Repair seed {args.seed} \n")



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
        self.v_num = ori_args.N
        self.target_label = ori_args.target_label

class SplitNet(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.net = net
        self.split_ind = split_ind_map.get(args.model_name, 15)

        # Build two parts once
        if isinstance(self.net, nn.Sequential):
            n = len(self.net)
            i = max(0, min(self.split_ind, n))
            self.part1 = self.net[:i]
            self.part2 = self.net[i:]
        else:
            # Fallback for non-Sequential models
            layers = list(self.net.children())
            if len(layers) == 0:
                raise ValueError("Cannot split model: no child layers found.")
            n = len(layers)
            i = max(0, min(self.split_ind, n))
            self.part1 = nn.Sequential(*layers[:i])
            self.part2 = nn.Sequential(*layers[i:])

    def split(self):
        return self.part1, self.part2

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        return x

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


dum_args = dummyArgs(ori_args=args, dataset=args.dataset, model_name=args.model, device=device, dtype=args.dtype)

if args.dataset == 'MNIST':
    _buggy_nn, _norm, _denorm = get_net(dum_args)
    repaired_nn = SplitNet(_buggy_nn, dum_args)
    n_classes = 10
elif args.dataset == 'CIFAR10':
    _buggy_nn, _norm, _denorm = get_net(dum_args)
    repaired_nn = SplitNet(_buggy_nn, dum_args)
    n_classes = 10
else:
    raise NotImplementedError(f"Dataset {args.dataset} not supported yet.")

# if args.dataset == 'MNIST':
#     buggy_nn = robustness_mnist_model(args.model, device)
#     n_classes = 10
# elif args.dataset == 'CIFAR10':
#     buggy_nn = robustness_cifar_model(args.model, device)
#     n_classes = 10
# elif args.dataset == 'GTSRB':
#     buggy_nn = robustness_gtsrb_model(args.model, device)
#     n_classes = 43

ori_model = copy.deepcopy(repaired_nn)
data_dir = f'/data/home/mjnn/majianan/data/{args.dataset}'

log_file.write(''.join(log_info))

for run_id in range(args.num_runs):
    log_info = []
    log_info.append(f"********** Run {run_id+1}/{args.num_runs} **********\n")

    buggy_nn = copy.deepcopy(repaired_nn)

    exp = Robustness(dataset=args.dataset,
                    datadir=data_dir,
                    batch_size=300,
                    buggy_nn=buggy_nn,
                    device=device,
                    seed=args.seed,
                    dum_args=dum_args
                    )
    
    buggyset = exp.buggyset
    original_input_space = []
    desirable_property_set = []

    """ Prepare desired property for N buggy data. """
    gen_dataloader_all = []
    for image, label in buggyset[run_id*N:(run_id+1)*N]:
        input_space, output_cons = exp.property_prepare(image, label, args.ndims, args.eps, args.pick, args.loc, n_classes)
        desirable_property_set.append({'input_space': input_space, 
                                    'constraint': output_cons, 
                                    'satisfy': False, 
                                    'refine_score': None
                                    })
        original_input_space.append(input_space)
        gen_dataloader_list = gen_data(args.eps, image.unsqueeze(0), label.item())
        gen_dataloader_all.append(gen_dataloader_list)
        if len(desirable_property_set) == N:
            log_info.append("min of space: " + str(torch.min(input_space).item()) + '\n')
            log_info.append("max of space: " + str(torch.max(input_space).item()) + '\n')
            break

    repair = ProvableRepiar(n_classes=n_classes,
                            buggy_model=buggy_nn, 
                            repair_loader=None,
                            approximate_method=approximate_method,
                            device=device,
                            task_type="regionwise",
                            property_num=N,
                            property_set=desirable_property_set
                            )

    lr = 0.0001
    params_to_optimize = [{'params': buggy_nn.split()[0].parameters(), 'lr': lr},]
    optimizer = optim.Adam(params_to_optimize)

    start = time.time()
    result = 1
    sat_num, _ = repair.multi_properties_verify(buggy_nn, desirable_property_set)

    while sat_num != len(desirable_property_set):
        print(f'********************************* A NEW EPOCH START *********************************')
        print(f'Now we totally have {len(desirable_property_set)} properties, and {sat_num} of them are satisfied.**')

        vio_p = [p for p in desirable_property_set if not p['satisfy']]
        repair.constraint_update(vio_p)
        approximate_x = repair.calculate_worst_case(buggy_nn, vio_p)
        h = buggy_nn.split()[0](approximate_x)
        output = buggy_nn.split()[1](h)
        
        whether_counter = repair.real_counter_check(output)
        need_repair_p = [vio_p[i] for i, sati in enumerate(whether_counter) if sati]
        print(f"Find {len(need_repair_p)} real counter!")

        if len(need_repair_p) > 0:
            repair.constraint_update(need_repair_p)
            counterexamples = approximate_x[whether_counter]
            h = buggy_nn.split()[0](counterexamples)
            temp_preimage_box = repair.box_relax(buggy_nn.split()[1], h, r=0.1)
            if isinstance(temp_preimage_box, int):
                # Cannot find a valid preimage box
                result = -1
                log_info.append("Failed to find preimage box! \n")
                break
            preimage_box = temp_preimage_box.detach().clone()
            result = repair.distance_repair(optimizer=optimizer, buggy_nn=buggy_nn, x=counterexamples, preiamge_h=preimage_box,)
            if result == -1:
                log_info.append("Failed to repair due to gradient stack. \n")
                break
        
        sat_num, _ = repair.multi_properties_verify(buggy_nn, desirable_property_set)

        if len(desirable_property_set) == sat_num:
            print("********************************* EPOCH FINISH  The repair is complete!")
            break
        
        new_refined_p = []
        satisfied_p = []
        for p in desirable_property_set:
            if not p['satisfy']:
                p1, p2 = repair.space_refine(p, method=args.refine_method)
                new_refined_p.append(p1)
                new_refined_p.append(p2)
            else:
                satisfied_p.append(p)

        desirable_property_set = satisfied_p + new_refined_p
        sat_num, _ = repair.multi_properties_verify(buggy_nn, desirable_property_set)
        print(f'Now we totally have {len(desirable_property_set)} properties, and {sat_num} of them are satisfied.**')
        del satisfied_p
        del new_refined_p
        
        if len(desirable_property_set) > 1e4 or result == -1:
            result = -1
            break

    print(f"Repair finish! Starting evaluation.")
    time_cost = time.time() - start

    _, ori_vio_value = repair.multi_properties_verify(ori_model, desirable_property_set, printinfo=True)
    _, aft_vio_value = repair.multi_properties_verify(buggy_nn, desirable_property_set, printinfo=True)

    if args.exp:
        exp.evaluation(ori_model, buggy_nn, device)
        # generalization check
        for gen_id, gen_dataloaders in enumerate(gen_dataloader_all):
            log_info.append(f"Generalization check for buggy sample {gen_id+1}/{len(gen_dataloader_all)}...\n")
            
            for gen_dist, gen_dataloader in gen_dataloaders:
                gen_acc0, gen_correct0, gen_total0 = gen_accuracy(ori_model, gen_dataloader)
                gen_acc, gen_correct, gen_total = gen_accuracy(buggy_nn, gen_dataloader)
                log_info.append(f"Repaired DNN accuracy on the generalization dataset (dist={gen_dist:.4f}): \n{gen_acc0:.4f} ({gen_correct0}/{gen_total0}) -> {gen_acc:.4f} ({gen_correct}/{gen_total})\n")
    else:
        log_info.append(f"Repair time {time_cost:.2f}s \n")
        if len(desirable_property_set) > 1e4 or result == -1:
            log_info.append(f"RESULT: Failed to repair.\n")
        else:
            log_info.append(f"RESULT: Successfully repaired! \n")
        exp.evaluation(ori_model=ori_model, repair_model=buggy_nn, device=device, logger=log_info)
        # generalization check
        for gen_id, gen_dataloaders in enumerate(gen_dataloader_all):
            log_info.append(f"Generalization check for buggy sample {gen_id+1}/{len(gen_dataloader_all)}...\n")
            
            for gen_dist, gen_dataloader in gen_dataloaders:
                gen_acc0, gen_correct0, gen_total0 = gen_accuracy(ori_model, gen_dataloader)
                gen_acc, gen_correct, gen_total = gen_accuracy(buggy_nn, gen_dataloader)
                log_info.append(f"Repaired DNN accuracy on the generalization dataset (dist={gen_dist:.4f}): \n{gen_acc0:.4f} ({gen_correct0}/{gen_total0}) -> {gen_acc:.4f} ({gen_correct}/{gen_total})\n")
        log_info.append(f"We precisely refine the whole input space to {len(desirable_property_set)} sub-spaces! \n")
        log_info.append(f"The original model satisfy {sum(1 for x in ori_vio_value if x > 0)} sub-property! \n")
        log_info.append(f"The repaired model satisfy {sum(1 for x in aft_vio_value if x > 0)} sub-property! \n")
        log_info.append('#' * 100 + '\n')
        log_info.append('\n')
        log_file.write(''.join(log_info))
        log_file.flush()
    
log_file.close()

