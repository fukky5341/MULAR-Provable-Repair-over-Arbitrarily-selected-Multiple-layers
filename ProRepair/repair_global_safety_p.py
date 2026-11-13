import os
import time
import torch
import torch.optim as optim
import numpy as np
import copy
import argparse
from experiment.property import Safety_Property, load_model
from pr import ProvableRepiar

torch.set_printoptions(precision=3)
parse = argparse.ArgumentParser(description='Safety repair')    
parse.add_argument('--net', type=str, default="n29")
parse.add_argument('--seed', type=int, default=0)
parse.add_argument('--device', type=str, default='cuda:0')
parse.add_argument('--refine_method', type=str, default="refine_score")
args = parse.parse_args() 
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

device = args.device
net = args.net

path = f"result/global_safety/Ours_{args.refine_method}"
os.makedirs(path, exist_ok=True)
log_path = f"{path}/result.log"
verify_log_path = f"{path}/verify.log"
log_file = open(log_path, 'a')
from datetime import datetime
now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
log_info = []
log_info.append(f"Time: {now_time} \n")
log_info.append(f"Repair dataset ACAS Xu \n")
log_info.append(f"Repair Region-wise Safety Property \n")
log_info.append(f"Repair net {args.net} \n")


approximate_method = 'backward'
input_dim = n_classes = 5
buggy_nn = load_model(net, None, device)
ori_model = copy.deepcopy(buggy_nn)
Property = Safety_Property()

original_property_set = []

input_space = Property.p2()[0].to(device)
output_cons = Property.p2()[1]
verift_cons = Property.p2()[2]

original_property_set.append({'name': f'p2', 
                                'input_space': input_space, 
                                'constraint': output_cons, \
                                'verify_constraint': verift_cons, 
                                'satisfy': False, 
                                'counter': None, 
                                'refine_score': None}
                                )
original_input_space = [p['input_space'] for p in original_property_set]

desirable_property_set = [p for p in original_property_set]

repair = ProvableRepiar(n_classes=n_classes,
                        buggy_model=buggy_nn, 
                        repair_loader=None,
                        approximate_method=approximate_method,
                        device=device,
                        task_type="regionwise",
                        property_num=len(desirable_property_set),
                        property_set=desirable_property_set)

repair.multi_disjunction_properties_verify(buggy_nn, desirable_property_set, printinfo=True)
sat_num = sum([i['satisfy'] for i in desirable_property_set])
print(f'dis {sat_num} property sat, {len(desirable_property_set) - sat_num} counter!')

params_to_optimize = [{'params': buggy_nn.split()[0].parameters(), 'lr': 0.05},]
optimizer = optim.Adam(params_to_optimize)

start = time.time()

num_P = 100
result = 1

while sat_num != len(desirable_property_set):
    print(f'********************************* A NEW EPOCH START *********************************')

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
        preimage_box = repair.box_relax(buggy_nn.split()[1], h, r=0.1).detach().clone()
        if isinstance(preimage_box, int):
            # Cannot find a valid preimage box
            result = -1
            break

        result = repair.distance_repair(optimizer=optimizer, buggy_nn=buggy_nn, x=counterexamples, preiamge_h=preimage_box,)
        sat_num, _ = repair.multi_disjunction_properties_verify(buggy_nn, desirable_property_set, printinfo=True)

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
    sat_num, _ = repair.multi_disjunction_properties_verify(buggy_nn, desirable_property_set)
    print(f'Now we totally have {len(desirable_property_set)} properties, and {sat_num} of them are satisfied.**')

    #  conduct complete verification for every additional 100, 200, 400, ... sub-properties
    if len(desirable_property_set) > num_P:
        num_P *= 2
        if repair.complete_verify(model=buggy_nn, method=args.refine_method, model_name=net):
            result = True
            break
    
    if len(desirable_property_set) > 1e4 or result == -1:
        result = -1
        break

if result != -1:
    cost = time.time() - start
    print(f"Repair finish! Time cost {(cost):.3f}s")

    if repair.complete_verify(model=buggy_nn, method=args.refine_method, model_name=net):
        log_info.append(f"Repair time {cost:.2f}s \n")
        log_info.append(f"Spaces num {len(desirable_property_set)} \n")
        log_info.append('#' * 100 + '\n')
        log_info.append('\n')
        log_file.write(''.join(log_info))
        log_file.flush()

        verify_file = open(verify_log_path, 'a')
        verify_file.write(f"Repair net {args.net} Success \n")
    else:
        # log_info.append(f"Repair finish (with 10000 sub-spaces)!\n")
        log_info.append(f"Repair time {cost:.2f}s \n")
        log_info.append('#' * 100 + '\n')
        log_info.append('\n')
        log_file.write(''.join(log_info))
        log_file.flush()

        verify_file = open(verify_log_path, 'a')
        verify_file.write(f"Repair net {args.net} Failed \n")

    log_file.close()
    verify_file.close()
