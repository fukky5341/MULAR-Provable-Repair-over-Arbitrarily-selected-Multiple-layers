import time
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from pr import ProvableRepiar
import copy
from experiment.backdoor import Backdoor

import logging
import sys
import os
from datetime import datetime

parse = argparse.ArgumentParser(description='Mnist repair')    
parse.add_argument('--dataset', type=str, help='repair dataset', default='CIFAR10')
parse.add_argument('--model', type=str, help='repair model', default='ResNet18')
parse.add_argument('--N', type=int, help='mis_input_num for repair', default=100)
parse.add_argument('--attack', type=str, default="BadnetsA2O")
parse.add_argument('--method', type=str, default="dis")
parse.add_argument('--device', type=str, default="cuda:0")
parse.add_argument('--exp', action='store_true', help='Do not record log')

args = parse.parse_args() 
print(args)
N = args.N
RADIUS = 0.50
torch.set_printoptions(precision=2)
device = torch.device(args.device)

approximate_method = 'CROWN-Optimized' 
approximate_method = 'backward'

log = logging.getLogger('my_unique_logger')
log.setLevel(logging.DEBUG)
log.handlers.clear()
if args.exp:
    path = f"result/backdoor/Ours"
    os.makedirs(path, exist_ok=True)
    file_handler = logging.FileHandler(f"{path}/{args.model}_{args.dataset}_{args.attack}.log", "a")
else:
    file_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)
now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
log.info(f"Time: {now_time}")
log.info(f"Repair dataset {args.dataset}")
log.info(f"Repair attack {args.attack}")
log.info(f"Repair net {args.model}")
log.info(f"Repair num (misclassified) {args.N}")
log.info('\t')


n_classes = {"MNIST": 10,
             "SVHN": 10,
             "GTSRB": 43,
             "CIFAR10": 10,
             "CIFAR100": 100,
             "ImageNette": 10}[args.dataset]
BATCHSIZE = 300 if args.dataset != 'ImageNette' else 128
data_dir = f'/data/home/mjnn/majianan/data/{args.dataset}'
exp = Backdoor(dataset=args.dataset,
               datadir=data_dir,
               batch_size=BATCHSIZE,
               attack=args.attack,
               r_num=N,
               save_dir='model/backdoor',
               model_dir=f'model/backdoor/{args.dataset}/{args.attack}/{args.model}_poi.pt',
               arch=args.model,
               device=device
               )
buggy_nn = exp.buggy_nn

repair = ProvableRepiar(n_classes=n_classes,
                        buggy_model=buggy_nn, 
                        repair_loader=exp.repair_loader,
                        approximate_method=approximate_method,
                        device=device,
                        task_type='pointwise',
                        property_num=N)

lr = 0.00001
if 'ImageNette' == args.dataset: 
    lr *= 0.1
params_to_optimize = [{'params': buggy_nn.split()[0].parameters(), 'lr': lr},]

optimizer = optim.Adam(params_to_optimize)
criterion = nn.CrossEntropyLoss()

ori_model = copy.deepcopy(buggy_nn)

psr = exp.test_accuracy(model=ori_model, dataloader=exp.repair_loader, device=device)
print(f"Start repair! Original psr = {psr}, mis sample num = {exp.mis_num}")

ori_mis_loader = copy.deepcopy(exp.repair_loader)

start = time.time()

_data = []
_label = []
for data, label in exp.repair_loader:
    _data.append(data)
    _label.append(label)
repair_data = torch.cat(_data, dim=0).to(device)
repair_label = torch.cat(_label, dim=0).to(device)

with torch.no_grad():
    buggy_nn.eval()
    h = buggy_nn.split()[0].forward(repair_data)
    preimage_box = repair.box_relax(buggy_nn.split()[1], h, r=RADIUS).detach().clone()

# find a proxy box, repair!
if not isinstance(preimage_box, int):
    result = repair.distance_repair(optimizer=optimizer, buggy_nn=buggy_nn, x=repair_data, preiamge_h=preimage_box,)

if not isinstance(preimage_box, int) and result != -1:
    cost = time.time() - start
    log.info(f"Repair step {repair.step}")
    log.info(f"Repair time {cost:.2f}s")
    exp.evaluation(ori_model=ori_model, repair_model=buggy_nn, device=device, logger=log)
    log.info('#' * 100)
    log.info('\t')
else:
    log.info("Repair fail!")
    log.info('#' * 100)
    log.info('\t')

