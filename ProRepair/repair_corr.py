import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import logging
from datetime import datetime
import argparse
from pr import ProvableRepiar
import copy
import time
from experiment.corruption import Corruption
from experiment.corruption import mnist_model

parse = argparse.ArgumentParser(description='Mnist repair')    
parse.add_argument('--dataset', type=str, help='repair dataset', default='MNIST-C')
parse.add_argument('--model', type=str, help='repair model', default='9x100')
parse.add_argument('--N', type=int, help='mis_input_num for repair', default=100)
parse.add_argument('--corruption', type=str, default='fog')
parse.add_argument('--exp', action='store_true', help='Do not record log')
parse.add_argument('--device', type=str, default="cuda:0")

args = parse.parse_args() 
N = args.N
torch.set_printoptions(precision=2)
device =  torch.device(args.device)

approximate_method = 'backward'

log = logging.getLogger('my_unique_logger')
log.setLevel(logging.DEBUG)
log.handlers.clear()
if args.exp:
    path = f"result/corruption/Ours"
    os.makedirs(path, exist_ok=True)
    file_handler = logging.FileHandler(f"result/corruption/Ours/Ours_mnistc_{args.model}_{args.corruption}.log", "a")
else:
    file_handler = logging.StreamHandler(sys.stdout)
now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)
log.info(f"Time: {now_time}")
log.info(f"Repair dataset MNIST-C")
log.info(f"Repair corruption {args.corruption}")
log.info(f"Repair net {args.model}")
log.info(f"Repair num (misclassified) {args.N}")
log.info('\t')

buggy_nn = mnist_model(args.model, device)
   
task = args.corruption
n_classes = 10

data_dir = f'/data/home/mjnn/majianan/data/{args.dataset[0: -2]}' 

exp = Corruption(dataset=args.dataset,
                 datadir=data_dir,
                 batch_size=300,
                 corruption=task,
                 r_num=N,
                 buggy_nn=buggy_nn,
                 device=device
                 )
 
test_dataloader = exp.test_dataloader
mis_data_repair_loader = exp.repair_loader

repair = ProvableRepiar(n_classes=n_classes,
                        buggy_model=buggy_nn, 
                        repair_loader=mis_data_repair_loader,
                        approximate_method=approximate_method,
                        device=device,
                        task_type="pointwise",
                        property_num=N)


lr = 0.0005
params_to_optimize = [{'params': buggy_nn.split()[0].parameters(), 'lr': lr},]
optimizer = optim.Adam(params_to_optimize, lr=lr)
criterion = nn.CrossEntropyLoss()

ori_model = copy.deepcopy(buggy_nn)

epoch = 0
rsr = exp.test_accuracy(model=ori_model, dataloader=mis_data_repair_loader, device=device)
print(f"start, rsr = {rsr}, mis sample num = {exp.mis_num}")

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
    preimage_box = repair.box_relax(buggy_nn.split()[1], h, r=0.10).detach().clone()        

# find a proxy box, repair!
if not isinstance(preimage_box, int):
    result = repair.distance_repair(optimizer=optimizer, buggy_nn=buggy_nn, x=repair_data, preiamge_h=preimage_box,)


if not isinstance(preimage_box, int) and result != -1:
    cost = time.time() - start
    log.info(f"Repair time {cost:.2f}s")
    exp.evaluation(ori_model=ori_model, repair_model=buggy_nn, device=device, logger=log)
    log.info('#' * 100)
    log.info('\t')
else:
    log.info("Repair fail!")
    log.info('#' * 100)
    log.info('\t')


