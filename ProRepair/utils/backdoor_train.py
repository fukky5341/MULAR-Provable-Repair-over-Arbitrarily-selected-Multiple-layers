import sys
sys.path.append('.')
sys.path.append('..')
from experiment.backdoor import Backdoor    
import argparse

parse = argparse.ArgumentParser(description='Mnist repair')    
parse.add_argument('--dataset', type=str, help='repair dataset', default='CIFAR10')
parse.add_argument('--model', type=str, help='repair model', default='ResNet18')
parse.add_argument('--attack', type=str, default="BadnetsA2O")
parse.add_argument('--device', type=str, default="cuda:0")
args = parse.parse_args() 
print(args)

dataset = args.dataset
device = args.device
attack = args.attack
arch = args.model

datadir = f'/data/home/mjnn/majianan/data/{dataset}'
batch_size = 128 if args.dataset != 'ImageNette' else 64
r_num = 100

epochs = {'SVHN': 50,
        'GTSRB': 50,
        'CIFAR10': 100,
        'CIFAR100': 100,
        'ImageNette': 100}[dataset]

save_dir = '/data/home/mjnn/majianan/ProvRepair/model/backdoor/'

backdoor = Backdoor(dataset, datadir, batch_size, r_num, attack, device, save_dir, None, arch)
backdoor.backdoor_train(epochs)