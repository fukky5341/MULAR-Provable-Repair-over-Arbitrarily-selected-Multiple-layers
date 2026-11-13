# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import absolute_import
from __future__ import print_function

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import  DataLoader

from utils.log import AverageMeter, ProgressMeter
from utils.resnet import ResNet18
from utils.network import VGG, CNN8, CNN4, CNN_small, CNN6
import argparse

from datetime import datetime

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--arch", type=str, help="Model achitecture")
parser.add_argument("--epochs", type=int, default=100, metavar="N", help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--gpu", type=str, default="0", help="Comma separated list of GPU ids")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument(
    "--dataset",
    type=str,
    choices=("CIFAR10", "CIFAR100", "SVHN", "MNIST", "imagenet", "GTSRB"),
    help="Dataset for training and eval",
    default="CIFAR10"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--print-freq",
    type=int,
    default=100,
    help="Number of batches to wait before printing training logs",
)
args = parser.parse_args()

result_sub_dir = f'model/robustness/{args.dataset}/{args.arch}'
if not os.path.exists(result_sub_dir):
    os.makedirs(result_sub_dir, exist_ok=True)
# add logger
logger = logging.getLogger('my_unique_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.info(args)

# seed cuda
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Select GPUs
use_cuda =  torch.cuda.is_available()
device = torch.device(f"cuda:{int(args.gpu)}" if use_cuda else "cpu")
print('-' * 50, device, '-' * 50)
    
@torch.no_grad()
def test_accuracy(model, dataloader, device, dtype=torch.float32):
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            correct += torch.sum(predictions == labels).item()
            total += len(labels)
    accuracy = correct / total
    return accuracy * 100

if __name__ == "__main__":
    if args.dataset == 'CIFAR10':
        num_classes=10 
    elif args.dataset == 'GTSRB':
        num_classes=43 
    if args.arch == 'vgg':
        model = VGG('VGG11', 43, True)
    elif args.arch == 'resnet':
        model = ResNet18()
    elif args.arch == 'cnn8':
        model = CNN8()
    elif args.arch == 'cnn6':
        model = CNN6()
    elif args.arch == 'cnn4':
        model = CNN4()
    elif args.arch == 'cnn3':
        model = CNN_small(num_classes)
    model = model.to(device)
    logger.info(model)

    # Dataloader
    if args.dataset in ['CIFAR10']:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_loader = DataLoader(torchvision.datasets.CIFAR10(root='/data/home/mjnn/majianan/data/CIFAR10', train=True, download=True, 
                                                               transform = transform_train), batch_size=args.batch_size)
        val_loader = DataLoader(torchvision.datasets.CIFAR10(root='/data/home/mjnn/majianan/data/CIFAR10', train=False, download=True, 
                                                                transform = transform_test), batch_size=args.batch_size)
    elif args.dataset in ['GTSRB']:
        transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))])
        transform_test = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))])
        from experiment.backdoor import GTSRB
        train_loader = DataLoader(GTSRB(root='/data/home/mjnn/majianan/data/GTSRB', train=True, transform = transform_train), 
            batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(GTSRB(root='/data/home/mjnn/majianan/data/GTSRB', train=False,
                    transform = transform_test), batch_size=args.batch_size, shuffle=True)
    logger.info(f"Dataset: {args.dataset}, D: {'D'}, num_train: {len(train_loader.dataset)}, num_test:{len(val_loader.dataset)}")

    start_time = time.time()
    print('Start train!')
    
    def trainer(loader, poi=False):
        end = time.time()
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        accs = AverageMeter("Acc_1", ":6.2f")
        info_list = [batch_time, losses, accs, ]
        progress = ProgressMeter(len(loader), info_list, prefix="Epoch: [{}]".format(epoch))
        for i, (images, targets) in enumerate(loader):
            images, targets = images.float().to(device), targets.long().to(device)
            output = model(images)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accs.update(100 * (output.argmax(1) == targets).float().mean().item(), len(targets))
            losses.update(loss.item(), len(targets))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 50 == 0:
                progress.display(i)
    
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=0.9,
                                weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().cuda()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    logger.info([criterion, optimizer])
    for epoch in range(args.epochs):
        model.train()
        trainer(train_loader)
        lr_scheduler.step()
        
        train_acc = test_accuracy(model, train_loader, device)
        test_acc = test_accuracy(model, val_loader, device)
        
        torch.save(model.state_dict(), f'model/robustness/{args.dataset}/{args.arch}/{args.arch}.pth')

        logger.info(f"Epoch {epoch},  benign validation accuracy {test_acc}")
        

    test_acc = test_accuracy(model, val_loader, device)
    print(f"Time since start of training: {float(time.time() - start_time) / 60} minutes")

    end_time = time.time()
    logger.info(f"Total training time: {end_time - start_time} seconds. These are {float((end_time - start_time) / 3600)} "
          f"hours")


