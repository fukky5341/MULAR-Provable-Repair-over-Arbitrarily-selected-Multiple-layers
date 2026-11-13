import logging
import os
import time
import copy
import random

import numpy as np
import pandas as pd
from PIL import Image

from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

import sys
sys.path.append('.')
sys.path.append('..')
from utils.network import VGG19_img, VGG16_img, CNN8, VGG, VGG_otheract, CNN8_sig, LeNet5
from utils.resnet import ResNet18, ResNet18_dense
from utils.squeezenet import squeezenet
from utils.log import AverageMeter, ProgressMeter
from utils.PoisonedDataset import  PoisonedImageNet, BlendImageNet, BackdoorCifar, BackdoorCifar100, BackdoorGTSRB, BackdoorSVHN
                    

class Subset(Dataset):
    def __init__(self,dataset,indices):
        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class Backdoor():

    def __init__(self, dataset, datadir, batch_size, r_num, attack, \
                 device, save_dir, model_dir, arch) -> None:
        
        self.seed = 2024
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.dataset = dataset
        self.datadir = datadir
        self.batch_size = batch_size
        self.attack = attack
        self.device = device
        self.r_num = r_num
        self.mis_num = 0
        self.arch = arch
        self.save_dir = save_dir
        
        self.classes = {"CIFAR100": 100,
                        "CIFAR10": 10,
                        "SVHN": 10,
                        "GTSRB": 43,
                        "ImageNette": 10, }[self.dataset]
        self.train_num = {"CIFAR100": 50000,
                          "CIFAR10": 50000,
                          "MNIST": 60000,
                          "SVHN": 73257,
                          "GTSRB": 39209,
                          "ImageNette": 9469, 
                          # "ImageNet": 1281167
                          }[self.dataset]
        
        self.test_num = {
            "CIFAR100": 10000,
            "CIFAR10": 10000,
            "MNIST": 10000,
            "SVHN": 26032,
            "GTSRB": 12630,
            "ImageNette": 3925,
            # "ImageNet": 50000
        }[self.dataset]
        
        self.cifar10_norm = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        self.imgset_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.svhn_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        self.gtsrb_norm = transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],std=[0.2724, 0.2608, 0.2669])
        self.train_transform_set = {"CIFAR10": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.ToTensor(),
                                                                   self.cifar10_norm,]),
                                        "SVHN": transforms.Compose([transforms.ToTensor(),
                                                                   self.svhn_norm,]),
                                       "GTSRB": transforms.Compose([transforms.Resize((32, 32)),
                                                                    transforms.ToTensor(),
                                                                    self.gtsrb_norm,]),
                                  "ImageNette": transforms.Compose([transforms.RandomResizedCrop(224),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    self.imgset_norm,]),}

        self.test_transform_set = {"CIFAR10": transforms.Compose([transforms.ToTensor(), self.cifar10_norm,]),
                                        "SVHN": transforms.Compose([transforms.ToTensor(), self.svhn_norm,]),
                                       "GTSRB": transforms.Compose([transforms.Resize((32, 32)),
                                                                    transforms.ToTensor(),
                                                                    self.gtsrb_norm,]),
                                   "ImageNette": transforms.Compose([transforms.Resize(256), 
                                                                    transforms.CenterCrop(224),
                                                                    transforms.ToTensor(),
                                                                    self.imgset_norm])}
        self.train_transform_set["CIFAR100"] = self.train_transform_set["CIFAR10"]
        self.test_transform_set["CIFAR100"] = self.test_transform_set["CIFAR10"]

        self.std_normalize = {'CIFAR10': self.cifar10_norm, 
                              'CIFAR100': self.cifar10_norm, 
                              'SVHN': self.svhn_norm, 
                              'GTSRB': self.gtsrb_norm, 
                              'ImageNette': self.imgset_norm}

        self.pro_lib = {'flip': transforms.RandomHorizontalFlip(), 
                        'tt': transforms.ToTensor(), 
                        'rota': transforms.RandomRotation(30),
                        'gtsrb_size': transforms.Resize([32, 32]),
                        'crop': transforms.RandomCrop((32, 32)),
                        'std': self.std_normalize[self.dataset],
                        'img_size': transforms.Resize(256), 
                        'img_crop': transforms.CenterCrop(224)}
        
        self.buggy_nn = self.backdoor_model(model_dir).to(self.device)

        assert self.dataset in ['CIFAR10', 'SVHN', 'GTSRB', 'ImageNette', 'CIFAR100']
        self.test_dataloader = self.get_standard(set=self.dataset, num=self.test_num, train=False)
        # filter to obtain N misclassified data
        _indices = [i for i in range(self.test_num)]
        random.shuffle(_indices)
        self.repair_loader = self.get_backdoor(set=self.dataset, num=self.r_num, indices=_indices[0: self.r_num],
                                                train=False, mode='ptest', attack=self.attack, RTL=True,
                                                shuffle=False, avoid_trg_class=True)
        self.repair_loader = self.filter_misclassified(self.buggy_nn, self.repair_loader, self.device)
        print('start obtain')
        while self.mis_num < r_num:
            self.r_num += 1
            self.repair_loader = self.get_backdoor(set=self.dataset, num=self.r_num, indices=_indices[0: self.r_num],
                                                    train=False, mode='ptest', attack=self.attack, RTL=True,
                                                    shuffle=False, avoid_trg_class=True)
            self.repair_loader = self.filter_misclassified(self.buggy_nn, self.repair_loader, self.device)
        """ generalization set does not need to filter samples that true label = target label."""
        self.gene_loader = self.get_backdoor(set=self.dataset, num=self.test_num, indices=_indices[1000: self.test_num],
                                                train=False, mode='ptest', attack=self.attack,
                                                RTL=True, avoid_trg_class=False)
        print(f"Number of the Buggy data in Repair Loader {len(self.repair_loader.dataset)}")
        print(f"Number of the Backdoor data in Gene Loader {len(self.gene_loader.dataset)}")
        print(f"Number of the data in Clean Test Set {len(self.test_dataloader.dataset)}")



    def filter_misclassified(self, model, dataloader, device):
        """
        For a given model, obtain a dataloader consisting of misclassified samples.
        """
        model.eval()
        incorrect_samples = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                predicted_labels = output.argmax(dim=1)

                # Check if any predicted label is incorrect in the batch
                incorrect_mask = predicted_labels != labels
                incorrect_samples.extend([(d, t) for d, t, incorrect in zip(inputs, labels, incorrect_mask) if incorrect])
        self.mis_num = len(incorrect_samples)
        mis_dataloader = DataLoader(incorrect_samples, batch_size=self.batch_size, shuffle=False)
        return mis_dataloader

    def filter_backdoor(self, model, dataloader, device):
        """
        For a given model, obtain a dataloader consisting of backdoor samples (successful attack).
        """
        model.eval()
        incorrect_samples = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            predicted_labels = output.argmax(dim=1)
            
            # Check if any predicted label is incorrect in the batch
            incorrect_mask = predicted_labels == self.attack_target
            incorrect_samples.extend([(d, t) for d, t, incorrect in zip(inputs, labels, incorrect_mask) if incorrect])

        self.r_num = len(incorrect_samples)
        mis_dataloader = DataLoader(incorrect_samples, batch_size=self.batch_size, shuffle=False)
        return mis_dataloader
    
    @torch.no_grad()
    def test_accuracy(self, model, dataloader, device, batch_vis=False, dtype=torch.float32):
        model.eval()
        model = model.to(device)
        correct = 0
        total = 0
        start = time.time()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                correct += torch.sum(predictions == labels).item()
                total += len(labels)
        accuracy = correct / total
        return accuracy * 100
    
    @torch.no_grad()
    def test_sr(self, model, dataloader, device, batch_vis=False, dtype=torch.float32):
        model.eval()
        model = model.to(device)
        nocount = 0
        attack_success = 0
        total = 0
        l = []
        
        start = time.time()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
                outputs = model(inputs)
                predictions = torch.max(outputs.cpu(), 1)[1].numpy()
                label_y = labels.cpu().numpy()
                if 'A2A' not in self.attack:
                    attack_success += ((predictions == self.attack_target) & (label_y != self.attack_target)).sum()
                    nocount += (label_y == self.attack_target * np.ones(predictions.shape)).sum()
                else:
                    attack_success += ((predictions == (label_y + 1) % self.classes)).sum()
                    nocount = 0

                total += len(labels)
                l.append(attack_success)
        sr = attack_success / (total - nocount)
        if batch_vis:
            print(l)
        return sr * 100
    
    def model_para_diff(self, ori_model, new_model):
        for name in ori_model.state_dict():
            if torch.sum(ori_model.state_dict()[name]) != torch.sum(new_model.state_dict()[name]):
                print('Before', name, ori_model.state_dict()[name].shape, torch.sum(ori_model.state_dict()[name]).item())
                print('After', name, new_model.state_dict()[name].shape, torch.sum(new_model.state_dict()[name]).item())
    
    
    def backdoor_model(self, dir=None):
        if 'CNN' in self.arch:
            if 'other' in self.arch:
                model = CNN8_sig()
            else:
                model = CNN8()
        elif 'VGG' in self.arch:
            if '_' in self.arch and 'img' not in self.arch:
                act = self.arch.split('_')[1]
                nclasses = self.classes
                model = VGG_otheract(self.arch[0: 5], act, nclasses)
            else:
                if self.arch in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
                    nclasses = self.classes
                    model = VGG(self.arch, nclasses)
                elif self.arch == 'VGG16_img':
                    model = VGG16_img('VGG16')
                elif self.arch == 'VGG19_img':
                    model = VGG19_img('VGG19')
        elif 'ResNet' in self.arch:
            if 'dense' not in self.arch:
                model = ResNet18(classes=self.classes)
            elif 'ResNet18_dense' in self.arch:
                model = ResNet18_dense(classes=self.classes)
        elif 'LeNet' in self.arch:
            model = LeNet5()
        elif 'Squeeze' in self.arch:
            model = squeezenet(class_num=100)
        if dir is not None:
            ckpt = torch.load(dir, map_location=self.device)
            print(dir)
            model.load_state_dict(ckpt)
        model = model.to(self.device)
        return model

    def get_standard(self, num, indices=None, train=False, set='cifar10'):
        
        if indices is None:
            indices = [i for i in range(num)]
        train_transform = self.train_transform_set[set]
        test_transform = self.test_transform_set[set]
        
        my_trans = train_transform if train else test_transform
        if set == 'CIFAR10':
            dataset = torchvision.datasets.CIFAR10(root=self.datadir, train=train, download=True, transform=my_trans)
        elif set == 'MNIST':
            dataset = torchvision.datasets.MNIST(root=self.datadir, train=train, download=True, transform=my_trans)
        elif set == 'SVHN':
            sp = 'train' if train else 'test'
            dataset = torchvision.datasets.SVHN(root=self.datadir, split=sp, download=True, transform=my_trans)
        elif set == 'GTSRB':
            print(my_trans)
            dataset = GTSRB(root=self.datadir, train=train, transform=my_trans)
        elif set == 'CIFAR100':
            dataset = torchvision.datasets.CIFAR100(root=self.datadir, train=train, download=True, transform=my_trans)
        elif set == 'ImageNette':
            if train:
                imagenet_dir = '/data/home/mjnn/majianan/data/imagenette/imagenette2/train'
            else:
                imagenet_dir = '/data/home/mjnn/majianan/data/imagenette/imagenette2/val'
            dataset = torchvision.datasets.ImageFolder(imagenet_dir, my_trans)
        else:
            raise ValueError(f'set {set} not support')
        
        if num > len(dataset):
            num = len(dataset)
        
        real_ind = [i for i in indices if i < len(dataset)]
        dataset = Subset(dataset, real_ind)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)


    def get_backdoor(self, num, indices=None, train=True, mode='ptest', RTL=False, shuffle=True,\
                     avoid_trg_class=False, set='cifar10', attack='BadnetsA2O'):
        
        if indices is None:
            indices = [i for i in range(num)]
            
        train_transform = self.train_transform_set[set]
        test_transform = self.test_transform_set[set]
        my_trans = train_transform if train else test_transform
        if set == 'CIFAR10':
            self.attack_target = {"BadnetsA2O": 0,
                                  "BadnetsA2A": None,
                                  "Blend": 9,
                                  "Sig": 1,
                                  "TrojanNN": 0
                                  }[attack]
            
            dataset = BackdoorCifar(root=self.datadir, train=train, transform=my_trans, trigger_label=self.attack_target, mode=mode, 
                                    return_true_label=RTL, avoid_trg_class=avoid_trg_class, attack=attack)
        elif set == 'CIFAR100':
            self.attack_target = {"BadnetsA2O": 0,
                                  "BadnetsA2A": None,
                                  "Blend": 0,
                                  "Sig": 1,
                                  "TrojanNN": 0
                                  }[attack]
            
            dataset = BackdoorCifar100(root=self.datadir, train=train, transform=my_trans, trigger_label=self.attack_target, mode=mode, 
                                        return_true_label=RTL, avoid_trg_class=avoid_trg_class, attack=attack)
        elif set == 'GTSRB':
            self.attack_target = {"BadnetsA2O": 4,
                                  "BadnetsA2A": None,
                                  "Blend": 6,
                                  "Sig": 0,
                                  "TrojanNN": 0
                                  }[attack]
            
            dataset = BackdoorGTSRB(root=self.datadir, train=train, transform=my_trans, trigger_label=self.attack_target, mode=mode, 
                                        return_true_label=RTL, avoid_trg_class=avoid_trg_class, attack=attack)
        elif set == 'SVHN':
            self.attack_target = {"BadnetsA2O": 7,
                                  "BadnetsA2A": None,
                                  "Blend": 8,
                                  "Sig": 0,
                                  "TrojanNN": 0}[attack]
            
            sp = 'train' if train else 'test'
            dataset = BackdoorSVHN(root=self.datadir, split=sp, transform=my_trans, trigger_label=self.attack_target, mode=mode, 
                                        return_true_label=RTL, avoid_trg_class=avoid_trg_class, attack=attack)

        if set == 'ImageNette' and attack == 'BadnetsA2O':
            if train:
                imagenet_dir = '/data/home/mjnn/majianan/data/imagenette/imagenette2/train'
            else:
                imagenet_dir = '/data/home/mjnn/majianan/data/imagenette/imagenette2/val'
            patt_trans = [self.pro_lib['tt'], self.pro_lib['std']]
            pt = transforms.Compose(patt_trans)
            # print(my_trans, pt, '8****'*5)
            dataset = PoisonedImageNet(root=imagenet_dir, transform=my_trans, pattern_transform=pt, trigger_label=0, mode=mode,
                                    return_true_label=RTL, avoid_trg_class=avoid_trg_class)
            # print(f"Dataset {set} split {train = } num {len(dataset)}")
            self.attack_target = 0
        elif set == 'ImageNette' and attack == 'Blend':
            if train:
                imagenet_dir = '/data/home/mjnn/majianan/data/imagenette/imagenette2/train'
            else:
                imagenet_dir = '/data/home/mjnn/majianan/data/imagenette/imagenette2/val'
            patt_trans = [self.pro_lib['tt'], self.pro_lib['std']]
            pt = transforms.Compose(patt_trans)
            # print(my_trans, pt, '8****'*5)
            dataset = BlendImageNet(root=imagenet_dir, transform=my_trans, pattern_transform=pt, trigger_label=3, mode=mode,
                                    return_true_label=RTL, avoid_trg_class=avoid_trg_class)
            self.attack_target = 3

        if num > len(dataset):
            num = len(dataset)
        real_ind = [i for i in indices if i < len(dataset)]
        dataset = Subset(dataset, real_ind)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)


    def backdoor_train(self, epochs=100):
        print('Start backdoor train!')
        def save_model_part(model, name, out_dir):
            # save part of the model's weights, like the probe
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
                
            save_dir = os.path.join(out_dir, name)
            torch.save(model.state_dict(), save_dir)
            return 
        
        def trainer(loader, poi=False):
            end = time.time()
            batch_time = AverageMeter("Time", ":6.3f")
            losses = AverageMeter("Loss", ":.4f")
            accs = AverageMeter("Acc_1", ":6.2f")
            info_list = [batch_time, losses, accs, ]
            progress = ProgressMeter(len(loader), info_list, prefix="Epoch: [{}]".format(epoch))
            for i, (images, targets) in enumerate(loader):
                images, targets = images.float().to(self.device), targets.long().to(self.device)
                output = self.buggy_nn(images)
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
        
        model_dir = os.path.join(self.save_dir, self.dataset)
        model_dir = os.path.join(model_dir, self.attack)
        
        if not os.path.exists(model_dir):
            print('\n new dir <%s>\n' % model_dir)
            os.makedirs(model_dir)
            
        log_file = open(os.path.join(model_dir, '%s_%slog.txt' % (self.arch, self.attack)), 'a')

        now = datetime.now()
        log_info = now.strftime("%Y-%m-%d %H:%M:%S")

        log_info += f'Training Info - Model {self.arch} - Attack {self.attack} - Dataset {self.dataset} \n'

        self.buggy_nn = self.backdoor_model()
        self.buggy_nn.to(self.device)


        loader_nor = self.get_standard(set=self.dataset, num=self.train_num, train=True)
        loader_bd = self.get_backdoor(set=self.dataset, num=self.train_num, train=True, mode='train', attack=self.attack)
        val_loader_nor = self.get_standard(set=self.dataset, num=self.test_num, train=False)
        val_loader_bd = self.get_backdoor(set=self.dataset, num=self.test_num, train=False, mode='ptest', attack=self.attack)

        # if 'sig' not in self.arch:
        lr = 0.1 if self.dataset != 'ImageNette' else 0.01

        # we can not conduct successful attack if lr=0.1 dataset=SVHN attack=Blend model=VGG11
        if self.dataset == 'SVHN' and self.attack == 'Blend' and self.arch == 'VGG11':
            lr = 0.02
            
        optimizer = torch.optim.SGD(self.buggy_nn.parameters(), 
                                    lr=lr, 
                                    momentum=0.9,
                                    weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss().cuda()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        for epoch in range(epochs):
            self.buggy_nn.train()
            trainer(loader_bd)
            lr_scheduler.step()
            
            train_acc = self.test_accuracy(self.buggy_nn, loader_nor, self.device)
            test_acc = self.test_accuracy(self.buggy_nn, val_loader_nor, self.device)
            sr = self.test_accuracy(self.buggy_nn, loader_bd, self.device)
            
            info = f'Epoch : {epoch}, train acc : {train_acc}, test acc : {test_acc}, SR : {sr} \n'
            log_info += info
            print(info)
            save_model_part(self.buggy_nn, f'{self.arch}_poi.pt', out_dir=model_dir)
            

        test_acc = self.test_accuracy(self.buggy_nn, val_loader_nor, self.device)
        test_sr = self.test_accuracy(self.buggy_nn, val_loader_bd, self.device)

        log_info += 'Acc on test set %s, ' % test_acc
        log_info += 'Attack success rate on test set %s \n ' % test_sr
        log_file.write(log_info)
        log_file.flush()


    @torch.no_grad()
    def evaluation(self, ori_model, repair_model, device, logger=None):
        if isinstance(device, list):
            device0, device1 = device[0], device[1]
        else:
            device0 = device1 = device
            
        ori_eval = []
        ori_eval.append(self.test_accuracy(ori_model, self.test_dataloader, device0))
        ori_eval.append(self.test_accuracy(ori_model, self.repair_loader, device0))
        ori_eval.append(self.test_sr(ori_model, self.gene_loader, device0))
        ori_eval.append(self.test_accuracy(ori_model, self.gene_loader, device0))

        aft_eval = []
        aft_eval.append(self.test_accuracy(repair_model, self.test_dataloader, device1))
        aft_eval.append(self.test_accuracy(repair_model, self.repair_loader, device1))
        aft_eval.append(self.test_sr(repair_model, self.gene_loader, device1))
        aft_eval.append(self.test_accuracy(repair_model, self.gene_loader, device1))
        if logger is None:
            print(f"Before repair, accuracy on test set: {ori_eval[0]:8.4f}")
            print(f" After repair, accuracy on test set: {aft_eval[0]:8.4f}")

            print(f"               accuracy on repair set: {ori_eval[1]:8.4f}")
            print(f"               accuracy on repair set: {aft_eval[1]:8.4f}")

            print(f"Before repair,      sr on gene set : {ori_eval[2]:8.4f}")
            print(f" After repair,      sr on gene set : {aft_eval[2]:8.4f}")

            print(f"Before repair, accuracy on gene set: {ori_eval[3]:8.4f}")
            print(f" After repair, accuracy on gene set: {aft_eval[3]:8.4f}")
        else:
            logger.info(f"Before repair, accuracy on test set: {ori_eval[0]:8.4f}")
            logger.info(f" After repair, accuracy on test set: {aft_eval[0]:8.4f}")

            logger.info(f"               accuracy on repair set: {ori_eval[1]:8.4f}")
            logger.info(f"               accuracy on repair set: {aft_eval[1]:8.4f}")

            logger.info(f"Before repair,      sr on gene set : {ori_eval[2]:8.4f}")
            logger.info(f" After repair,      sr on gene set : {aft_eval[2]:8.4f}")

            logger.info(f"Before repair, accuracy on gene set: {ori_eval[3]:8.4f}")
            logger.info(f" After repair, accuracy on gene set: {aft_eval[3]:8.4f}")
        return ori_eval, aft_eval
        

class GTSRB(Dataset):
    def __init__(self, root, train=True, transform=None,):
        self.root = root
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),])

        if train:
            csv_path = os.path.join(root, "Train.csv")
        else:
            csv_path = os.path.join(root, "Test.csv")
        print(csv_path)
        df = pd.read_csv(csv_path)

        self.img_paths = list(df["Path"])
        self.class_ids = list(df["ClassId"])

        self._samples = [(os.path.join(root, path), label) for path, label in zip(df["Path"], df["ClassId"])]

    def __len__(self):
        return len(self.class_ids)

    def __getitem__(self, index):
        # img_path = os.path.join(self.root,self.img_paths[index])
        # img = Image.open(img_path)
        # label = self.class_ids[index]
        # label = torch.tensor(label).long()
        # img = self.transform(img)

        # return img,label
        path, target = self._samples[index]
        sample = Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(target).long()

        return sample, target

if __name__ == '__main__':
    

    dataset = 'ImageNette'
    arch = 'VGG19'

    dataset = 'CIFAR10'
    arch = 'ResNet18_dense'
    arch = 'ResNet18'

    dataset = 'CIFAR100'
    arch = 'Squeezenet'

    dataset = 'GTSRB'
    arch = 'VGG11'

    dataset = 'SVHN'
    arch = 'VGG11'
    arch = 'ResNet18'
    
    datadir = f'/data/home/mjnn/majianan/data/{dataset}'
    batch_size = 128
    r_num = 100
    attack = 'BadnetsA2A'
    attack = 'BadnetsA2O'
    attack = 'Sig'
    attack = 'Blend'

    epochs = {'SVHN': 50,
              'GTSRB': 50,
              'CIFAR10': 100,
              'CIFAR100': 100,
              'ImageNette': 100}[dataset]
    device = torch.device('cuda:0')
    save_dir = '/data/home/mjnn/majianan/ProvRepair/model/backdoor/'


    backdoor = Backdoor(dataset, datadir, batch_size, r_num, attack, device, save_dir, None, arch)
    backdoor.backdoor_train(epochs)
    pass