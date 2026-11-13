import os
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Subset

import time

from utils.reader import from_eran
from utils.network import *
from utils.resnet import ResNet18

MNIST_CORRUPTION = ["brightness", "canny_edges", "dotted_line",
                    "fog", "glass_blur", "identity", "impulse_noise",
                    "motion_blur", "rotate", "scale", "shear", "shot_noise",
                    "spatter", "stripe", "translate", "zigzag"]

class Corruption():
    """
        ==============================
        
        self.test_dataloader: dataloader for test accuracy;
        self.repair_loader: dataloader for repair;
        self.gene_loader  : dataloader for evaluating the generalization (independent from the repair_loader)
    """

    def __init__(self, dataset, datadir, batch_size, corruption, r_num, buggy_nn, device) -> None:

        self.dataset = dataset
        self.datadir = datadir
        self.batch_size = batch_size
        self.corruption = corruption
        self.r_num = r_num
        self.mis_num = 0
        self.device = device
        
        self.seed = 2024
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        print(f"Dataset: {self.dataset} Repair num: {self.r_num}")
        if self.dataset == 'MNIST-C':
            self.transform_test=torchvision.transforms.ToTensor()

            self.test_dataloader = DataLoader(torchvision.datasets.MNIST(root=self.datadir, train=False, 
                                              download=True, transform=self.transform_test), batch_size=self.batch_size)
            self.mnist_c_loader(self.corruption, self.r_num)
            self.filter_misclassified(buggy_nn, self.repair_loader, self.device)

            # filter to obtain N misclassified data
            while self.mis_num < r_num:
                self.r_num += 1
                self.mnist_c_loader(self.corruption, self.r_num)
                self.repair_loader = self.filter_misclassified(buggy_nn, self.repair_loader, self.device)
            # pass

    def mnist_c_loader(self, corruption, index, num_workers=1):
        path = os.path.join(self.datadir, 'mnist_c')
        if corruption != 'all':
            image_path = os.path.join(path, f'{corruption}/test_images.npy')
            label_path = os.path.join(path, f'{corruption}/test_labels.npy')
            data = torch.from_numpy(np.load(image_path)).permute(0, 3, 1, 2).float() / 255.0
            labels = torch.from_numpy(np.load(label_path))  
            
            repair_data = data[0: index]
            repair_label = labels[0: index]
            
            GENE_SET_IND = 1000
            gene_data = data[GENE_SET_IND:]
            gene_label = labels[GENE_SET_IND:]
            
            repair_set = TensorDataset(repair_data, repair_label)
            gene_set = TensorDataset(gene_data, gene_label)
            
            self.repair_loader = DataLoader(repair_set, batch_size=self.batch_size, shuffle=False)
            self.gene_loader = DataLoader(gene_set, batch_size=self.batch_size, shuffle=False)
        else:
            # load all corruptions:
            all_data = {}
            all_labels = {}
            for c in MNIST_CORRUPTION:
                image_path = os.path.join(path, f'{c}/test_images.npy')
                label_path = os.path.join(path, f'{c}/test_labels.npy')
                data = torch.from_numpy(np.load(image_path)).permute(0, 3, 1, 2).float() / 255.0
                labels = torch.from_numpy(np.load(label_path))  
                all_data[c] = data
                all_labels[c] = labels
            repair_data = []
            repair_label = []
            gene_data = []
            gene_label = []
            p = 0
            while len(repair_data) < index:
                for c in MNIST_CORRUPTION:
                    repair_data.append(all_data[c][p: p+1])
                    repair_label.append(all_labels[c][p: p+1])
                    if len(repair_data) == index:
                        break
                p += 1 

            GENE_SET_IND = 1000
            for c in MNIST_CORRUPTION:
                gene_data.append(all_data[c][GENE_SET_IND:])
                gene_label.append(all_labels[c][GENE_SET_IND:])
            repair_data = torch.cat(repair_data, dim=0)
            repair_label = torch.cat(repair_label, dim=0)
            gene_data = torch.cat(gene_data, dim=0)
            gene_label = torch.cat(gene_label, dim=0)
            repair_set = TensorDataset(repair_data, repair_label)
            gene_set = TensorDataset(gene_data, gene_label)
            
            self.repair_loader = DataLoader(repair_set, batch_size=self.batch_size, shuffle=False)
            self.gene_loader = DataLoader(gene_set, batch_size=self.batch_size, shuffle=False)
        

    def filter_misclassified(self, model, dataloader, device):
        """
        For a given model, filter out correct samples to obtain a dataloader consisting of misclassified samples.
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

        # Create a new DataLoader for incorrect samples
        self.mis_num = len(incorrect_samples)
        # self.r_num = len(incorrect_samples)
        # print(f"{self.mis_num =}")
        mis_dataloader = DataLoader(incorrect_samples, batch_size=self.batch_size, shuffle=False)
        return mis_dataloader


    def evaluation(self, ori_model, repair_model, device, dtype=torch.float32, logger=None):
        EVAL = [self.test_dataloader, self.repair_loader, self.gene_loader]

        ori_eval = []
        for loader in EVAL:
            ori_eval.append(self.test_accuracy(ori_model, loader, device, dtype=dtype))

        aft_eval = []    
        for loader in EVAL:
            aft_eval.append(self.test_accuracy(repair_model, loader, device, dtype=dtype))
        
        if logger is None:
            print(f"Before repair, accuracy on test set: {ori_eval[0]:8.4f}")
            print(f" After repair, accuracy on test set: {aft_eval[0]:8.4f}")

            print(f"             accuracy on repair set: {ori_eval[1]:8.4f}")
            print(f"             accuracy on repair set: {aft_eval[1]:8.4f}")

            print(f"Before repair,      generalization : {ori_eval[2]:8.4f}")
            print(f" After repair,      generalization : {aft_eval[2]:8.4f}")
        else:
            logger.info(f"Before repair, accuracy on test set: {ori_eval[0]:8.4f}")
            logger.info(f" After repair, accuracy on test set: {aft_eval[0]:8.4f}")

            logger.info(f"             accuracy on repair set: {ori_eval[1]:8.4f}")
            logger.info(f"             accuracy on repair set: {aft_eval[1]:8.4f}")

            logger.info(f"Before repair,      generalization : {ori_eval[2]:8.4f}")
            logger.info(f" After repair,      generalization : {aft_eval[2]:8.4f}")

        
        
    def test_accuracy(self, model, dataloader, device, dtype=torch.float32, batch_vis=False):
        model.eval()
        model = model.to(device)

        correct = 0
        total = 0
        l = []
        start = time.time()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device, dtype=dtype), labels.to(device)
                outputs = model(inputs)
                    
                _, predictions = torch.max(outputs, 1)
                correct += torch.sum(predictions == labels).item()
                total += len(labels)
                l.append(correct)
        accuracy = correct / total
        if batch_vis:
            print(l)
        return accuracy * 100
    
    def model_para_diff(self, ori_model, new_model):
        for name in ori_model.state_dict():
            if torch.sum(ori_model.state_dict()[name]) != torch.sum(new_model.state_dict()[name]):
                print('Before', name, ori_model.state_dict()[name].shape, torch.sum(ori_model.state_dict()[name]).item())
                print('After', name, new_model.state_dict()[name].shape, torch.sum(new_model.state_dict()[name]).item())


    

def mnist_model(arch, device):
    read_para = {}
    if arch == '3x100':
        buggy_nn = FNN_3_100().to(device)
        read_nn = from_eran('model/mnist/mnist_relu_3_100.tf')
    elif arch == '6x100':
        buggy_nn = FNN_6_100().to(device)
        ckpt = torch.load('model/mnist/mnist_relu_6_100.pth')
        buggy_nn.load_state_dict(ckpt)
    elif arch == '9x100':
        buggy_nn = FNN_9_100().to(device)
        read_nn = from_eran('model/mnist/mnist_relu_9_100.tf')
    elif arch == '9x200':
        buggy_nn = FNN_9_200().to(device)
        read_nn = from_eran('model/mnist/mnist_relu_9_200.tf')
    
    if arch not in ['3xgelu', '6x100', '6xgelu', '9x100_gelu', '9x200_gelu']:
        for name in read_nn.state_dict():
            read_para['classifier.' + name] = read_nn.state_dict()[name]
        buggy_nn.load_state_dict(read_para)
        buggy_nn.eval()
        
    return buggy_nn
 