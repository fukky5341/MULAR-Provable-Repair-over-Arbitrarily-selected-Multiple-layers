import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
import numpy as np
import time
from utils.reader import from_eran
from utils.network import *
from utils.resnet import ResNet18
# from .backdoor import GTSRB

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from input_space.generate_input import clean_points, split_by_misclassification
from input_space.advex import adv_dataset



class Robustness():
    """
        ==============================
        
        self.test_dataloader: dataloader for test accuracy;
        self.repair_loader: dataloader for repair;
        self.gene_loader  : dataloader for evaluating the generalization (independent from the repair_loader)
    """

    def __init__(self, dataset, datadir, batch_size, buggy_nn, seed, device, dum_args) -> None:

        self.seed = seed
        # torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.dataset = dataset
        self.datadir = datadir
        self.batch_size = batch_size
        self.device = device
        
        print(f"Dataset: {self.dataset}")
        if self.dataset == 'MNIST':
            if dum_args.repair_task == 'Local Robustness':
                c_points = clean_points(dum_args)
                self.test_dataloader = c_points.dataloader(batch_size=self.batch_size, shuffle=False)
                data = torch.load(f"safe_radii/adv_{dum_args.model_name}.pt")
                data_keys = list(data.keys())
                self.buggyset = []
                for key in data_keys:
                    item = data[key]
                    self.buggyset.append((item['center'].to(dum_args.device, dum_args.dtype), torch.tensor(item['label'])))
                self.normalize = self.unnormalize = None
            elif dum_args.repair_task == 'LocalCounterexample':
                c_points = clean_points(dum_args)
                pos_xs, neg_xs = split_by_misclassification(c_points, buggy_nn)
                self.test_dataloader = c_points.dataloader(batch_size=self.batch_size, shuffle=False)
                self.buggyset = []
                for i in range(len(neg_xs.images)):
                    self.buggyset.append((neg_xs.images[i], neg_xs.labels[i]))
                self.normalize = self.unnormalize = None
            else:
                raise NotImplementedError(f"Repair task {dum_args.repair_task} not supported yet.")
        elif self.dataset == 'CIFAR10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            self.normalize = transforms.Normalize(mean=mean, std=std)
            self.unnormalize = transforms.Normalize(mean=-mean / std, std=1 / std)
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,])
            self.train_dataloader = DataLoader(torchvision.datasets.CIFAR10(root=self.datadir, train=True, 
                                                                download=True, transform=self.transform_test), 
                                                                batch_size=self.batch_size, shuffle=False)
            self.test_dataloader = DataLoader(torchvision.datasets.CIFAR10(root=self.datadir, train=False, 
                                                                download=True, transform=self.transform_test), 
                                                                batch_size=self.batch_size, shuffle=False)
            self.buggyset = self.filter_misclassified(buggy_nn, self.test_dataloader, self.device)
            # self.buggyset = self.filter_none(buggy_nn, self.train_dataloader, self.device)
            print(f"len of buggyset = {len(self.buggyset)}")
        
        # print(f"Dataset: {self.dataset}")
        # if self.dataset == 'MNIST':
        #     self.transform_test=torchvision.transforms.ToTensor()
        #     self.train_dataloader = DataLoader(torchvision.datasets.MNIST(root=self.datadir, train=True, 
        #                                                              download=True, transform=self.transform_test), 
        #                                                              batch_size=self.batch_size, shuffle=False)
        #     self.test_dataloader = DataLoader(torchvision.datasets.MNIST(root=self.datadir, train=False, 
        #                                                              download=True, transform=self.transform_test), 
        #                                                              batch_size=self.batch_size, shuffle=False)
            
        #     self.buggyset = self.filter_misclassified(buggy_nn, self.train_dataloader, self.device)
        #     self.normalize = self.unnormalize = None
        # elif self.dataset == 'CIFAR10':
        #     mean = torch.tensor([0.4914, 0.4822, 0.4465])
        #     std = torch.tensor([0.2023, 0.1994, 0.2010])
        #     self.normalize = transforms.Normalize(mean=mean, std=std)
        #     self.unnormalize = transforms.Normalize(mean=-mean / std, std=1 / std)
        #     self.transform_test = transforms.Compose([
        #         transforms.ToTensor(),
        #         self.normalize,])
        #     self.train_dataloader = DataLoader(torchvision.datasets.CIFAR10(root=self.datadir, train=True, 
        #                                                         download=True, transform=self.transform_test), 
        #                                                         batch_size=self.batch_size, shuffle=False)
        #     self.test_dataloader = DataLoader(torchvision.datasets.CIFAR10(root=self.datadir, train=False, 
        #                                                         download=True, transform=self.transform_test), 
        #                                                         batch_size=self.batch_size, shuffle=False)
        #     self.buggyset = self.filter_misclassified(buggy_nn, self.train_dataloader, self.device)
        #     # self.buggyset = self.filter_none(buggy_nn, self.train_dataloader, self.device)
        #     print(f"len of buggyset = {len(self.buggyset)}")
        # elif self.dataset == 'GTSRB':
        #     mean = torch.tensor([0.3403, 0.3121, 0.3214])
        #     std = torch.tensor([0.2724, 0.2608, 0.2669])

        #     self.normalize = transforms.Normalize(mean=mean, std=std)
        #     self.unnormalize = transforms.Normalize(mean=-mean / std, std=1 / std)
        #     self.transform_test = transforms.Compose([transforms.Resize((32, 32)),
        #                                     transforms.ToTensor(),
        #                                     self.normalize])
        #     self.train_dataloader = DataLoader(GTSRB(root=self.datadir, train=True, transform=self.transform_test), 
        #                                                         batch_size=self.batch_size, shuffle=False)
        #     self.test_dataloader = DataLoader(GTSRB(root=self.datadir, train=False, transform=self.transform_test), 
        #                                                         batch_size=self.batch_size, shuffle=False)
        #     self.buggyset = self.filter_misclassified(buggy_nn, self.train_dataloader, self.device)

        #     print(f"len of buggyset = {len(self.buggyset)}")
        

    def property_prepare(self, image, label, ndims, r, pick, loc, nclasses):
        if pick == 'nonzero':
            x_l, x_u = self.perturb_input_nonzero(image, ndims, r)
        elif 'patch' in pick:
            x_l, x_u = self.perturb_input_patch(image, ndims, r, loc)
        elif 'all' in pick:
            # perturb all pixels
            x_l, x_u = self.perturb_input_all(image, ndims, r)

        input_space = torch.stack([x_l, x_u], dim=-1)
        output_constraint_coff = torch.zeros((nclasses - 1, nclasses), device=self.device)
        output_constraint_bias = torch.zeros((nclasses - 1), device=self.device)
        row = 0
        for j in range(nclasses):
            if j != label:
                output_constraint_coff[row, label] = 1.0
                output_constraint_coff[row, j] = -1.0
                row += 1
        output_constraint = [output_constraint_coff, output_constraint_bias]
        return input_space, output_constraint
    
    def perturb_input_nonzero(self, tensor, k, r):
        if self.normalize is not None and self.unnormalize is not None:
            tensor = self.unnormalize(tensor)

        flat_tensor = tensor.flatten()
        non_zero_indices = torch.nonzero(flat_tensor, as_tuple=True)[0]
        num_elements = min(k, len(non_zero_indices))
        indices_to_modify = non_zero_indices[:num_elements]
        mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
        mask[indices_to_modify] = True

        tensor_l = torch.where(mask, flat_tensor - r, flat_tensor)
        tensor_u = torch.where(mask, flat_tensor + r, flat_tensor)

        # tensor_l = tensor_l.view(tensor.shape)
        # tensor_u = tensor_u.view(tensor.shape).view(tensor.shape)
        tensor_l = torch.clamp(tensor_l, min=0.0, max=1.0).view(tensor.shape)
        tensor_u = torch.clamp(tensor_u, min=0.0, max=1.0).view(tensor.shape)

        if self.normalize is not None and self.unnormalize is not None:
            tensor_l = self.normalize(tensor_l)
            tensor_u = self.normalize(tensor_u)

        return tensor_l, tensor_u
    
    def perturb_input_all(self, tensor, k, r):
        if self.normalize is not None and self.unnormalize is not None:
            tensor = self.unnormalize(tensor)

        tensor_l = torch.clamp(tensor - r, min=0.0, max=1.0).view(tensor.shape)
        tensor_u = torch.clamp(tensor + r, min=0.0, max=1.0).view(tensor.shape)

        if self.normalize is not None and self.unnormalize is not None:
            tensor_l = self.normalize(tensor_l)
            tensor_u = self.normalize(tensor_u)

        return tensor_l, tensor_u
    
    def perturb_input_patch(self, tensor, k, r, location):
        if self.normalize is not None and self.unnormalize is not None:
            tensor = self.unnormalize(tensor)
        assert len(tensor.shape) == 3
        inds =  np.arange(tensor.numel(), dtype=int).reshape(tensor.shape)
        if location == 'center':
            startx = tensor.shape[1] // 2 - (k // 2)
            starty = tensor.shape[2] // 2 - (k // 2)
            indices_to_modify = inds[:, startx: startx + k, starty: starty + k]
        elif location == 'bottom-right':
            startx = tensor.shape[1] - k
            starty = tensor.shape[2] - k
            indices_to_modify = inds[:, startx: startx + k, starty: starty + k]
        elif location == 'bottom-left':
            startx = tensor.shape[1] - k
            indices_to_modify = inds[:, startx: startx + k, 0: k]
        elif location == 'top-right':
            starty = tensor.shape[2] - k
            indices_to_modify = inds[:, 0: k, starty: starty + k]
        elif location == 'top-left':
            indices_to_modify = inds[:, 0: k, 0: k]
        # print(indices_to_modify)
        indices_to_modify = indices_to_modify.flatten()
        flat_tensor = tensor.flatten()
        mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
        mask[indices_to_modify] = True

        tensor_l = torch.where(mask, flat_tensor - r, flat_tensor)
        tensor_u = torch.where(mask, flat_tensor + r, flat_tensor)

        # tensor_l = tensor_l.view(tensor.shape)
        # tensor_u = tensor_u.view(tensor.shape).view(tensor.shape)
        tensor_l = torch.clamp(tensor_l, min=0.0, max=1.0).view(tensor.shape)
        tensor_u = torch.clamp(tensor_u, min=0.0, max=1.0).view(tensor.shape)

        if self.normalize is not None and self.unnormalize is not None:
            tensor_l = self.normalize(tensor_l)
            tensor_u = self.normalize(tensor_u)

        return tensor_l, tensor_u


    def filter_misclassified(self, model, dataloader, device):
        """ For a given model, filter out correct samples to obtain a dataloader consisting of misclassified samples. """
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
        return incorrect_samples
    

    def filter_none(self, model, dataloader, device):
        model.eval()

        samples = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                samples.extend([(d, t) for d, t in zip(inputs, labels)])
        return samples


    def evaluation(self, ori_model, repair_model, device, logger=None):

        ori_acc = self.test_accuracy(ori_model, self.test_dataloader, device)
        aft_acc = self.test_accuracy(repair_model, self.test_dataloader, device)
        
        if logger is None:
            print(f"Before repair, accuracy on test set: {ori_acc:8.4f}")
            print(f" After repair, accuracy on test set: {aft_acc:8.4f}")

        else:
            logger.append(f"Before repair, accuracy on test set: {ori_acc:8.4f} \n")
            logger.append(f" After repair, accuracy on test set: {aft_acc:8.4f} \n")

                
    def test_accuracy(self, model, dataloader, device, dtype=torch.float32, batch_vis=False):
        model.eval()

        correct = 0
        total = 0
        l = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device, dtype=dtype)
                labels = labels.to(device)
                labels = labels.view(-1)

                outputs = model(inputs)

                _, predictions = torch.max(outputs, dim=1)

                correct_inp = predictions == labels
                batch_correct = correct_inp.sum().item()
                correct += batch_correct
                total += labels.size(0)

                l.append(batch_correct)  # per-batch, not cumulative

        accuracy = correct / total

        if batch_vis:
            print(l)

        return accuracy * 100
    



def robustness_mnist_model(arch, device):
    read_para = {}
    if arch == '3x100':
        buggy_nn = FNN_3_100().to(device)
        read_nn = from_eran('model/robustness/MNIST/mnist_relu_3_100.tf')
    # elif arch == '3xgelu':
    #     buggy_nn = FNN_3_100_gelu().to(device)
    #     checkpoint = torch.load('model/mnist/mnist_gelu_3_100.pth')
    #     for name in checkpoint:
    #         read_para['classifier.' + name] = checkpoint[name]
    #     buggy_nn.load_state_dict(read_para)
    # elif arch == '6xgelu':
    #     buggy_nn = FNN_6_100_gelu().to(device)
    #     ckpt = torch.load('model/mnist/mnist_gelu_6_100.pth')
    #     buggy_nn.load_state_dict(ckpt)
    # elif arch == '9x100_gelu':
    #     buggy_nn = FNN_9_100_gelu().to(device)
    #     ckpt = torch.load('model/mnist/mnist_gelu_9_100.pth')
    #     buggy_nn.load_state_dict(ckpt)
    # elif arch == '6x100':
    #     buggy_nn = FNN_6_100().to(device)
    #     ckpt = torch.load('model/mnist/mnist_relu_6_100.pth')
    #     buggy_nn.load_state_dict(ckpt)
    # elif arch == '9x100':
    #     buggy_nn = FNN_9_100().to(device)
    #     read_nn = from_eran('model/mnist/mnist_relu_9_100.tf')
    # elif arch == '9x200':
    #     buggy_nn = FNN_9_200().to(device)
    #     read_nn = from_eran('model/mnist/mnist_relu_9_200.tf')
    
    if arch not in ['3xgelu', '6x100', '6xgelu', '9x100_gelu']:
        for name in read_nn.state_dict():
            read_para['classifier.' + name] = read_nn.state_dict()[name]
        buggy_nn.load_state_dict(read_para)
        buggy_nn.eval()
        
    return buggy_nn

def robustness_gtsrb_model(arch, device):
    read_para = {}
    if arch == 'cnn3':
        buggy_nn = CNN_small(43)
        checkpoint = torch.load('model/robustness/GTSRB/cnn3/cnn3.pth')
    elif arch == 'cnn4':
        buggy_nn = CNN4()
        checkpoint = torch.load('model/robustness/GTSRB/cnn4/cnn4.pth')
    buggy_nn.load_state_dict(checkpoint)
    buggy_nn = buggy_nn.to(device)
    buggy_nn.eval()    
    return buggy_nn

def robustness_cifar_model(arch, device):
    read_para = {}
    if arch == 'cnn4':
        buggy_nn = CNN4()
        checkpoint = torch.load('model/robustness/CIFAR10/cnn4/cnn4.pth')
    elif arch == 'cnn6':
        buggy_nn = CNN6()
        checkpoint = torch.load('model/robustness/CIFAR10/cnn6/cnn6.pth')
    elif arch == 'vgg11':
        buggy_nn = VGG('VGG11')
        checkpoint = torch.load('model/cifar/vgg/vgg11.pth')
    elif arch == 'resnet18':
        buggy_nn = ResNet18()
        checkpoint = torch.load('model/cifar/resnet/resnet18.pth')
    buggy_nn.load_state_dict(checkpoint)
    buggy_nn = buggy_nn.to(device)
    buggy_nn.eval()    
    return buggy_nn
