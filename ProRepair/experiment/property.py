import numpy as np
import torch
import argparse
import random
from utils.network import FNN
import itertools
from torch.utils.data import Subset, DataLoader, TensorDataset
#property 7 N19
#  ": "[(-0.3284,0.6799), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5)]",
#    "assert": "(FA x . TRUE => arg_min(x) != 3 && arg_min(x) != 4)"


BATCH_SIZE = 32
DRAWNDOWN_SIZE = 32 * 313
COUNTEREG_SIZE = 32 * 313
Fidelity_SIZE = 5000

class Safety_Property():
    def __init__(self) -> None:
        self.unnormalizeLB = [0.0, -3.141593, -3.141593, 100.0, 0.0]
        self.unnormalizeUB = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]

        # COC, WL, WR, SL, SR = list(range(5))
     
    def p1(self, ):
        """
        input space: after normalize
        output_constraint: y[0] <= 1500 (before normalization); c * y + d >=0; output_constraint = [C; D]
        """
        input_space = torch.Tensor([
                                    [    0.6,  0.6798577687], 
                                    [   -0.5,           0.5], 
                                    [   -0.5,           0.5], 
                                    [   0.45,           0.5], 
                                    [   -0.5,         -0.45],
                                ])
        output_constraint = [torch.Tensor([[  -1, 0, 0, 0, 0], # - y[0] + 3.9911256459 >= 0
                                           [  -1, 0, 0, 0, 0], # - y[0] + 3.9911256459 >= 0
                                           [  -1, 0, 0, 0, 0], # - y[0] + 3.9911256459 >= 0
                                           [  -1, 0, 0, 0, 0], # - y[0] + 3.9911256459 >= 0
                                          ]),
                             torch.Tensor([3.9911256459, 
                                           3.9911256459, 
                                           3.9911256459, 
                                           3.9911256459, ]),]
        
        verify_constraint = [[torch.Tensor([ -1, 0, 0, 0, 0]), torch.tensor(3.9911256459)],
                             [torch.Tensor([ -1, 0, 0, 0, 0]), torch.tensor(3.9911256459)],
                             [torch.Tensor([ -1, 0, 0, 0, 0]), torch.tensor(3.9911256459)],
                             [torch.Tensor([ -1, 0, 0, 0, 0]), torch.tensor(3.9911256459)],
                             ]
        return input_space, output_constraint, verify_constraint
    
    def p2(self, ):
        """
        input space: after normalize
        output_constraint: y[0] is minimal ; c * y + d >=0; output_constraint = [C; D]
        """
        input_space = torch.Tensor([
                                    [    0.6,  0.6798577687], 
                                    [   -0.5,           0.5], 
                                    [   -0.5,           0.5], 
                                    [   0.45,           0.5], 
                                    [   -0.5,         -0.45],
                                ])
        output_constraint = [torch.Tensor([[  -1, 1, 0, 0, 0], # - y[0] + y[1] >=0
                                           [  -1, 0, 1, 0, 0], # - y[0] + y[2] >=0
                                           [  -1, 0, 0, 1, 0], # - y[0] + y[3] >=0
                                           [  -1, 0, 0, 0, 1], # - y[0] + y[4] >=0
                                          ]),
                             torch.Tensor([0.0, 
                                           0.0,
                                           0.0, 
                                           0.0, 
                                          ]),]
        """"[(a and b and ...) or (c and d and ...) or ...]"""
        verify_constraint = [[torch.Tensor([  -1, 1, 0, 0, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  -1, 0, 1, 0, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  -1, 0, 0, 1, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  -1, 0, 0, 0, 1,]), torch.tensor(0.0)],]

        return input_space, output_constraint, verify_constraint

    def p3(self, ):
        """
        input space: after normalize
        output_constraint: y[0] is maximal ; c * y + d >=0; output_constraint = [C; D]
        """
        input_space = torch.Tensor([
                                    [  -0.3035311561, -0.2985528119], 
                                    [  -0.0095492966,  0.0095492966], 
                                    [   0.4933803236,           0.5], 
                                    [            0.3,           0.5], 
                                    [            0.3,           0.5],
                                ])
        output_constraint = [torch.Tensor([
                                        [  1, -1, 0, 0, 0], # y[0] - y[1] >=0
                                        [  1, 0, -1, 0, 0], # y[0] - y[2] >=0
                                        [  1, 0, 0, -1, 0], # y[0] - y[3] >=0
                                        [  1, 0, 0, 0, -1], # y[0] - y[4] >=0
                                ]),
                             torch.Tensor([
                                        0.0, 
                                        0.0,
                                        0.0, 
                                        0.0, 
                                ]),]
        """"[(a and b and ...) or (c and d and ...) or ...]"""
        verify_constraint = [[torch.Tensor([  1, -1, 0, 0, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  1, 0, -1, 0, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  1, 0, 0, -1, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  1, 0, 0, 0, -1,]), torch.tensor(0.0)],]
        return input_space, output_constraint, verify_constraint
    
    def p4(self, ):
        """
        input space: after normalize
        output_constraint: y[0] is maximal ; c * y + d >=0; output_constraint = [C; D]
        """
        input_space = torch.Tensor([
                                    [  -0.3035311561, -0.2985528119], 
                                    [  -0.0095492966,  0.0095492966], 
                                    [              0,             0], 
                                    [   0.3181818182,           0.5], 
                                    [   0.0833333333,  0.1666666667],
                                ])
        output_constraint = [torch.Tensor([
                                        [  1, -1, 0, 0, 0], # y[0] - y[1] >=0
                                        [  1, 0, -1, 0, 0], # y[0] - y[2] >=0
                                        [  1, 0, 0, -1, 0], # y[0] - y[3] >=0
                                        [  1, 0, 0, 0, -1], # y[0] - y[4] >=0
                                ]),
                             torch.Tensor([
                                        0.0, 
                                        0.0,
                                        0.0, 
                                        0.0, 
                                ]),]
        """"[(a and b and ...) or (c and d and ...) or ...]"""
        verify_constraint = [[torch.Tensor([  1, -1, 0, 0, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  1, 0, -1, 0, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  1, 0, 0, -1, 0,]), torch.tensor(0.0)],
                             [torch.Tensor([  1, 0, 0, 0, -1,]), torch.tensor(0.0)],]
        return input_space, output_constraint, verify_constraint
    
    def p8(self, ):
        """
        input space: after normalize
        output_constraint:  the score for “weak left” is minimal or the score for COC is minimal. 
                            c * y + d >=0; output_constraint = [C; D]
        """

        input_space = torch.Tensor([
                                    [  -0.3284228772,  0.6798577687], 
                                    [           -0.5,        -0.375], 
                                    [   -0.031847133,   0.031847133], 
                                    [   -0.045454545,           0.5], 
                                    [             0.,           0.5],
                                ])
        output_constraint = [torch.Tensor([
                                        [  -1, 0, 1, 0, 0], # -y[0] + y[2] >=0
                                        [  -1, 0, 0, 1, 0], # -y[0] + y[3] >=0
                                        [  -1, 0, 0, 0, 1], # -y[0] + y[4] >=0
                                        [  0, -1, 1, 0, 0], # -y[1] + y[2] >=0
                                        [  0, -1, 0, 1, 0], # -y[1] + y[3] >=0
                                        [  0, -1, 0, 0, 1], # -y[1] + y[4] >=0
                                ]),
                             torch.Tensor([
                                        0.0, 
                                        0.0,
                                        0.0, 
                                        0.0, 
                                        0.0, 
                                        0.0, 
                                ]),]
        """"[(a and b and ...) or (c and d and ...) or ...]"""
        # verify_constraint = [[torch.Tensor([[  -1, 0, 1, 0, 0], 
        #                                    [  -1, 0, 0, 1, 0],
        #                                    [  -1, 0, 0, 0, 1], ]), torch.tensor([0.0, 0.0, 0.0])], 
        #                      [torch.Tensor([[  0, -1, 1, 0, 0], 
        #                                    [  0, -1, 0, 1, 0],
        #                                    [  0, -1, 0, 0, 1], ]), torch.tensor([0.0, 0.0, 0.0])]
        #                     ] 
        verify_constraint = [[torch.Tensor([
                                        [  -1, 0, 1, 0, 0], # -y[0] + y[2] >=0
                                        [  -1, 0, 0, 1, 0], # -y[0] + y[3] >=0
                                        [  -1, 0, 0, 0, 1], # -y[0] + y[4] >=0
                                        [  0, -1, 1, 0, 0], # -y[1] + y[2] >=0
                                        [  0, -1, 0, 1, 0], # -y[1] + y[3] >=0
                                        [  0, -1, 0, 0, 1], # -y[1] + y[4] >=0
                                ]),
                             torch.Tensor([
                                        0.0, 
                                        0.0,
                                        0.0, 
                                        0.0, 
                                        0.0, 
                                        0.0, 
                                ]),]]

        return input_space, output_constraint, verify_constraint

    def property_partition(self, property, l):
        """ Preprocessing of desirable property. """
        def space_partition(space, l):
            assert space.shape == (5, 2), f"input error! {space.shape}"
            ndim_bounds = []
            for lb, ub in space:
                dim_num =  max(int(torch.ceil((ub - lb) / l)), 1)
                end_points = torch.linspace(lb, ub, dim_num + 1, dtype=space.dtype, device=space.device)
                bounds = []
                for i in range(len(end_points) - 1):
                    bounds.append((end_points[i], end_points[i+1]))
                ndim_bounds.append(bounds)
            return torch.stack(tuple(torch.tensor(i) for i in itertools.product(*ndim_bounds))).to(space.device)

        refined_spaces = space_partition(property['input_space'], l)
        refined_property = []
        for space in refined_spaces:
            new_p = property.copy()
            new_p.update({'input_space': space})
            refined_property.append(new_p)
        return refined_property
    

class Point_wise_Safety_Property():
    def __init__(self, net, N_vio, N_sat=0, p=2):
        BATCH_SIZE = 256
        correct_data = torch.load(f'data/safety/{net}/drawdown.pt')
        correct_data_test = torch.load(f'data/safety/{net}/drawdown_test.pt')
        mis_data = torch.load(f'data/safety/{net}/counterexample.pt')
        mis_data_test = torch.load(f'data/safety/{net}/counterexample_test.pt')
        fidelty_set = torch.load(f'data/fidelity_data/AcasNetID_{net[1]},{net[2]}_-normed-test.pt')

        random.seed(2024)
        if p == 2:
            ind = random.sample(range(len(correct_data)), N_sat)
            correct_labels = torch.full((N_sat,), 0)
            cor_dataset = TensorDataset(correct_data[ind], correct_labels)
            self.sat_loader = DataLoader(dataset=cor_dataset, batch_size=BATCH_SIZE, shuffle=False)

            ind = random.sample(range(len(mis_data)), N_vio)
            vio_labels = torch.full((N_vio,), 0)
            vio_dataset = TensorDataset(mis_data[ind], vio_labels)
            self.vio_loader = DataLoader(dataset=vio_dataset, batch_size=BATCH_SIZE, shuffle=False)

            sat_testset = TensorDataset(correct_data_test, torch.full((len(correct_data_test),), 0))
            vio_testset = TensorDataset(mis_data_test, torch.full((len(mis_data_test),), 0))
            self.sat_test_loader = DataLoader(dataset=sat_testset, batch_size=BATCH_SIZE, shuffle=False)
            self.vio_test_loader = DataLoader(dataset=vio_testset, batch_size=BATCH_SIZE, shuffle=False)


    def property_check(self, model, loader, p, device):
        if p == 2:
            cons = torch.Tensor([[  -1, 1, 0, 0, 0], # - y[0] + y[1] >=0
                                [  -1, 0, 1, 0, 0], # - y[0] + y[2] >=0
                                [  -1, 0, 0, 1, 0], # - y[0] + y[3] >=0
                                [  -1, 0, 0, 0, 1], # - y[0] + y[4] >=0
                                ]).to(device)
        else:
            raise 'Property not support!'
        
        sat = 0
        all = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            cons_values = torch.matmul(output, cons.T)
            sat += torch.sum(torch.min(cons_values, dim=1)[0] >= 0).item()
            all += len(y)
        return sat / all



    def fidelity_test(self, model, loader, p, device):
        cor = 0
        all = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = torch.min(output.cpu(), 1)[1].numpy()
            cor += (pred == y.cpu().numpy()).sum()
            all += len(y)
        return cor / all



def load_model(net, path, device):
    mapping = {
        'classifier.0.weight': 'dense1.weight',
        'classifier.0.bias': 'dense1.bias',
        'classifier.2.weight': 'dense2.weight',
        'classifier.2.bias': 'dense2.bias',
        'classifier.4.weight': 'dense3.weight',
        'classifier.4.bias': 'dense3.bias',
        'classifier.6.weight': 'dense4.weight',
        'classifier.6.bias': 'dense4.bias',
        'classifier.8.weight': 'dense5.weight',
        'classifier.8.bias': 'dense5.bias',
        'classifier.10.weight': 'dense6.weight',
        'classifier.10.bias': 'dense6.bias',
        'classifier.12.weight': 'out.weight',
        'classifier.12.bias': 'out.bias'}
    buggy_nn = FNN()
    if path is None:
        ckpt = torch.load(f'/data/home/mjnn/majianan/ProvRepair/model/acasxu/{net}.pth')
    else:
        ckpt = torch.load(path)
    para = {}
    for new, old in mapping.items():
        if old in ckpt:
            para[new] = ckpt[old]
    if para == {}:
        mapping = {
        'classifier.0.weight': '0.weight',
        'classifier.0.bias': '0.bias',
        'classifier.2.weight': '2.weight',
        'classifier.2.bias': '2.bias',
        'classifier.4.weight': '4.weight',
        'classifier.4.bias': '4.bias',
        'classifier.6.weight': '6.weight',
        'classifier.6.bias': '6.bias',
        'classifier.8.weight': '8.weight',
        'classifier.8.bias': '8.bias',
        'classifier.10.weight': '10.weight',
        'classifier.10.bias': '10.bias',
        'classifier.12.weight': '12.weight', 
        'classifier.12.bias': '12.bias'
        }
        for new, old in mapping.items():
            if old in ckpt:
                para[new] = ckpt[old]
    buggy_nn.load_state_dict(para)
    buggy_nn.eval()
    buggy_nn.to(device)
    return buggy_nn



def __generate_x_7():
    x_0 = np.random.uniform(low=-0.3284, high=0.6799, size=(BATCH_SIZE, 1))
    x_1 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_2 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_3 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_4 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x = np.array([x_0, x_1, x_2, x_3, x_4])
    return np.transpose(x.reshape(5,32))

def __generate_x_2():
    x_0 = np.random.uniform(low=0.6, high=0.6799, size=(BATCH_SIZE, 1))
    x_1 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_2 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_3 = np.random.uniform(low=0.45, high=0.5, size=(BATCH_SIZE, 1))
    x_4 = np.random.uniform(low=-0.5, high=-0.45, size=(BATCH_SIZE, 1))
    x = np.array([x_0, x_1, x_2, x_3, x_4])
    return np.transpose(x.reshape(5,32))

def __generate_x_3():
    x_0 = np.random.uniform(low=-0.3035311561, high=-0.2985528119, size=(BATCH_SIZE, 1))
    x_1 = np.random.uniform(low=-0.0095492966, high=0.0095492966, size=(BATCH_SIZE, 1))
    x_2 = np.random.uniform(low=0.4933803236, high=0.5, size=(BATCH_SIZE, 1))
    x_3 = np.random.uniform(low=0.3, high=0.5, size=(BATCH_SIZE, 1))
    x_4 = np.random.uniform(low=0.3, high=0.5, size=(BATCH_SIZE, 1))
    x = np.array([x_0, x_1, x_2, x_3, x_4])
    return np.transpose(x.reshape(5,32))

def __generate_x_8():
    x_0 = np.random.uniform(low=-0.3284228772, high=0.6798577687, size=(BATCH_SIZE, 1))
    x_1 = np.random.uniform(low=-0.5, high=-0.375, size=(BATCH_SIZE, 1))
    x_2 = np.random.uniform(low=-0.031847133, high=0.031847133, size=(BATCH_SIZE, 1))
    x_3 = np.random.uniform(low=-0.045454545, high=0.5, size=(BATCH_SIZE, 1))
    x_4 = np.random.uniform(low=0., high=0.5, size=(BATCH_SIZE, 1))
    x = np.array([x_0, x_1, x_2, x_3, x_4])
    return np.transpose(x.reshape(5,32))


def property_satisfied_8(pre_y):
    #pre_y = np.argmin(y, axis=1)
    if pre_y == 0 or pre_y == 1:
        return True
    return False

def property_satisfied_7(pre_y):
    #pre_y = np.argmin(y, axis=1)
    if pre_y != 3 and pre_y != 4:
        return True
    return False

def property_satisfied_2(pre_y):
    #pre_y = np.argmin(y, axis=1)
    if pre_y == 0:
        return True
    return False

def property_satisfied_3(pre_y):
    #pre_y = np.argmin(y, axis=1)
    if pre_y != 0:
        return True
    return False


def gen_data_set_Fidelity_2(model, data_path, ty):
    """generate drawndown and counter example data set"""
    # property 2
    n_dd = 0
    Fid = torch.empty([Fidelity_SIZE, 5])
    print(Fid.shape)
    dd_saved = 0

    while True:
        x = __generate_x_2()
        x = torch.Tensor(x).cuda()
        print(x.shape, x.device)
        y = model(x.reshape(BATCH_SIZE,5))
        pre_y =  torch.argmin(y, axis=1)
        for i in range (0, BATCH_SIZE):
            if (property_satisfied_2(pre_y[i])):
                if n_dd < Fidelity_SIZE:
                    Fid[n_dd] = x[i].detach().clone()
                    n_dd = n_dd + 1
                    print('n_dd:{}'.format(n_dd))

        if n_dd >= Fidelity_SIZE and dd_saved == 0:
            dd_saved = 1
            # dd = torch.Tensor(dd)
            print('Fidelity ', Fid.shape)
            print('Fidelity ', Fid)
            torch.save(Fid, data_path)
        if dd_saved == 1:
            break
    return n_dd

def gen_data_set_Fidelity_3(model, data_path, ty):
    """generate drawndown and counter example data set"""
    # property 2
    n_dd = 0
    Fid = torch.empty([Fidelity_SIZE, 5])
    print(Fid.shape)
    dd_saved = 0

    while True:
        x = __generate_x_3()
        x = torch.Tensor(x).cuda()
        print(x.shape, x.device)
        y = model(x.reshape(BATCH_SIZE,5))
        pre_y =  torch.argmin(y, axis=1)
        for i in range (0, BATCH_SIZE):
            if (property_satisfied_3(pre_y[i])):
                if n_dd < Fidelity_SIZE:
                    Fid[n_dd] = x[i].detach().clone()
                    n_dd = n_dd + 1
                    print('n_dd:{}'.format(n_dd))

        if n_dd >= Fidelity_SIZE and dd_saved == 0:
            dd_saved = 1
            # dd = torch.Tensor(dd)
            print('Fidelity ', Fid.shape)
            print('Fidelity ', Fid)
            torch.save(Fid, data_path)
        if dd_saved == 1:
            break
    return n_dd


def gen_data_set_Fidelity_7(model, data_path, ty):
    """generate drawndown and counter example data set"""
    # property 2
    n_dd = 0
    Fid = torch.empty([Fidelity_SIZE, 5])
    print(Fid.shape)
    dd_saved = 0
    while True:
        x = __generate_x_7()
        x = torch.Tensor(x).cuda()
        print(x.shape, x.device)
        y = model(x.reshape(BATCH_SIZE,5))
        pre_y =  torch.argmin(y, axis=1)
        for i in range (0, BATCH_SIZE):
            if (property_satisfied_7(pre_y[i])):
                if n_dd < Fidelity_SIZE:
                    Fid[n_dd] = x[i].detach().clone()
                    n_dd = n_dd + 1
                    print('n_dd:{}'.format(n_dd))

        if n_dd >= Fidelity_SIZE and dd_saved == 0:
            dd_saved = 1
            # dd = torch.Tensor(dd)
            print('Fidelity ', Fid.shape)
            print('Fidelity ', Fid)
            torch.save(Fid, data_path)
        if dd_saved == 1:
            break
    return n_dd


def gen_data_set_Fidelity_8(model, data_path, ty):
    """generate drawndown and counter example data set"""
    # property 2
    n_dd = 0
    Fid = torch.empty([Fidelity_SIZE, 5])
    print(Fid.shape)
    dd_saved = 0
    while True:
        x = __generate_x_8()
        x = torch.Tensor(x).cuda()
        print(x.shape, x.device)
        y = model(x.reshape(BATCH_SIZE,5))
        pre_y =  torch.argmin(y, axis=1)
        for i in range (0, BATCH_SIZE):
            if (property_satisfied_8(pre_y[i])):
                if n_dd < Fidelity_SIZE:
                    Fid[n_dd] = x[i].detach().clone()
                    n_dd = n_dd + 1
                    print('n_dd:{}'.format(n_dd))

        if n_dd >= Fidelity_SIZE and dd_saved == 0:
            dd_saved = 1
            # dd = torch.Tensor(dd)
            print('Fidelity ', Fid.shape)
            print('Fidelity ', Fid)
            torch.save(Fid, data_path)
        if dd_saved == 1:
            break
    return n_dd




# model = FNN_gen().cuda()

# parse = argparse.ArgumentParser(description='Safety repair')    
# parse.add_argument('--para1', type=int)
# parse.add_argument('--para2', type=int)
# args = parse.parse_args() 
# para1 = args.para1
# para2 = args.para2

# model_path = '/data/home/mjnn/majianan/PRDNN/experiments/safety/n' + str(para1) + str(para2) + '/model/n' + str(para1) + str(para2) + '.pth'
# data_path_2 = '/data/home/mjnn/majianan/PRDNN/experiments/prodrawdown/n' + str(para1) + str(para2) + '_drawdown_test_2.pt'
# data_path_3 = '/data/home/mjnn/majianan/PRDNN/experiments/prodrawdown/n' + str(para1) + str(para2) + '_drawdown_test_3.pt'
# data_path_7 = '/data/home/mjnn/majianan/PRDNN/experiments/prodrawdown/n' + str(para1) + str(para2) + '_drawdown_test_7.pt'
# data_path_8 = '/data/home/mjnn/majianan/PRDNN/experiments/prodrawdown/n' + str(para1) + str(para2) + '_drawdown_test_8.pt'
# model.load_state_dict(torch.load(model_path))
# if not (para1 == 1 and para2 == 9 or para1 == 2 and para2 == 9):
#     gen_data_set_Fidelity_2(model, data_path_2, '')
#     gen_data_set_Fidelity_3(model, data_path_3, '')
#     gen_data_set_Fidelity_7(model, data_path_7, '')
# elif para1 == 1 and para2 == 9:
#     gen_data_set_Fidelity_2(model, data_path_2, '')
#     gen_data_set_Fidelity_7(model, data_path_7, '')
#     gen_data_set_Fidelity_8(model, data_path_8, '')
# elif para1 == 2 and para2 == 9:
#     # gen_data_set_Fidelity_2(model, data_path_2, '')
#     gen_data_set_Fidelity_3(model, data_path_3, '')
#     # gen_data_set_Fidelity_8(model, data_path_8, '')
