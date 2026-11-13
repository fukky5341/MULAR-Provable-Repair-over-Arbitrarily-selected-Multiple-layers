import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from PIL import Image
import time
from gurobipy import *
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class ProvableRepiar():
    
    def __init__(self, n_classes, buggy_model,  repair_loader, approximate_method, \
                 device, task_type, property_num, property_set=None) -> None:

        self.device = device
        self.n_classes = n_classes
        self.model = buggy_model
        self.repair_loader = repair_loader
        self.approximate_method = approximate_method     
        self.task_type = task_type
        self.property_num = property_num

        if self.task_type == 'pointwise':          
            """ self.mis_label is the true labels. """
            misclassified_labels = []
            for data, labels in repair_loader:
                misclassified_labels.extend(labels)
            self.mis_label = torch.tensor(misclassified_labels)

            """ self.output_constraint and self.output_constraint_bias:
                    cofficients and biases of linear constraint on f(x), e.g. C and D in (C * f(x) + D >= 0) """
            self.output_constraint = torch.zeros((self.property_num, self.n_classes - 1, self.n_classes), device=self.device)
            self.output_constraint_bias = torch.zeros((self.property_num, self.n_classes - 1), device=self.device)
            for i in range(self.property_num):
                label = self.mis_label[i]
                row = 0
                for j in range(self.n_classes):
                    if j != label:
                        self.output_constraint[i, row, label] = 1.0
                        self.output_constraint[i, row, j] = -1.0
                        row += 1

        elif self.task_type == 'regionwise':
            assert property_set is not None, "Missing property_set!"
            assert self.property_num == len(property_set), "property num error!"

            self.verify_method = self.approximate_method
            # self.verify_method = 'CROWN-Optimized'

            self.property_set = property_set

            cons_dim = self.property_set[0]['constraint'][0].shape[0]
            self.output_constraint = torch.zeros((self.property_num, cons_dim, self.n_classes), device=self.device)
            self.output_constraint_bias = torch.zeros((self.property_num, cons_dim), device=self.device)

            for i in range(self.property_num):
                self.output_constraint[i] = self.property_set[i]['constraint'][0]
                self.output_constraint_bias[i] = self.property_set[i]['constraint'][1]

    def constraint_update(self, property_set):
        assert property_set is not None, "Missing property_set!"
        self.property_num = len(property_set)
        self.property_set = property_set

        cons_dim = self.property_set[0]['constraint'][0].shape[0]
        self.output_constraint = torch.zeros((self.property_num, cons_dim, self.n_classes), device=self.device)
        self.output_constraint_bias = torch.zeros((self.property_num, cons_dim), device=self.device)

        for i in range(self.property_num):
            self.output_constraint[i] = self.property_set[i]['constraint'][0]
            self.output_constraint_bias[i] = self.property_set[i]['constraint'][1]


    @torch.no_grad()
    def linear_relaxation(self, relax_model, center, radius):
        """ Conduct linear relaxation on 'realx_model', input space: Box(center, r). """

        lirpa_model = BoundedModule(relax_model, torch.empty_like(center), device=self.device)
        ptb = PerturbationLpNorm(norm=np.inf, eps=radius)
        true_input = BoundedTensor(center, ptb)
        required_A = defaultdict(set)
        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
        if 'Optimized' in self.approximate_method:
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 100, 'lr_alpha': 0.05}})
        
        """  lb, ub shape:  sample num * (constraint_num_each_property) 
             lower_A shape: sample num * (constraint_num_each_property) * input_dim"""
        lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=self.approximate_method, return_A=True,
                                                needed_A_dict=required_A, C=self.output_constraint)
        lower_A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA']
        return lb, lower_A


    @torch.no_grad()
    def box_relax(self, relax_model, feature, r):
        center = feature.detach().clone()
        new_center = feature.detach().clone()
        lb, lower_A = self.linear_relaxation(relax_model=relax_model, center=center, radius=r)
        outputs = relax_model(feature)
        step = 0
        while torch.min(lb) < 0:
            if step > 200:
                print("Box relax fail!")
                return -1
            for i in range(self.property_num):
                """ compute the concrete values on linear constraint expression
                    and check whether satisfy constraint, shape: nclass - 1 """

                cons_vio = torch.where(lb[i] >= 0, 0, 1).float()

                if torch.sum(cons_vio) == 0:
                    new_center[i] = center[i].detach().clone()
                else:

                    k = torch.einsum('i...,i->...', lower_A[i], cons_vio)

                    direction = torch.where(k < 0, -r, r)
                    new_center[i] = (center[i] + direction).detach().clone()
            center = new_center.detach().clone()
            lb, lower_A = self.linear_relaxation(relax_model=relax_model, center=new_center, radius=r)
            step += 1
        print(step)
        return new_center

    def distance_repair(self, optimizer, buggy_nn, x, preiamge_h, disjunction=False, exp=None):
        step = 0
        buggy_nn.eval()
        psr = self.property_satisfy_rate(buggy_nn, x, disjunction)
        print(f"Start distance repair x:{x.shape}, psr: {psr}")
        start = time.time()
        preiamge_h = preiamge_h.flatten(start_dim=1)
        while psr != 100.0:
            step += 1
            if step > 500:
                # repair failed
                return -1
            optimizer.zero_grad()
            ori_h = buggy_nn.split()[0].forward(x).flatten(start_dim=1)
            loss = torch.norm((ori_h - preiamge_h), p=2, dim=1).mean()
            print(f"{step = } loss :", loss.item())
            loss.backward()
            optimizer.step()
            psr = self.property_satisfy_rate(buggy_nn, x, disjunction)
            # print(f'after opt {psr = } time {time.time() - start}')
        self.step = step
        return step

    @torch.no_grad()
    def property_satisfy_rate(self, model, x, disjunction=False):
                                                                  # self.output_constraint shape: (self.property_num, n - 1, n)
        outputs = model(x)                                        # shape: (self.property_num, n)
        outputs_expanded = outputs.unsqueeze(1)                   # shape: (self.property_num, 1, n)
        cons_values = (self.output_constraint * outputs_expanded).sum(dim=2) \
                                        + self.output_constraint_bias          # shape: (self.property_num, n - 1)
        if not disjunction:
            sat = torch.sum(torch.min(cons_values, dim=1)[0] >= 0).item()
        else:
            sat = torch.sum(torch.max(cons_values, dim=1)[0] >= 0).item()
        return sat / self.property_num * 100

    @torch.no_grad()
    def real_counter_check(self, output):
        # output-shape: (self.property_num, n)
        outputs_expanded = output.unsqueeze(1)                   # shape: (self.property_num, 1, n)
        cons_values = (self.output_constraint * outputs_expanded).sum(dim=2) \
                                        + self.output_constraint_bias          # shape: (self.property_num, n - 1)
        sat = torch.min(cons_values, dim=1)[0] < 0
        # print(sat)
        # print(cons_values[sat])
        # print(self.output_constraint[sat])
        return sat

    # @torch.no_grad()
    def multi_properties_verify(self, model, property_set, printinfo=False):
        """ Verify a property set (includes multiple same type properties)
            And update set['satisfy'] """
        
        input_lb = torch.stack([property['input_space'][..., 0] for property in property_set])
        input_ub = torch.stack([property['input_space'][..., 1] for property in property_set])

        lirpa_model = BoundedModule(model, torch.empty_like(input_lb), device=self.device)
        ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
        true_input = BoundedTensor(torch.empty_like(input_lb), ptb)
        C = torch.stack([property['constraint'][0] for property in property_set]).to(self.device)
        required_A = defaultdict(set)
        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
        
        if 'Optimized' in self.verify_method:
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.01, }})
        lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=self.verify_method, return_A=True, needed_A_dict=required_A, C=C)

        lower_A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA']
        bias = torch.stack([property['constraint'][1] for property in property_set]).to(self.device)
        lb += bias
        sat = 0
        violate_matrix = lb < 0
        for i in range(len(property_set)):
            property_set[i]['satisfy'] = torch.min(lb[i]) >= 0
            if not property_set[i]['satisfy']:
                """ refine_score shape = input shape"""
                property_set[i]['refine_score'] = lower_A[i][violate_matrix[i]].sum(dim=0)
            sat += property_set[i]['satisfy']
        if printinfo:
            print(f"All {len(property_set)} properties, {sat} sat!")
        print('lb!', lb)
        return sat, torch.min(lb, dim=1)[0]

    
    @torch.no_grad()
    def multi_disjunction_properties_verify(self, model, property_set, printinfo=False):
   
        input_lb = torch.stack([property['input_space'][..., 0] for property in property_set])
        input_ub = torch.stack([property['input_space'][..., 1] for property in property_set])

        lirpa_model = BoundedModule(model, torch.empty_like(input_lb), device=self.device)
        ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
        true_input = BoundedTensor(torch.empty_like(input_lb), ptb)

        C = []
        for property in property_set:
            # print(torch.stack([disj[0] for disj in property['verify_constraint']]).shape)
            C.append(torch.stack([disj[0] for disj in property['verify_constraint']]))
        C = torch.stack(C, dim=0).to(self.device)

        required_A = defaultdict(set)
        required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
        # if 'Optimized' in self.verify_method:
        #     lirpa_model.set_bound_opts({'optimize_boun d_args': {'iteration': 20, 'lr_alpha': 0.01, }})

        lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=self.verify_method, return_A=True, needed_A_dict=required_A, C=C)
        lower_A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA']

        B = []
        for property in property_set:
            B.append(torch.stack([disj[1] for disj in property['verify_constraint']]))
        bias = torch.stack(B, dim=0).to(self.device)
        lb += bias
        violate_matrix = lb < 0
        sat = 0
        for i in range(len(property_set)):
            property_set[i]['satisfy'] = torch.max(lb[i]) >= 0
            if not property_set[i]['satisfy']:
                """ refine_score shape = input shape"""
                property_set[i]['refine_score'] = lower_A[i][violate_matrix[i]].sum(dim=0)
            else:
                sat += 1
        if printinfo:
            print(f"All {len(property_set)} properties, {sat} sat!")
            # print(torch.max(lb, dim=1)[0])
        return sat, torch.max(lb, dim=1)[0]

    
    @torch.no_grad()
    def space_refine(self, p, method='mag'):
        p1, p2 = copy.deepcopy(p), copy.deepcopy(p)
        if method == 'mag':
            """ focus on magnitude"""
            score = p['input_space'][..., 1] - p['input_space'][..., 0]
            candidate_dim = torch.argmax(score)
        elif method == 'refine_score':
            """ focus on linear cofficients (somewhat like a Integrated Gradients)"""
            score = torch.abs(p['refine_score']) * (p['input_space'][..., 1] - p['input_space'][..., 0])
            candidate_dim = torch.argmax(score)
        elif method == 'random':
            """ randomly select a dimension for refinement"""
            input_shape = p['input_space'][..., 0].shape
            score = p['input_space'][..., 0]
            candidate_dim = torch.randint(0, input_shape.numel(), (1,)).item() 
    

        flat_l = p2['input_space'][..., 0].view(-1)
        flat_u = p1['input_space'][..., 1].view(-1)
        mean = (flat_l[candidate_dim] + flat_u[candidate_dim]) / 2.
        flat_l[candidate_dim] = mean
        flat_u[candidate_dim] = mean
        flat_l = flat_l.reshape(score.shape)
        flat_u = flat_u.reshape(score.shape)
        p1['satisfy'] = p2['satisfy'] = False

        return p1, p2



    def calculate_worst_case(self, model, property_set):
        """ Given a model and a set of property,
                For each property, calculate a approximate worst case (a vertice of the box)."""
        
        with torch.no_grad():
            input_lb = torch.stack([property['input_space'][..., 0] for property in property_set])
            input_ub = torch.stack([property['input_space'][..., 1] for property in property_set])
            lirpa_model = BoundedModule(model, torch.empty_like(input_lb), device=self.device)
            ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
            true_input = BoundedTensor(torch.zeros_like(input_lb), ptb)
            C = torch.stack([property['constraint'][0] for property in property_set]).to(self.device)
            required_A = defaultdict(set)
            required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
            if 'Optimized' in self.approximate_method:
                lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 10, 'lr_alpha': 0.1, }})

            lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=self.approximate_method, 
                                                    return_A=True, needed_A_dict=required_A, C=C)
            lower_A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA']

            bias = torch.stack([property['constraint'][1] for property in property_set]).to(self.device)
            lb += bias
            """ violate_matrix: property_num * constraint_num_each_property
                                1 means violate"""
            violate_matrix = lb < 0
            # print(violate_matrix, lower_A)
            result = torch.zeros_like(lower_A[:, 0, ...])
            for i in range(lower_A.shape[0]):
                result[i] += lower_A[i][violate_matrix[i]].sum(dim=0)
            # print(result)
            worst_case = torch.where(result > 0, input_lb, input_ub)
        # print(input_lb, input_ub)
        # print(worst_case)

        return worst_case




    def complete_verify(self, model, method, model_name):
        from utils.prepare_yaml import update_yaml_path
        yaml_path = 'utils/myverify.yaml'
        model_path = f'result/global_safety/Ours_{method}/{model_name}_repair.pth'
        torch.save(model.state_dict(), model_path)
        update_yaml_path(yaml_path, model_path)
        
        original_sys_path = sys.path.copy()
        complete_verifier_path = os.path.join(os.path.dirname(__file__), 'complete_verifier')
        external_utils_path = os.path.join(os.path.dirname(__file__), 'utils')
        if external_utils_path in sys.path:
            sys.path.remove(external_utils_path)
        sys.path.insert(0, complete_verifier_path)
        if 'utils' in sys.modules:
            del sys.modules['utils']

        try:
            from complete_verifier.abcrown import ABCROWN
            arg = ['--config', yaml_path]
            abcrown = ABCROWN(args=arg)
            result = abcrown.main()
        finally:
            sys.path = original_sys_path
            from utils.network import FNN
            if 'safe' in result.keys():
                return True
            else:
                return False