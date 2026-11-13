import gc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_network.dual_network_ind import DualNetwork_Ind



""" helper """
def compute_input_shapes(net, input_shape):
    """
    shapes: [input_shape, layer1_out_shape, layer2_out_shape, ..., layerN_out_shape]
    shapes[i+1] = output shape AFTER layer i
    """

    def _numel(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    shapes = []
    cur = input_shape
    shapes.append(cur)

    for layer in net:

        # -------------------------
        # Linear
        # -------------------------
        if isinstance(layer, nn.Linear):
            if isinstance(cur, tuple):
                cur = _numel(cur)

            cur = layer.out_features
            shapes.append(cur)

        # -------------------------
        # Conv2d
        # -------------------------
        elif isinstance(layer, nn.Conv2d):
            if not (isinstance(cur, tuple) and len(cur) == 3):
                raise ValueError(f"Conv2d expects (C,H,W), got {cur}")

            C, H, W = cur

            k_h, k_w = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
            s_h, s_w = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
            p_h, p_w = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
            d_h, d_w = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation)

            out_h = (H + 2*p_h - d_h*(k_h - 1) - 1) // s_h + 1
            out_w = (W + 2*p_w - d_w*(k_w - 1) - 1) // s_w + 1

            cur = (layer.out_channels, out_h, out_w)
            shapes.append(cur)

        # -------------------------
        # Flatten
        # -------------------------
        elif isinstance(layer, nn.Flatten):
            if isinstance(cur, tuple):
                cur = _numel(cur)
            shapes.append(cur)

        # -------------------------
        # Activations (IMPORTANT FIX)
        # -------------------------
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Identity)):
            # shape unchanged but MUST append
            shapes.append(cur)

        else:
            raise NotImplementedError(f"Unsupported layer {type(layer)}")

    return shapes


@dataclass
class BackPropStruct:
    """
    Out = C @ In + b
    """
    n_C_lb: Optional[torch.Tensor] = None
    n_b_lb: Optional[torch.Tensor] = None
    n_C_ub: Optional[torch.Tensor] = None
    n_b_ub: Optional[torch.Tensor] = None

    def populate(self, n_C_lb, n_b_lb, n_C_ub, n_b_ub):
        self.n_C_lb = n_C_lb
        self.n_b_lb = n_b_lb
        self.n_C_ub = n_C_ub
        self.n_b_ub = n_b_ub

    def clone_detached(self):
        return BackPropStruct(
            n_C_lb=None if self.n_C_lb is None else self.n_C_lb.detach().clone(),
            n_b_lb=None if self.n_b_lb is None else self.n_b_lb.detach().clone(),
            n_C_ub=None if self.n_C_ub is None else self.n_C_ub.detach().clone(),
            n_b_ub=None if self.n_b_ub is None else self.n_b_ub.detach().clone(),
        )

    def delete_old(self):
        del self.n_C_lb, self.n_b_lb, self.n_C_ub, self.n_b_ub
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class IndividualBounds:
    def __init__(self, net, lb_inp, ub_inp,  eps=None, device="cpu"):
        self.net = net

        self.lb_inp = lb_inp.to(device)
        self.ub_inp = ub_inp.to(device)

        self.eps = eps
        self.device = device

        self.shapes = compute_input_shapes(net=self.net, input_shape=self.lb_inp.shape)

        self.affine_layer_indices = []
        self.lbs = None
        self.ubs = None
        self.alpha_params: Dict[int, torch.nn.Parameter] = {}
        self.saved_coeffs: Dict[Tuple[int, int], BackPropStruct] = {}

        for ind, layer in enumerate(self.net):
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                self.affine_layer_indices.append(ind)

        for p in self.net.parameters():
            p.requires_grad_(False)

        self.dual_network = None

    # ------------------------------------------------------------
    # shape helpers
    # ------------------------------------------------------------

    def get_layer_size(self, layer_index):
        shape = self.shapes[layer_index+1]  # output shape of the layer

        if isinstance(shape, int):
            return shape
        prod = 1
        for x in shape:
            prod *= x
        return prod

    def initialize_back_prop_struct(self, layer_idx):
        layer_size = self.get_layer_size(layer_index=layer_idx)
        dtype = self.lb_inp.dtype
        return BackPropStruct(
            n_C_lb=torch.eye(layer_size, device=self.device, dtype=dtype),
            n_b_lb=torch.zeros(layer_size, device=self.device, dtype=dtype),
            n_C_ub=torch.eye(layer_size, device=self.device, dtype=dtype),
            n_b_ub=torch.zeros(layer_size, device=self.device, dtype=dtype),
        )

    def pos_neg_weight_decomposition(self, coef):
        neg_comp = torch.where(coef < 0, coef, torch.zeros_like(coef, device=self.device))
        pos_comp = torch.where(coef >= 0, coef, torch.zeros_like(coef, device=self.device))
        return neg_comp, pos_comp

    def has_inconsistent_bounds(self, lb, ub):
        return not torch.all(lb <= ub + 1e-6)

    # ------------------------------------------------------------
    # alpha
    # ------------------------------------------------------------

    def initialize_alpha(self):
        self.alpha_params = {}
        for layer_idx, layer in enumerate(self.net):
            if isinstance(layer, nn.ReLU):
                alpha_init = torch.full_like(
                    self.lbs[layer_idx],
                    0.5,
                    device=self.device,
                    requires_grad=True,
                )
                self.alpha_params[layer_idx] = torch.nn.Parameter(alpha_init)

    def save_coeff(self, target_layer_idx, r_layer_idx, back_prop_struct):
        self.saved_coeffs[(target_layer_idx, r_layer_idx)] = back_prop_struct.clone_detached()

    # ------------------------------------------------------------
    # affine handlers
    # ------------------------------------------------------------

    def handle_linear_individual(self, linear_wt, bias, back_prop_struct):
        n_b_lb = back_prop_struct.n_b_lb + back_prop_struct.n_C_lb.matmul(bias)
        n_b_ub = back_prop_struct.n_b_ub + back_prop_struct.n_C_ub.matmul(bias)

        n_C_lb = back_prop_struct.n_C_lb.matmul(linear_wt)
        n_C_ub = back_prop_struct.n_C_ub.matmul(linear_wt)

        back_prop_struct.populate(n_C_lb, n_b_lb, n_C_ub, n_b_ub)
        return back_prop_struct

    def handle_conv_individual(
        self, conv_weight, conv_bias, back_prop_struct,
        preconv_shape, postconv_shape, stride, padding, groups=1, dilation=(1, 1)
    ):
        kernel_hw = conv_weight.shape[-2:]
        h_padding = (
            preconv_shape[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_hw[0] - 1)
        ) % stride[0]
        w_padding = (
            preconv_shape[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_hw[1] - 1)
        ) % stride[1]
        output_padding = (h_padding, w_padding)

        coef_shape = back_prop_struct.n_C_lb.shape
        n_C_lb = back_prop_struct.n_C_lb.view((coef_shape[0], *postconv_shape))
        n_C_ub = back_prop_struct.n_C_ub.view((coef_shape[0], *postconv_shape))

        n_b_lb = back_prop_struct.n_b_lb + (n_C_lb.sum((2, 3)) * conv_bias).sum(1)
        n_b_ub = back_prop_struct.n_b_ub + (n_C_ub.sum((2, 3)) * conv_bias).sum(1)

        n_C_lb = F.conv_transpose2d(
            n_C_lb, conv_weight, None, stride, padding, output_padding, groups, dilation
        )
        n_C_ub = F.conv_transpose2d(
            n_C_ub, conv_weight, None, stride, padding, output_padding, groups, dilation
        )

        n_C_lb = n_C_lb.view((coef_shape[0], -1))
        n_C_ub = n_C_ub.view((coef_shape[0], -1))

        back_prop_struct.populate(n_C_lb, n_b_lb, n_C_ub, n_b_ub)
        return back_prop_struct

    # ------------------------------------------------------------
    # activation / reshape handlers
    # ------------------------------------------------------------

    def handle_relu_individual(self, back_prop_struct, n_lb_layer, n_ub_layer, layer_idx):
        relu_active = (n_lb_layer >= 0)
        relu_passive = (n_ub_layer <= 0)
        relu_unsettled = (~relu_active) & (~relu_passive)

        lambda_lb = torch.zeros_like(n_lb_layer, device=self.device)
        lambda_ub = torch.zeros_like(n_lb_layer, device=self.device)
        mu_ub = torch.zeros_like(n_lb_layer, device=self.device)

        lambda_lb = torch.where(relu_active, torch.ones_like(n_lb_layer), lambda_lb)
        lambda_ub = torch.where(relu_active, torch.ones_like(n_lb_layer), lambda_ub)

        lambda_ub = torch.where(
            relu_unsettled,
            n_ub_layer / (n_ub_layer - n_lb_layer + 1e-15),
            lambda_ub
        )
        mu_ub = torch.where(
            relu_unsettled,
            -(n_ub_layer * n_lb_layer) / (n_ub_layer - n_lb_layer + 1e-15),
            mu_ub
        )

        if layer_idx in self.alpha_params:
            alpha = torch.clamp(self.alpha_params[layer_idx], 0.0, 1.0)
        else:
            alpha = torch.where(
                n_ub_layer < -n_lb_layer,
                torch.zeros_like(n_lb_layer),
                torch.ones_like(n_lb_layer),
            )

        lambda_lb = torch.where(relu_unsettled, alpha, lambda_lb)

        neg_C_lb, pos_C_lb = self.pos_neg_weight_decomposition(back_prop_struct.n_C_lb)
        neg_C_ub, pos_C_ub = self.pos_neg_weight_decomposition(back_prop_struct.n_C_ub)

        n_b_lb = back_prop_struct.n_b_lb + neg_C_lb @ mu_ub
        n_b_ub = back_prop_struct.n_b_ub + pos_C_ub @ mu_ub

        n_C_lb = neg_C_lb * lambda_ub + pos_C_lb * lambda_lb
        n_C_ub = neg_C_ub * lambda_lb + pos_C_ub * lambda_ub

        back_prop_struct.populate(n_C_lb, n_b_lb, n_C_ub, n_b_ub)
        return back_prop_struct

    def handle_flatten_individual(self, back_prop_struct):
        return back_prop_struct

    # ------------------------------------------------------------
    # dispatcher
    # ------------------------------------------------------------

    def handle_layer_individual(self, back_prop_struct, layer, layer_idx, n_lbs, n_ubs):
        if isinstance(layer, nn.Linear):
            return self.handle_linear_individual(layer.weight, layer.bias, back_prop_struct)

        elif isinstance(layer, nn.Conv2d):
            return self.handle_conv_individual(
                conv_weight=layer.weight,
                conv_bias=layer.bias,
                back_prop_struct=back_prop_struct,
                preconv_shape=self.shapes[layer_idx],
                postconv_shape=self.shapes[layer_idx + 1],
                stride=layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride),
                padding=layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding),
                dilation=layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation),
                groups=layer.groups,
            )

        elif isinstance(layer, nn.ReLU):
            return self.handle_relu_individual(
                back_prop_struct=back_prop_struct,
                n_lb_layer=n_lbs[layer_idx],  # preactivaiton bounds
                n_ub_layer=n_ubs[layer_idx],
                layer_idx=layer_idx,
            )

        elif isinstance(layer, nn.Flatten):
            return self.handle_flatten_individual(back_prop_struct)

        elif isinstance(layer, (nn.Identity, nn.Sigmoid, nn.Tanh)):
            raise NotImplementedError(f"{type(layer)} not implemented yet in backsubstitution")

        else:
            raise NotImplementedError(f"Layer {type(layer)} is not implemented")

    # ------------------------------------------------------------
    # concretization
    # ------------------------------------------------------------

    def concretize_bounds_individual(self, back_prop_struct, n_lb_layer, n_ub_layer):
        neg_comp_lb, pos_comp_lb = self.pos_neg_weight_decomposition(back_prop_struct.n_C_lb)
        neg_comp_ub, pos_comp_ub = self.pos_neg_weight_decomposition(back_prop_struct.n_C_ub)

        lb = neg_comp_lb @ n_ub_layer + pos_comp_lb @ n_lb_layer + back_prop_struct.n_b_lb
        ub = neg_comp_ub @ n_lb_layer + pos_comp_ub @ n_ub_layer + back_prop_struct.n_b_ub
        return lb, ub

    # ------------------------------------------------------------
    # one target layer backward pass
    # ------------------------------------------------------------

    def backsubstitution_individual(self, layer_idx, n_lbs, n_ubs, save_coeffs=False):
        back_prop_struct = None
        n_lb = None
        n_ub = None

        for r_layer_idx in reversed(range(layer_idx + 1)):
            if back_prop_struct is None:
                back_prop_struct = self.initialize_back_prop_struct(layer_idx=layer_idx)

            curr_layer = self.net[r_layer_idx]
            back_prop_struct = self.handle_layer_individual(
                back_prop_struct=back_prop_struct,
                layer=curr_layer,
                layer_idx=r_layer_idx,
                n_lbs=n_lbs,
                n_ubs=n_ubs,
            )

            if save_coeffs:
                self.save_coeff(layer_idx, r_layer_idx, back_prop_struct)

            if r_layer_idx == 0:
                new_n_lb, new_n_ub = self.concretize_bounds_individual(
                    back_prop_struct=back_prop_struct,
                    n_lb_layer=n_lbs[r_layer_idx],
                    n_ub_layer=n_ubs[r_layer_idx],
                )
                n_lb = new_n_lb if n_lb is None else torch.max(n_lb, new_n_lb)
                n_ub = new_n_ub if n_ub is None else torch.min(n_ub, new_n_ub)

        if self.has_inconsistent_bounds(n_lb, n_ub):
            return None, None

        return n_lb, n_ub

    # ------------------------------------------------------------
    # full pass
    # ------------------------------------------------------------

    def run_backsubstitution_individual(self, save_coeffs=False):
        lb0 = self.lb_inp.detach().clone().view(-1)
        ub0 = self.ub_inp.detach().clone().view(-1)
        self.lbs = [lb0]
        self.ubs = [ub0]

        for layer_idx, _layer in enumerate(self.net):
            if isinstance(_layer, (nn.Flatten)):
                self.lbs.append(self.lbs[-1].detach().clone())
                self.ubs.append(self.ubs[-1].detach().clone())
                continue
            curr_n_lb, curr_n_ub = self.backsubstitution_individual(
                layer_idx=layer_idx,
                n_lbs=self.lbs,
                n_ubs=self.ubs,
                save_coeffs=save_coeffs,
            )
            if curr_n_lb is None or curr_n_ub is None:
                return None, None

            self.lbs.append(curr_n_lb.detach().clone())
            self.ubs.append(curr_n_ub.detach().clone())

        return self.lbs, self.ubs

    # ------------------------------------------------------------
    # alpha optimization
    # ------------------------------------------------------------

    def optimize_alpha(self, alpha_steps=20, alpha_lr=1e-1, save_coeffs=False, verbose=False):
        if len(self.alpha_params) == 0:
            return

        optimizer = torch.optim.Adam(self.alpha_params.values(), lr=alpha_lr)

        for step in range(alpha_steps):
            optimizer.zero_grad()

            self.saved_coeffs = {}
            lb0 = self.lb_inp.detach().clone().view(-1)
            ub0 = self.ub_inp.detach().clone().view(-1)
            temp_lbs = [lb0]
            temp_ubs = [ub0]
            objective_terms = []

            for layer_idx, _layer in enumerate(self.net):
                curr_lb, curr_ub = self.backsubstitution_individual(
                    layer_idx=layer_idx,
                    n_lbs=temp_lbs,
                    n_ubs=temp_ubs,
                    save_coeffs=save_coeffs,
                )
                if curr_lb is None or curr_ub is None:
                    continue

                temp_lbs.append(curr_lb)
                temp_ubs.append(curr_ub)
                objective_terms.append(curr_lb.mean())

            if len(objective_terms) == 0:
                return

            objective = torch.stack(objective_terms).sum()
            loss = -objective
            loss.backward()
            optimizer.step()

            for p in self.alpha_params.values():
                p.data.clamp_(0.0, 1.0)

            if verbose and (step == 0 or step == alpha_steps - 1 or (step + 1) % 10 == 0):
                print(f"[alpha step {step+1:03d}] objective = {objective.item():.6f}")

        self.saved_coeffs = {}
        return self.run_backsubstitution_individual(save_coeffs=save_coeffs)

    # ------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------
    def run(self, optimize_alpha=True, alpha_steps=20, alpha_lr=1e-1, save_coeffs=False, verbose=False):
        with torch.no_grad():
            self.run_backsubstitution_individual(save_coeffs=False)

        # # logging before optimization
        # if verbose:
        #     print("Initial bounds (before alpha optimization):")
        #     for idx, (lb, ub) in enumerate(zip(self.lbs, self.ubs)):
        #         print(f"Layer {idx}: lb mean = {lb.mean().item():.6f}, ub mean = {ub.mean().item():.6f}")
        #     print("Output layer")
        #     print(f"  lb: {self.lbs[-1]}")
        #     print(f"  ub: {self.ubs[-1]}")

        if optimize_alpha:
            self.initialize_alpha()
            return self.optimize_alpha(
                alpha_steps=alpha_steps,
                alpha_lr=alpha_lr,
                save_coeffs=save_coeffs,
                verbose=verbose,
            )

        if save_coeffs:
            self.saved_coeffs = {}
            return self.run_backsubstitution_individual(save_coeffs=True)

        return self.lbs, self.ubs
    
    # ------------------------------------------------------------
    # Dual network interaction
    # ------------------------------------------------------------
    # helper
    def filter_satisfied_specs(self, C, out_lb, out_ub):
        """
        Filter out already-satisfied rows of C using independent output bounds.

        Args:
            C:
                shape (out_dim,) or (num_specs, out_dim)
            out_lb, out_ub:
                output lower/upper bounds, shape (out_dim,)

        Returns:
            filtered_C, keep_mask, satisfied_mask
        """
        if C.dim() == 1:
            C = C.unsqueeze(0)

        C_pos = torch.clamp(C, min=0)
        C_neg = torch.clamp(C, max=0)

        # lower bound on C f(x)
        spec_lb = (C_pos * out_lb.unsqueeze(0)).sum(dim=1) + (C_neg * out_ub.unsqueeze(0)).sum(dim=1)
        satisfied_mask = spec_lb >= 0  # shape (num_specs,)

        keep_mask = ~satisfied_mask
        filtered_C = C[keep_mask]

        return filtered_C, spec_lb
    
    def build_dual_network(self, C, relu_precise=False, perform_dual=False):
        # keep alpha params optimized during backsubstitution
        alpha_params = {
            k: torch.nn.Parameter(v.detach().clone())
            for k, v in self.alpha_params.items()
        }

        lbs = [x.detach() for x in self.lbs]
        ubs = [x.detach() for x in self.ubs]

        self.dual_network = DualNetwork_Ind(
            C=C,
            ori_net=self.net,
            shapes=self.shapes,
            lbs=lbs,
            ubs=ubs,
            relu_precise=relu_precise,
            alpha_params=alpha_params,
        )
        self.dual_network.build_dual_network_individual()
        return self.dual_network
    
    def compute_dual_min_objective(self, C, relu_precise=False, optimize_alpha=True, alpha_steps=30, alpha_lr=1e-2, verbose=False):
        dual_net = self.build_dual_network(C=C, relu_precise=relu_precise)
        if dual_net is None:
            return None, None

        if optimize_alpha:
            dual_net.optimize_alpha(steps=alpha_steps, lr=alpha_lr, verbose=verbose)

        return dual_net.get_minimized_objective(), dual_net
    
    # ------------------------------------------------------------
    # Estimation of input splitting importance
    # ------------------------------------------------------------
    '''
    Option 1 (Dual network style)
        score_i = |A| * (ub_i - lb_i) * 0.5
    '''
    def estimate_input_importance(self, Cs, topk=5):
        split_indices_list = []
        scores_list = []
        for C in Cs:
            dual_net = self.build_dual_network(C=C)
            split_indices, scores = dual_net.best_input_split(topk=topk)
            split_indices_list.append(split_indices)
            scores_list.append(scores)
        # pick the most important split among all Cs
        all_scores = torch.cat(scores_list)
        all_indices = torch.cat(split_indices_list)
        _, topk_indices = torch.topk(all_scores, k=1)
        return all_indices[topk_indices]


    '''
    Option 2 (Prorepair style)
        score_i = (ub_j - lb_j) * sum_j |w_ij|
    '''
    def estimate_input_importance_linear(self, LinearLayer):
        lb = self.lbs[0]  # 1D
        ub = self.ubs[0]  # 1D
        width = ub - lb  # 1D

        W = LinearLayer.weight  # shape (out_dim, in_dim)
        # sum over output dimension
        w_sum_input_wise = W.abs().sum(dim=0)  # shape (in_dim,)

        score = width * w_sum_input_wise  # shape (in_dim,)
        return score
    
    def estimate_input_importance_conv(self, ConvLayer):
        lb = self.lbs[0]  # 1D
        ub = self.ubs[0]  # 1D
        width = ub - lb  # 1D

        W = ConvLayer.weight  # shape (out_c, in_c, k_h, k_w)

        # collapse spatial + output dims
        influence_per_channel = torch.abs(W).sum(dim=(0, 2, 3))  # (in_c,)

        # expand to input size
        # assumes input shape (C, H, W)
        C, H, W_in = self.shapes[0]
        influence = influence_per_channel.repeat_interleave(H * W_in)

        score = width * influence
        return score

    def estimate_input_importance_prorepair(self, topk=5):
        assert self.lbs is not None and self.ubs is not None, "Bounds must be computed before estimating input importance"
        # first linear layer
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                score = self.estimate_input_importance_linear(first_linear)
                break
            elif isinstance(layer, nn.Conv2d):
                first_linear = layer
                score = self.estimate_input_importance_conv(first_linear)
                break
        
        topk_scores, topk_indices = torch.topk(score, k=topk)

        return topk_indices, topk_scores


class BatchBounds(IndividualBounds):
    """Backsubstitution bounds for independent boxes stacked on dimension 0."""

    def __init__(self, net, lb_inp, ub_inp, device="cpu"):
        if lb_inp.ndim < 2 or lb_inp.shape != ub_inp.shape:
            raise ValueError("Batched bounds require matching tensors shaped (batch, ...)")
        self.batch_size = lb_inp.shape[0]
        self.net = net
        self.lb_inp = lb_inp.to(device)
        self.ub_inp = ub_inp.to(device)
        self.eps = None
        self.device = device
        self.shapes = compute_input_shapes(net, self.lb_inp.shape[1:])
        self.affine_layer_indices = [
            i for i, layer in enumerate(net)
            if isinstance(layer, (nn.Linear, nn.Conv2d))
        ]
        self.lbs = None
        self.ubs = None
        self.alpha_params = {}
        self.saved_coeffs = {}
        self.dual_network = None
        for p in self.net.parameters():
            p.requires_grad_(False)

    def initialize_back_prop_struct(self, layer_idx):
        size = self.get_layer_size(layer_idx)
        dtype = self.lb_inp.dtype
        eye = torch.eye(size, device=self.device, dtype=dtype)
        coeff = eye.unsqueeze(0).expand(self.batch_size, -1, -1).clone()
        bias = torch.zeros(self.batch_size, size, device=self.device, dtype=dtype)
        return BackPropStruct(coeff, bias, coeff.clone(), bias.clone())

    def handle_linear_individual(self, linear_wt, bias, back_prop_struct):
        n_b_lb = back_prop_struct.n_b_lb
        n_b_ub = back_prop_struct.n_b_ub
        if bias is not None:
            n_b_lb = n_b_lb + torch.matmul(back_prop_struct.n_C_lb, bias)
            n_b_ub = n_b_ub + torch.matmul(back_prop_struct.n_C_ub, bias)
        n_C_lb = torch.matmul(back_prop_struct.n_C_lb, linear_wt)
        n_C_ub = torch.matmul(back_prop_struct.n_C_ub, linear_wt)
        back_prop_struct.populate(n_C_lb, n_b_lb, n_C_ub, n_b_ub)
        return back_prop_struct

    def handle_conv_individual(
        self, conv_weight, conv_bias, back_prop_struct,
        preconv_shape, postconv_shape, stride, padding, groups=1, dilation=(1, 1)
    ):
        kernel_hw = conv_weight.shape[-2:]
        h_padding = (
            preconv_shape[1] + 2 * padding[0] - 1
            - dilation[0] * (kernel_hw[0] - 1)
        ) % stride[0]
        w_padding = (
            preconv_shape[2] + 2 * padding[1] - 1
            - dilation[1] * (kernel_hw[1] - 1)
        ) % stride[1]
        output_padding = (h_padding, w_padding)
        batch, targets = back_prop_struct.n_C_lb.shape[:2]
        n_C_lb = back_prop_struct.n_C_lb.reshape(batch * targets, *postconv_shape)
        n_C_ub = back_prop_struct.n_C_ub.reshape(batch * targets, *postconv_shape)
        n_b_lb = back_prop_struct.n_b_lb
        n_b_ub = back_prop_struct.n_b_ub
        if conv_bias is not None:
            n_b_lb = n_b_lb + (
                (n_C_lb.sum((2, 3)) * conv_bias).sum(1).view(batch, targets)
            )
            n_b_ub = n_b_ub + (
                (n_C_ub.sum((2, 3)) * conv_bias).sum(1).view(batch, targets)
            )
        n_C_lb = F.conv_transpose2d(
            n_C_lb, conv_weight, None, stride, padding, output_padding, groups, dilation
        ).reshape(batch, targets, -1)
        n_C_ub = F.conv_transpose2d(
            n_C_ub, conv_weight, None, stride, padding, output_padding, groups, dilation
        ).reshape(batch, targets, -1)
        back_prop_struct.populate(n_C_lb, n_b_lb, n_C_ub, n_b_ub)
        return back_prop_struct

    def handle_relu_individual(self, back_prop_struct, n_lb_layer, n_ub_layer, layer_idx):
        active = n_lb_layer >= 0
        passive = n_ub_layer <= 0
        unsettled = (~active) & (~passive)
        lambda_lb = torch.where(active, torch.ones_like(n_lb_layer), torch.zeros_like(n_lb_layer))
        lambda_ub = torch.where(active, torch.ones_like(n_lb_layer), torch.zeros_like(n_lb_layer))
        mu_ub = torch.zeros_like(n_lb_layer)
        lambda_ub = torch.where(
            unsettled, n_ub_layer / (n_ub_layer - n_lb_layer + 1e-15), lambda_ub
        )
        mu_ub = torch.where(
            unsettled,
            -(n_ub_layer * n_lb_layer) / (n_ub_layer - n_lb_layer + 1e-15),
            mu_ub,
        )
        if layer_idx in self.alpha_params:
            alpha = self.alpha_params[layer_idx].clamp(0.0, 1.0)
        else:
            alpha = torch.where(
                n_ub_layer < -n_lb_layer,
                torch.zeros_like(n_lb_layer),
                torch.ones_like(n_lb_layer),
            )
        lambda_lb = torch.where(unsettled, alpha, lambda_lb)
        neg_lb, pos_lb = self.pos_neg_weight_decomposition(back_prop_struct.n_C_lb)
        neg_ub, pos_ub = self.pos_neg_weight_decomposition(back_prop_struct.n_C_ub)
        n_b_lb = back_prop_struct.n_b_lb + (neg_lb * mu_ub.unsqueeze(1)).sum(-1)
        n_b_ub = back_prop_struct.n_b_ub + (pos_ub * mu_ub.unsqueeze(1)).sum(-1)
        n_C_lb = neg_lb * lambda_ub.unsqueeze(1) + pos_lb * lambda_lb.unsqueeze(1)
        n_C_ub = neg_ub * lambda_lb.unsqueeze(1) + pos_ub * lambda_ub.unsqueeze(1)
        back_prop_struct.populate(n_C_lb, n_b_lb, n_C_ub, n_b_ub)
        return back_prop_struct

    def concretize_bounds_individual(self, back_prop_struct, n_lb_layer, n_ub_layer):
        neg_lb, pos_lb = self.pos_neg_weight_decomposition(back_prop_struct.n_C_lb)
        neg_ub, pos_ub = self.pos_neg_weight_decomposition(back_prop_struct.n_C_ub)
        lb = ((neg_lb * n_ub_layer.unsqueeze(1)).sum(-1)
              + (pos_lb * n_lb_layer.unsqueeze(1)).sum(-1)
              + back_prop_struct.n_b_lb)
        ub = ((neg_ub * n_lb_layer.unsqueeze(1)).sum(-1)
              + (pos_ub * n_ub_layer.unsqueeze(1)).sum(-1)
              + back_prop_struct.n_b_ub)
        return lb, ub

    def run_backsubstitution_individual(self, save_coeffs=False):
        self.lbs = [self.lb_inp.detach().clone().flatten(start_dim=1)]
        self.ubs = [self.ub_inp.detach().clone().flatten(start_dim=1)]
        for layer_idx, layer in enumerate(self.net):
            if isinstance(layer, nn.Flatten):
                self.lbs.append(self.lbs[-1].detach().clone())
                self.ubs.append(self.ubs[-1].detach().clone())
                continue
            curr_lb, curr_ub = self.backsubstitution_individual(
                layer_idx, self.lbs, self.ubs, save_coeffs=save_coeffs
            )
            if curr_lb is None or curr_ub is None:
                return None, None
            self.lbs.append(curr_lb.detach().clone())
            self.ubs.append(curr_ub.detach().clone())
        return self.lbs, self.ubs

    def optimize_alpha(self, alpha_steps=20, alpha_lr=1e-1, save_coeffs=False, verbose=False):
        if not self.alpha_params:
            return self.lbs, self.ubs
        optimizer = torch.optim.Adam(self.alpha_params.values(), lr=alpha_lr)
        for step in range(alpha_steps):
            optimizer.zero_grad()
            temp_lbs = [self.lb_inp.detach().clone().flatten(start_dim=1)]
            temp_ubs = [self.ub_inp.detach().clone().flatten(start_dim=1)]
            terms = []
            for layer_idx, _layer in enumerate(self.net):
                curr_lb, curr_ub = self.backsubstitution_individual(
                    layer_idx, temp_lbs, temp_ubs, save_coeffs=False
                )
                if curr_lb is None or curr_ub is None:
                    continue
                temp_lbs.append(curr_lb)
                temp_ubs.append(curr_ub)
                terms.append(curr_lb.mean(dim=1).sum())
            if not terms:
                return self.lbs, self.ubs
            objective = torch.stack(terms).sum()
            (-objective).backward()
            optimizer.step()
            for p in self.alpha_params.values():
                p.data.clamp_(0.0, 1.0)
            if verbose and (step == 0 or step == alpha_steps - 1):
                print(f"[batch alpha step {step+1:03d}] objective = {objective.item():.6f}")
        return self.run_backsubstitution_individual(save_coeffs=save_coeffs)
