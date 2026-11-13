from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from FL.util import FLmode
from repair.logging import logging



class ObjectiveType(Enum):
    SLACK_INTERVAL = "slack_interval"
    HINGE_MARGIN = "hinge_margin"
    POLYTOPE_DISTANCE = "polytope_distance"

EPS = 1e-6
BOUND_EPS = 1e-7


@dataclass
class LPSolver:

    weight: torch.Tensor | None = None  # repaired layer weight
    bias: torch.Tensor | None = None  # repaired layer bias
    modifiable_range: float = 1.0

    enable_repair: bool = True

    model: gp.Model = field(init=False)
    dw: list = field(init=False)
    db: list = field(init=False)

    objective_terms: list = field(default_factory=list)

    # linear
    n_out: int = None
    n_in: int = None

    # conv
    C_out: int = None
    C_in: int = None
    K_h: int = None
    K_w: int = None

    def __post_init__(self):

        self.model = gp.Model("repair_lp")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("TimeLimit", 7200)  # seconds

        # set shape
        if self.enable_repair:
            assert self.weight is not None and self.bias is not None
            if self.weight.ndim == 2:
                self.n_out, self.n_in = self.weight.shape
                self.dw = [[None for _ in range(self.n_in)] for _ in range(self.n_out)]
                self.db = [None for _ in range(self.n_out)]
            elif self.weight.ndim == 4:
                self.C_out, self.C_in, self.K_h, self.K_w = self.weight.shape
                self.dw = np.empty((self.C_out, self.C_in, self.K_h, self.K_w), dtype=object)  # np.array 4D for conv weights
                self.db = [None for _ in range(self.C_out)]  # 1D list for bias
            else:
                raise ValueError("Unsupported weight shape")
            

        self.model.update()

    def dispose(self):
        if self.model is not None:
            self.model.dispose()
            self.model = None

    # -- helper functions for iterating over parameters ---
    def iter_dw(self):
        if self.weight.ndim == 2:
            for i in range(self.n_out):
                for j in range(self.n_in):
                    yield self.dw[i][j]
        elif self.weight.ndim == 4:
            for co in range(self.C_out):
                for ci in range(self.C_in):
                    for kh in range(self.K_h):
                        for kw in range(self.K_w):
                            yield self.dw[co, ci, kh, kw]

    def iter_db(self):
        if self.weight.ndim == 2:
            for i in range(self.n_out):
                yield self.db[i]
        elif self.weight.ndim == 4:
            for co in range(self.C_out):
                yield self.db[co]


    def build_net(self, net, z_lb, z_ub, bounder, with_modif=False):
        model = self.model

        # convert bounds to numpy vectors
        z_lb = z_lb.detach().cpu().numpy().reshape(-1)
        z_ub = z_ub.detach().cpu().numpy().reshape(-1)

        alpha_params = {
            k: torch.nn.Parameter(v.detach().clone())
            for k, v in bounder.alpha_params.items()
        }

        # --- input variables ---
        input_dim = z_lb.shape[0]
        x = model.addMVar(input_dim, lb=z_lb, ub=z_ub, name="input")
        vars = [x]

        # --- build net structure ---
        for layer_idx, layer in enumerate(net):
            prev = vars[-1]
            lb = bounder.lbs[layer_idx+1].detach().cpu().numpy()
            lb = lb - BOUND_EPS
            ub = bounder.ubs[layer_idx+1].detach().cpu().numpy()
            ub = ub + BOUND_EPS

            # == flatten layer ==
            if isinstance(layer, torch.nn.Flatten):
                vars.append(prev)

            # == linear layer ==
            elif isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()

                out_dim = W.shape[0]
                y = model.addMVar(out_dim, lb=lb, ub=ub, name=f"layer_{layer_idx}")

                # last layer with modifiable parameters
                if with_modif and (layer_idx == len(net) - 1):
                    for i in range(out_dim):
                        expr = gp.LinExpr()
                        for j in range(W.shape[1]):
                            if self.dw[i][j] is not None:
                                expr += (W[i,j] + self.dw[i][j]) * prev[j]
                            else:
                                expr += W[i,j] * prev[j]
                        if self.db[i] is not None:
                            expr += b[i] + self.db[i]
                        else:
                            expr += b[i]
                        model.addConstr(y[i] <= expr + EPS)
                        model.addConstr(y[i] >= expr - EPS)
                else:
                    model.addConstr(y <= W @ prev + b + EPS)
                    model.addConstr(y >= W @ prev + b - EPS)
                
                vars.append(y)

            # == conv layer ==
            elif isinstance(layer, torch.nn.Conv2d):
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()

                stride_h, stride_w = layer.stride
                pad_h, pad_w = layer.padding
                dil_h, dil_w = layer.dilation

                C_out, C_in, K_h, K_w = W.shape
                C_in0, H_in, W_in = bounder.shapes[layer_idx]
                C_out0, H_out, W_out = bounder.shapes[layer_idx + 1]

                prev_vars = prev.reshape(C_in0, H_in, W_in)

                total_out = C_out0 * H_out * W_out

                y = model.addMVar(
                    total_out,
                    lb=lb.flatten(),
                    ub=ub.flatten(),
                    name=f"conv_{layer_idx}"
                )

                # ---- repaired layer ----
                for co in range(C_out0):
                    for i in range(H_out):
                        for j in range(W_out):

                            out_idx = co * H_out * W_out + i * W_out + j

                            expr = gp.LinExpr()

                            in_i = i * stride_h - pad_h
                            in_j = j * stride_w - pad_w

                            for ci in range(C_in0):
                                for kh in range(K_h):
                                    for kw in range(K_w):

                                        ih = in_i + kh * dil_h
                                        iw = in_j + kw * dil_w

                                        if 0 <= ih < H_in and 0 <= iw < W_in:

                                            var = prev_vars[ci, ih, iw]

                                            if with_modif and (layer_idx == len(net) - 1):
                                                if self.dw[co, ci, kh, kw] is not None:
                                                    expr += (W[co, ci, kh, kw] +
                                                            self.dw[co, ci, kh, kw]) * var
                                                else:
                                                    expr += W[co, ci, kh, kw] * var
                                            else:
                                                expr += W[co, ci, kh, kw] * var

                            if with_modif and (layer_idx == len(net) - 1) and self.db[co] is not None:
                                expr += b[co] + self.db[co]
                                model.addConstr(y[out_idx] <= expr + EPS)
                                model.addConstr(y[out_idx] >= expr - EPS)
                            else:
                                expr += b[co]
                                model.addConstr(y[out_idx] <= expr + EPS)
                                model.addConstr(y[out_idx] >= expr - EPS)

                vars.append(y)
            
            # == ReLU layer ==
            elif isinstance(layer, torch.nn.ReLU):
                n = lb.shape[0]
                y = model.addMVar(n, lb=np.maximum(lb, 0), ub=np.maximum(ub, 0), name=f"relu_{layer_idx}")

                # previous layer's bounds
                pre_lb = bounder.lbs[layer_idx].detach().cpu().numpy()
                pre_ub = bounder.ubs[layer_idx].detach().cpu().numpy()

                for i in range(n):
                    l = pre_lb[i].item()
                    u = pre_ub[i].item()
                    if u <= 0:
                        model.addConstr(y[i] == 0)
                    elif l >= 0:
                        model.addConstr(y[i] == prev[i])
                    else:
                        slope = u / (u - l + 1e-6)
                        intercept = -l * slope
                        alpha = alpha_params.get((layer_idx, i), None)
                        if alpha is None:
                            alpha_val = slope
                        else:
                            alpha_val = alpha.item()

                        # lower bound
                        model.addConstr(y[i] >= alpha_val * prev[i])
                        # upper bound
                        model.addConstr(y[i] <= slope * prev[i] + intercept)
                
                vars.append(y)
            
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

        model.update()

        return model, vars

    def minimize(self, obj):
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            self.model.computeIIS()
            self.model.write("model.ilp")
            return None
        return self.model.objVal


    # ========
    # Repair
    # ========

    def set_parameter_mask(self, args, weight_mask, bias_mask):
        assert weight_mask.shape == self.weight.shape
        assert bias_mask.shape == self.bias.shape

        # weight
        # linear layer: (out_dim, in_dim)
        if self.weight.ndim == 2:
            for i in range(self.n_out):
                neuron_mask = weight_mask[i]
                for j in range(self.n_in):
                    if args.flmode in (
                        FLmode.PARAMETER,
                        FLmode.ARACHNE,
                        FLmode.ORIGINAL_ARACHNE,
                    ):
                        if weight_mask[i,j]:
                            self.dw[i][j] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"dw_{i}_{j}")
                        else:
                            self.dw[i][j] = None
                    elif args.flmode == FLmode.NEURON:
                        if neuron_mask[j]:
                            self.dw[i][j] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"dw_{i}_{j}")
                        else:
                            self.dw[i][j] = None
                    else:
                        raise ValueError("Invalid FL mode")
        # conv layer: (C_out, C_in, K_h, K_w)
        elif self.weight.ndim == 4:
            for co in range(self.C_out):
                neuron_selected = weight_mask[co]
                for ci in range(self.C_in):
                    for kh in range(self.K_h):
                        for kw in range(self.K_w):
                            if args.flmode in (
                                FLmode.PARAMETER,
                                FLmode.ARACHNE,
                                FLmode.ORIGINAL_ARACHNE,
                            ):
                                selected = weight_mask[co, ci, kh, kw]

                            elif args.flmode == FLmode.NEURON:
                                selected = neuron_selected  # same for all weights in this filter

                            else:
                                raise ValueError("Invalid FL mode")

                            if selected:
                                self.dw[co, ci, kh, kw] = self.model.addVar(
                                    lb=-self.modifiable_range,
                                    ub=self.modifiable_range,
                                    name=f"dw_{co}_{ci}_{kh}_{kw}"
                                )
                            else:
                                self.dw[co, ci, kh, kw] = None
        
        # bias
        out_len = self.n_out if self.n_out is not None else self.C_out
        for i in range(out_len):
            if bias_mask[i]:
                self.db[i] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"db_{i}")
            else:
                self.db[i] = None

        self.model.update()

    def set_repairable_parameters_all(self):
        # weight
        if self.weight.ndim == 2:
            for i in range(self.n_out):
                for j in range(self.n_in):
                    self.dw[i][j] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"dw_{i}_{j}")
        elif self.weight.ndim == 4:
            for co in range(self.C_out):
                for ci in range(self.C_in):
                    for kh in range(self.K_h):
                        for kw in range(self.K_w):
                            self.dw[co, ci, kh, kw] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"dw_{co}_{ci}_{kh}_{kw}")
        else:
            raise ValueError("Unsupported weight shape")

        # bias
        out_len = self.n_out if self.n_out is not None else self.C_out
        for i in range(out_len):
            self.db[i] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"db_{i}")

        self.model.update()

    def build_net_with_subspace(self, net, z_lb, z_ub, subspace):
        '''
        Build the net structure with the given subspace constraints.

        args
        --------
        - net: the part of the DNN up to the repaired layer (inclusive)
        - z_lb, z_ub: input bounds
        - subspace: the safe subspace for the base region
            - subspace.lb, subspace.ub: safe subspace bounds for the output of the repaired layer
            - subspace.bounder: a bounder object with
                - bounder.shapes: the shape of layers
                - bounder.alpha_params: the parameters for relu relaxation
        
        net structure
        --------------------
        - modifiable parameters are only in the last layer (liner layer)
        - bounder.lbs/ubs[i+1] are the concrete bounds for the output of layer i  

        Subspace constraints
        --------------------
        - subspace.lb <= net_output <= subspace.ub
        '''
        # build net structure with modifiable parameters in the last layer
        model, vars = self.build_net(net, z_lb, z_ub, subspace.bounder, with_modif=True)

        # convert bounds to numpy vectors
        C_lb = subspace.lb.detach().cpu().numpy().reshape(-1)
        C_ub = subspace.ub.detach().cpu().numpy().reshape(-1)

        # --- subspace constraints ---
        output = vars[-1]
        for i in range(output.shape[0]):
            model.addConstr(output[i] >= C_lb[i])
            model.addConstr(output[i] <= C_ub[i])

        model.update()

        return model, vars


    # --- objective ---
    def compute_priority_weight(self, net, n_region, subspace):
        """
        Compute importance weight for a repaired region based on
        how far its current output is from the safe subspace.
        """

        violation = 0.0
        center_point = n_region.center_point
        center_output = net(center_point).squeeze(0)  # (1, out_dim) -> (out_dim,)

        for i in range(len(subspace.lb)):

            y = float(center_output[i])

            if y < subspace.lb[i]:
                violation += subspace.lb[i] - y

            elif y > subspace.ub[i]:
                violation += y - subspace.ub[i]

        return 1.0 + violation**2 / (1e-6 + len(subspace.lb))
    
    def _objective_slack_interval(self, expr, lb, ub):
        s_low = self.model.addVar(lb=0)
        s_up = self.model.addVar(lb=0)

        self.model.addConstr(expr + s_low >= lb)
        self.model.addConstr(expr - s_up <= ub)

        return s_low + s_up
    
    def _objective_hinge(self, expr, lb):
        slack = self.model.addVar(lb=0)
        self.model.addConstr(expr + slack >= lb)

        return slack

    def _objective_polytope_distance(self, expr, lb, ub):

        v_low = self.model.addVar(lb=0)
        v_up = self.model.addVar(lb=0)

        # violation constraints
        self.model.addConstr(v_low >= lb - expr)
        self.model.addConstr(v_up >= expr - ub)

        return v_low * v_low + v_up * v_up
    
    def set_objective(self, args, net, n_bounder, n_region, subspace, obj_weight=1.0):
        obj_type = args.obj_type
        use_priority_weight = args.use_priority_weight

        z_lb = n_region.lb
        z_ub = n_region.ub
        _, vars = self.build_net(net, z_lb, z_ub, n_bounder, with_modif=True)

        if use_priority_weight:
            prior_weight = float(self.compute_priority_weight(net, n_region, subspace))
            logging(f"priority weight for region {n_region.data_id}: {prior_weight:.4f}", logfile=args.logfile)

        terms = []
        out_var = vars[-1]  # 1D
        out_len = out_var.shape[0]
        for i in range(out_len):
            expr = out_var[i]

            if obj_type == ObjectiveType.SLACK_INTERVAL:
                obj = self._objective_slack_interval(expr, subspace.lb[i], subspace.ub[i])
                terms.append(obj_weight * obj)

            elif obj_type == ObjectiveType.POLYTOPE_DISTANCE:
                lbi = subspace.lb[i]
                ubi = subspace.ub[i]
                poly = self._objective_polytope_distance(expr, lbi, ubi)

                if use_priority_weight:
                    terms.append(prior_weight * poly)
                else:
                    terms.append(obj_weight * poly)

        self.model.update()
        self.objective_terms.extend(terms)

    def set_objective_ibp(self, args, net, bounder, region, subspace, obj_weight=1.0):
        obj_type = args.obj_type
        use_priority_weight = args.use_priority_weight

        _lbs, _ubs = bounder.lbs, bounder.ubs
        lb_in_repair_layer = _lbs[-2]
        ub_in_repair_layer = _ubs[-2]

        W = self.weight.detach().cpu().numpy()
        b = self.bias.detach().cpu().numpy()

        prior_weight = 1.0
        if use_priority_weight:
            prior_weight = float(self.compute_priority_weight(net, region, subspace))
            logging(f"priority weight for region {region.data_id}: {prior_weight:.4f}", logfile=args.logfile)

        terms = []
        exprs = []
        repair_layer = net[-1]
        if isinstance(repair_layer, torch.nn.Linear):
            out_dim = W.shape[0]
            for i in range(out_dim):
                expr_lb = gp.LinExpr()
                expr_ub = gp.LinExpr()
                for j in range(W.shape[1]):
                    if self.dw[i][j] is not None:
                        expr_lb += (W[i,j] + self.dw[i][j]) * (lb_in_repair_layer[j] if W[i,j] >= 0 else ub_in_repair_layer[j])
                        expr_ub += (W[i,j] + self.dw[i][j]) * (ub_in_repair_layer[j] if W[i,j] >= 0 else lb_in_repair_layer[j])
                    else:
                        expr_lb += W[i,j] * (lb_in_repair_layer[j] if W[i,j] >= 0 else ub_in_repair_layer[j])
                        expr_ub += W[i,j] * (ub_in_repair_layer[j] if W[i,j] >= 0 else lb_in_repair_layer[j])
                if self.db[i] is not None:
                    expr_lb += b[i] + self.db[i]
                    expr_ub += b[i] + self.db[i]
                else:
                    expr_lb += b[i]
                    expr_ub += b[i]
                
                exprs.append((i, expr_lb, expr_ub))

        elif isinstance(repair_layer, torch.nn.Conv2d):
            C_out, C_in, K_h, K_w = W.shape

            stride_h, stride_w = repair_layer.stride
            pad_h, pad_w = repair_layer.padding
            dil_h, dil_w = repair_layer.dilation

            C_in0, H_in, W_in = bounder.shapes[-2]
            C_out0, H_out, W_out = bounder.shapes[-1]

            in_lb = lb_in_repair_layer.reshape(C_in0, H_in, W_in)
            in_ub = ub_in_repair_layer.reshape(C_in0, H_in, W_in)

            for co in range(C_out0):
                for i in range(H_out):
                    for j in range(W_out):

                        expr_lb = gp.LinExpr()
                        expr_ub = gp.LinExpr()

                        in_i = i * stride_h - pad_h
                        in_j = j * stride_w - pad_w

                        for ci in range(C_in0):
                            for kh in range(K_h):
                                for kw in range(K_w):

                                    ih = in_i + kh * dil_h
                                    iw = in_j + kw * dil_w

                                    if 0 <= ih < H_in and 0 <= iw < W_in:

                                        w_val = W[co, ci, kh, kw]

                                        if self.dw[co, ci, kh, kw] is not None:
                                            w_eff = w_val + self.dw[co, ci, kh, kw]
                                        else:
                                            w_eff = w_val

                                        # IBP bound selection
                                        if w_val >= 0:
                                            expr_lb += w_eff * in_lb[ci, ih, iw]
                                            expr_ub += w_eff * in_ub[ci, ih, iw]
                                        else:
                                            expr_lb += w_eff * in_ub[ci, ih, iw]
                                            expr_ub += w_eff * in_lb[ci, ih, iw]

                        # bias
                        if self.db[co] is not None:
                            expr_lb += b[co] + self.db[co]
                            expr_ub += b[co] + self.db[co]
                        else:
                            expr_lb += b[co]
                            expr_ub += b[co]

                        idx = co * H_out * W_out + i * W_out + j
                        exprs.append((idx, expr_lb, expr_ub))

        for idx, expr_lb, expr_ub in exprs:
            sub_lbi = subspace.lb[idx]
            sub_ubi = subspace.ub[idx]

            if obj_type == ObjectiveType.SLACK_INTERVAL:
                obj = self._objective_slack_interval(expr_lb, sub_lbi, sub_ubi) \
                        + self._objective_slack_interval(expr_ub, sub_lbi, sub_ubi)
                terms.append(obj_weight * obj)

            elif obj_type == ObjectiveType.POLYTOPE_DISTANCE:
                poly = self._objective_polytope_distance(expr_lb, sub_lbi, sub_ubi) \
                            + self._objective_polytope_distance(expr_ub, sub_lbi, sub_ubi)

                if use_priority_weight:
                    terms.append(prior_weight * poly)
                else:
                    terms.append(obj_weight * poly)

        self.model.update()
        self.objective_terms.extend(terms)

    def set_objective_lastlayer(self, args, net, bounder, region):
        
        _lbs, _ubs = bounder.lbs, bounder.ubs
        lb_in_repair_layer = _lbs[-2]
        ub_in_repair_layer = _ubs[-2]

        W = self.weight.detach().cpu().numpy()
        b = self.bias.detach().cpu().numpy()

        terms = []
        exprs = []
        repair_layer = net[-1]
        if isinstance(repair_layer, torch.nn.Linear):
            out_dim = W.shape[0]
            for i in range(out_dim):
                expr_lb = gp.LinExpr()
                expr_ub = gp.LinExpr()
                for j in range(W.shape[1]):
                    if self.dw[i][j] is not None:
                        expr_lb += (W[i,j] + self.dw[i][j]) * (lb_in_repair_layer[j] if W[i,j] >= 0 else ub_in_repair_layer[j])
                        expr_ub += (W[i,j] + self.dw[i][j]) * (ub_in_repair_layer[j] if W[i,j] >= 0 else lb_in_repair_layer[j])
                    else:
                        expr_lb += W[i,j] * (lb_in_repair_layer[j] if W[i,j] >= 0 else ub_in_repair_layer[j])
                        expr_ub += W[i,j] * (ub_in_repair_layer[j] if W[i,j] >= 0 else lb_in_repair_layer[j])
                if self.db[i] is not None:
                    expr_lb += b[i] + self.db[i]
                    expr_ub += b[i] + self.db[i]
                else:
                    expr_lb += b[i]
                    expr_ub += b[i]
                
                exprs.append((i, expr_lb, expr_ub))

        elif isinstance(repair_layer, torch.nn.Conv2d):
            C_out, C_in, K_h, K_w = W.shape

            stride_h, stride_w = repair_layer.stride
            pad_h, pad_w = repair_layer.padding
            dil_h, dil_w = repair_layer.dilation

            C_in0, H_in, W_in = bounder.shapes[-2]
            C_out0, H_out, W_out = bounder.shapes[-1]

            in_lb = lb_in_repair_layer.reshape(C_in0, H_in, W_in)
            in_ub = ub_in_repair_layer.reshape(C_in0, H_in, W_in)

            for co in range(C_out0):
                for i in range(H_out):
                    for j in range(W_out):

                        expr_lb = gp.LinExpr()
                        expr_ub = gp.LinExpr()

                        in_i = i * stride_h - pad_h
                        in_j = j * stride_w - pad_w

                        for ci in range(C_in0):
                            for kh in range(K_h):
                                for kw in range(K_w):

                                    ih = in_i + kh * dil_h
                                    iw = in_j + kw * dil_w

                                    if 0 <= ih < H_in and 0 <= iw < W_in:

                                        w_val = W[co, ci, kh, kw]

                                        if self.dw[co, ci, kh, kw] is not None:
                                            w_eff = w_val + self.dw[co, ci, kh, kw]
                                        else:
                                            w_eff = w_val

                                        # IBP bound selection
                                        if w_val >= 0:
                                            expr_lb += w_eff * in_lb[ci, ih, iw]
                                            expr_ub += w_eff * in_ub[ci, ih, iw]
                                        else:
                                            expr_lb += w_eff * in_ub[ci, ih, iw]
                                            expr_ub += w_eff * in_lb[ci, ih, iw]

                        # bias
                        if self.db[co] is not None:
                            expr_lb += b[co] + self.db[co]
                            expr_ub += b[co] + self.db[co]
                        else:
                            expr_lb += b[co]
                            expr_ub += b[co]

                        idx = co * H_out * W_out + i * W_out + j
                        exprs.append((idx, expr_lb, expr_ub))
        
        def output_constraint_objective(spec_lb):
            t = self.model.addVar(lb=0)
            self.model.addConstr(t >= -spec_lb)
            return t

        spec_C = region.spec.C  # (num_constraints, out_dim)
        assert spec_C.shape[1] == len(exprs), "Output dimension mismatch between region spec and net output"
        
        for c_dim in range(spec_C.shape[0]):
            c = spec_C[c_dim]  # (out_dim,)
            pos_C = torch.clamp(c, min=0)
            pos_C_np = pos_C.detach().cpu().numpy()
            neg_C = torch.clamp(c, max=0)
            neg_C_np = neg_C.detach().cpu().numpy()
            spec_lb = gp.LinExpr()
            for (idx, e_lb, e_ub) in exprs:
                if pos_C_np[idx] != 0:
                    spec_lb += pos_C_np[idx] * e_lb
                if neg_C_np[idx] != 0:
                    spec_lb += neg_C_np[idx] * e_ub

            # output constraint: spec_lb > 0
            obj = output_constraint_objective(spec_lb)
            terms.append(obj)

        self.model.update()
        self.objective_terms.extend(terms)

    # add a regularization term to encourage smaller parameter changes
    # and collected objective terms into a single objective function
    def build_objective_l2(self, strength_dist=10.0, lambda_reg=1.0):
        num_w = sum(1 for var in self.iter_dw() if var is not None)
        num_b = sum(1 for var in self.iter_db() if var is not None)
        lambda_w = lambda_reg / (max(1, num_w))
        lambda_b = lambda_reg / (max(1, num_b))

        # ---- weight regularization ----
        param_reg_w = gp.quicksum(
            var * var
            for var in self.iter_dw()
            if var is not None
        )

        # ---- bias regularization ----
        param_reg_b = gp.quicksum(
            var * var
            for var in self.iter_db()
            if var is not None
        )

        num_terms = float(max(1, len(self.objective_terms)))

        obj = (
            strength_dist * (gp.quicksum(self.objective_terms) / num_terms)
            + lambda_w * param_reg_w
            + lambda_b * param_reg_b
        )

        self.model.setObjective(obj, GRB.MINIMIZE)

    def build_objective_l1(self, strength_dist=10.0, lambda_reg=1.0):
        num_w = sum(1 for var in self.iter_dw() if var is not None)
        num_b = sum(1 for var in self.iter_db() if var is not None)
        lambda_w = lambda_reg / (max(1, num_w))
        lambda_b = lambda_reg / (max(1, num_b))

        t_w = []
        t_b = []

        # ---- weight abs ----
        for var in self.iter_dw():
            if var is not None:
                t = self.model.addVar(lb=0)
                self.model.addConstr(t >= var)
                self.model.addConstr(t >= -var)
                t_w.append(t)

        # ---- bias abs ----
        for var in self.iter_db():
            if var is not None:
                t = self.model.addVar(lb=0)
                self.model.addConstr(t >= var)
                self.model.addConstr(t >= -var)
                t_b.append(t)

        param_reg = lambda_w * gp.quicksum(t_w) + lambda_b * gp.quicksum(t_b)

        num_terms = float(max(1, len(self.objective_terms)))

        obj = (
            strength_dist * (gp.quicksum(self.objective_terms) / num_terms)
            + param_reg
        )

        self.model.setObjective(obj, GRB.MINIMIZE)


    # --- solve ---
    def repair_solve(self):
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            status = self.model.status
            print(f"Optimization failed with status {status}")
            return None

        W = self.weight.detach().cpu().numpy()
        B = self.bias.detach().cpu().numpy()
        new_W = W.copy()
        new_B = B.copy()

        if self.weight.ndim == 2:
            for i in range(self.n_out):
                for j in range(self.n_in):
                    if self.dw[i][j] is not None:
                        new_W[i,j] += self.dw[i][j].X
        elif self.weight.ndim == 4:
            for co in range(self.C_out):
                for ci in range(self.C_in):
                    for kh in range(self.K_h):
                        for kw in range(self.K_w):
                            if self.dw[co, ci, kh, kw] is not None:
                                new_W[co, ci, kh, kw] += self.dw[co, ci, kh, kw].X

        out_len = self.n_out if self.n_out is not None else self.C_out
        for i in range(out_len):
            if self.db[i] is not None:
                new_B[i] += self.db[i].X

        return (
            torch.tensor(new_W),
            torch.tensor(new_B)
        )
