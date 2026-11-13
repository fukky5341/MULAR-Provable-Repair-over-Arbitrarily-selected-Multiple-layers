from dataclasses import dataclass, field
import os
from pathlib import Path
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB

EPS = 1e-6
BOUND_EPS = 1e-7
GIB = 1024 ** 3
SOFT_MEMORY_FRACTION = 0.75


def _positive_memory_limit(value: str) -> int | None:
    """Parse a cgroup byte limit, ignoring `max` and invalid values."""
    try:
        limit = int(value.strip())
    except (TypeError, ValueError):
        return None
    return limit if 0 < limit < (1 << 60) else None


def _cgroup_memory_limits() -> list[int]:
    limits = []
    candidates = {
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    }
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative_path = line.split(":", 2)
            del hierarchy
            relative_path = relative_path.lstrip("/")
            if not controllers:
                candidates.add(
                    Path("/sys/fs/cgroup") / relative_path / "memory.max"
                )
            elif "memory" in controllers.split(","):
                candidates.add(
                    Path("/sys/fs/cgroup/memory")
                    / relative_path
                    / "memory.limit_in_bytes"
                )
    except (OSError, ValueError):
        pass

    for path in candidates:
        try:
            limit = _positive_memory_limit(path.read_text())
        except OSError:
            continue
        if limit is not None:
            limits.append(limit)
    return limits


def _slurm_memory_limits() -> list[int]:
    limits = []
    try:
        memory_per_node_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    except ValueError:
        memory_per_node_mb = 0
    if memory_per_node_mb > 0:
        limits.append(memory_per_node_mb * 1024 ** 2)

    try:
        memory_per_cpu_mb = int(os.environ.get("SLURM_MEM_PER_CPU", "0"))
        cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", "0"))
    except ValueError:
        memory_per_cpu_mb = cpus = 0
    if memory_per_cpu_mb > 0 and cpus > 0:
        limits.append(memory_per_cpu_mb * cpus * 1024 ** 2)
    return limits


def default_soft_memory_limit_gb() -> float | None:
    """Reserve headroom while respecting physical and job memory limits."""
    limits = _cgroup_memory_limits() + _slurm_memory_limits()
    try:
        physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        physical_memory = 0
    if physical_memory > 0:
        limits.append(physical_memory)
    if not limits:
        return None
    return min(limits) / GIB * SOFT_MEMORY_FRACTION

@dataclass
class PreparedLP:

    net: torch.nn.Module

    modifiable_range: float = 1.0

    enable_repair: bool = True

    soft_mem_limit_gb: float | None = None

    model: gp.Model = field(init=False)
    dw: list = field(init=False)
    dbs: list = field(init=False) # db for each layer
    last_failure_reason: str | None = field(init=False, default=None)

    def __post_init__(self):

        self.model = gp.Model("repair_lp")
        self.model.setParam("OutputFlag", 1)
        self.model.setParam("TimeLimit", 3600*12)  # seconds
        if self.soft_mem_limit_gb is None:
            self.soft_mem_limit_gb = default_soft_memory_limit_gb()
        if self.soft_mem_limit_gb is not None and self.soft_mem_limit_gb > 0:
            self.model.setParam("SoftMemLimit", self.soft_mem_limit_gb)

        # set modifiable parameters list
        dw = []
        dbs = []
        if self.enable_repair:
            for layer_idx, layer in enumerate(self.net):
                if isinstance(layer, torch.nn.Linear):
                    weight = layer.weight
                    n_out, n_in = weight.shape
                    if dw == []:
                        dw = [[None for _ in range(n_in)] for _ in range(n_out)]
                    db = [None for _ in range(n_out)]
                    dbs.append(db)
                elif isinstance(layer, torch.nn.Conv2d):
                    weight = layer.weight
                    C_out, C_in, K_h, K_w = weight.shape
                    if dw == []:
                        dw = np.empty((C_out, C_in, K_h, K_w), dtype=object)  # np.array 4D for conv weights
                    db = [None for _ in range(C_out)]
                    dbs.append(db)
                else:
                    dbs.append(None)
            
            self.dw = dw
            self.dbs = dbs
        self.model.update()
    
    def dispose(self):
        if self.model is not None:
            self.model.dispose()
            self.model = None


    def build_net(self, net, z_lb, z_ub, bounder):
        model = self.model

        # convert bounds to numpy vectors
        z_lb = z_lb.detach().cpu().numpy().reshape(-1)
        z_ub = z_ub.detach().cpu().numpy().reshape(-1)

        alpha_params = {
            k: torch.nn.Parameter(v.detach().clone())
            for k, v in bounder.alpha_params.items()
        }

        # --- input variables ---
        vars = [(None, None)]  # placeholder for input layer

        # --- build net structure ---
        first_linear = True
        for layer_idx, layer in enumerate(net):
            prev_vlb, prev_vub = vars[-1]

            assert (layer_idx == 0 and prev_vlb is None and prev_vub is None) \
                    or (layer_idx > 0 and prev_vlb is not None and prev_vub is not None)

            pre_lb = bounder.lbs[layer_idx].detach().cpu().numpy()
            pre_lb = pre_lb - BOUND_EPS
            pre_ub = bounder.ubs[layer_idx].detach().cpu().numpy()
            pre_ub = pre_ub + BOUND_EPS

            lb = bounder.lbs[layer_idx+1].detach().cpu().numpy()
            lb = lb - BOUND_EPS
            ub = bounder.ubs[layer_idx+1].detach().cpu().numpy()
            ub = ub + BOUND_EPS

            # == flatten layer ==
            if isinstance(layer, torch.nn.Flatten):
                vars.append((prev_vlb, prev_vub))

            # == linear layer ==
            elif isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()

                if first_linear:
                    dw_layer = self.dw
                    first_linear = False
                else:
                    dw_layer = None
                db_layer = self.dbs[layer_idx]

                out_dim = W.shape[0]
                y_lb = model.addMVar(out_dim, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"layer_{layer_idx}_lb")
                y_ub = model.addMVar(out_dim, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"layer_{layer_idx}_ub")

                for i in range(out_dim):
                    expr_lb = gp.LinExpr()
                    expr_ub = gp.LinExpr()
                    for j in range(W.shape[1]):
                        if dw_layer is not None:  # constant input bounds
                            v_lb = model.addVar(lb=-GRB.INFINITY, name=f"v_{layer_idx}_{i}_{j}_lb")
                            v_ub = model.addVar(lb=-GRB.INFINITY, name=f"v_{layer_idx}_{i}_{j}_ub")
                            if self.dw[i][j] is not None:
                                model.addConstr(v_ub >= (W[i,j] + dw_layer[i][j]) * pre_ub[j].item())
                                model.addConstr(v_ub >= (W[i,j] + dw_layer[i][j]) * pre_lb[j].item())
                                model.addConstr(v_lb <= (W[i,j] + dw_layer[i][j]) * pre_ub[j].item())
                                model.addConstr(v_lb <= (W[i,j] + dw_layer[i][j]) * pre_lb[j].item())
                            else:
                                model.addConstr(v_ub >= W[i,j] * pre_ub[j].item())
                                model.addConstr(v_ub >= W[i,j] * pre_lb[j].item())
                                model.addConstr(v_lb <= W[i,j] * pre_ub[j].item())
                                model.addConstr(v_lb <= W[i,j] * pre_lb[j].item())
                            expr_lb += v_lb
                            expr_ub += v_ub
                        else:  # variable input bounds
                            if W[i,j] >= 0:
                                expr_lb += W[i,j] * prev_vlb[j]
                                expr_ub += W[i,j] * prev_vub[j]
                            else:
                                expr_lb += W[i,j] * prev_vub[j]
                                expr_ub += W[i,j] * prev_vlb[j]
                    if db_layer is not None and db_layer[i] is not None:
                        expr_lb += b[i] + db_layer[i]
                        expr_ub += b[i] + db_layer[i]
                    else:
                        expr_lb += b[i]
                        expr_ub += b[i]
                    model.addConstr(y_lb[i] == expr_lb)
                    model.addConstr(y_ub[i] == expr_ub)
                
                for i in range(out_dim):
                    model.addConstr(y_ub[i] >= y_lb[i])
                vars.append((y_lb, y_ub))

            # == conv layer ==
            elif isinstance(layer, torch.nn.Conv2d):
                W = layer.weight.detach().cpu().numpy()
                b = layer.bias.detach().cpu().numpy()

                if first_linear:
                    dw_layer = self.dw
                    first_linear = False
                else:
                    dw_layer = None
                db_layer = self.dbs[layer_idx]

                stride_h, stride_w = layer.stride
                pad_h, pad_w = layer.padding
                dil_h, dil_w = layer.dilation

                C_out, C_in, K_h, K_w = W.shape
                C_in0, H_in, W_in = bounder.shapes[layer_idx]
                C_out0, H_out, W_out = bounder.shapes[layer_idx + 1]

                prev_vlb_c = prev_vlb.reshape(C_in0, H_in, W_in)
                prev_vub_c = prev_vub.reshape(C_in0, H_in, W_in)
                pre_lb_c = pre_lb.reshape(C_in0, H_in, W_in)
                pre_ub_c = pre_ub.reshape(C_in0, H_in, W_in)

                total_out = C_out0 * H_out * W_out

                y_lb = model.addMVar(total_out, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"conv_{layer_idx}_lb")
                y_ub = model.addMVar(total_out, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"conv_{layer_idx}_ub")


                # ---- repaired layer ----
                for co in range(C_out0):
                    for i in range(H_out):
                        for j in range(W_out):

                            out_idx = co * H_out * W_out + i * W_out + j

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

                                            if dw_layer is not None:  # constant input bounds
                                                v_lb = model.addVar(lb=-GRB.INFINITY, name=f"v_{layer_idx}_{co}_{ci}_{kh}_{kw}_lb")
                                                v_ub = model.addVar(lb=-GRB.INFINITY, name=f"v_{layer_idx}_{co}_{ci}_{kh}_{kw}_ub")
                                                if dw_layer[co, ci, kh, kw] is not None:
                                                    model.addConstr(v_ub >= (W[co, ci, kh, kw] + dw_layer[co, ci, kh, kw]) * pre_ub_c[ci, ih, iw].item())
                                                    model.addConstr(v_ub >= (W[co, ci, kh, kw] + dw_layer[co, ci, kh, kw]) * pre_lb_c[ci, ih, iw].item())
                                                    model.addConstr(v_lb <= (W[co, ci, kh, kw] + dw_layer[co, ci, kh, kw]) * pre_ub_c[ci, ih, iw].item())
                                                    model.addConstr(v_lb <= (W[co, ci, kh, kw] + dw_layer[co, ci, kh, kw]) * pre_lb_c[ci, ih, iw].item())
                                                else:
                                                    model.addConstr(v_ub >= W[co, ci, kh, kw] * pre_ub_c[ci, ih, iw].item())
                                                    model.addConstr(v_ub >= W[co, ci, kh, kw] * pre_lb_c[ci, ih, iw].item())
                                                    model.addConstr(v_lb <= W[co, ci, kh, kw] * pre_ub_c[ci, ih, iw].item())
                                                    model.addConstr(v_lb <= W[co, ci, kh, kw] * pre_lb_c[ci, ih, iw].item())
                                                expr_lb += v_lb
                                                expr_ub += v_ub
                                            else:  # variable input bounds
                                                if W[co, ci, kh, kw] >= 0:
                                                    expr_lb += W[co, ci, kh, kw] * prev_vlb_c[ci, ih, iw]
                                                    expr_ub += W[co, ci, kh, kw] * prev_vub_c[ci, ih, iw]
                                                else:
                                                    expr_lb += W[co, ci, kh, kw] * prev_vub_c[ci, ih, iw]
                                                    expr_ub += W[co, ci, kh, kw] * prev_vlb_c[ci, ih, iw]

                            if db_layer is not None and db_layer[co] is not None:
                                expr_lb += b[co] + db_layer[co]
                                expr_ub += b[co] + db_layer[co]
                                model.addConstr(y_lb[out_idx] == expr_lb)
                                model.addConstr(y_ub[out_idx] == expr_ub)
                            else:
                                expr_lb += b[co]
                                expr_ub += b[co]
                                model.addConstr(y_lb[out_idx] == expr_lb)
                                model.addConstr(y_ub[out_idx] == expr_ub)

                for i in range(total_out):
                    model.addConstr(y_ub[i] >= y_lb[i])
                vars.append((y_lb, y_ub))
            
            # == ReLU layer ==
            elif isinstance(layer, torch.nn.ReLU):
                n = lb.shape[0]
                y_lb = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"relu_{layer_idx}_lb")
                y_ub = model.addMVar(n, lb=0, ub=GRB.INFINITY, name=f"relu_{layer_idx}_ub")

                for i in range(n):
                    l = pre_lb[i].item()
                    u = pre_ub[i].item()
                    
                    if l >= 0:
                        slope = 1
                    elif u <= 0:
                        slope = 0
                    else:
                        slope = u / (u - l + 1e-6)
                        assert 0 <= slope <= 1, f"slope out of range: {slope}"

                    alpha = alpha_params.get((layer_idx, i), None)
                    if alpha is None:
                        alpha_val = slope
                    else:
                        assert 0 <= alpha.item() <= 1, f"alpha value out of range: {alpha.item()}"
                        alpha_val = alpha.item()

                    # lower bound
                    model.addConstr(y_lb[i] == alpha_val * prev_vlb[i])
                    # upper bound
                    model.addConstr(y_ub[i] >= prev_vub[i])
                    model.addConstr(y_ub[i] >= 0)
                
                for i in range(n):
                    model.addConstr(y_ub[i] >= y_lb[i])
                vars.append((y_lb, y_ub))
            
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")
            
        model.update()

        return model, vars
    
    def init_repair_vars(self, net):
        # note: only support modifying all parameters in the first linear/conv layer and all biases
        first_linear = True
        for layer_idx, layer in enumerate(net):
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight
                n_out, n_in = weight.shape
                if first_linear:
                    dw_layer = self.dw
                    first_linear = False
                    for i in range(n_out):
                        for j in range(n_in):
                            dw_layer[i][j] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"dw_{layer_idx}_{i}_{j}")
                db_layer = self.dbs[layer_idx]
                for i in range(n_out):
                    db_layer[i] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"db_{layer_idx}_{i}")
            elif isinstance(layer, torch.nn.Conv2d):
                weight = layer.weight
                C_out, C_in, K_h, K_w = weight.shape
                if first_linear:
                    dw_layer = self.dw
                    first_linear = False
                    for co in range(C_out):
                        for ci in range(C_in):
                            for kh in range(K_h):
                                for kw in range(K_w):
                                    dw_layer[co, ci, kh, kw] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"dw_{layer_idx}_{co}_{ci}_{kh}_{kw}")
                db_layer = self.dbs[layer_idx]
                for i in range(C_out):
                    db_layer[i] = self.model.addVar(lb=-self.modifiable_range, ub=self.modifiable_range, name=f"db_{layer_idx}_{i}")
            else:
                continue
        self.model.update()

    def add_repair_constraints(self, vars, C):
        out_lb, out_ub = vars[-1]
        for i in range(C.shape[0]):
            c = C[i].detach().cpu().numpy()
            c_pos = np.maximum(c, 0)
            c_neg = np.minimum(c, 0)
            out_constr_expr = c_pos @ out_lb + c_neg @ out_ub
            self.model.addConstr(out_constr_expr >= EPS, name=f"repair_constr_{i}")
        
        self.model.update()

    def set_objective(self):
        model = self.model
        # objective: minimize the total modification
        obj_expr = gp.LinExpr()

        # dw
        if self.dw != []:
            dw_np = np.array(self.dw) if isinstance(self.dw, list) else self.dw
            dw_flat = dw_np.flatten()
            for dw_var in dw_flat:
                if dw_var is not None:
                    t = model.addVar(lb=0)
                    model.addConstr(t >= dw_var)
                    model.addConstr(t >= -dw_var)
                    obj_expr += t
        # db
        if self.dbs != []:
            for db_layer in self.dbs:
                if db_layer is not None:
                    for db_var in db_layer:
                        if db_var is not None:
                            t = model.addVar(lb=0)
                            model.addConstr(t >= db_var)
                            model.addConstr(t >= -db_var)
                            obj_expr += t
        
        self.model.update()
        self.model.setObjective(obj_expr, GRB.MINIMIZE)

    def repair_solve(self):
        self.last_failure_reason = None
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            status = self.model.status
            if status == GRB.TIME_LIMIT:
                time_limit = self.model.Params.TimeLimit
                self.last_failure_reason = (
                    f"Gurobi time limit reached ({time_limit:g} seconds)"
                )
            elif status == getattr(GRB, "MEM_LIMIT", None):
                soft_limit = self.model.Params.SoftMemLimit
                self.last_failure_reason = (
                    f"Gurobi soft memory limit reached "
                    f"({soft_limit:g} GB)"
                )
            elif status == GRB.INTERRUPTED:
                self.last_failure_reason = "Gurobi optimization interrupted"
            else:
                self.last_failure_reason = (
                    f"Gurobi optimization failed with status {status}"
                )
            print(self.last_failure_reason)
            return None
        
        # extract modified parameters
        first_linear = True
        for layer_idx, layer in enumerate(self.net):
            if isinstance(layer, torch.nn.Linear):
                # dw
                if first_linear:
                    first_linear = False

                    weight = layer.weight
                    n_out, n_in = weight.shape
                    new_W = weight.detach().cpu().numpy().copy()
                    dw_layer = self.dw
                    for i in range(n_out):
                        for j in range(n_in):
                            if dw_layer[i][j] is not None:
                                new_W[i,j] += dw_layer[i][j].X
                    
                    # update weight
                    layer.weight.data = torch.tensor(new_W)

                # db
                db_layer = self.dbs[layer_idx]
                if db_layer is not None:
                    bias = layer.bias
                    new_b = bias.detach().cpu().numpy().copy()
                    for i in range(bias.shape[0]):
                        if db_layer[i] is not None:
                            new_b[i] += db_layer[i].X
                    # update bias
                    layer.bias.data = torch.tensor(new_b)
            elif isinstance(layer, torch.nn.Conv2d):
                # dw
                if first_linear:
                    first_linear = False

                    weight = layer.weight
                    C_out, C_in, K_h, K_w = weight.shape
                    new_W = weight.detach().cpu().numpy().copy()
                    dw_layer = self.dw
                    for co in range(C_out):
                        for ci in range(C_in):
                            for kh in range(K_h):
                                for kw in range(K_w):
                                    if dw_layer[co, ci, kh, kw] is not None:
                                        new_W[co, ci, kh, kw] += dw_layer[co, ci, kh, kw].X
                    # update weight
                    layer.weight.data = torch.tensor(new_W)
                # db
                db_layer = self.dbs[layer_idx]
                if db_layer is not None:
                    bias = layer.bias
                    new_b = bias.detach().cpu().numpy().copy()
                    for i in range(bias.shape[0]):
                        if db_layer[i] is not None:
                            new_b[i] += db_layer[i].X
                    # update bias
                    layer.bias.data = torch.tensor(new_b)
            else:
                continue

        return self.net
