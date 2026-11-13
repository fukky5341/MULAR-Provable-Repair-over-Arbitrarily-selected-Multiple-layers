'''
small 2-layer network
    - first layer has two ReLU neurons
    - second layer has one linear output

let the hidden neurons be x11, x12 and the output be x21
inputs: x01, x02

Example: a violating case is reachable, but last-layer repair cannot complete
because the hidden activation interval still starts at zero, so the strict
margin constraint cannot be guaranteed by last-layer changes alone.

inputs:
    - 2 input bounds: [0, 1], [0,1]

first layer: two ReLU neurons with pre-activations
    z11 = w11 * x01 + w12 * x02
    z12 = w21 * x01 + w22 * x02
second layer:
    x21 = v11 * x11 + v12 * x12

violation specification: x21 <= 3

last-layer repair:
    - hidden activation lower bound is 0
    - the lower bound of x21 cannot exceed 3 without changing
      the hidden layer, so last-layer repair fails
'''

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace


def _parse_csv_floats(raw: str, expected_len: int | None = None) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(values)}")
    return values


@dataclass(frozen=True)
class NetworkConfig:
    input_lb: tuple[float, float] = (0.0, 0.0)
    input_ub: tuple[float, float] = (1.0, 1.0)
    hidden_w: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 1.0), (-1.0, 1.0))
    output_w: tuple[float, float] = (1.0, -1.0)


def affine_box_bounds(weights: tuple[float, float], lower: tuple[float, float], upper: tuple[float, float]) -> tuple[float, float]:
    candidates = (
        weights[0] * lower[0] + weights[1] * lower[1],
        weights[0] * lower[0] + weights[1] * upper[1],
        weights[0] * upper[0] + weights[1] * lower[1],
        weights[0] * upper[0] + weights[1] * upper[1],
    )
    return min(candidates), max(candidates)


def matrix2x2_from_csv(raw: str) -> tuple[tuple[float, float], tuple[float, float]]:
    values = _parse_csv_floats(raw, 4)
    return (values[0], values[1]), (values[2], values[3])


def vector2_from_csv(raw: str) -> tuple[float, float]:
    values = _parse_csv_floats(raw, 2)
    return values[0], values[1]


def matrix2x2_delta(
    before: tuple[tuple[float, float], tuple[float, float]],
    after: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (after[0][0] - before[0][0], after[0][1] - before[0][1]),
        (after[1][0] - before[1][0], after[1][1] - before[1][1]),
    )


def matrix2x2_add(
    left: tuple[tuple[float, float], tuple[float, float]],
    right: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (left[0][0] + right[0][0], left[0][1] + right[0][1]),
        (left[1][0] + right[1][0], left[1][1] + right[1][1]),
    )


def vector2_delta(before: tuple[float, float], after: tuple[float, float]) -> tuple[float, float]:
    return after[0] - before[0], after[1] - before[1]


def print_matrix2x2(label: str, matrix: tuple[tuple[float, float], tuple[float, float]]) -> None:
    print(f"{label}:")
    print(f"  [{matrix[0][0]}, {matrix[0][1]}]")
    print(f"  [{matrix[1][0]}, {matrix[1][1]}]")


def print_vector2(label: str, vector: tuple[float, float]) -> None:
    print(f"{label}: [{vector[0]}, {vector[1]}]")


def relu_interval_bounds(z_lb: float, z_ub: float) -> tuple[float, float]:
    """Propagate bounds through ReLU using the convex relaxation."""
    if z_ub <= 0.0:
        return 0.0, 0.0
    if z_lb >= 0.0:
        return z_lb, z_ub
    return 0.0, z_ub


def _add_relu_relaxation(model, z_var, x_var, z_lb, z_ub, name_prefix):
    """Add the linear relaxation of x = max(0, z)."""
    if z_ub <= 0.0:
        model.addConstr(x_var == 0.0, name=f"{name_prefix}_inactive")
        return None
    if z_lb >= 0.0:
        model.addConstr(x_var == z_var, name=f"{name_prefix}_active")
        return None

    slope = z_ub / (z_ub - z_lb)
    intercept = -z_lb * slope
    model.addConstr(x_var >= 0.0, name=f"{name_prefix}_x_ge_0")
    model.addConstr(x_var >= z_var, name=f"{name_prefix}_x_ge_z")
    model.addConstr(
        x_var <= slope * z_var + intercept,
        name=f"{name_prefix}_x_le_relaxation",
    )
    model.addConstr(x_var <= z_ub, name=f"{name_prefix}_x_le_ub")
    return None


def _add_relu_mip(model, z_var, x_var, z_lb, z_ub, name_prefix):
    """Add the exact big-M ReLU formulation with one binary variable."""
    if z_ub <= 0.0:
        model.addConstr(x_var == 0.0, name=f"{name_prefix}_inactive")
        return None
    if z_lb >= 0.0:
        model.addConstr(x_var == z_var, name=f"{name_prefix}_active")
        return None

    a = model.addVar(vtype="B", name=f"{name_prefix}_a")
    model.addConstr(x_var >= z_var, name=f"{name_prefix}_x_ge_z")
    model.addConstr(x_var >= 0.0, name=f"{name_prefix}_x_ge_0")
    model.addConstr(x_var <= z_var - z_lb * (1 - a), name=f"{name_prefix}_x_le_z")
    model.addConstr(x_var <= z_ub * a, name=f"{name_prefix}_x_le_ub")
    return a


def build_network_model(config: NetworkConfig, *, relaxation_only: bool = False):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - import guard for user envs
        raise RuntimeError("gurobipy is required to run this example") from exc

    model = gp.Model("example")
    model.Params.OutputFlag = 0

    # Inputs
    x01 = model.addVar(lb=config.input_lb[0], ub=config.input_ub[0], name="x01")
    x02 = model.addVar(lb=config.input_lb[1], ub=config.input_ub[1], name="x02")

    # Layer 1 pre-activation.
    hidden_vars = []
    hidden_bounds = []
    for idx, weights in enumerate(config.hidden_w, start=1):
        hidden_lb, hidden_ub = affine_box_bounds(weights, config.input_lb, config.input_ub)
        hidden_bounds.append((hidden_lb, hidden_ub))
        z_var = model.addVar(lb=hidden_lb, ub=hidden_ub, name=f"z1{idx}")
        x_var = model.addVar(lb=0.0, name=f"x1{idx}")
        model.addConstr(z_var == weights[0] * x01 + weights[1] * x02, name=f"z1{idx}_def")

        x_lb, x_ub = relu_interval_bounds(hidden_lb, hidden_ub)
        model.addConstr(x_var >= x_lb, name=f"x1{idx}_lb")
        model.addConstr(x_var <= x_ub, name=f"x1{idx}_ub")

        if relaxation_only:
            _add_relu_relaxation(model, z_var, x_var, hidden_lb, hidden_ub, f"relu1{idx}")
        else:
            _add_relu_mip(model, z_var, x_var, hidden_lb, hidden_ub, f"relu1{idx}")

        hidden_vars.append((z_var, x_var))

    # Layer 2 (linear)
    x21 = model.addVar(lb=-GRB.INFINITY, name="x21")
    model.addConstr(
        x21 == config.output_w[0] * hidden_vars[0][1] + config.output_w[1] * hidden_vars[1][1],
        name="x21_def",
    )

    return model, {
        "x01": x01,
        "x02": x02,
        "z11": hidden_vars[0][0],
        "z12": hidden_vars[1][0],
        "x11": hidden_vars[0][1],
        "x12": hidden_vars[1][1],
        "x21": x21,
    }


def exact_neuron_bounds(config: NetworkConfig) -> dict[str, tuple[float, float]]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - import guard for user envs
        raise RuntimeError("gurobipy is required to run this example") from exc

    model, vars_ = build_network_model(config, relaxation_only=False)
    bounds: dict[str, tuple[float, float]] = {}
    for name, var in vars_.items():
        model.setObjective(var, GRB.MINIMIZE)
        model.optimize()
        lower = None if model.Status != GRB.OPTIMAL else model.ObjVal

        model.setObjective(var, GRB.MAXIMIZE)
        model.optimize()
        upper = None if model.Status != GRB.OPTIMAL else model.ObjVal

        bounds[name] = (lower, upper)
    return bounds


def check_violation(config: NetworkConfig, eps=1e-6, relaxation_only: bool = False):
    try:
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - import guard for user envs
        raise RuntimeError("gurobipy is required to run this example") from exc

    model, vars_ = build_network_model(config, relaxation_only=relaxation_only)
    x21 = vars_["x21"]
    model.setObjective(x21, GRB.MINIMIZE)
    model.optimize()

    violation_reachable = model.Status == GRB.OPTIMAL and x21.X <= 3.0 + eps
    solution = None
    if violation_reachable:
        solution = {name: var.X for name, var in vars_.items()}
        solution["x21"] = x21.X
    return violation_reachable, solution, model.ObjVal if model.Status == GRB.OPTIMAL else None


def check_last_layer_repair(config: NetworkConfig, modifiable_range=1.0, eps=1e-6):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - import guard for user envs
        raise RuntimeError("gurobipy is required to run this example") from exc

    model = gp.Model("last_layer_repair")
    model.Params.OutputFlag = 0

    hidden_bounds = [
        relu_interval_bounds(*affine_box_bounds(weights, config.input_lb, config.input_ub))
        for weights in config.hidden_w
    ]

    # Repair variables for the last layer only.
    dw = [model.addVar(lb=-modifiable_range, ub=modifiable_range, name=f"dw_{j}") for j in range(2)]

    # Conservative lower bound of x21 under the repaired last layer.
    coeff0 = config.output_w[0] + dw[0]
    coeff1 = config.output_w[1] + dw[1]
    margin_lb = (
        coeff0 * (hidden_bounds[0][0] if config.output_w[0] >= 0 else hidden_bounds[0][1])
        + coeff1 * (hidden_bounds[1][0] if config.output_w[1] >= 0 else hidden_bounds[1][1])
    )

    slack = model.addVar(lb=0.0, name="slack")
    model.addConstr(slack >= 3.0 + eps - margin_lb, name="slack_def")
    model.setObjective(slack, GRB.MINIMIZE)
    model.optimize()

    slack_value = None if model.Status != GRB.OPTIMAL else slack.X
    repaired_output_w = None
    if model.Status == GRB.OPTIMAL:
        repaired_output_w = (
            config.output_w[0] + dw[0].X,
            config.output_w[1] + dw[1].X,
        )
    repaired_config = (
        replace(config, output_w=repaired_output_w)
        if repaired_output_w is not None
        else None
    )
    return model.Status == GRB.OPTIMAL and slack.X <= eps, slack_value, hidden_bounds, repaired_config


def iter_integer_bounds(min_value: int, max_value: int):
    for lower in range(min_value, max_value):
        for upper in range(lower + 1, max_value + 1):
            yield float(lower), float(upper)


def iter_integer_matrices(min_value: int, max_value: int):
    values = range(min_value, max_value + 1)
    for w11 in values:
        for w12 in values:
            for w21 in values:
                for w22 in values:
                    yield ((float(w11), float(w12)), (float(w21), float(w22)))


def search_integer_example(
    *,
    weight_min: int = -2,
    weight_max: int = 2,
    bound_min: int = 0,
    bound_max: int = 2,
    after_hidden_w_delta: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 1.0), (-1.0, -1.0)),
    output_w: tuple[float, float] = (1.0, -1.0),
    modifiable_range: float = 1.0,
    eps: float = 1e-6,
):
    for input_lb_pair in iter_integer_bounds(bound_min, bound_max):
        for input_ub_pair in iter_integer_bounds(bound_min, bound_max):
            if not all(lower < upper for lower, upper in zip(input_lb_pair, input_ub_pair)):
                continue

            for hidden_w in iter_integer_matrices(weight_min, weight_max):
                config = NetworkConfig(
                    input_lb=input_lb_pair,
                    input_ub=input_ub_pair,
                    hidden_w=hidden_w,
                    output_w=output_w,
                )
                violates, _, _ = check_violation(config, eps=eps)
                if not violates:
                    continue

                repair_ok, _, _, _ = check_last_layer_repair(
                    config,
                    modifiable_range=modifiable_range,
                    eps=eps,
                )
                if repair_ok:
                    continue

                after_config = replace(
                    config,
                    hidden_w=matrix2x2_add(config.hidden_w, after_hidden_w_delta),
                )
                after_violates, _, _ = check_violation(after_config, eps=eps)
                if after_violates:
                    return config, after_config

    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_lb", type=str, default="1,1")
    parser.add_argument("--input_ub", type=str, default="2,2")
    # parser.add_argument("--hidden_w", type=str, default="1,-1,-1,1")
    parser.add_argument("--hidden_w", type=str, default="2,0,-2,0")
    # parser.add_argument("--output_w", type=str, default="1,-1")
    parser.add_argument("--output_w", type=str, default="2,-1")
    parser.add_argument("--modifiable_range", type=float, default=1.0)
    # parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--search_integer_example", action="store_true")
    parser.add_argument("--search_weight_min", type=int, default=-2)
    parser.add_argument("--search_weight_max", type=int, default=2)
    parser.add_argument("--search_bound_min", type=int, default=0)
    parser.add_argument("--search_bound_max", type=int, default=2)
    parser.add_argument("--after_hidden_w_delta", type=str, default="1,1,-1,-1")
    cli = parser.parse_args()

    config = NetworkConfig(
        input_lb=tuple(_parse_csv_floats(cli.input_lb, 2)),
        input_ub=tuple(_parse_csv_floats(cli.input_ub, 2)),
        hidden_w=matrix2x2_from_csv(cli.hidden_w),
        output_w=vector2_from_csv(cli.output_w),
    )
    after_hidden_w_delta = matrix2x2_from_csv(cli.after_hidden_w_delta)

    if cli.search_integer_example:
        found_config, found_after_config = search_integer_example(
            weight_min=cli.search_weight_min,
            weight_max=cli.search_weight_max,
            bound_min=cli.search_bound_min,
            bound_max=cli.search_bound_max,
            after_hidden_w_delta=after_hidden_w_delta,
            output_w=config.output_w,
            modifiable_range=cli.modifiable_range,
            eps=cli.eps,
        )
        if found_config is None:
            print("No integer example found in the requested search range.")
            raise SystemExit(1)
        config = found_config
        print("Found integer example.")
        print(f"initial config: {config}")
        print(f"after config: {found_after_config}")
        print("hidden bounds before delta:")
        for idx, weights in enumerate(config.hidden_w, start=1):
            print(f"  x1{idx}: {relu_interval_bounds(*affine_box_bounds(weights, config.input_lb, config.input_ub))}")
        print("hidden bounds after delta:")
        for idx, weights in enumerate(found_after_config.hidden_w, start=1):
            print(f"  x1{idx}: {relu_interval_bounds(*affine_box_bounds(weights, found_after_config.input_lb, found_after_config.input_ub))}")

    hidden_bounds = [
        relu_interval_bounds(*affine_box_bounds(weights, config.input_lb, config.input_ub))
        for weights in config.hidden_w
    ]
    print(f"hidden bounds: {hidden_bounds}")
    print(f"config: {config}")

    try:
        before_bounds = exact_neuron_bounds(config)
        print("exact bounds before repair:")
        for name in ["x01", "x02", "z11", "z12", "x11", "x12", "x21"]:
            print(f"  {name}: {before_bounds[name]}")

        ok, sol, obj = check_violation(config, eps=cli.eps)
        print("violation reachable:", ok)
        print("exact min x21:", obj)
        if sol is not None:
            print(sol)

        repair_ok, slack, hidden_bounds, repaired_config = check_last_layer_repair(
            config,
            modifiable_range=cli.modifiable_range,
            eps=cli.eps,
        )
        print("last-layer repair completes:", repair_ok)
        print("repair slack:", slack)
        print("hidden bounds:", hidden_bounds)
        if repaired_config is not None:
            print_matrix2x2("hidden_w before repair", config.hidden_w)
            print_matrix2x2("hidden_w after repair", repaired_config.hidden_w)
            print("hidden_w delta:", matrix2x2_delta(config.hidden_w, repaired_config.hidden_w))
            print_vector2("output_w before repair", config.output_w)
            print_vector2("output_w after repair", repaired_config.output_w)
            print("output_w delta:", vector2_delta(config.output_w, repaired_config.output_w))
            after_bounds = exact_neuron_bounds(repaired_config)
            print("exact bounds after repair:")
            for name in ["x01", "x02", "z11", "z12", "x11", "x12", "x21"]:
                print(f"  {name}: {after_bounds[name]}")
    except RuntimeError as exc:
        print(exc)
