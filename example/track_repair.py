#!/usr/bin/env python3
"""Track the single-layer and multilayer repairs from ``design.md``.

The script intentionally prints every important bound and search decision so
that the numerical execution can be compared with ``repair_report.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog


Bounds = list[tuple[float, float]]


@dataclass
class Trace:
    """Write trace messages immediately to the terminal and an optional file."""

    output: Path | None = None

    def __post_init__(self) -> None:
        self._file = None
        if self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.output.open("w", encoding="utf-8", buffering=1)

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        if self._file is not None:
            print(message, file=self._file, flush=True)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def fmt_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}"


def fmt_bounds(bounds: Sequence[tuple[float, float]]) -> str:
    return " x ".join(f"[{fmt_number(lo)}, {fmt_number(hi)}]" for lo, hi in bounds)


def affine_bounds(weights: np.ndarray, inputs: Bounds) -> Bounds:
    """Sound interval propagation through a zero-bias affine layer."""

    result: Bounds = []
    for row in weights:
        lower = 0.0
        upper = 0.0
        for weight, (in_lower, in_upper) in zip(row, inputs):
            products = (weight * in_lower, weight * in_upper)
            lower += min(products)
            upper += max(products)
        result.append((lower, upper))
    return result


def relu_bounds(inputs: Bounds) -> Bounds:
    return [(max(0.0, lower), max(0.0, upper)) for lower, upper in inputs]


def propagate(
    inputs: Bounds, w1: np.ndarray, w2: np.ndarray
) -> tuple[Bounds, Bounds, tuple[float, float]]:
    preactivation = affine_bounds(w1, inputs)
    hidden = relu_bounds(preactivation)
    output = affine_bounds(w2.reshape(1, -1), hidden)[0]
    return preactivation, hidden, output


def input_box(radius_scale: float) -> Bounds:
    return [
        (5.5 - 2.5 * radius_scale, 5.5 + 2.5 * radius_scale),
        (4.0 - 4.0 * radius_scale, 4.0 + 4.0 * radius_scale),
    ]


def binary_search_safe_input(
    w1: np.ndarray,
    w2: np.ndarray,
    tolerance: float,
    trace: Trace,
) -> float:
    lower = 0.0
    upper = 1.0
    iteration = 0
    trace.log("Binary search for the largest safe input radius:")
    while upper - lower > tolerance:
        iteration += 1
        candidate = (lower + upper) / 2.0
        output = propagate(input_box(candidate), w1, w2)[2]
        safe = output[0] >= 3.0
        trace.log(
            f"  {iteration:02d}: r={candidate:.8f}, "
            f"output=[{output[0]:.8f}, {output[1]:.8f}], "
            f"{'safe' if safe else 'unsafe'}"
        )
        if safe:
            lower = candidate
        else:
            upper = candidate
    trace.log(f"  result: r in [{lower:.8f}, {upper:.8f}]")
    return lower


def scaled_intermediate_box(center: np.ndarray, radius: np.ndarray, scale: float) -> Bounds:
    return [
        (float(c - scale * r), float(c + scale * r))
        for c, r in zip(center, radius)
    ]


def intermediate_is_safe(box: Bounds, w2: np.ndarray) -> tuple[bool, tuple[float, float]]:
    output = affine_bounds(w2.reshape(1, -1), relu_bounds(box))[0]
    return output[0] >= 3.0, output


def search_intermediate_box(
    initial: Bounds,
    w2: np.ndarray,
    tolerance: float,
    trace: Trace,
) -> tuple[float, Bounds]:
    center = np.array([(lo + hi) / 2.0 for lo, hi in initial])
    radius = np.array([(hi - lo) / 2.0 for lo, hi in initial])
    safe, output = intermediate_is_safe(initial, w2)
    if not safe:
        raise RuntimeError(f"Initial intermediate box is unexpectedly unsafe: {output}")

    trace.log("Exponential search for the intermediate safe box:")
    accepted = 1.0
    trial = 2.0
    while True:
        trial_box = scaled_intermediate_box(center, radius, trial)
        safe, output = intermediate_is_safe(trial_box, w2)
        trace.log(
            f"  scale={trial:.6f}, box={fmt_bounds(trial_box)}, "
            f"output={fmt_bounds([output])}: {'safe' if safe else 'unsafe'}"
        )
        if not safe:
            rejected = trial
            break
        accepted = trial
        trial *= 2.0

    trace.log("Binary refinement of the intermediate radius:")
    iteration = 0
    while rejected - accepted > tolerance:
        iteration += 1
        candidate = (accepted + rejected) / 2.0
        candidate_box = scaled_intermediate_box(center, radius, candidate)
        safe, output = intermediate_is_safe(candidate_box, w2)
        trace.log(
            f"  {iteration:02d}: scale={candidate:.8f}, "
            f"lower(output)={output[0]:.8f}, {'safe' if safe else 'unsafe'}"
        )
        if safe:
            accepted = candidate
        else:
            rejected = candidate
    return accepted, scaled_intermediate_box(center, radius, accepted)


def repair_last_layer(
    hidden: Bounds,
    original: np.ndarray,
    modification_limit: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Maximize the sound output lower bound for nonnegative activations."""

    coefficient = np.array(
        [lower if weight >= 0.0 else upper for weight, (lower, upper) in zip(original, hidden)]
    )
    bounds = [
        (float(weight - modification_limit), float(weight + modification_limit))
        for weight in original
    ]
    result = linprog(-coefficient, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Last-layer optimization failed: {result.message}")
    repaired = result.x
    # Preserve irrelevant weights to make the reported optimum deterministic.
    repaired[np.isclose(coefficient, 0.0)] = original[np.isclose(coefficient, 0.0)]
    output = affine_bounds(repaired.reshape(1, -1), hidden)[0]
    return repaired, output


def repair_first_layer(target: Bounds) -> np.ndarray:
    """Solve the first-layer hinge-bound objective as a linear program."""

    # Variables: w11, w12, w21, w22, then lower/upper violation slacks per row.
    objective = np.array([0.0] * 4 + [1.0] * 4)
    constraints: list[list[float]] = []
    rhs: list[float] = []

    # For X=[3,8]x[0,8] and the fixed weight signs:
    # row 1 lower=3*w11+8*w12, upper=8*w11
    # row 2 lower=8*w21,        upper=3*w21+8*w22.
    actual_lower = [
        np.array([3.0, 8.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 8.0, 0.0]),
    ]
    actual_upper = [
        np.array([8.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 3.0, 8.0]),
    ]

    for row in range(2):
        lower_slack = 4 + 2 * row
        upper_slack = lower_slack + 1

        # target_lower - actual_lower <= lower_slack
        constraint = np.zeros(8)
        constraint[:4] = -actual_lower[row]
        constraint[lower_slack] = -1.0
        constraints.append(constraint.tolist())
        rhs.append(-target[row][0])

        # actual_upper - target_upper <= upper_slack
        constraint = np.zeros(8)
        constraint[:4] = actual_upper[row]
        constraint[upper_slack] = -1.0
        constraints.append(constraint.tolist())
        rhs.append(target[row][1])

    variable_bounds = [
        (1.5, 2.5),
        (-1.5, -0.5),
        (-2.5, -1.5),
        (0.5, 1.5),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
    ]
    result = linprog(
        objective,
        A_ub=np.asarray(constraints),
        b_ub=np.asarray(rhs),
        bounds=variable_bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"First-layer optimization failed: {result.message}")
    return result.x[:4].reshape(2, 2)


def log_network_bounds(trace: Trace, inputs: Bounds, w1: np.ndarray, w2: np.ndarray) -> None:
    preactivation, hidden, output = propagate(inputs, w1, w2)
    trace.log(f"  preactivation: {fmt_bounds(preactivation)}")
    trace.log(f"  post-ReLU:     {fmt_bounds(hidden)}")
    trace.log(f"  output:        {fmt_bounds([output])}")


def log_relu_states(trace: Trace, preactivation: Bounds) -> None:
    for index, (lower, upper) in enumerate(preactivation, start=1):
        if lower < 0.0 < upper:
            slope = upper / (upper - lower)
            trace.log(
                f"  ReLU {index}:       unstable; convex upper slope "
                f"u/(u-l)={slope:.8f}"
            )
        elif lower >= 0.0:
            trace.log(f"  ReLU {index}:       stable active")
        else:
            trace.log(f"  ReLU {index}:       stable inactive")


def assert_close(actual: Iterable[float], expected: Iterable[float], tolerance: float = 2e-5) -> None:
    if not np.allclose(list(actual), list(expected), atol=tolerance, rtol=0.0):
        raise AssertionError(f"Expected {list(expected)}, got {list(actual)}")


def run(tolerance: float, trace: Trace) -> None:
    inputs: Bounds = [(3.0, 8.0), (0.0, 8.0)]
    w1 = np.array([[2.0, -1.0], [-2.0, 1.0]])
    w2 = np.array([3.0, -3.0])

    trace.log("=== Original network ===")
    log_network_bounds(trace, inputs, w1, w2)
    original_pre, original_hidden, original_output = propagate(inputs, w1, w2)
    log_relu_states(trace, original_pre)
    assert_close(original_output, (-6.0, 48.0))
    trace.log("  verification: FAILURE (lower bound -6 < 3)\n")

    trace.log("=== Single-layer repair (last layer only) ===")
    single_w2, single_output = repair_last_layer(original_hidden, w2, 0.5)
    trace.log(f"  repaired W2:   {single_w2.tolist()}")
    trace.log(f"  output:        {fmt_bounds([single_output])}")
    trace.log("  repair result: FAILURE (best lower bound -5 < 3)\n")
    assert_close(single_w2, (3.0, -2.5))
    assert_close(single_output, (-5.0, 48.0))

    trace.log("=== Multilayer repair ===")
    searched_radius = binary_search_safe_input(w1, w2, tolerance, trace)
    exact_radius = 2.0 / 3.0
    if abs(searched_radius - exact_radius) > tolerance:
        raise AssertionError("Safe-input binary search did not converge to 2/3")
    # Use the exact analytical limit after demonstrating numerical convergence.
    safe_input = input_box(exact_radius)
    safe_pre, _, safe_output = propagate(safe_input, w1, w2)
    trace.log(f"  exact r*:      2/3 = {exact_radius:.8f}")
    trace.log(f"  safe input:    {fmt_bounds(safe_input)}")
    trace.log(f"  safe output:   {fmt_bounds([safe_output])}\n")
    assert_close([v for pair in safe_pre for v in pair], (1.0, 13.0, -13.0, -1.0))

    scale, intermediate = search_intermediate_box(safe_pre, w2, tolerance, trace)
    trace.log(f"  final scale:   {scale:.8f}")
    trace.log(f"  safe subspace: {fmt_bounds(intermediate)}\n")
    if abs(scale - 1.0) > tolerance:
        raise AssertionError("Intermediate search should return scale 1")

    # Use the exact safe target rather than the tolerance-perturbed search box.
    repaired_w1 = repair_first_layer(safe_pre)
    first_pre, first_hidden, first_output = propagate(inputs, repaired_w1, w2)
    trace.log("First-layer subrepair:")
    trace.log(f"  repaired W1:   {repaired_w1.tolist()}")
    trace.log(f"  preactivation: {fmt_bounds(first_pre)}")
    trace.log(f"  post-ReLU:     {fmt_bounds(first_hidden)}")
    trace.log(f"  output:        {fmt_bounds([first_output])}")
    trace.log("  status:        still unsafe (lower bound 21/8 < 3)\n")
    assert_close(repaired_w1.ravel(), (13.0 / 8.0, -0.5, -13.0 / 8.0, 0.5))
    assert_close(first_output, (21.0 / 8.0, 39.0))

    repaired_w2, final_output = repair_last_layer(first_hidden, w2, 0.5)
    trace.log("Last-layer subrepair:")
    trace.log(f"  repaired W2:   {repaired_w2.tolist()}")
    trace.log(f"  output:        {fmt_bounds([final_output])}")
    trace.log("  repair result: SUCCESS (lower bound 49/16 > 3)")
    assert_close(repaired_w2, (3.5, -3.0))
    assert_close(final_output, (49.0 / 16.0, 91.0 / 2.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="binary-search tolerance (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path that receives the same line-buffered trace as stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive")
    trace = Trace(args.output)
    try:
        run(args.tolerance, trace)
    finally:
        trace.close()


if __name__ == "__main__":
    main()
