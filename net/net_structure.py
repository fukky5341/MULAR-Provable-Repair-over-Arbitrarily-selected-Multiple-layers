"""Print network structure summaries for selected models.

Reports for each model:
- number of learnable layers
- layer types
- total number of neurons
- total number of parameters (weights and biases)

Output: latex table
- model name
    map = {
        "mnist_relu_9_100.tf": "\\mnistO",
        "mnist_256x4.onnx": "\\mnistT",
        "cifar10_cnn1.onnx": "\\cifarNet",
        "gtsrb_cnn.onnx": "\\gtsrbNet",}
- dataset
    map = {
        "mnist_relu_9_100.tf": "\\mnist",
        "mnist_256x4.onnx": "\\mnist",
        "cifar10_cnn1.onnx": "\\cifar",
        "gtsrb_cnn.onnx": "\\gtsrb",}
- number of layers
    - e.g., "5 (FC x3, Conv x2, ReLU x4)"
- total neurons
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelSummary:
    model_path: Path
    num_layers: int
    layer_types: list[str]
    total_neurons: int
    total_parameters: int


MODEL_NAME_MAP = {
    "mnist_relu_9_100.tf": r"\mnistO",
    "mnist_256x4.onnx": r"\mnistT",
    "cifar10_cnn1.onnx": r"\cifarNet",
    "gtsrb_cnn.onnx": r"\gtsrbNet",
}

DATASET_MAP = {
    "mnist_relu_9_100.tf": r"\mnist",
    "mnist_256x4.onnx": r"\mnist",
    "cifar10_cnn1.onnx": r"\cifar",
    "gtsrb_cnn.onnx": r"\gtsrb",
}

LAYER_LABEL_MAP = {
    "Linear": "FC",
    "Gemm": "FC",
    "MatMul": "FC",
    "Conv": "Conv",
    "Relu": "ReLU",
    "ReLU": "ReLU",
}


def _layer_breakdown_text(layer_types: list[str]) -> str:
    type_counter = Counter(LAYER_LABEL_MAP.get(layer_type, layer_type) for layer_type in layer_types)
    ordered_keys = ["FC", "Conv", "ReLU"]
    parts: list[str] = []
    for key in ordered_keys:
        if key in type_counter:
            parts.append(f"{key} x{type_counter[key]}")
            del type_counter[key]
    for key in sorted(type_counter):
        parts.append(f"{key} x{type_counter[key]}")
    return ", ".join(parts)


def _to_latex_row(summary: ModelSummary) -> str:
    model_name = MODEL_NAME_MAP.get(summary.model_path.name, summary.model_path.name)
    dataset = DATASET_MAP.get(summary.model_path.name, "-")
    layer_info = f"{summary.num_layers} ({_layer_breakdown_text(summary.layer_types)})"
    return (
        f"{model_name} & {dataset} & {layer_info} & "
        f"{summary.total_neurons} & {summary.total_parameters} \\\\"
    )


def _render_latex_table(summaries: list[ModelSummary]) -> str:
    lines = [
        r"\begin{tabular}{l l l r r}",
        r"\toprule",
        r"Model & Dataset & Layers & Total neurons & Total parameters \\",
        r"\midrule",
    ]
    lines.extend(_to_latex_row(summary) for summary in summaries)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def summarize_tf_model(model_path: Path) -> ModelSummary:
    lines = [line.strip() for line in model_path.read_text().splitlines() if line.strip()]
    layer_types: list[str] = []
    total_neurons = 0
    total_parameters = 0

    i = 0
    while i + 2 < len(lines):
        activation_name = lines[i]
        weights = ast.literal_eval(lines[i + 1])
        biases = ast.literal_eval(lines[i + 2])

        out_features = len(weights)
        if out_features != len(biases):
            raise ValueError(
                f"Malformed tf model block at line {i + 1}: "
                f"weight rows ({out_features}) != bias size ({len(biases)})."
            )

        layer_types.append("Linear")
        total_neurons += out_features
        total_parameters += sum(len(row) for row in weights) + len(biases)

        i += 3

    if i != len(lines):
        raise ValueError(
            f"Malformed tf model format in {model_path}: remaining lines after parsing."
        )

    return ModelSummary(
        model_path=model_path,
        num_layers=len(layer_types),
        layer_types=layer_types,
        total_neurons=total_neurons,
        total_parameters=total_parameters,
    )


def _onnx_value_shapes(onnx_model: Any) -> dict[str, tuple[int | None, ...]]:
    def _extract_shape(value_info: Any) -> tuple[int | None, ...]:
        tensor_type = value_info.type.tensor_type
        dims: list[int | None] = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        return tuple(dims)

    shapes: dict[str, tuple[int | None, ...]] = {}

    all_value_infos = (
        list(onnx_model.graph.value_info)
        + list(onnx_model.graph.input)
        + list(onnx_model.graph.output)
    )

    for value_info in all_value_infos:
        shapes[value_info.name] = _extract_shape(value_info)

    return shapes


def _neuron_count_from_onnx_output_shape(shape: tuple[int | None, ...]) -> int:
    # Ignore batch dimension and unknown dimensions.
    if len(shape) == 0:
        return 0
    dims = shape[1:] if len(shape) > 1 else shape
    product = 1
    known = False
    for d in dims:
        if d is None:
            continue
        known = True
        product *= d
    return product if known else 0


def summarize_onnx_model(model_path: Path) -> ModelSummary:
    try:
        import onnx
        from onnx import shape_inference
    except ImportError as exc:
        raise ImportError(
            "Failed to import onnx. Run this script in your project environment."
        ) from exc

    model = onnx.load(model_path.as_posix())
    inferred = shape_inference.infer_shapes(model)
    value_shapes = _onnx_value_shapes(inferred)

    layer_types: list[str] = []
    total_neurons = 0
    parameter_names: set[str] = set()
    initializer_sizes = {
        initializer.name: int(math.prod(initializer.dims))
        for initializer in inferred.graph.initializer
    }

    neuron_ops = {"Gemm", "MatMul", "Conv"}

    for node in inferred.graph.node:
        if node.op_type not in neuron_ops:
            continue

        layer_types.append(node.op_type)
        parameter_names.update(
            name for name in node.input if name in initializer_sizes
        )

        # Use first output for per-layer neuron estimate.
        if len(node.output) == 0:
            continue
        out_name = node.output[0]
        shape = value_shapes.get(out_name)
        if shape is None:
            continue
        total_neurons += _neuron_count_from_onnx_output_shape(shape)

    return ModelSummary(
        model_path=model_path,
        num_layers=len(layer_types),
        layer_types=layer_types,
        total_neurons=total_neurons,
        total_parameters=sum(initializer_sizes[name] for name in parameter_names),
    )


def summarize_model(model_path: Path) -> ModelSummary:
    suffix = model_path.suffix.lower()
    if suffix == ".tf":
        return summarize_tf_model(model_path)
    if suffix == ".onnx":
        return summarize_onnx_model(model_path)
    raise ValueError(f"Unsupported model format: {model_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    model_paths = [
        base_dir.parent / "models" / "mnist" / "mnist_relu_9_100.tf",
        base_dir / "onnx" / "mnist_256x4.onnx",
        base_dir / "onnx" / "cifar10_cnn1.onnx",
        base_dir / "onnx" / "gtsrb_cnn.onnx",
    ]

    summaries: list[ModelSummary] = []
    for model_path in model_paths:
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            print()
            continue

        try:
            summary = summarize_model(model_path)
        except Exception as exc:
            print(f"Model: {model_path}")
            print(f"  Failed to summarize: {exc}")
            print()
            continue

        summaries.append(summary)

    if not summaries:
        print("No model summaries available.")
        return

    print(_render_latex_table(summaries))


if __name__ == "__main__":
    main()
