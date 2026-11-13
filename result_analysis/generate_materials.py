"""Generate exactly the paper materials listed in README.md from original_result.

Run from the repository root:

    uv run python result_analysis/generate_materials.py

Use ``--original`` to reproduce the same artifact set in
``result_analysis/original_materials`` instead of ``result_analysis/materials``.

By default this first rebuilds the canonical CSV files from both the legacy and
current result layouts. Pass ``--no-refresh`` to reuse existing CSV files. The
output directory is replaced on each run so stale or renamed artifacts cannot
remain alongside the paper materials.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "provable-repair-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

try:
    from .analysis import load_all_csv_rows
    from .collect_results import collect_results
    from .policy_merge import (
        POLICY_STAGES,
        collect_policy_stage_rows,
        replace_layerwise_comparison_rows,
    )
except ImportError:  # Direct script execution.
    from analysis import load_all_csv_rows
    from collect_results import collect_results
    from policy_merge import (
        POLICY_STAGES,
        collect_policy_stage_rows,
        replace_layerwise_comparison_rows,
    )


POLICY_LAYER_SETS = {
    model: [stage.label for stage in stages]
    for model, stages in POLICY_STAGES.items()
}
COMPARISON_APPROACHES = {"aprnn", "prorepair", "prepared", "lastlayer", "ours"}
APPROACH_ORDER = ["aprnn", "prorepair", "prepared", "lastlayer", "ours"]
APPROACH_COLOR_INDICES = {
    "aprnn": 0,
    "prorepair": 3,
    "prepared": 2,
    "lastlayer": 1,
    "ours": 4,
}
APPROACH_NAMES = {
    "aprnn": "APRNN",
    "prorepair": "ProRepair",
    "prepared": "PREPARED",
    "lastlayer": "LLR",
    "ours": "MULAR",
}
TEX_APPROACH_NAMES = {
    "aprnn": r"\aprnn",
    "prorepair": r"\prorepair",
    "prepared": r"\prepared",
    "lastlayer": r"\lastlayerTool",
    "ours": r"\tool",
}
RQ5_DISPLAY_NAMES = {
    "pareto10": r"MULAR$_{\mathrm{Arachne}}$",
    "pareto10_random": r"MULAR$_{\mathrm{Random}}$",
}
NETWORK_NAMES = {
    "mnist 9x100": "MNIST 9x100",
    "mnist_256x4": "MNIST 256x4",
    "cifar10_cnn1": "CIFAR-10 CNN1",
    "gtsrb_cnn": "GTSRB CNN",
}
NETWORK_DATASETS = {
    "mnist 9x100": "MNIST",
    "mnist_256x4": "MNIST",
    "cifar10_cnn1": "CIFAR-10",
    "gtsrb_cnn": "GTSRB",
}
PAPER_NETWORK_SLUGS = {
    "mnist 9x100": "mnistf1",
    "mnist_256x4": "mnistf2",
    "cifar10_cnn1": "cifar",
    "gtsrb_cnn": "gtsrb",
}
PAPER_MATERIAL_PATHS = {
    "rq1/success_count_nonzero_mnistf1.csv",
    "rq1/success_count_nonzero_mnistf2.csv",
    "rq1/success_count_all_mnistf1.csv",
    "rq1/success_count_all_mnistf2.csv",
    "rq1/success_count_all_cifar.csv",
    "rq1/success_count_all_gtsrb.csv",
    "rq1/prepared_outcomes_cifar.csv",
    "rq1/prepared_outcomes_gtsrb.csv",
    "rq1/pairwise_success_count_mnistf1.csv",
    "rq2/repair_acc_drop_nonzero_mnistf1.png",
    "rq2/repair_acc_drop_nonzero_mnistf2.png",
    "rq2/repair_acc_drop_all_mnistf1.png",
    "rq2/repair_acc_drop_all_mnistf2.png",
    "rq2/repair_acc_drop_all_cifar.png",
    "rq2/repair_acc_drop_all_gtsrb.png",
    "rq2/pairwise_repair_acc_drop_all_mular_vs_prepared_mnistf1.png",
    "rq2/pairwise_repair_acc_drop_all_mular_vs_prepared_mnistf2.png",
    "rq3/repair_time_nonzero_mnistf1.png",
    "rq3/repair_time_nonzero_mnistf2.png",
    "rq3/repair_time_all_mnistf1.png",
    "rq3/repair_time_all_mnistf2.png",
    "rq3/repair_time_all_cifar.png",
    "rq3/repair_time_all_gtsrb.png",
    "rq4/pairwise_success_count_diff_layers_mnistf1.csv",
    "rq4/pairwise_success_count_diff_layers_mnistf2.csv",
    "rq4/pairwise_success_count_diff_layers_cifar.csv",
    "rq4/pairwise_success_count_diff_layers_gtsrb.csv",
    "rq4/pairwise_repair_acc_drop_diff_layers_mnistf1.csv",
    "rq4/pairwise_repair_acc_drop_diff_layers_mnistf2.csv",
    "rq4/pairwise_repair_acc_drop_diff_layers_cifar.csv",
    "rq4/pairwise_repair_acc_drop_diff_layers_gtsrb.csv",
    "rq5/success_count_fault_localization_mnistf1.csv",
    "rq5/success_count_fault_localization_cifar.csv",
    "rq5/repair_acc_drop_fault_localization_mnistf1.png",
    "rq5/repair_acc_drop_fault_localization_cifar.png",
    "rq5/repair_time_fault_localization_mnistf1.png",
    "rq5/repair_time_fault_localization_cifar.png",
}
FIGURE_LABEL_FONTSIZE = 14
FIGURE_TICK_FONTSIZE = 14
FIGURE_LEGEND_FONTSIZE = 14
FIGURE_ANNOTATION_FONTSIZE = 14
RQ4_LAYER_SETS = {
    "layers_12_14_16",
    "layers_10_12_14",
    "layers_6_10_14",
    "layers_8_10_12",
}
PREPARED_OUTCOME_NAMES = {
    "repair_failed": "Repair failed",
    "gurobi_time_limit": "Gurobi time limit",
    "time_limit": "Time limit",
    "memory_limit": "Memory limit",
    "interrupted": "Interrupted",
    "exception": "Exception",
    "log_ended_before_result": "Log ended before result",
}


def _layer_set_name(label: str) -> str:
    layers = label.removeprefix("layers_").split("_")
    return "[" + ", ".join(layers) + "]"


def _layer_count_name(label: str) -> str:
    count = len(label.removeprefix("layers_").split("_"))
    return f"{count} layer" if count == 1 else f"{count} layers"


def _series(row: pd.Series) -> str:
    approach = str(row.get("approach", "unknown"))
    if approach == "ours" and pd.notna(row.get("layer_set_label")):
        return str(row["layer_set_label"])
    if approach == "FL":
        return str(row.get("variant", "FL"))
    return approach


def _prepare_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if rows.empty:
        return rows
    for column in [
        "buggy_points_num",
        "ndims",
        "total_repair_time_sec",
        "original_accuracy",
        "repaired_accuracy",
        "accuracy_drop",
    ]:
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    if "accuracy_drop" not in rows.columns:
        rows["accuracy_drop"] = rows["original_accuracy"] - rows["repaired_accuracy"]
    else:
        calculated = rows["original_accuracy"] - rows["repaired_accuracy"]
        rows["accuracy_drop"] = rows["accuracy_drop"].fillna(calculated)
    rows["evaluation_series"] = rows.apply(_series, axis=1)
    return rows


def summarize(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = keys + [
        "total_runs",
        "successes",
        "failures",
        "incomplete",
        "success_rate",
        "mean_runtime_sec_success",
        "median_runtime_sec_success",
        "mean_repaired_accuracy_success",
        "mean_accuracy_drop_success",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict] = []
    for values, group in rows.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        successful = group[group["result"] == "success"]
        total = len(group)
        records.append(
            {
                **dict(zip(keys, values)),
                "total_runs": total,
                "successes": int((group["result"] == "success").sum()),
                "failures": int((group["result"] == "failed").sum()),
                "incomplete": int((group["result"] == "incomplete").sum()),
                "success_rate": float((group["result"] == "success").sum() / total),
                "mean_runtime_sec_success": successful["total_repair_time_sec"].mean(),
                "median_runtime_sec_success": successful["total_repair_time_sec"].median(),
                "mean_repaired_accuracy_success": successful["repaired_accuracy"].mean(),
                "mean_accuracy_drop_success": successful["accuracy_drop"].mean(),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _tex_approach_labels(frame: pd.DataFrame) -> pd.DataFrame:
    replacements = {
        **TEX_APPROACH_NAMES,
        **RQ5_DISPLAY_NAMES,
        **{
            APPROACH_NAMES[approach]: tex_name
            for approach, tex_name in TEX_APPROACH_NAMES.items()
        },
    }
    return frame.replace(replacements).rename(columns=replacements)


def _write_table(frame: pd.DataFrame, directory: Path, name: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{name}.csv"
    tex_path = directory / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        _tex_approach_labels(frame).to_latex(
            index=False,
            float_format=lambda value: f"{value:.4g}",
            na_rep="--",
            escape=False,
        ),
        encoding="utf-8",
    )
    return csv_path, tex_path


def success_count_matrix(
    summary: pd.DataFrame,
    x_column: str,
    *,
    row_column: str = "approach",
    row_order: list[str] | None = None,
    row_names: dict[str, str] | None = None,
    left_header: str = "Approach",
) -> pd.DataFrame:
    """Pivot a single-network summary into approach rows and setting columns."""

    row_order = APPROACH_ORDER if row_order is None else row_order
    row_names = APPROACH_NAMES if row_names is None else row_names
    if summary.empty:
        return pd.DataFrame({left_header: [row_names.get(name, name) for name in row_order]})
    values = sorted(summary[x_column].dropna().unique(), key=float)
    pivot = summary.pivot_table(
        index=row_column,
        columns=x_column,
        values="successes",
        aggfunc="sum",
    ).reindex(index=row_order, columns=values)
    matrix = pivot.map(lambda value: "0" if pd.isna(value) else str(int(value)))
    matrix.index = [row_names.get(name, name) for name in matrix.index]
    matrix.index.name = left_header
    matrix.columns = [int(value) if float(value).is_integer() else value for value in matrix.columns]
    return matrix.reset_index()


def _write_network_success_tables(
    summary: pd.DataFrame,
    directory: Path,
    *,
    rq: str,
    x_column: str,
    x_symbol: str | None,
    row_column: str = "approach",
    row_order: list[str] | None = None,
    row_order_by_model: dict[str, list[str]] | None = None,
    row_names: dict[str, str] | None = None,
    left_header: str = "Approach",
) -> list[Path]:
    """Write one publication-style success-count table for every network."""

    written: list[Path] = []
    directory.mkdir(parents=True, exist_ok=True)
    for model, model_summary in summary.groupby("model_name", sort=True):
        model_row_order = row_order_by_model.get(str(model)) if row_order_by_model else row_order
        matrix = success_count_matrix(
            model_summary,
            x_column,
            row_column=row_column,
            row_order=model_row_order,
            row_names=row_names,
            left_header=left_header,
        )
        slug = str(model).lower().replace(" ", "_").replace("/", "_")
        csv_path = directory / f"{rq}_success_counts_{slug}.csv"
        tex_path = directory / f"{rq}_success_counts_{slug}.tex"
        matrix.to_csv(csv_path, index=False)
        tex_matrix = _tex_approach_labels(matrix.rename(
            columns={
                column: rf"${x_symbol}={column}$" if x_symbol else str(column)
                for column in matrix.columns
                if column != left_header
            }
        ))
        display_model = NETWORK_NAMES.get(str(model), str(model).replace("_", r"\_"))
        tex_path.write_text(
            tex_matrix.to_latex(
                index=False,
                escape=False,
                column_format="l" + "c" * (len(tex_matrix.columns) - 1),
                caption=f"Number of successful repair instances for {display_model}.",
                label=f"tab:{rq}-success-{slug.replace('_', '-')}",
            ),
            encoding="utf-8",
        )
        written.extend([csv_path, tex_path])
    return written


def prepared_outcome_matrix(model_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize PREPARED completion and failure modes by number of polytopes."""

    prepared = model_rows[model_rows["approach"] == "prepared"].copy()
    values = sorted(prepared["buggy_points_num"].dropna().unique(), key=float)
    records: list[dict[str, object]] = []
    for _, row in prepared.iterrows():
        if row["result"] == "success":
            outcome = "Successful"
        else:
            reason = row.get("failure_reason")
            outcome = PREPARED_OUTCOME_NAMES.get(
                str(reason),
                "Other unsuccessful",
            )
        records.append(
            {
                "buggy_points_num": row["buggy_points_num"],
                "outcome": outcome,
            }
        )
    counts = pd.DataFrame(records)
    present_failure_rows = [
        label
        for label in PREPARED_OUTCOME_NAMES.values()
        if not counts.empty and (counts["outcome"] == label).any()
    ]
    if not counts.empty and (counts["outcome"] == "Other unsuccessful").any():
        present_failure_rows.append("Other unsuccessful")
    row_order = ["Successful", *present_failure_rows]
    if counts.empty:
        matrix = pd.DataFrame(index=row_order, columns=values).fillna(0)
    else:
        matrix = (
            counts.groupby(["outcome", "buggy_points_num"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=row_order, columns=values, fill_value=0)
        )
    matrix.loc["Total unsuccessful"] = [
        int(
            (
                (prepared["buggy_points_num"] == value)
                & (prepared["result"] != "success")
            ).sum()
        )
        for value in values
    ]
    matrix.index.name = "Outcome"
    matrix.columns = [
        int(value) if float(value).is_integer() else value for value in matrix.columns
    ]
    return matrix.astype(int).reset_index()


def _write_rq2_prepared_outcome_tables(
    rows: pd.DataFrame,
    directory: Path,
) -> list[Path]:
    """Write one PREPARED outcome/failure table for each RQ2 network."""

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    prepared = rows[rows["approach"] == "prepared"]
    for model, model_rows in prepared.groupby("model_name", sort=True):
        matrix = prepared_outcome_matrix(model_rows)
        slug = str(model).lower().replace(" ", "_").replace("/", "_")
        csv_path = directory / f"rq2_prepared_outcomes_{slug}.csv"
        tex_path = directory / f"rq2_prepared_outcomes_{slug}.tex"
        matrix.to_csv(csv_path, index=False)
        tex_matrix = matrix.rename(columns={"Outcome": r"\numBox"})
        display_model = NETWORK_NAMES.get(str(model), str(model).replace("_", r"\_"))
        tex_path.write_text(
            tex_matrix.to_latex(
                index=False,
                escape=False,
                column_format="l" + "c" * (len(tex_matrix.columns) - 1),
                caption=(
                    f"Repair outcomes and failure modes of \\prepared for "
                    f"{display_model}."
                ),
                label=f"tab:rq2-prepared-outcomes-{slug.replace('_', '-')}",
            ),
            encoding="utf-8",
        )
        written.extend([csv_path, tex_path])
    return written


def _display_series_name(value: str) -> str:
    if value in APPROACH_NAMES:
        return APPROACH_NAMES[value]
    if value.startswith("layers_"):
        return _layer_set_name(value)
    return RQ5_DISPLAY_NAMES.get(value, value)


def _series_sort_key(value: str) -> tuple[int, int | str]:
    if value in APPROACH_ORDER:
        return 0, APPROACH_ORDER.index(value)
    if value.startswith("layers_"):
        numbers = [int(item) for item in value.removeprefix("layers_").split("_")]
        return 1, len(numbers) * 1000 + numbers[0]
    return 2, value


def _series_color_index(value: str, fallback_index: int) -> int:
    return APPROACH_COLOR_INDICES.get(value, fallback_index)


def _accuracy_legend_kwargs(prefix: str, model: str) -> dict[str, object]:
    """Return the requested per-RQ legend placement for accuracy plots."""

    if prefix == "rq1":
        return {"loc": "lower right"}
    if prefix == "rq2":
        if model == "cifar10_cnn1":
            return {"loc": "lower right"}
        if model in {"mnist 9x100", "mnist_256x4"}:
            return {"loc": "lower left"}
        return {"loc": "upper right"}
    if prefix == "rq5" and model.startswith("cifar"):
        return {"loc": "upper right"}
    if prefix == "rq5" and model.startswith("mnist"):
        return {"loc": "lower left"}
    return {"loc": "upper left", "bbox_to_anchor": (1.01, 1.0)}


def _accuracy_x_label(x: str) -> str:
    return "#Dim" if x == "ndims" else "#P"


def _accuracy_legend_fontsize(prefix: str) -> int:
    if prefix == "rq5":
        return 18
    if prefix == "rq2":
        return FIGURE_LEGEND_FONTSIZE + 2
    return FIGURE_LEGEND_FONTSIZE


def _accuracy_plot_fontsize(prefix: str) -> int:
    """Return the axis, tick, and annotation size for accuracy figures."""

    if prefix == "rq5":
        return 18
    if prefix == "rq2":
        return FIGURE_LABEL_FONTSIZE + 2
    return FIGURE_LABEL_FONTSIZE


def _pairwise_accuracy_fontsize(prefix: str) -> int:
    """Return text size for diagonal pairwise accuracy-drop figures."""

    return 18 if prefix in {"rq2", "rq5"} else FIGURE_LABEL_FONTSIZE


def _accuracy_drop_y_label(prefix: str) -> str:
    if prefix in {"rq1", "rq2", "rq5"}:
        return "Acc. (original - repaired)"
    return "Accuracy degradation (original - repaired)"


def _accuracy_category_spacing(x: str) -> float:
    return 0.82 if x == "ndims" else 0.70


def _runtime_legend_kwargs(prefix: str, model: str = "") -> dict[str, object]:
    if prefix == "rq1":
        return {"loc": "upper right"}
    if prefix == "rq2":
        if model == "cifar10_cnn1":
            return {"loc": "upper right"}
        return {"loc": "upper left"}
    if prefix in {"rq3", "rq4", "rq5"}:
        return {"loc": "upper left"}
    return {"loc": "upper left", "bbox_to_anchor": (1.01, 1.0)}


def _runtime_legend_fontsize(prefix: str) -> int:
    if prefix == "rq1":
        return FIGURE_LEGEND_FONTSIZE + 4
    if prefix == "rq2":
        return 22
    if prefix == "rq5":
        return 20
    return FIGURE_LEGEND_FONTSIZE


def _runtime_axis_label_fontsize(prefix: str) -> int:
    """Return the configured axis-label size for runtime figures."""

    if prefix == "rq1":
        return FIGURE_LABEL_FONTSIZE + 4
    if prefix in {"rq2", "rq5"}:
        return 22
    return FIGURE_LABEL_FONTSIZE


def _runtime_tick_fontsize(prefix: str) -> int:
    if prefix == "rq1":
        return FIGURE_TICK_FONTSIZE + 4
    return 22 if prefix in {"rq2", "rq5"} else FIGURE_TICK_FONTSIZE


def _runtime_y_label(prefix: str) -> str:
    return (
        "Mean Time"
        if prefix in {"rq1", "rq2", "rq5"}
        else "Mean repair time (seconds)"
    )


def _runtime_y_label_fontsize(prefix: str) -> int:
    return _runtime_axis_label_fontsize(prefix)


def _runtime_series_name(prefix: str, series: str) -> str:
    if prefix == "rq3" and series.startswith("layers_"):
        return _layer_count_name(series)
    return _display_series_name(series)


def _write_accuracy_boxplots(
    rows: pd.DataFrame,
    directory: Path,
    *,
    x: str,
    prefix: str,
    output_names: dict[str, str] | None = None,
) -> list[Path]:
    """Write one grouped accuracy-degradation box plot per network."""

    successful = rows[
        (rows["result"] == "success")
        & rows["accuracy_drop"].notna()
        & rows[x].notna()
        & rows["evaluation_series"].notna()
    ].copy()
    if successful.empty:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    colors = list(plt.get_cmap("tab10").colors)
    for model, model_rows in successful.groupby("model_name", sort=True):
        x_values = sorted(model_rows[x].unique(), key=float)
        series_values = sorted(
            model_rows["evaluation_series"].astype(str).unique(),
            key=_series_sort_key,
        )
        group_width = 0.82
        category_spacing = _accuracy_category_spacing(x)
        series_spacing = group_width / max(len(series_values), 1) * 0.72
        box_width = group_width / max(len(series_values), 1) * 0.60
        fig_width = max(6.5, 0.65 * len(x_values) + 3.0)
        fig, axis = plt.subplots(figsize=(fig_width, 4.8))
        legend_handles: list[Patch] = []
        for series_index, series in enumerate(series_values):
            color_index = _series_color_index(series, series_index)
            color = colors[color_index % len(colors)]
            data: list[list[float]] = []
            positions: list[float] = []
            for x_index, x_value in enumerate(x_values):
                values = model_rows[
                    (model_rows["evaluation_series"].astype(str) == series)
                    & (model_rows[x] == x_value)
                ]["accuracy_drop"].astype(float)
                if values.empty:
                    continue
                offset = (
                    series_index - (len(series_values) - 1) / 2
                ) * series_spacing
                data.append(values.tolist())
                positions.append(x_index * category_spacing + offset)
            if not data:
                continue
            artists = axis.boxplot(
                data,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                showmeans=True,
                medianprops={"color": "black", "linewidth": 1.2},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": color,
                    "markeredgecolor": "black",
                    "markeredgewidth": 0.9,
                    "markersize": 4.5,
                },
            )
            for box in artists["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.4)
            legend_handles.append(Patch(facecolor=color, alpha=0.4, label=_display_series_name(series)))
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="gray",
                markeredgecolor="black",
                label="Mean",
                markersize=5,
            )
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        category_positions = [
            x_index * category_spacing for x_index in range(len(x_values))
        ]
        axis.set_xticks(category_positions)
        axis.set_xticklabels(
            [int(value) if float(value).is_integer() else value for value in x_values]
        )
        cluster_half_width = (
            (len(series_values) - 1) / 2 * series_spacing + box_width / 2
        )
        outer_padding = max(0.04, box_width * 0.25)
        axis.set_xlim(
            -cluster_half_width - outer_padding,
            category_positions[-1] + cluster_half_width + outer_padding,
        )
        accuracy_fontsize = _accuracy_plot_fontsize(prefix)
        axis.set_xlabel(_accuracy_x_label(x), fontsize=accuracy_fontsize)
        axis.set_ylabel(
            _accuracy_drop_y_label(prefix),
            fontsize=accuracy_fontsize,
        )
        axis.tick_params(axis="both", labelsize=accuracy_fontsize)
        axis.grid(axis="y", alpha=0.25)
        axis.invert_yaxis()
        axis.legend(
            handles=legend_handles,
            fontsize=_accuracy_legend_fontsize(prefix),
            **_accuracy_legend_kwargs(prefix, str(model)),
        )
        fig.tight_layout()
        slug = str(model).lower().replace(" ", "_").replace("/", "_")
        filename = (
            output_names.get(str(model))
            if output_names is not None
            else None
        )
        output = directory / (filename or f"{prefix}_accuracy_drop_{slug}.png")
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(output)
    legacy_combined = directory / f"{prefix}_accuracy_drop.png"
    if legacy_combined.exists():
        legacy_combined.unlink()
    return written


def _write_runtime_figures(
    summary: pd.DataFrame,
    directory: Path,
    *,
    x: str,
    prefix: str,
    style_prefix: str | None = None,
    output_names: dict[str, str] | None = None,
) -> list[Path]:
    """Write one successful-repair runtime figure per network."""

    available = summary[
        summary["mean_runtime_sec_success"].notna()
        & summary[x].notna()
        & summary["evaluation_series"].notna()
    ].copy()
    if available.empty:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    colors = list(plt.get_cmap("tab10").colors)
    style_prefix = prefix if style_prefix is None else style_prefix
    for model, model_rows in available.groupby("model_name", sort=True):
        fig, axis = plt.subplots(figsize=(7.0, 4.8))
        series_values = sorted(
            model_rows["evaluation_series"].astype(str).unique(),
            key=_series_sort_key,
        )
        for series_index, series in enumerate(series_values):
            plotted = model_rows[
                model_rows["evaluation_series"].astype(str) == series
            ].sort_values(x)
            color = colors[_series_color_index(series, series_index) % len(colors)]
            axis.plot(
                plotted[x],
                plotted["mean_runtime_sec_success"],
                marker="o",
                color=color,
                alpha=0.55,
                linewidth=2.0,
                label=_runtime_series_name(prefix, series),
            )
        x_values = sorted(model_rows[x].dropna().unique(), key=float)
        axis.set_xticks(x_values)
        axis.set_xticklabels(
            [int(value) if float(value).is_integer() else value for value in x_values]
        )
        axis_label_fontsize = _runtime_axis_label_fontsize(style_prefix)
        axis.set_xlabel(_accuracy_x_label(x), fontsize=axis_label_fontsize)
        axis.set_ylabel(
            _runtime_y_label(style_prefix),
            fontsize=_runtime_y_label_fontsize(style_prefix),
        )
        axis.tick_params(
            axis="both", labelsize=_runtime_tick_fontsize(style_prefix)
        )
        axis.grid(alpha=0.25)
        axis.legend(
            fontsize=_runtime_legend_fontsize(style_prefix),
            **_runtime_legend_kwargs(style_prefix, str(model)),
        )
        fig.tight_layout()
        slug = str(model).lower().replace(" ", "_").replace("/", "_")
        filename = (
            output_names.get(str(model))
            if output_names is not None
            else None
        )
        output = directory / (filename or f"{prefix}_runtime_{slug}.png")
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(output)
    legacy_combined = directory / f"{prefix}_runtime.png"
    if legacy_combined.exists():
        legacy_combined.unlink()
    return written


def _write_figures(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    directory: Path,
    x: str,
    prefix: str,
) -> list[Path]:
    outputs = _write_accuracy_boxplots(rows, directory, x=x, prefix=prefix)
    outputs.extend(_write_runtime_figures(summary, directory, x=x, prefix=prefix))
    return outputs


def pairwise_accuracy(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare configurations on matched N/run pairs where both succeeded."""

    output: list[dict] = []
    for model, model_rows in rows.groupby("model_name", dropna=False):
        labels = sorted(model_rows["evaluation_series"].dropna().unique())
        for left, right in itertools.combinations(labels, 2):
            key = ["inp_eps", "buggy_points_num", "run_index"]
            left_rows = model_rows[model_rows["evaluation_series"] == left]
            right_rows = model_rows[model_rows["evaluation_series"] == right]
            if (
                "repair_data_ids" in model_rows.columns
                and left_rows["repair_data_ids"].notna().all()
                and right_rows["repair_data_ids"].notna().all()
            ):
                key.append("repair_data_ids")
            left_rows = left_rows[key + ["result", "repaired_accuracy"]].rename(
                columns={"result": "left_result", "repaired_accuracy": "left_accuracy"}
            )
            right_rows = right_rows[key + ["result", "repaired_accuracy"]].rename(
                columns={"result": "right_result", "repaired_accuracy": "right_accuracy"}
            )
            matched = left_rows.merge(right_rows, on=key, how="inner")
            valid = matched[
                (matched["left_result"] == "success")
                & (matched["right_result"] == "success")
                & matched["left_accuracy"].notna()
                & matched["right_accuracy"].notna()
            ]
            delta = valid["left_accuracy"] - valid["right_accuracy"]
            output.append(
                {
                    "model_name": model,
                    "left": left,
                    "right": right,
                    "matched_runs": len(matched),
                    "both_successful_with_accuracy": len(valid),
                    "left_wins": int((delta > 1e-12).sum()),
                    "ties": int((delta.abs() <= 1e-12).sum()),
                    "right_wins": int((delta < -1e-12).sum()),
                }
            )
    return pd.DataFrame(output)


def _matched_accuracy_drop_pair(
    model_rows: pd.DataFrame,
    left: str,
    right: str,
) -> pd.DataFrame:
    """Match two approaches/configurations on the same repair runs."""

    key = ["inp_eps", "buggy_points_num", "run_index"]
    left_rows = model_rows[model_rows["evaluation_series"] == left]
    right_rows = model_rows[model_rows["evaluation_series"] == right]
    if (
        "repair_data_ids" in model_rows.columns
        and not left_rows.empty
        and not right_rows.empty
        and left_rows["repair_data_ids"].notna().all()
        and right_rows["repair_data_ids"].notna().all()
    ):
        key.append("repair_data_ids")
    left_frame = left_rows[
        key + ["result", "accuracy_drop"]
    ].rename(columns={"result": "left_result", "accuracy_drop": "left_drop"})
    right_frame = right_rows[
        key + ["result", "accuracy_drop"]
    ].rename(columns={"result": "right_result", "accuracy_drop": "right_drop"})
    matched = left_frame.merge(right_frame, on=key, how="inner")
    return matched[
        (matched["left_result"] == "success")
        & (matched["right_result"] == "success")
        & matched["left_drop"].notna()
        & matched["right_drop"].notna()
    ].copy()


def _matched_runtime_pair(
    model_rows: pd.DataFrame,
    left: str,
    right: str,
) -> pd.DataFrame:
    """Match two configurations with successful, reported repair runtimes."""

    key = ["inp_eps", "buggy_points_num", "run_index"]
    left_rows = model_rows[model_rows["evaluation_series"] == left]
    right_rows = model_rows[model_rows["evaluation_series"] == right]
    if (
        "repair_data_ids" in model_rows.columns
        and not left_rows.empty
        and not right_rows.empty
        and left_rows["repair_data_ids"].notna().all()
        and right_rows["repair_data_ids"].notna().all()
    ):
        key.append("repair_data_ids")
    left_frame = left_rows[key + ["result", "total_repair_time_sec"]].rename(
        columns={"result": "left_result", "total_repair_time_sec": "left_runtime"}
    )
    right_frame = right_rows[key + ["result", "total_repair_time_sec"]].rename(
        columns={"result": "right_result", "total_repair_time_sec": "right_runtime"}
    )
    matched = left_frame.merge(right_frame, on=key, how="inner")
    return matched[
        (matched["left_result"] == "success")
        & (matched["right_result"] == "success")
        & matched["left_runtime"].notna()
        & matched["right_runtime"].notna()
    ].copy()


def pairwise_success_asymmetry(
    model_rows: pd.DataFrame,
    *,
    label_names: dict[str, str] | None = None,
    left_header: str = "Approach",
) -> pd.DataFrame:
    """Count matched runs where the row succeeds and the column does not."""

    label_names = {} if label_names is None else label_names

    def display_name(label: str) -> str:
        return label_names.get(label, _display_series_name(label))

    labels = sorted(
        model_rows["evaluation_series"].dropna().astype(str).unique(),
        key=_series_sort_key,
    )
    display_labels = [display_name(label) for label in labels]
    matrix = pd.DataFrame(0, index=display_labels, columns=display_labels, dtype=int)
    matrix.index.name = left_header
    for row_label, column_label in itertools.permutations(labels, 2):
        key = ["inp_eps", "buggy_points_num", "run_index"]
        row_runs = model_rows[model_rows["evaluation_series"].astype(str) == row_label]
        column_runs = model_rows[
            model_rows["evaluation_series"].astype(str) == column_label
        ]
        if (
            "repair_data_ids" in model_rows.columns
            and not row_runs.empty
            and not column_runs.empty
            and row_runs["repair_data_ids"].notna().all()
            and column_runs["repair_data_ids"].notna().all()
        ):
            key.append("repair_data_ids")
        row_results = row_runs[key + ["result"]].rename(
            columns={"result": "row_result"}
        )
        column_results = column_runs[key + ["result"]].rename(
            columns={"result": "column_result"}
        )
        matched = row_results.merge(column_results, on=key, how="inner")
        count = (
            (matched["row_result"] == "success")
            & (matched["column_result"] != "success")
        ).sum()
        matrix.loc[
            display_name(row_label), display_name(column_label)
        ] = int(count)
    return matrix.reset_index()


def pairwise_accuracy_wins(
    model_rows: pd.DataFrame,
    *,
    label_names: dict[str, str] | None = None,
    left_header: str = "Approach",
) -> pd.DataFrame:
    """Count matched successful runs where the row has higher accuracy."""

    label_names = {} if label_names is None else label_names

    def display_name(label: str) -> str:
        return label_names.get(label, _display_series_name(label))

    labels = sorted(
        model_rows["evaluation_series"].dropna().astype(str).unique(),
        key=_series_sort_key,
    )
    display_labels = [display_name(label) for label in labels]
    matrix = pd.DataFrame(0, index=display_labels, columns=display_labels, dtype=int)
    matrix.index.name = left_header
    for row_label, column_label in itertools.permutations(labels, 2):
        key = ["inp_eps", "buggy_points_num", "run_index"]
        row_runs = model_rows[model_rows["evaluation_series"].astype(str) == row_label]
        column_runs = model_rows[
            model_rows["evaluation_series"].astype(str) == column_label
        ]
        if (
            "repair_data_ids" in model_rows.columns
            and not row_runs.empty
            and not column_runs.empty
            and row_runs["repair_data_ids"].notna().all()
            and column_runs["repair_data_ids"].notna().all()
        ):
            key.append("repair_data_ids")
        row_results = row_runs[key + ["result", "repaired_accuracy"]].rename(
            columns={
                "result": "row_result",
                "repaired_accuracy": "row_accuracy",
            }
        )
        column_results = column_runs[
            key + ["result", "repaired_accuracy"]
        ].rename(
            columns={
                "result": "column_result",
                "repaired_accuracy": "column_accuracy",
            }
        )
        matched = row_results.merge(column_results, on=key, how="inner")
        valid = matched[
            (matched["row_result"] == "success")
            & (matched["column_result"] == "success")
            & matched["row_accuracy"].notna()
            & matched["column_accuracy"].notna()
        ]
        count = (valid["row_accuracy"] - valid["column_accuracy"] > 1e-12).sum()
        matrix.loc[display_name(row_label), display_name(column_label)] = int(count)
    return matrix.reset_index()


def _mark_pairwise_diagonal(matrix: pd.DataFrame, left_header: str) -> pd.DataFrame:
    """Render self-comparisons as not applicable while retaining real zeros."""

    formatted = matrix.astype(object).copy()
    for index, row_label in formatted[left_header].items():
        if row_label in formatted.columns:
            formatted.at[index, row_label] = "--"
    return formatted


def _write_pairwise_success_tables(
    rows: pd.DataFrame,
    directory: Path,
    *,
    rq: str = "rq2",
    label_names: dict[str, str] | None = None,
    left_header: str = "Approach",
) -> list[Path]:
    """Write a row-success/column-failure matrix for each network."""

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model, model_rows in rows.groupby("model_name", sort=True):
        matrix = pairwise_success_asymmetry(
            model_rows,
            label_names=label_names,
            left_header=left_header,
        )
        matrix = _mark_pairwise_diagonal(matrix, left_header)
        slug = str(model).lower().replace(" ", "_").replace("/", "_")
        csv_path = directory / f"{rq}_pairwise_success_{slug}.csv"
        tex_path = directory / f"{rq}_pairwise_success_{slug}.tex"
        matrix.to_csv(csv_path, index=False)
        tex_matrix = _tex_approach_labels(matrix)
        display_model = NETWORK_NAMES.get(str(model), str(model).replace("_", r"\_"))
        tex_path.write_text(
            tex_matrix.to_latex(
                index=False,
                escape=False,
                column_format="l" + "c" * (len(tex_matrix.columns) - 1),
                caption=(
                    "Pairwise repair-success counts for "
                    f"{display_model}. Each cell counts matched instances where "
                    "the row approach succeeds and the column approach does not."
                ),
                label=f"tab:{rq}-pairwise-success-{slug.replace('_', '-')}",
            ),
            encoding="utf-8",
        )
        written.extend([csv_path, tex_path])
    return written


def _write_pairwise_accuracy_win_tables(
    rows: pd.DataFrame,
    directory: Path,
    *,
    rq: str,
    label_names: dict[str, str] | None = None,
    left_header: str = "Approach",
) -> list[Path]:
    """Write a higher-repaired-accuracy win matrix for each network."""

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model, model_rows in rows.groupby("model_name", sort=True):
        matrix = pairwise_accuracy_wins(
            model_rows,
            label_names=label_names,
            left_header=left_header,
        )
        matrix = _mark_pairwise_diagonal(matrix, left_header)
        slug = str(model).lower().replace(" ", "_").replace("/", "_")
        csv_path = directory / f"{rq}_pairwise_accuracy_wins_{slug}.csv"
        tex_path = directory / f"{rq}_pairwise_accuracy_wins_{slug}.tex"
        matrix.to_csv(csv_path, index=False)
        tex_matrix = _tex_approach_labels(matrix)
        display_model = NETWORK_NAMES.get(str(model), str(model).replace("_", r"\_"))
        tex_path.write_text(
            tex_matrix.to_latex(
                index=False,
                escape=False,
                column_format="l" + "c" * (len(tex_matrix.columns) - 1),
                caption=(
                    "Pairwise repaired-accuracy wins for "
                    f"{display_model}. Each cell counts matched successful "
                    "repairs where the row has higher repaired accuracy than "
                    "the column; ties are excluded."
                ),
                label=f"tab:{rq}-pairwise-accuracy-wins-{slug.replace('_', '-')}",
            ),
            encoding="utf-8",
        )
        written.extend([csv_path, tex_path])
    return written


def _write_pairwise_accuracy_drop_figures(
    rows: pd.DataFrame,
    directory: Path,
    *,
    prefix: str = "rq2",
    pairs_by_model: dict[str, list[tuple[str, str]]] | None = None,
    output_names: dict[tuple[str, str, str], str] | None = None,
) -> list[Path]:
    """Write each approach/configuration pair as a separate diagonal figure."""

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model, model_rows in rows.groupby("model_name", sort=True):
        model_slug = str(model).lower().replace(" ", "_").replace("/", "_")
        legacy_combined = directory / f"{prefix}_pairwise_accuracy_drop_{model_slug}.png"
        if legacy_combined.exists():
            legacy_combined.unlink()
        for stale in directory.glob(
            f"{prefix}_pairwise_accuracy_drop_{model_slug}_*_vs_*.png"
        ):
            stale.unlink()
        labels = sorted(
            model_rows["evaluation_series"].dropna().astype(str).unique(),
            key=_series_sort_key,
        )
        pairs = (
            pairs_by_model.get(str(model), [])
            if pairs_by_model is not None
            else list(itertools.combinations(labels, 2))
        )
        if not pairs:
            continue
        for left, right in pairs:
            fig, axis = plt.subplots(figsize=(6.5, 5.2))
            matched = _matched_accuracy_drop_pair(model_rows, left, right)
            accuracy_fontsize = _pairwise_accuracy_fontsize(prefix)
            axis.scatter(
                matched["left_drop"],
                matched["right_drop"],
                marker="o",
                color="tab:blue",
                s=30,
                alpha=0.7,
                edgecolors="none",
            )
            if matched.empty:
                low, high = -0.01, 0.01
            else:
                low = min(float(matched["left_drop"].min()), float(matched["right_drop"].min()))
                high = max(float(matched["left_drop"].max()), float(matched["right_drop"].max()))
                padding = max((high - low) * 0.06, 0.005)
                low -= padding
                high += padding
            axis.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1)
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(
                f"{_display_series_name(left)} degradation",
                fontsize=accuracy_fontsize,
            )
            axis.set_ylabel(
                f"{_display_series_name(right)} degradation",
                fontsize=accuracy_fontsize,
            )
            axis.tick_params(axis="both", labelsize=accuracy_fontsize)
            if not matched.empty:
                delta = matched["right_drop"] - matched["left_drop"]
                above = int((delta > 1e-12).sum())
                below = int((delta < -1e-12).sum())
                on_diagonal = len(matched) - above - below
                axis.text(
                    0.02,
                    0.98,
                    f"n: {len(matched)}\nabove: {above}\nbelow: {below}\n"
                    f"on diagonal: {on_diagonal}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=accuracy_fontsize,
                )
            axis.grid(alpha=0.25)
            fig.tight_layout()
            left_slug = str(left).lower().replace(" ", "_").replace("/", "_")
            right_slug = str(right).lower().replace(" ", "_").replace("/", "_")
            filename = (
                output_names.get((str(model), left, right))
                if output_names is not None
                else None
            )
            output = directory / (
                filename
                or (
                    f"{prefix}_pairwise_accuracy_drop_{model_slug}_"
                    f"{left_slug}_vs_{right_slug}.png"
                )
            )
            fig.savefig(output, dpi=200, bbox_inches="tight")
            plt.close(fig)
            written.append(output)
    return written


def _write_pairwise_runtime_figures(
    rows: pd.DataFrame,
    directory: Path,
    *,
    prefix: str,
) -> list[Path]:
    """Write matched-run diagonal runtime comparisons for each pair."""

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model, model_rows in rows.groupby("model_name", sort=True):
        model_slug = str(model).lower().replace(" ", "_").replace("/", "_")
        labels = sorted(
            model_rows["evaluation_series"].dropna().astype(str).unique(),
            key=_series_sort_key,
        )
        for left, right in itertools.combinations(labels, 2):
            matched = _matched_runtime_pair(model_rows, left, right)
            fig, axis = plt.subplots(figsize=(6.5, 5.2))
            axis.scatter(
                matched["left_runtime"],
                matched["right_runtime"],
                marker="o",
                color="tab:blue",
                s=30,
                alpha=0.7,
                edgecolors="none",
            )
            if matched.empty:
                low, high = 0.0, 1.0
            else:
                low = min(
                    float(matched["left_runtime"].min()),
                    float(matched["right_runtime"].min()),
                )
                high = max(
                    float(matched["left_runtime"].max()),
                    float(matched["right_runtime"].max()),
                )
                padding = max((high - low) * 0.06, 1.0)
                low = max(0.0, low - padding)
                high += padding
            axis.plot(
                [low, high],
                [low, high],
                linestyle="--",
                color="gray",
                linewidth=1,
            )
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(
                f"{_display_series_name(left)} runtime (seconds)",
                fontsize=FIGURE_LABEL_FONTSIZE,
            )
            axis.set_ylabel(
                f"{_display_series_name(right)} runtime (seconds)",
                fontsize=FIGURE_LABEL_FONTSIZE,
            )
            axis.tick_params(axis="both", labelsize=FIGURE_TICK_FONTSIZE)
            if not matched.empty:
                delta = matched["right_runtime"] - matched["left_runtime"]
                above = int((delta > 1e-12).sum())
                below = int((delta < -1e-12).sum())
                on_diagonal = len(matched) - above - below
                axis.text(
                    0.02,
                    0.98,
                    f"n: {len(matched)}\nabove: {above}\nbelow: {below}\n"
                    f"on diagonal: {on_diagonal}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=FIGURE_ANNOTATION_FONTSIZE,
                )
            axis.grid(alpha=0.25)
            fig.tight_layout()
            left_slug = str(left).lower().replace(" ", "_").replace("/", "_")
            right_slug = str(right).lower().replace(" ", "_").replace("/", "_")
            output = directory / (
                f"{prefix}_pairwise_runtime_{model_slug}_"
                f"{left_slug}_vs_{right_slug}.png"
            )
            fig.savefig(output, dpi=200, bbox_inches="tight")
            plt.close(fig)
            written.append(output)
    return written


def build_comparison_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace raw layer-wise OURS rows with one sequential-policy row per run."""

    comparison, merged_ours = replace_layerwise_comparison_rows(rows)
    comparison = comparison[comparison["approach"].isin(COMPARISON_APPROACHES)].copy()
    return comparison, merged_ours


def _metric_matrix(
    summary: pd.DataFrame,
    x_column: str,
    value_column: str,
    *,
    row_column: str,
    row_order: list[str],
    row_names: dict[str, str],
    left_header: str = "Approach",
) -> pd.DataFrame:
    """Pivot one summarized metric into the paper's CSV table layout."""

    values = sorted(summary[x_column].dropna().unique(), key=float)
    pivot = summary.pivot_table(
        index=row_column,
        columns=x_column,
        values=value_column,
        aggfunc="first",
    ).reindex(index=row_order, columns=values)
    pivot.index = [row_names.get(name, name) for name in pivot.index]
    pivot.index.name = left_header
    pivot.columns = [
        int(value) if float(value).is_integer() else value for value in values
    ]
    return pivot.reset_index()


def _paper_comparison_subset(
    rows: pd.DataFrame,
    perturbation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if perturbation == "nonzero":
        subset = rows[
            (rows["perturbation"] == "nonzero")
            & (rows["buggy_points_num"] == 1)
            & rows["ndims"].notna()
        ].copy()
        x_column = "ndims"
    else:
        subset = rows[
            (rows["perturbation"] == "all")
            & rows["buggy_points_num"].notna()
        ].copy()
        x_column = "buggy_points_num"
    summary = summarize(
        subset,
        ["model_name", "inp_eps", x_column, "approach", "evaluation_series"],
    )
    return subset, summary, x_column


def _write_paper_success_csv(
    summary: pd.DataFrame,
    model: str,
    x_column: str,
    output: Path,
) -> None:
    matrix = success_count_matrix(summary[summary["model_name"] == model], x_column)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output, index=False)


def _write_marked_pairwise_csv(
    matrix: pd.DataFrame,
    output: Path,
    left_header: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _mark_pairwise_diagonal(matrix, left_header).to_csv(output, index=False)


def _rq_materials(rows: pd.DataFrame, output_root: Path) -> dict[str, int]:
    """Generate only the paper artifacts listed in README.md."""

    for rq in ("rq1", "rq2", "rq3", "rq4", "rq5"):
        (output_root / rq).mkdir(parents=True, exist_ok=True)

    comparison_rows, _merged_ours = build_comparison_rows(rows)
    nonzero, nonzero_summary, nonzero_x = _paper_comparison_subset(
        comparison_rows, "nonzero"
    )
    all_rows, all_summary, all_x = _paper_comparison_subset(
        comparison_rows, "all"
    )

    # RQ1: repair-success counts and PREPARED failure outcomes.
    for perturbation, summary, x_column, models in (
        (
            "nonzero",
            nonzero_summary,
            nonzero_x,
            ("mnist 9x100", "mnist_256x4"),
        ),
        ("all", all_summary, all_x, tuple(PAPER_NETWORK_SLUGS)),
    ):
        for model in models:
            slug = PAPER_NETWORK_SLUGS[model]
            _write_paper_success_csv(
                summary,
                model,
                x_column,
                output_root / "rq1" / f"success_count_{perturbation}_{slug}.csv",
            )
    for model in ("cifar10_cnn1", "gtsrb_cnn"):
        slug = PAPER_NETWORK_SLUGS[model]
        matrix = prepared_outcome_matrix(all_rows[all_rows["model_name"] == model])
        matrix.to_csv(
            output_root / "rq1" / f"prepared_outcomes_{slug}.csv",
            index=False,
        )
    mnistf1_all = all_rows[all_rows["model_name"] == "mnist 9x100"]
    _write_marked_pairwise_csv(
        pairwise_success_asymmetry(mnistf1_all),
        output_root / "rq1" / "pairwise_success_count_mnistf1.csv",
        "Approach",
    )

    # RQ2: accuracy degradation for both perturbation regimes.
    _write_accuracy_boxplots(
        nonzero[nonzero["model_name"].isin(["mnist 9x100", "mnist_256x4"])],
        output_root / "rq2",
        x=nonzero_x,
        prefix="rq2",
        output_names={
            model: f"repair_acc_drop_nonzero_{PAPER_NETWORK_SLUGS[model]}.png"
            for model in ("mnist 9x100", "mnist_256x4")
        },
    )
    _write_accuracy_boxplots(
        all_rows[all_rows["model_name"].isin(PAPER_NETWORK_SLUGS)],
        output_root / "rq2",
        x=all_x,
        prefix="rq2",
        output_names={
            model: f"repair_acc_drop_all_{slug}.png"
            for model, slug in PAPER_NETWORK_SLUGS.items()
        },
    )
    pair_models = ("mnist 9x100", "mnist_256x4")
    _write_pairwise_accuracy_drop_figures(
        all_rows[all_rows["model_name"].isin(pair_models)],
        output_root / "rq2",
        prefix="rq2",
        pairs_by_model={model: [("prepared", "ours")] for model in pair_models},
        output_names={
            (model, "prepared", "ours"): (
                "pairwise_repair_acc_drop_all_mular_vs_prepared_"
                f"{PAPER_NETWORK_SLUGS[model]}.png"
            )
            for model in pair_models
        },
    )

    # RQ3: successful-repair runtime for both perturbation regimes.
    _write_runtime_figures(
        nonzero_summary[
            nonzero_summary["model_name"].isin(["mnist 9x100", "mnist_256x4"])
        ],
        output_root / "rq3",
        x=nonzero_x,
        prefix="rq3",
        style_prefix="rq1",
        output_names={
            model: f"repair_time_nonzero_{PAPER_NETWORK_SLUGS[model]}.png"
            for model in ("mnist 9x100", "mnist_256x4")
        },
    )
    _write_runtime_figures(
        all_summary[all_summary["model_name"].isin(PAPER_NETWORK_SLUGS)],
        output_root / "rq3",
        x=all_x,
        prefix="rq3",
        style_prefix="rq2",
        output_names={
            model: f"repair_time_all_{slug}.png"
            for model, slug in PAPER_NETWORK_SLUGS.items()
        },
    )

    # RQ4: effect of using different repairable-layer sets in MULAR.
    layer_rows = collect_policy_stage_rows(rows)
    layer_rows = layer_rows[layer_rows["perturbation"] == "all"].copy()
    layer_names = {
        label: _layer_count_name(label)
        for labels in POLICY_LAYER_SETS.values()
        for label in labels
    }
    for model, slug in PAPER_NETWORK_SLUGS.items():
        model_rows = layer_rows[layer_rows["model_name"] == model]
        _write_marked_pairwise_csv(
            pairwise_success_asymmetry(
                model_rows,
                label_names=layer_names,
                left_header="Repairable layers",
            ),
            output_root / "rq4" / f"pairwise_success_count_diff_layers_{slug}.csv",
            "Repairable layers",
        )
        _write_marked_pairwise_csv(
            pairwise_accuracy_wins(
                model_rows,
                label_names=layer_names,
                left_header="Repairable layers",
            ),
            output_root / "rq4" / f"pairwise_repair_acc_drop_diff_layers_{slug}.csv",
            "Repairable layers",
        )

    # RQ5: fault-localization variants.
    rq5 = rows[
        (rows["approach"] == "FL")
        & rows["variant"].isin(["pareto10", "pareto10_random"])
        & rows["model_name"].isin(["mnist 9x100", "cifar10_cnn1"])
    ].copy()
    rq5_summary = summarize(
        rq5,
        ["model_name", "inp_eps", "buggy_points_num", "variant", "evaluation_series"],
    )
    variant_order = ["pareto10", "pareto10_random"]
    variant_names = {
        "pareto10": "MULAR_Arachne",
        "pareto10_random": "MULAR_Random",
    }
    for model in ("mnist 9x100", "cifar10_cnn1"):
        slug = PAPER_NETWORK_SLUGS[model]
        model_summary = rq5_summary[rq5_summary["model_name"] == model]
        matrix = _metric_matrix(
            model_summary,
            "buggy_points_num",
            "successes",
            row_column="variant",
            row_order=variant_order,
            row_names=variant_names,
        )
        matrix.to_csv(
            output_root / "rq5" /
            f"success_count_fault_localization_{slug}.csv",
            index=False,
        )

    _write_accuracy_boxplots(
        rq5,
        output_root / "rq5",
        x="buggy_points_num",
        prefix="rq5",
        output_names={
            model: (
                "repair_acc_drop_fault_localization_"
                f"{PAPER_NETWORK_SLUGS[model]}.png"
            )
            for model in ("mnist 9x100", "cifar10_cnn1")
        },
    )
    _write_runtime_figures(
        rq5_summary,
        output_root / "rq5",
        x="buggy_points_num",
        prefix="rq5",
        output_names={
            model: (
                "repair_time_fault_localization_"
                f"{PAPER_NETWORK_SLUGS[model]}.png"
            )
            for model in ("mnist 9x100", "cifar10_cnn1")
        },
    )

    return {
        "RQ1": len(nonzero) + len(all_rows),
        "RQ2": len(nonzero) + len(all_rows),
        "RQ3": len(nonzero) + len(all_rows),
        "RQ4": len(layer_rows),
        "RQ5": len(rq5),
    }


def _write_network_summary(repository: Path, output_root: Path) -> tuple[Path, Path]:
    """Write architecture statistics for the four evaluated networks."""

    repository_text = str(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    from net.net_structure import summarize_model

    model_paths = {
        "mnist 9x100": repository / "models/mnist/mnist_relu_9_100.tf",
        "mnist_256x4": repository / "net/onnx/mnist_256x4.onnx",
        "cifar10_cnn1": repository / "net/onnx/cifar10_cnn1.onnx",
        "gtsrb_cnn": repository / "net/onnx/gtsrb_cnn.onnx",
    }
    rows: list[dict[str, object]] = []
    for model_name, model_path in model_paths.items():
        summary = summarize_model(model_path)
        layer_counts = Counter(
            "FC" if layer_type in {"Linear", "Gemm", "MatMul"}
            else "Conv" if layer_type == "Conv"
            else layer_type
            for layer_type in summary.layer_types
        )
        layer_types = ", ".join(
            f"{layer_type} x{count}"
            for layer_type, count in layer_counts.items()
        )
        rows.append(
            {
                "network": NETWORK_NAMES[model_name],
                "dataset": NETWORK_DATASETS[model_name],
                "layers": summary.num_layers,
                "layer_types": layer_types,
                "neurons": summary.total_neurons,
                "parameters": summary.total_parameters,
            }
        )

    frame = pd.DataFrame(rows)
    csv_path = output_root / "network_summary.csv"
    tex_path = output_root / "network_summary.tex"
    frame.to_csv(csv_path, index=False)
    tex_frame = frame.copy()
    tex_frame["layer_types"] = tex_frame["layer_types"].str.replace(
        " x", r" $\times$ ", regex=False
    )
    for column in ["neurons", "parameters"]:
        tex_frame[column] = tex_frame[column].map(lambda value: f"{value:,}")
    tex_frame = tex_frame.rename(
        columns={
            "network": "Network",
            "dataset": "Dataset",
            "layers": r"\# Layers",
            "layer_types": "Layer types",
            "neurons": r"\# Neurons",
            "parameters": r"\# Parameters",
        }
    )
    tex_path.write_text(
        tex_frame.to_latex(
            index=False,
            escape=False,
            column_format="llrlrr",
            caption="Architectures of the evaluated neural networks.",
            label="tab:network-summary",
        ),
        encoding="utf-8",
    )
    return csv_path, tex_path


def _write_report(rows: pd.DataFrame, output_root: Path, rq_counts: dict[str, int]) -> Path:
    inventory = summarize(
        rows,
        ["approach", "variant", "model_name", "repair_task", "perturbation"],
    )
    _write_table(inventory, output_root, "inventory")
    failures = (
        rows.groupby(["approach", "result", "failure_reason"], dropna=False)
        .size()
        .reset_index(name="runs")
    )
    _write_table(failures, output_root, "status_and_failure_reasons")

    arbitrary = sorted(
        rows[
            (rows["approach"] == "ours")
            & (rows["model_name"] == "mnist 9x100")
            & rows["layer_set_label"].isin(RQ4_LAYER_SETS)
        ]["layer_set_label"].dropna().unique()
    )
    status_counts = rows["result"].value_counts().to_dict()
    lines = [
        "# Generated evaluation materials",
        "",
        f"Canonical run records: **{len(rows):,}**.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python result_analysis/generate_materials.py",
        "```",
        "",
        "This rebuilds `original_result/csv` from both legacy aggregate logs and current per-run logs, then regenerates this directory.",
        "",
        "## Network summary",
        "",
        "`network_summary.csv` and `network_summary.tex` describe the four evaluated architectures. Layers count learnable FC/Conv layers; neurons count their per-sample output units (including convolutional spatial positions); parameters include weights and biases.",
        "",
        "## Result status",
        "",
        f"- Successful: {status_counts.get('success', 0):,}",
        f"- Failed: {status_counts.get('failed', 0):,}",
        f"- Incomplete/abruptly ended: {status_counts.get('incomplete', 0):,}",
        "",
        "## Coverage",
        "",
        *[f"- {name}: {count:,} run records" for name, count in rq_counts.items()],
        "",
        "## Interpretation notes",
        "",
        "- For RQ1/RQ2, `ours` is one sequential result per run: try the last layer first, expand only after failure, stop at first success, and sum all attempted repair runtimes. If expansion reaches a multi-layer stage, the shared input-subspace construction time is added exactly once.",
        "- Older multi-stage logs that do not report input construction retain repair-only runtime and are marked `input_construction_time_missing` in `ours_sequential_policy_runs.csv`.",
        "- Raw per-layer results are retained only for RQ3/RQ4; they are not counted as separate OURS approaches in RQ1/RQ2.",
        "- If a required next layer-set result is unavailable after earlier failures, the merged result is marked `incomplete`.",
        "- Legacy MNIST 9x100 RQ1 layer labels are normalized as `[14] -> [16]`, `[12,14] -> [14,16]`, and `[10,12,14] -> [12,14,16]`; this makes the legacy `[14]` configuration the last-layer baseline.",
        "- Success rates use every logged run. Abrupt logs without a success or failure result are `incomplete`, not successful.",
        "- Runtime and accuracy summaries use successful runs because failed runs do not consistently report those values.",
        "- Pairwise accuracy tables match configurations by epsilon, N, run index, and repair-data IDs when available, then compare only pairs where both repairs succeeded and reported accuracy.",
        "- The RQ3 `layers_9` data for `mnist_256x4` comes from the equivalent last-layer experiment.",
        f"- RQ4 uses the arbitrary layer sets actually present in the results: {', '.join(arbitrary) or 'none'}.",
        "",
        "The auditable per-run merge is in `ours_sequential_policy_runs.csv`; aggregate layer usage is in `ours_sequential_policy_usage.csv`.",
        "",
        "RQ1--RQ5 contain one LaTeX success-count table per network, with the compared approaches/configurations in rows and dimensions or N in columns.",
        "",
        "Each RQ directory contains CSV and LaTeX tables, one grouped accuracy-degradation box plot and one runtime figure per network when numeric data is available. Diamonds mark box-plot means and use the corresponding box color; runtime lines are semi-transparent so overlaps remain visible.",
        "RQ2 contains a separate diagonal accuracy-degradation figure for each network and approach pair; all matched repair runs use the same point style.",
    ]
    report = output_root / "README.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _reset_material_output(output_root: Path) -> None:
    """Clear the dedicated material directory before exact regeneration."""

    resolved = output_root.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in forbidden:
        raise ValueError(f"Refusing to clear unsafe output directory: {resolved}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def _validate_paper_materials(output_root: Path) -> None:
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    missing = sorted(PAPER_MATERIAL_PATHS - actual)
    unexpected = sorted(actual - PAPER_MATERIAL_PATHS)
    if missing or unexpected:
        raise RuntimeError(
            "Generated material set does not match README.md: "
            f"missing={missing}, unexpected={unexpected}"
        )


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    analysis_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=repository / "original_result")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        type=Path,
        help="Write generated artifacts to a custom directory.",
    )
    output_group.add_argument(
        "--original",
        action="store_true",
        help="Write generated artifacts to result_analysis/original_materials.",
    )
    parser.add_argument("--no-refresh", action="store_true", help="Reuse original_result/csv")
    args = parser.parse_args()
    output_root = args.output or analysis_directory / (
        "original_materials" if args.original else "materials"
    )

    if not args.no_refresh:
        written = collect_results(args.result_root, clean=True)
        print(f"Collected {len(written)} experiment CSV files.")
    rows = _prepare_rows(load_all_csv_rows(args.result_root))
    if rows.empty:
        raise SystemExit("No result rows were found.")
    _reset_material_output(output_root)
    _rq_materials(rows, output_root)
    _validate_paper_materials(output_root)
    print(
        f"Generated {len(PAPER_MATERIAL_PATHS)} paper materials in "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
