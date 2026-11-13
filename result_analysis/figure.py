from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from result_analysis.analysis import DISPLAY_NAME_MAP, aggregate_results_from_csvs, default_repair_root, load_all_csv_rows

COLORS = {
    "aprnn": "#7E9D5B",
    "prorepair": "#FE8104",
    "prepared": "#4C78A8",
    "lastlayer": "#E45756",
    "ours": "#2FA32F",
    "FL": "#BC1BD8",
}

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16

APPROACHES = ["aprnn", "prorepair", "prepared", "lastlayer", "ours"]
APPROACHES_MNIST_ALL = ["prorepair", "prepared", "lastlayer", "ours"]
APPROACHES_CIFAR_GTSRB = ["prepared", "lastlayer", "ours"]
LAYER_LABEL_RE = re.compile(r"^layer\d+$")
MERGED_DIR_RE = re.compile(r"^merged_(?P<threshold>[^/]+)$")
GREEDY_MERGED_DIR = "merged_greedy_best_accuracy"


def _x_axis_config(perturbation: str | None) -> tuple[str, str]:
    if (perturbation or "").lower() == "nonzero":
        return "ndims", "#DIM"
    return "buggy_points_num", "#P"


def _plot_approaches(perturbation: str | None, model_name: str | None = None) -> list[str]:
    if (perturbation or "").lower() == "nonzero":
        return list(APPROACHES)
    if "mnist" in model_name.lower() if model_name else False:
        return list(APPROACHES_MNIST_ALL)
    if "cifar" in model_name.lower() or "gtsrb" in model_name.lower() if model_name else False:
        return list(APPROACHES_CIFAR_GTSRB)
    return list(APPROACHES)


def _display_name(approach: str) -> str:
    return DISPLAY_NAME_MAP.get(approach, approach)


def _parse_label_list(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _normalize_text(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(value)).lower()


def _text_matches(value: object, target: str) -> bool:
    value_norm = _normalize_text(value)
    target_norm = _normalize_text(target)
    if value_norm == target_norm:
        return True
    # Some repair tasks are written as "A + B" in the experiment map but
    # stored as "AAndB" in the merged CSV metadata. Treat the conjunction as
    # optional so both forms match the same slice.
    return value_norm.replace("and", "") == target_norm.replace("and", "")


def _threshold_to_dir_label(threshold: float) -> str:
    return f"merged_{threshold:g}pct"


def _merge_strategy_dir_label(merge_strategy: str, threshold: float | None) -> str | None:
    if merge_strategy == "greedy_best_accuracy":
        return GREEDY_MERGED_DIR
    if threshold is None:
        return None
    return _threshold_to_dir_label(threshold)


def _is_merged_ours_csv(source_csv: str, threshold: float | None, merge_strategy: str = "threshold") -> bool:
    csv_path = Path(source_csv)
    merged_label = _merge_strategy_dir_label(merge_strategy, threshold)
    merged_dir = csv_path.parent
    ours_dir = csv_path.parent.parent

    if merged_dir is None or ours_dir is None:
        return False

    if not ours_dir.name.startswith("ours_"):
        return False

    match = MERGED_DIR_RE.fullmatch(merged_dir.name)
    if match is None:
        return False

    if merged_label is None:
        return True
    return merged_dir.name == merged_label


def _filter_ours_rows(
    runs: pd.DataFrame,
    ours_threshold: float | None,
    merge_strategy: str = "threshold",
) -> pd.DataFrame:
    if runs.empty or "approach" not in runs.columns or "source_csv" not in runs.columns:
        return runs

    is_ours = runs["approach"] == "ours"
    if ours_threshold is None:
        # Default to the raw per-layer OURS CSVs and ignore merged outputs.
        keep_ours = ~runs.loc[is_ours, "source_csv"].map(
            lambda source_csv: _is_merged_ours_csv(str(source_csv), None, merge_strategy=merge_strategy)
        )
    else:
        # Use only the merged outputs for the requested threshold.
        keep_ours = runs.loc[is_ours, "source_csv"].map(
            lambda source_csv: _is_merged_ours_csv(str(source_csv), ours_threshold, merge_strategy=merge_strategy)
        )

    keep_mask = ~is_ours
    keep_mask.loc[is_ours] = keep_ours.values
    return runs[keep_mask].copy()


def _label_merged_ours_rows(
    runs: pd.DataFrame,
    ours_threshold: float | None,
    merge_strategy: str = "threshold",
) -> pd.DataFrame:
    """
    For merged OURS outputs, use the successful layer label as the series label.

    This lets merged rows participate in plots that expect labels like
    `lastlayer`, `layer14`, `layer12`, ...
    """
    if ours_threshold is None or runs.empty:
        return runs
    if "approach" not in runs.columns or "source_csv" not in runs.columns:
        return runs
    if "successful_layer_label" not in runs.columns:
        return runs

    out = runs.copy()
    is_ours = out["approach"] == "ours"
    is_merged = out.loc[is_ours, "source_csv"].map(
        lambda source_csv: _is_merged_ours_csv(str(source_csv), ours_threshold, merge_strategy=merge_strategy)
    )
    merged_mask = is_merged.fillna(False).to_numpy(dtype=bool)
    idx = out.index[is_ours][merged_mask]
    if len(idx) == 0:
        return out

    out.loc[idx, "series_label"] = out.loc[idx, "successful_layer_label"].fillna(out.loc[idx, "series_label"])
    return out


def _should_use_merged_ours(series_labels: list[str] | None) -> bool:
    if series_labels is None:
        return True
    return "ours" in series_labels


def _is_layer_label(label: str) -> bool:
    return bool(LAYER_LABEL_RE.fullmatch(label))


def _build_color_map(labels: list[str]) -> dict[str, str]:
    color_map: dict[str, str] = {}
    layer_labels = [label for label in labels if _is_layer_label(label)]
    layer_cmap = plt.get_cmap("Greens")
    if layer_labels:
        for i, label in enumerate(layer_labels):
            if len(layer_labels) == 1:
                shade = 0.65
            else:
                shade = 0.45 + 0.35 * (i / max(1, len(layer_labels) - 1))
            color_map[label] = layer_cmap(shade)

    palette = plt.get_cmap("tab20")
    palette_i = 0
    for label in labels:
        if label in color_map:
            continue
        if label in COLORS:
            color_map[label] = COLORS[label]
        else:
            color_map[label] = palette(palette_i % 20)
            palette_i += 1
    return color_map


def _filter_df(
    df: pd.DataFrame,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if model_name is not None:
        out = out[out["model_name"].map(lambda value: _text_matches(value, model_name))]
    if repair_task is not None:
        out = out[out["repair_task"].map(lambda value: _text_matches(value, repair_task))]
    if perturbation is not None:
        out = out[out["perturbation"].map(lambda value: _text_matches(value, perturbation))]
    if inp_eps is not None:
        out = out[out["inp_eps"].astype(str) == str(inp_eps)]
    if ndims is not None:
        out = out[out["ndims"] == ndims]
    return out


def _series_col_from_df(df: pd.DataFrame, series_labels: list[str] | None = None) -> str:
    if series_labels is not None and "series_label" in df.columns:
        return "series_label"
    return "approach"


def _series_labels_for_plot(df: pd.DataFrame, perturbation: str | None, model_name: str | None, series_labels: list[str] | None = None) -> list[str]:
    if series_labels is not None:
        return series_labels
    return _plot_approaches(perturbation, model_name)


def make_figure1_success_accuracy(
    root: Path | None = None,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
    ours_threshold: float | None = None,
    ours_merge_strategy: str = "threshold",
    series_labels: list[str] | None = None,
    run_nums: int = 20,
    save_path: Path | None = None,
    legend_loc: str = "center right",
    legend_offset: float = 0.0,
) -> plt.Figure:
    """Figure 1 with dynamic x-axis (N for all, ndims for nonzero)."""
    runs = load_all_csv_rows(root=root)
    if runs.empty:
        raise ValueError("No CSV rows found. Generate CSVs first.")

    runs = _filter_df(runs, model_name, repair_task, perturbation, inp_eps, ndims)
    if runs.empty:
        raise ValueError("No rows left after filters.")

    series_col = _series_col_from_df(runs, series_labels=series_labels)
    series_to_plot = _series_labels_for_plot(runs, perturbation, model_name, series_labels=series_labels)

    if _should_use_merged_ours(series_labels):
        runs = _filter_ours_rows(runs, ours_threshold, merge_strategy=ours_merge_strategy)
        runs = _label_merged_ours_rows(runs, ours_threshold, merge_strategy=ours_merge_strategy)
        if runs.empty:
            raise ValueError("No rows left after selecting OURS threshold.")

    runs = runs[runs[series_col].isin(series_to_plot)]
    x_col, x_label = _x_axis_config(perturbation)
    runs = runs[runs[x_col].notna()].copy()
    if runs.empty:
        raise ValueError(f"No rows with valid '{x_col}' after filters.")

    success_runs = runs[runs["result"] == "success"].copy()

    x_values = sorted(runs[x_col].dropna().unique())
    if not x_values:
        raise ValueError(f"No valid '{x_col}' values found for repaired accuracy plotting.")

    x_positions_base = np.arange(len(x_values), dtype=float)
    x_position_map = {x_val: x_positions_base[i] for i, x_val in enumerate(x_values)}

    approaches = series_to_plot
    is_cifar_or_gtsrb = bool(model_name and ("cifar" in model_name.lower() or "gtsrb" in model_name.lower()))
    if (perturbation or "").lower() == "nonzero":
        fig_size = (10, 6)
    else:
        fig_size = (7, 6) if is_cifar_or_gtsrb else (10, 6)
    fig, ax_acc = plt.subplots(figsize=fig_size)

    # Reduce overlap: use wider center spread and narrower boxes.
    offset_span = 0.30
    box_width = 0.12
    offsets = np.linspace(-offset_span, offset_span, len(approaches))

    # Reserve lower space for success-count labels.
    ax_acc.set_ylim(-0.08, 1.05)

    color_map = _build_color_map(approaches)

    for i, approach in enumerate(approaches):
        color = color_map.get(approach, COLORS.get(approach, None))
        x_positions = x_positions_base + offsets[i]

        # boxplot data
        box_data = []
        mean_vals = []
        for x_val in x_values:
            vals = success_runs.loc[
                (success_runs[series_col] == approach) & (success_runs[x_col] == x_val),
                "repaired_accuracy",
            ].dropna()
            box_data.append(vals.to_numpy())
            mean_vals.append(vals.mean() if len(vals) else np.nan)

        bp = ax_acc.boxplot(
            box_data,
            positions=x_positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for element in ["whiskers", "caps", "medians"]:
            for artist in bp[element]:
                artist.set_color(color)

        ax_acc.plot(x_positions, mean_vals, "o", color=color, label=_display_name(approach))

    # ax_acc.set_title("Repaired Accuracy", fontsize=TITLE_FONTSIZE)
    ax_acc.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
    ax_acc.set_ylabel("Repaired accuracy (over successful runs)", fontsize=LABEL_FONTSIZE)

    # Add dotted baseline for original accuracy.
    orig_vals = runs["original_accuracy"].dropna()
    if not orig_vals.empty:
        orig_acc = float(orig_vals.iloc[0])
        # ax_acc.axhline(orig_acc, linestyle=":", linewidth=2, color="#333333", label=f"original acc ({orig_acc:.2f})")
        ax_acc.axhline(orig_acc, linestyle=":", linewidth=2, color="#333333", label=f"original acc")

    ax_acc.set_xticks(x_positions_base)
    ax_acc.set_xticklabels([str(v) for v in x_values])
    ax_acc.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_acc.grid(axis="y", linestyle="--", alpha=0.35)

    ax_acc.text(
        0.01,
        0.02,
        "",
        transform=ax_acc.transAxes,
        fontsize=TICK_FONTSIZE-2,
        color="#555555",
    )
    ax_acc.legend(
        loc=legend_loc,
        bbox_to_anchor=(0, legend_offset, 1, 1),
        fontsize=TICK_FONTSIZE,
        labelspacing=0.25,
    )

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def make_figure2_time(
    root: Path | None = None,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
    ours_threshold: float | None = None,
    ours_merge_strategy: str = "threshold",
    series_labels: list[str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figure 2 with dynamic x-axis (N for all, ndims for nonzero)."""
    runs = load_all_csv_rows(root=root)
    if runs.empty:
        raise ValueError("No CSV rows found. Generate CSVs first.")

    runs = _filter_df(runs, model_name, repair_task, perturbation, inp_eps, ndims)
    if runs.empty:
        raise ValueError("No rows left after filters.")

    series_col = _series_col_from_df(runs, series_labels=series_labels)
    series_to_plot = _series_labels_for_plot(runs, perturbation, model_name, series_labels=series_labels)

    if _should_use_merged_ours(series_labels):
        runs = _filter_ours_rows(runs, ours_threshold, merge_strategy=ours_merge_strategy)
        runs = _label_merged_ours_rows(runs, ours_threshold, merge_strategy=ours_merge_strategy)
        if runs.empty:
            raise ValueError("No rows left after selecting OURS threshold.")

    runs = runs[runs[series_col].isin(series_to_plot)].copy()
    if runs.empty:
        raise ValueError("No rows left after selecting series.")

    fig, ax = plt.subplots(figsize=(6, 5))
    x_col, x_label = _x_axis_config(perturbation)

    runs = runs[runs[x_col].notna()].copy()
    if runs.empty:
        raise ValueError(f"No rows with valid '{x_col}' after filters.")

    x_values = sorted(runs[x_col].dropna().unique())
    if not x_values:
        raise ValueError("No valid x-axis values found after filters.")

    x_positions_base = np.arange(len(x_values), dtype=float)
    x_position_map = {x_val: x_positions_base[i] for i, x_val in enumerate(x_values)}

    color_map = _build_color_map(series_to_plot)
    for approach in series_to_plot:
        df = (
            runs[runs[series_col] == approach]
            .groupby([series_col, x_col], dropna=False)
            .apply(lambda g: pd.Series({"avg_total_repair_time_sec_success": g.loc[g["result"] == "success", "total_repair_time_sec"].mean()}))
            .reset_index()
            .sort_values(x_col)
        )
        if df.empty:
            continue

        x_vals = df[x_col].tolist()
        x_positions = np.array([x_position_map[v] for v in x_vals], dtype=float)
        ax.plot(
            x_positions,
            df["avg_total_repair_time_sec_success"],
            marker="o",
            linewidth=2,
            label=_display_name(approach),
            color=color_map.get(approach, COLORS.get(approach, None)),
        )

    # ax.set_title("Average Successful Repair Time", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Average repair time (sec)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_xticks(x_positions_base)
    ax.set_xticklabels([str(int(v)) for v in x_values])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=TICK_FONTSIZE)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return fig


model_name_map = {
    0: "mnist 9x100",
    1: "mnist_256x4",
    2: "mnist_256x6",
    3: "mnist_conv",
    4: "cifar10_cnn1",
    5: "cifar10_cnn2",
    6: "gtsrb_cnn",
}

experiment_map = {
    # 0: (0, "Local Counterexample", "nonzero", 1, 20, "center right", 0.0),
    # 1: (0, "Local Counterexample", "all", 1, 20, "center left", -0.17),
    # 2: (1, "Local Counterexample", "nonzero", 1, 20, "center right", 0.0),
    # 3: (1, "Local Counterexample", "all", 1, 20, "center left", -0.15),
    0: (0, "Local Robustness", "nonzero", 1, 20, "lower left", 0.0, ''),
    1: (0, "Local Robustness", "all", 1, 20, "lower left", 0.0, ''),
    2: (1, "Local Robustness", "nonzero", 1, 20, "lower left", 0.0, ''),
    3: (1, "Local Robustness", "all", 1, 20, "lower left", 0.0, ''),
    4: (4, "Corruption + Perturbation", "all", 1, 10, "upper right", 0.0, ''),
    5: (6, "Adversarial + Perturbation", "all", 1, 10, "upper right", 0.0, ''),
    6: (0, "Local Robustness", "nonzero", 1, 20, "lower left", 0.0, 'lastlayer,layer14,layer12,layer10'),
    7: (0, "Local Robustness", "all", 1, 20, "lower left", 0.0, 'lastlayer,layer14,layer12,layer10'),
    8: (1, "Local Robustness", "nonzero", 1, 20, "lower left", 0.0, 'lastlayer,layer7,layer5'),
    9: (1, "Local Robustness", "all", 1, 20, "lower left", 0.0, 'lastlayer,layer7,layer5'),
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison figures from repair CSV results.")
    parser.add_argument("--root", type=Path, default=default_repair_root(), help="repair_result root path")
    parser.add_argument("--exp_id", type=int, default=5)
    parser.add_argument("--ndims", type=int, default=None)
    parser.add_argument(
        "--ours-threshold",
        type=float,
        default=100,
        help="Use only merged OURS CSVs from merged_<threshold>pct (for example 1, 10, 20, 100).",
    )
    parser.add_argument(
        "--ours-merge-strategy",
        choices=["threshold", "greedy_best_accuracy"],
        # default="greedy_best_accuracy",
        default="threshold",
        help="Choose the merged OURS family to use when OURS is compared.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("result_analysis/figures"))
    parser.add_argument(
        "--series-labels",
        type=str,
        default=None,
        help="Comma-separated labels to plot, e.g. lastlayer,layer14,layer12,layer10.",
    )
    args = parser.parse_args()

    model_id, repair_task, perturbation, inp_eps, run_nums, legend_loc, legend_offset, series_label_str = experiment_map[args.exp_id]

    model_name = model_name_map.get(model_id, None)
    if args.series_labels is not None:
        series_labels = _parse_label_list(args.series_labels)
    elif series_label_str:
        series_labels = _parse_label_list(series_label_str)
    else:
        series_labels = None

    suffix = "_".join(series_labels) if series_labels is not None else "approaches"
    if args.ours_merge_strategy == "greedy_best_accuracy":
        suffix = f"{suffix}_oursmerged_greedy_best_accuracy"
    elif args.ours_threshold is not None:
        suffix = f"{suffix}_oursmerged{args.ours_threshold:g}pct"
    suffix = suffix.replace(" ", "_").replace("/", "_")
    model_dir = (model_name or "unknown_model").replace(" ", "_").replace("/", "_")
    task_dir = (repair_task or "unknown_task").replace(" ", "_").replace("/", "_")
    out_dir = args.out_dir / model_dir / task_dir
    fig1_path = out_dir / f"{perturbation}_{suffix}_figure1_success_accuracy.png"
    fig2_path = out_dir / f"{perturbation}_{suffix}_figure2_time.png"

    make_figure1_success_accuracy(
        root=args.root,
        model_name=model_name,
        repair_task=repair_task,
        perturbation=perturbation,
        inp_eps=inp_eps,
        ndims=args.ndims,
        ours_threshold=args.ours_threshold,
        ours_merge_strategy=args.ours_merge_strategy,
        series_labels=series_labels,
        run_nums=run_nums,
        legend_loc=legend_loc,
        legend_offset=legend_offset,
        save_path=fig1_path,
    )
    make_figure2_time(
        root=args.root,
        model_name=model_name,
        repair_task=repair_task,
        perturbation=perturbation,
        inp_eps=inp_eps,
        ndims=args.ndims,
        ours_threshold=args.ours_threshold,
        ours_merge_strategy=args.ours_merge_strategy,
        series_labels=series_labels,
        save_path=fig2_path,
    )

    print(f"Saved: {fig1_path}")
    print(f"Saved: {fig2_path}")



if __name__ == "__main__":
    main()
