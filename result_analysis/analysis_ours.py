from __future__ import annotations

'''
keep comments for future reference.

1.
how performance varies according to the different start layers
- success number
- failure number
- accuracy (drop)
- time
metrics above can be analyzed by the figure in figure.py. so can you produce figures by using figure.py?


2.
instance-wise comparison
- for each instance, which layer's repair is successful and what is the repaired accuracy
this can be analyzed by table. so can you produce a table for instance-wise comparison?


3.
merge our approaches with different start layers into one
assumptions:
    - we run our approach starting with last layer first, and then move to the earlier layer. if repair is successful at a certain layer, we do not try earlier layers.
    - run order example: lastlayer -> layer3 -> layer2 -> layer1 -> layer0 (descending order)
    - lastlayer is a part of our approach, and it is equivalent to starting with the last layer. so we can directly use the results from lastlayer as a first layer to try in our approach.
merge policy:
    given: accuracy drop thresholds
    on each performance for current layer,
        1. we first check if repair is successful. if not we continue to the next layer.
        2. if successful, we check if repaired accuracy drop is within the threshold. if not, we continue to the next layer. if yes, we keep the performance and do not try earlier layers.
        3. if we cannot find any successful repair within the threshold, we keep the performance with the smallest accuracy drop among all layers.
        4. repair time is the sum of repair time across all layers we tried until we find the successful repair within the threshold, or we tried all layers.
can you make a function that merges the performance of different start layers into one according to the policy above for a given accuracy drop threshold?
by the function above, can you produce a csv across different thresholds (e.g., 1%, 2%, 5%, 10%, 15%, 20%, 30%, 40%, 50%)? the csv should contain the same columns as the one for each start layer, but with an additional column for the accuracy drop threshold.
csv file location can be "csv/ours_{model_name}_{repair_task}_{threshold}/merged_{accuracy_drop_threshold}/eps{eps}_N{num_polys}.csv".
'''

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "provable-repair-matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from result_analysis.analysis import default_repair_root, load_all_csv_rows, parse_result_filename
from result_analysis.figure import COLORS, DISPLAY_NAME_MAP, LABEL_FONTSIZE, TITLE_FONTSIZE, TICK_FONTSIZE

GROUP_KEYS = ["model_name", "repair_task", "perturbation", "inp_eps", "buggy_points_num", "ndims"]
RUN_KEY = "run_index"
INSTANCE_KEYS = GROUP_KEYS + [RUN_KEY]

OURS_DIR_RE = re.compile(r"^ours_(?P<rest>.+)$")
LASTLAYER_DIR_RE = re.compile(r"^lastlayer_(?P<rest>.+)$")
LAYER_DIR_RE = re.compile(r"^layer(?P<layer>\d+)$")
DEFAULT_THRESHOLDS = (1, 2, 5, 10, 15, 20, 30, 40, 50, 100)
OURS_LAYER_COLOR = COLORS.get("ours", "#2FA32F")
LASTLAYER_COLOR = COLORS.get("lastlayer", "#E45756")


def _find_ancestor_dir(path: Path, prefix: str) -> Path | None:
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent
    return None


def _parse_approach_dirname(dirname: str) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Parse directory names like:
      ours_mnist 9x100_LocalRobustness_all
      lastlayer_mnist 9x100_LocalRobustness_all

    Returns:
      (approach, model_name, repair_task, perturbation)
    """
    if dirname.startswith("ours_"):
        approach = "ours"
        rest = dirname[len("ours_") :]
    elif dirname.startswith("lastlayer_"):
        approach = "lastlayer"
        rest = dirname[len("lastlayer_") :]
    else:
        return None, None, None, None

    if "_" not in rest:
        return approach, None, None, None

    prefix, perturbation = rest.rsplit("_", 1)
    if "_" not in prefix:
        return approach, None, None, perturbation

    model_name, repair_task = prefix.rsplit("_", 1)
    return approach, model_name, repair_task, perturbation


def _infer_start_layer(csv_path: Path, df: pd.DataFrame) -> int | None:
    for parent in csv_path.parents:
        match = LAYER_DIR_RE.fullmatch(parent.name)
        if match:
            return int(match.group("layer"))

    if "first_layer_to_repair" in df.columns:
        values = df["first_layer_to_repair"].dropna().unique()
        if len(values) == 1:
            return int(values[0])

    return None


def _numeric_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return float(value)


def _accuracy_drop(row: pd.Series | dict) -> float:
    original = _numeric_or_none(row.get("original_accuracy"))
    repaired = _numeric_or_none(row.get("repaired_accuracy"))
    if original is None or repaired is None:
        return float("inf")
    return original - repaired


def _threshold_to_fraction(threshold: float) -> float:
    # The requested thresholds are usually expressed as percentages
    # (for example 1, 2, 5, 10). Internally we compare them against the
    # accuracy drop stored in [0, 1] space.
    return threshold / 100.0 if threshold > 1 else threshold


def _threshold_label(threshold: float) -> str:
    return f"{threshold:g}pct"


def _candidate_sort_key(row: pd.Series | dict) -> tuple[int, int]:
    layer_label = row.get("start_layer_label")
    start_layer = row.get("start_layer")
    is_lastlayer = 1 if str(layer_label) == "lastlayer" else 0
    layer_value = -1 if start_layer is None or pd.isna(start_layer) else int(start_layer)
    return is_lastlayer, layer_value


def _start_layer_sort_key(label: object) -> tuple[int, int]:
    text = "" if label is None or pd.isna(label) else str(label)
    if text == "lastlayer":
        return (0, 0)
    match = re.fullmatch(r"layer(\d+)", text)
    if match:
        return (1, -int(match.group(1)))
    return (2, 0)


def _start_layer_order(labels: Iterable[object]) -> list[str]:
    cleaned = {str(label) for label in labels if label is not None and not pd.isna(label)}
    return sorted(cleaned, key=_start_layer_sort_key)


def _start_layer_display_label(label: object) -> str:
    text = "" if label is None or pd.isna(label) else str(label)
    if text == "lastlayer":
        return "lastlayer"
    return text


def _load_csv_with_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    approach_dir = _find_ancestor_dir(csv_path, "ours_") or _find_ancestor_dir(csv_path, "lastlayer_")
    if approach_dir is None:
        return pd.DataFrame()

    approach, model_name, repair_task, perturbation = _parse_approach_dirname(approach_dir.name)
    if approach not in {"ours", "lastlayer"}:
        return pd.DataFrame()

    out = df.copy()
    out["source_csv"] = str(csv_path)
    out["approach"] = approach
    out["model_name"] = model_name
    out["repair_task"] = repair_task
    out["perturbation"] = perturbation
    out["start_layer"] = _infer_start_layer(csv_path, out)
    out["start_layer_label"] = "lastlayer" if approach == "lastlayer" else (
        f"layer{int(out['start_layer'].dropna().iloc[0])}" if "start_layer" in out.columns and out["start_layer"].notna().any() else None
    )

    # Ensure we have a consistent numeric input eps column for filtering/grouping.
    file_meta = parse_result_filename(csv_path)
    if "inp_eps" not in out.columns:
        out["inp_eps"] = file_meta["inp_eps"] if file_meta["inp_eps"] is not None else out.get("eps")
    if "buggy_points_num" not in out.columns:
        out["buggy_points_num"] = file_meta["buggy_points_num"]
    if "ndims" not in out.columns:
        out["ndims"] = file_meta["ndims"]

    return out


def load_ours_csv_rows(root: Path | None = None) -> pd.DataFrame:
    """Load OURS and LastLayer rows from either canonical or legacy CSV paths."""
    root = default_repair_root() if root is None else Path(root)
    rows = load_all_csv_rows(root=root)
    if rows.empty:
        return rows
    rows = rows[rows["approach"].isin(["ours", "lastlayer"])].copy()
    rows.loc[rows["approach"] == "lastlayer", "start_layer_label"] = "lastlayer"
    ours = (rows["approach"] == "ours") & rows["start_layer"].notna()
    rows.loc[ours, "start_layer_label"] = rows.loc[ours, "start_layer"].astype(int).map(
        lambda layer: f"layer{layer}"
    )
    return rows


def load_start_layer_rows(root: Path | None = None) -> pd.DataFrame:
    """Load only the start-layer rows used for OURS layer-wise analysis."""
    runs = load_ours_csv_rows(root=root)
    if runs.empty:
        return runs
    return runs[runs["approach"].isin(["ours", "lastlayer"])].copy()


def _filter_runs(
    runs: pd.DataFrame,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
) -> pd.DataFrame:
    out = runs.copy()
    if model_name is not None:
        out = out[out["model_name"] == model_name]
    if repair_task is not None:
        out = out[out["repair_task"] == repair_task]
    if perturbation is not None:
        out = out[out["perturbation"] == perturbation]
    if inp_eps is not None:
        out = out[out["inp_eps"].astype(str) == str(inp_eps)]
    if ndims is not None:
        out = out[out["ndims"] == ndims]
    return out


def _filter_start_layer_runs(
    runs: pd.DataFrame,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
) -> pd.DataFrame:
    out = _filter_runs(runs, model_name=model_name, repair_task=repair_task, perturbation=perturbation, inp_eps=inp_eps, ndims=ndims)
    if out.empty:
        return out
    return out[out["approach"].isin(["ours", "lastlayer"])].copy()


def _layer_frame_label(group: pd.DataFrame) -> str:
    label = group["start_layer_label"].dropna().iloc[0] if group["start_layer_label"].notna().any() else None
    if label is not None:
        return str(label)
    start_layer = group["start_layer"].dropna().iloc[0] if group["start_layer"].notna().any() else None
    return f"layer{int(start_layer)}" if start_layer is not None else "unknown"


def _success_failure_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in runs.groupby("start_layer_label", dropna=False):
        total_num = len(group)
        success_num = int((group["result"] == "success").sum())
        failure_num = total_num - success_num
        success_df = group[group["result"] == "success"]
        rows.append(
            {
                "start_layer_label": _start_layer_display_label(label),
                "total_num": total_num,
                "success_num": success_num,
                "failure_num": failure_num,
                "success_rate": (success_num / total_num) if total_num else 0.0,
                "avg_repaired_accuracy_success": success_df["repaired_accuracy"].mean() if success_num else np.nan,
                "avg_accuracy_drop_success": (
                    (success_df["original_accuracy"] - success_df["repaired_accuracy"]).mean() if success_num else np.nan
                ),
                "avg_total_repair_time_sec_success": success_df["total_repair_time_sec"].mean() if success_num else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["__sort_key"] = out["start_layer_label"].map(_start_layer_sort_key)
    return out.sort_values("__sort_key").drop(columns=["__sort_key"]).reset_index(drop=True)


def make_start_layer_figure_success_failure(
    root: Path | None = None,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot success/failure counts for each start layer."""
    runs = load_start_layer_rows(root=root)
    if runs.empty:
        raise ValueError("No CSV rows found. Generate CSVs first.")

    runs = _filter_start_layer_runs(runs, model_name, repair_task, perturbation, inp_eps, ndims)
    if runs.empty:
        raise ValueError("No rows left after filters.")

    summary = _success_failure_summary(runs)
    if summary.empty:
        raise ValueError("No start-layer summary data found.")

    labels = summary["start_layer_label"].tolist()
    x = np.arange(len(labels))
    success_vals = summary["success_num"].to_numpy()
    failure_vals = summary["failure_num"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    ax.bar(x, success_vals, color=OURS_LAYER_COLOR, label="success")
    ax.bar(x, failure_vals, bottom=success_vals, color="#d9d9d9", label="failure")

    for idx, (s, f) in enumerate(zip(success_vals, failure_vals, strict=False)):
        ax.text(idx, s / 2 if s else 0.05, str(int(s)), ha="center", va="center", color="white", fontsize=TICK_FONTSIZE)
        if f:
            ax.text(idx, s + f / 2, str(int(f)), ha="center", va="center", color="#444444", fontsize=TICK_FONTSIZE)

    ax.set_title("Success and Failure Count by Start Layer", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Start layer", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Run count", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=TICK_FONTSIZE)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def make_start_layer_figure_accuracy_drop(
    root: Path | None = None,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot repaired accuracy drop for each start layer."""
    runs = load_start_layer_rows(root=root)
    if runs.empty:
        raise ValueError("No CSV rows found. Generate CSVs first.")

    runs = _filter_start_layer_runs(runs, model_name, repair_task, perturbation, inp_eps, ndims)
    if runs.empty:
        raise ValueError("No rows left after filters.")

    runs = runs[runs["result"] == "success"].copy()
    if runs.empty:
        raise ValueError("No successful runs found for accuracy-drop plotting.")

    runs["accuracy_drop"] = runs["original_accuracy"] - runs["repaired_accuracy"]
    summary = _success_failure_summary(runs)
    if summary.empty:
        raise ValueError("No start-layer summary data found.")

    labels = summary["start_layer_label"].tolist()
    x = np.arange(len(labels))
    values = [runs.loc[runs["start_layer_label"] == label, "accuracy_drop"].dropna().to_numpy() for label in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    box = ax.boxplot(values, positions=x, widths=0.5, patch_artist=True, showfliers=False, manage_ticks=False)
    for patch in box["boxes"]:
        patch.set_facecolor(OURS_LAYER_COLOR)
        patch.set_alpha(0.35)
        patch.set_edgecolor(OURS_LAYER_COLOR)
    for element in ["whiskers", "caps", "medians"]:
        for artist in box[element]:
            artist.set_color(OURS_LAYER_COLOR)

    means = [np.mean(v) if len(v) else np.nan for v in values]
    ax.plot(x, means, "o", color=OURS_LAYER_COLOR, label="mean drop")

    ax.set_title("Accuracy Drop by Start Layer", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Start layer", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy drop", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=TICK_FONTSIZE)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def make_start_layer_figure_time(
    root: Path | None = None,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot average successful repair time by start layer."""
    runs = load_start_layer_rows(root=root)
    if runs.empty:
        raise ValueError("No CSV rows found. Generate CSVs first.")

    runs = _filter_start_layer_runs(runs, model_name, repair_task, perturbation, inp_eps, ndims)
    if runs.empty:
        raise ValueError("No rows left after filters.")

    summary = _success_failure_summary(runs)
    if summary.empty:
        raise ValueError("No start-layer summary data found.")

    labels = summary["start_layer_label"].tolist()
    x = np.arange(len(labels))
    y = summary["avg_total_repair_time_sec_success"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    ax.plot(x, y, marker="o", linewidth=2, color=OURS_LAYER_COLOR, label="avg success time")

    ax.set_title("Average Successful Repair Time by Start Layer", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Start layer", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Repair time (sec)", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=TICK_FONTSIZE)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def make_start_layer_figures(
    root: Path | None = None,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, plt.Figure]:
    """Generate all start-layer figures and optionally save them."""
    out_dir = Path(out_dir) if out_dir is not None else None
    figures = {
        "success_failure": make_start_layer_figure_success_failure(
            root=root,
            model_name=model_name,
            repair_task=repair_task,
            perturbation=perturbation,
            inp_eps=inp_eps,
            ndims=ndims,
            save_path=(out_dir / "start_layer_success_failure.png") if out_dir is not None else None,
        ),
        "accuracy_drop": make_start_layer_figure_accuracy_drop(
            root=root,
            model_name=model_name,
            repair_task=repair_task,
            perturbation=perturbation,
            inp_eps=inp_eps,
            ndims=ndims,
            save_path=(out_dir / "start_layer_accuracy_drop.png") if out_dir is not None else None,
        ),
        "time": make_start_layer_figure_time(
            root=root,
            model_name=model_name,
            repair_task=repair_task,
            perturbation=perturbation,
            inp_eps=inp_eps,
            ndims=ndims,
            save_path=(out_dir / "start_layer_time.png") if out_dir is not None else None,
        ),
    }
    return figures


def make_start_layer_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance for each start layer."""
    required = INSTANCE_KEYS + ["approach", "result", "total_repair_time_sec", "total_steps", "original_accuracy", "repaired_accuracy"]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for (model_name, repair_task, perturbation, inp_eps, buggy_points_num, ndims, approach, start_layer_label), group in runs.groupby(
        ["model_name", "repair_task", "perturbation", "inp_eps", "buggy_points_num", "ndims", "approach", "start_layer_label"],
        dropna=False,
    ):
        success_df = group[group["result"] == "success"]
        total_num = len(group)
        success_num = len(success_df)
        rows.append(
            {
                "model_name": model_name,
                "repair_task": repair_task,
                "perturbation": perturbation,
                "inp_eps": inp_eps,
                "buggy_points_num": buggy_points_num,
                "ndims": ndims,
                "approach": approach,
                "start_layer_label": start_layer_label,
                "total_num": int(total_num),
                "success_num": int(success_num),
                "success_rate": (success_num / total_num) if total_num else 0.0,
                "avg_total_repair_time_sec_success": success_df["total_repair_time_sec"].mean() if success_num else None,
                "avg_total_steps_success": success_df["total_steps"].mean() if success_num else None,
                "original_accuracy": group["original_accuracy"].dropna().iloc[0] if group["original_accuracy"].notna().any() else None,
                "avg_repaired_accuracy_success": success_df["repaired_accuracy"].mean() if success_num else None,
            }
        )

    return pd.DataFrame(rows).sort_values(GROUP_KEYS + ["approach", "start_layer_label"]).reset_index(drop=True)


def make_instance_wise_table(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table that compares each run instance across layers.

    One row corresponds to one instance key:
      (model_name, repair_task, perturbation, inp_eps, buggy_points_num, ndims, run_index)
    """
    required = INSTANCE_KEYS + ["approach", "start_layer_label", "result", "repaired_accuracy"]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    base = runs[required].copy()
    base["status"] = base["result"].astype(str).str.lower()

    rows: list[dict[str, object]] = []
    for instance_key, group in base.groupby(INSTANCE_KEYS, dropna=False):
        row: dict[str, object] = {
            "model_name": instance_key[0],
            "repair_task": instance_key[1],
            "perturbation": instance_key[2],
            "inp_eps": instance_key[3],
            "buggy_points_num": instance_key[4],
            "ndims": instance_key[5],
            "run_index": instance_key[6],
        }
        for _, item in group.iterrows():
            label = str(item["start_layer_label"]) if pd.notna(item["start_layer_label"]) else "unknown"
            row[f"result_{label}"] = item["status"]
            row[f"repaired_accuracy_{label}"] = item["repaired_accuracy"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(GROUP_KEYS + [RUN_KEY]).reset_index(drop=True)


def _merge_instance_for_threshold(group: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Merge multiple start layers for one run instance according to the policy:
      - check layers in descending order, with lastlayer first
      - stop at the first successful repair within threshold
      - if none satisfy threshold, keep the row with the smallest accuracy drop
      - repair time is summed over all tried layers
    """
    candidates = sorted(group.to_dict("records"), key=_candidate_sort_key, reverse=True)
    threshold_frac = _threshold_to_fraction(threshold)

    cumulative_time = 0.0
    cumulative_steps = 0.0
    chosen_row: dict[str, object] | None = None
    chosen_time: float | None = None
    chosen_steps: float | None = None
    chosen_success_layer_label: str | None = None
    chosen_success_layer: int | None = None
    fallback_row: dict[str, object] | None = None
    fallback_drop = float("inf")

    for row in candidates:
        time_val = _numeric_or_none(row.get("total_repair_time_sec")) or 0.0
        step_val = _numeric_or_none(row.get("total_steps")) or 0.0
        cumulative_time += time_val
        cumulative_steps += step_val

        drop = _accuracy_drop(row)
        if drop < fallback_drop:
            fallback_drop = drop
            fallback_row = row

        if str(row.get("result")).lower() == "success" and drop <= threshold_frac:
            chosen_row = row
            chosen_time = cumulative_time
            chosen_steps = cumulative_steps
            chosen_success_layer_label = str(row.get("start_layer_label")) if row.get("start_layer_label") is not None else None
            chosen_success_layer = int(row["start_layer"]) if row.get("start_layer") is not None and not pd.isna(row.get("start_layer")) else None
            break

    if chosen_row is None:
        chosen_row = fallback_row if fallback_row is not None else candidates[0]
        chosen_time = cumulative_time
        chosen_steps = cumulative_steps
        if str(chosen_row.get("result")).lower() == "success":
            chosen_success_layer_label = str(chosen_row.get("start_layer_label")) if chosen_row.get("start_layer_label") is not None else None
            chosen_success_layer = int(chosen_row["start_layer"]) if chosen_row.get("start_layer") is not None and not pd.isna(chosen_row.get("start_layer")) else None

    out = dict(chosen_row)
    out["accuracy_drop_threshold"] = threshold_frac
    out["total_repair_time_sec"] = chosen_time
    if "total_steps" in out:
        out["total_steps"] = int(chosen_steps) if chosen_steps is not None else None
    out["successful_layer_label"] = chosen_success_layer_label
    out["successful_layer"] = chosen_success_layer
    return pd.Series(out)


def _merge_instance_greedy_best_accuracy(group: pd.DataFrame) -> pd.Series:
    """
    Merge multiple start layers for one run instance by choosing the row with
    the best repaired accuracy among successful repairs.

    If no successful repair exists, fall back to the row with the best repaired
    accuracy among all candidates.

    Ties are broken by the same layer ordering used elsewhere, with lastlayer
    preferred over earlier layers.
    """
    candidates = sorted(group.to_dict("records"), key=_candidate_sort_key, reverse=True)

    success_candidates: list[dict[str, object]] = []
    fallback_row: dict[str, object] | None = None
    fallback_repaired_accuracy = float("-inf")
    fallback_drop = float("inf")
    cumulative_time = 0.0
    cumulative_steps = 0.0
    best_success_layer_label: str | None = None
    best_success_layer: int | None = None

    for row in candidates:
        time_val = _numeric_or_none(row.get("total_repair_time_sec")) or 0.0
        step_val = _numeric_or_none(row.get("total_steps")) or 0.0
        cumulative_time += time_val
        cumulative_steps += step_val

        repaired_accuracy = _numeric_or_none(row.get("repaired_accuracy"))
        drop = _accuracy_drop(row)

        if repaired_accuracy is None:
            continue

        if str(row.get("result")).lower() == "success":
            success_candidates.append(row)
        else:
            is_better = repaired_accuracy > fallback_repaired_accuracy
            if not is_better and repaired_accuracy == fallback_repaired_accuracy:
                is_better = drop < fallback_drop

            if is_better:
                fallback_row = row
                fallback_repaired_accuracy = repaired_accuracy
                fallback_drop = drop

    if success_candidates:
        best_row = None
        best_repaired_accuracy = float("-inf")
        best_drop = float("inf")
        for row in success_candidates:
            repaired_accuracy = _numeric_or_none(row.get("repaired_accuracy"))
            if repaired_accuracy is None:
                continue
            drop = _accuracy_drop(row)
            is_better = repaired_accuracy > best_repaired_accuracy
            if not is_better and repaired_accuracy == best_repaired_accuracy:
                is_better = drop < best_drop
            if is_better:
                best_row = row
                best_repaired_accuracy = repaired_accuracy
                best_drop = drop
                best_success_layer_label = str(row.get("start_layer_label")) if row.get("start_layer_label") is not None else None
                best_success_layer = int(row["start_layer"]) if row.get("start_layer") is not None and not pd.isna(row.get("start_layer")) else None
    else:
        best_row = fallback_row if fallback_row is not None else candidates[0]
        if str(best_row.get("result")).lower() == "success":
            best_success_layer_label = str(best_row.get("start_layer_label")) if best_row.get("start_layer_label") is not None else None
            best_success_layer = int(best_row["start_layer"]) if best_row.get("start_layer") is not None and not pd.isna(best_row.get("start_layer")) else None
        else:
            best_success_layer_label = None
            best_success_layer = None

    out = dict(best_row)
    out["merge_strategy"] = "greedy_best_accuracy"
    out["accuracy_drop_threshold"] = None
    out["total_repair_time_sec"] = cumulative_time
    if "total_steps" in out:
        out["total_steps"] = int(cumulative_steps)
    out["successful_layer_label"] = best_success_layer_label
    out["successful_layer"] = best_success_layer
    return pd.Series(out)


def merge_ours_layers(
    runs: pd.DataFrame,
    accuracy_drop_threshold: float | None = None,
    merge_strategy: str = "threshold",
) -> pd.DataFrame:
    """Merge all rows into one per instance using the requested strategy."""
    required = INSTANCE_KEYS + ["approach", "start_layer_label", "result", "total_repair_time_sec", "total_steps", "original_accuracy", "repaired_accuracy"]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows: list[pd.Series] = []
    if merge_strategy not in {"threshold", "greedy_best_accuracy"}:
        raise ValueError(f"Unknown merge_strategy: {merge_strategy}")

    for _, group in runs.groupby(INSTANCE_KEYS, dropna=False):
        if merge_strategy == "threshold":
            if accuracy_drop_threshold is None:
                raise ValueError("accuracy_drop_threshold is required for threshold merge strategy")
            rows.append(_merge_instance_for_threshold(group, accuracy_drop_threshold))
        else:
            rows.append(_merge_instance_greedy_best_accuracy(group))

    if not rows:
        return pd.DataFrame()

    merged = pd.DataFrame(rows)
    merged = merged.sort_values(INSTANCE_KEYS).reset_index(drop=True)
    return merged


def _build_merged_output_path(
    row: pd.Series,
    threshold: float | None,
    output_root: Path,
    merge_strategy: str = "threshold",
) -> Path:
    model_name = str(row["model_name"])
    repair_task = str(row["repair_task"])
    perturbation = str(row["perturbation"])
    if merge_strategy == "greedy_best_accuracy":
        threshold_dir = "greedy_best_accuracy"
    else:
        if threshold is None:
            raise ValueError("threshold is required for threshold merge strategy")
        threshold_dir = _threshold_label(threshold)
    source_csv = Path(str(row["source_csv"]))
    stem = source_csv.stem

    out_dir = output_root / f"ours_{model_name}_{repair_task}_{perturbation}" / f"merged_{threshold_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.csv"


def write_merged_csvs(
    root: Path | None = None,
    output_root: Path | None = None,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    merge_strategy: str = "threshold",
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
) -> list[Path]:
    """Generate merged CSVs for each threshold and return the written paths."""
    root = default_repair_root() if root is None else Path(root)
    output_root = (root / "csv") if output_root is None else Path(output_root)

    runs = load_ours_csv_rows(root=root)
    if runs.empty:
        return []

    runs = _filter_runs(runs, model_name=model_name, repair_task=repair_task, perturbation=perturbation, inp_eps=inp_eps, ndims=ndims)
    if runs.empty:
        return []

    written: list[Path] = []
    experiment_keys = GROUP_KEYS

    if merge_strategy == "threshold":
        for threshold in thresholds:
            merged = merge_ours_layers(runs, accuracy_drop_threshold=threshold, merge_strategy=merge_strategy)
            if merged.empty:
                continue

            for _, group in merged.groupby(experiment_keys, dropna=False):
                out_path = _build_merged_output_path(group.iloc[0], threshold, output_root, merge_strategy=merge_strategy)
                group.to_csv(out_path, index=False)
                written.append(out_path)
    elif merge_strategy == "greedy_best_accuracy":
        merged = merge_ours_layers(runs, merge_strategy=merge_strategy)
        if merged.empty:
            return written

        for _, group in merged.groupby(experiment_keys, dropna=False):
            out_path = _build_merged_output_path(group.iloc[0], None, output_root, merge_strategy=merge_strategy)
            group.to_csv(out_path, index=False)
            written.append(out_path)
    else:
        raise ValueError(f"Unknown merge_strategy: {merge_strategy}")

    return written


def build_threshold_summaries(
    root: Path | None = None,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Return a merged DataFrame for each default threshold."""
    runs = load_ours_csv_rows(root=root)
    if runs.empty:
        return {}

    runs = _filter_runs(runs, model_name=model_name, repair_task=repair_task, perturbation=perturbation, inp_eps=inp_eps, ndims=ndims)
    if runs.empty:
        return {}

    return {str(threshold): merge_ours_layers(runs, threshold) for threshold in DEFAULT_THRESHOLDS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OURS and LastLayer CSV results.")
    parser.add_argument("--root", type=Path, default=default_repair_root(), help="repair_result root path")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output root for merged CSVs; defaults to repair_result/csv")
    parser.add_argument("--make-figures", action="store_true", help="Generate start-layer figures as well.")
    parser.add_argument(
        "--figure-out-dir",
        type=Path,
        default=Path("result_analysis/figures"),
        help="Directory where start-layer figures will be saved.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--repair-task", type=str, default=None)
    parser.add_argument("--perturbation", type=str, default=None)
    parser.add_argument("--inp-eps", type=str, default=None)
    parser.add_argument("--ndims", type=int, default=None)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="1,10,20,40,50,100",
        # default="1,2,5,10,15,20,30,40,50,100",
        help="Comma-separated accuracy-drop thresholds in percent.",
    )
    parser.add_argument(
        "--merge-strategy",
        choices=["threshold", "greedy_best_accuracy", "both"],
        default="threshold",
        # default="greedy_best_accuracy",
        help="Choose threshold merge, greedy best-accuracy merge, or both.",
    )
    args = parser.parse_args()

    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    written: list[Path] = []
    strategies = [args.merge_strategy] if args.merge_strategy != "both" else ["threshold", "greedy_best_accuracy"]
    for strategy in strategies:
        written.extend(
            write_merged_csvs(
                root=args.root,
                output_root=args.out_dir,
                thresholds=thresholds,
                merge_strategy=strategy,
                model_name=args.model_name,
                repair_task=args.repair_task,
                perturbation=args.perturbation,
                inp_eps=args.inp_eps,
                ndims=args.ndims,
            )
        )

    print(f"Wrote {len(written)} merged CSV files.")

    if args.make_figures:
        figures = make_start_layer_figures(
            root=args.root,
            model_name=args.model_name,
            repair_task=args.repair_task,
            perturbation=args.perturbation,
            inp_eps=args.inp_eps,
            ndims=args.ndims,
            out_dir=args.figure_out_dir,
        )
        print(f"Wrote {len(figures)} start-layer figures to {args.figure_out_dir}.")


if __name__ == "__main__":
    main()
