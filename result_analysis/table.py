from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from result_analysis.analysis import DISPLAY_NAME_MAP, default_repair_root, load_all_csv_rows

GROUP_KEYS = ["model_name", "repair_task", "perturbation", "inp_eps", "buggy_points_num", "ndims"]
RUN_KEY = "run_index"
PAIRWISE_INSTANCE_KEY = "__pairwise_instance_id"

LATEX_DISPLAY_NAME_MAP = {
    "aprnn": "\\aprnn",
    "prorepair": "\\prorepair",
    "prepared": "\\prepared",
    "lastlayer": "\\lastlayerTool",
    "ours": "\\tool",
    "FL": "\\FLTool",
}


def _display_name(approach: str) -> str:
    return LATEX_DISPLAY_NAME_MAP.get(approach, approach)


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
    return value_norm.replace("and", "") == target_norm.replace("and", "")


def _label_display_name(label: str) -> str:
    if label in LATEX_DISPLAY_NAME_MAP:
        return LATEX_DISPLAY_NAME_MAP[label]
    return label


def _approaches_for_perturbation(perturbation: str, model_name: str | None = None) -> list[str]:
    if perturbation.lower() == "nonzero":
        return ["aprnn", "prorepair", "prepared", "lastlayer", "ours"]
    if model_name is not None and "mnist" in model_name.lower():
        return ["prorepair", "prepared", "lastlayer", "ours"]
    if model_name is not None and ("cifar" in model_name.lower() or "gtsrb" in model_name.lower()):
        return ["prepared", "lastlayer", "ours"]
    return ["aprnn", "prorepair", "prepared", "lastlayer", "ours"]


def _pairwise_approaches() -> list[str]:
    return ["prepared", "lastlayer", "ours"]


def _labels_for_table(
    perturbation: str,
    model_name: str | None = None,
    series_labels: list[str] | None = None,
) -> list[str]:
    if series_labels is not None:
        return series_labels
    return _approaches_for_perturbation(perturbation, model_name=model_name)


def _should_use_merged_ours(series_labels: list[str] | None) -> bool:
    if series_labels is None:
        return True
    return "ours" in series_labels


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _threshold_to_dir_label(threshold: float) -> str:
    return f"merged_{threshold:g}pct"


def _merge_strategy_dir_label(merge_strategy: str, threshold: float | None) -> str | None:
    if merge_strategy == "greedy_best_accuracy":
        return "merged_greedy_best_accuracy"
    if threshold is None:
        return None
    return _threshold_to_dir_label(threshold)


def _threshold_to_name_suffix(threshold: float | None) -> str:
    if threshold is None:
        return ""
    return f"_merged_{threshold:g}pct"


def _merge_strategy_name_suffix(merge_strategy: str, threshold: float | None) -> str:
    if merge_strategy == "greedy_best_accuracy":
        return "_merged_greedy_best_accuracy"
    return _threshold_to_name_suffix(threshold)


def _nested_output_dir(out_dir: Path, model_name: str | None, repair_task: str | None) -> Path:
    model_dir = (model_name or "unknown_model").replace(" ", "_").replace("/", "_")
    task_dir = (repair_task or "unknown_task").replace(" ", "_").replace("/", "_")
    return out_dir / model_dir / task_dir


def _nested_pairwise_output_dir(out_dir: Path, model_name: str | None, repair_task: str | None) -> Path:
    return out_dir / "pairwise" / _nested_output_dir(Path(), model_name, repair_task)


def _is_merged_ours_csv(source_csv: str, threshold: float | None, merge_strategy: str = "threshold") -> bool:
    csv_path = Path(source_csv)
    merged_dir = csv_path.parent
    ours_dir = csv_path.parent.parent

    if merged_dir is None or ours_dir is None:
        return False
    if not ours_dir.name.startswith("ours_"):
        return False

    match = re.fullmatch(r"merged_(?P<threshold>[^/]+)", merged_dir.name)
    if match is None:
        return False

    merged_label = _merge_strategy_dir_label(merge_strategy, threshold)
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
        keep_ours = ~runs.loc[is_ours, "source_csv"].map(
            lambda source_csv: _is_merged_ours_csv(str(source_csv), None, merge_strategy=merge_strategy)
        )
    else:
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

    This is only relevant when the comparison itself includes `ours`.
    """
    if ours_threshold is None or runs.empty:
        return runs
    if "approach" not in runs.columns or "source_csv" not in runs.columns:
        return runs
    if "successful_layer_label" not in runs.columns:
        return runs

    out = runs.copy()
    is_ours = out["approach"] == "ours"
    if not is_ours.any():
        return out

    merged_mask = out.loc[is_ours, "source_csv"].map(
        lambda source_csv: _is_merged_ours_csv(str(source_csv), ours_threshold, merge_strategy=merge_strategy)
    )
    idx = out.index[is_ours][merged_mask.fillna(False).to_numpy(dtype=bool)]
    if len(idx) == 0:
        return out

    out.loc[idx, "series_label"] = out.loc[idx, "successful_layer_label"].fillna(out.loc[idx, "series_label"])
    return out


def _filter_runs(
    runs: pd.DataFrame,
    model_name: str | None = None,
    repair_task: str | None = None,
    perturbation: str | None = None,
    inp_eps: str | None = None,
    ndims: int | None = None,
    buggy_points_nums: list[int] | None = None,
) -> pd.DataFrame:
    out = runs.copy()
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
    if buggy_points_nums is not None:
        out = out[out["buggy_points_num"].isin(buggy_points_nums)]
    return out


def make_pairwise_win_table(runs: pd.DataFrame, approaches: list[str]) -> pd.DataFrame:
    """
    Cell (a, b) = number of instances where approach a succeeded and approach b
    failed, counted only over instances where both approaches are present.

    This is a strict two-approach comparison for each pair (a, b). Each cell is
    computed independently from the others using the shared row order for the
    two approaches after filtering to the experiment slice.
    """
    required = ["approach", "result"]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    matrix = pd.DataFrame(0, index=approaches, columns=approaches, dtype=int)
    for a in approaches:
        for b in approaches:
            if a == b:
                continue

            pair = runs[required].copy()
            pair = pair[pair["approach"].isin([a, b])]
            if pair.empty:
                matrix.loc[a, b] = 0
                continue

            pair["is_success"] = pair["result"].astype(str).str.lower().eq("success")

            # Pair the rows in their filtered order. Each approach has one row per
            # shared run instance, so the row position acts as the common key.
            pair[PAIRWISE_INSTANCE_KEY] = pair.groupby("approach", sort=False).cumcount()
            status = (
                pair.pivot(index=PAIRWISE_INSTANCE_KEY, columns="approach", values="is_success")
                .reindex(columns=[a, b])
                .dropna()
            )
            if status.empty:
                matrix.loc[a, b] = 0
                continue

            matrix.loc[a, b] = int((status[a].astype(bool) & ~status[b].astype(bool)).sum())

    matrix.index = [_display_name(a) for a in matrix.index]
    matrix.columns = [_display_name(a) for a in matrix.columns]
    return matrix


def make_pairwise_win_table_by_label(
    runs: pd.DataFrame,
    labels: list[str],
    label_col: str = "series_label",
) -> pd.DataFrame:
    """
    Same pairwise-win table, but compare an arbitrary label column.

    This is used for start-layer comparisons where `series_label` contains values
    like `lastlayer`, `layer14`, `layer12`, ...
    """
    required = [label_col, "result"]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for a in labels:
        for b in labels:
            if a == b:
                continue

            pair = runs[required].copy()
            pair = pair[pair[label_col].isin([a, b])]
            if pair.empty:
                matrix.loc[a, b] = 0
                continue

            pair["is_success"] = pair["result"].astype(str).str.lower().eq("success")

            # Pair rows by their filtered order within each label.
            pair[PAIRWISE_INSTANCE_KEY] = pair.groupby(label_col, sort=False).cumcount()
            status = (
                pair.pivot(index=PAIRWISE_INSTANCE_KEY, columns=label_col, values="is_success")
                .reindex(columns=[a, b])
                .dropna()
            )
            if status.empty:
                matrix.loc[a, b] = 0
                continue

            matrix.loc[a, b] = int((status[a].astype(bool) & ~status[b].astype(bool)).sum())

    matrix.index = [_label_display_name(a) for a in matrix.index]
    matrix.columns = [_label_display_name(a) for a in matrix.columns]
    return matrix


def make_pairwise_relation_summary_table(
    runs: pd.DataFrame,
    labels: list[str],
    label_col: str = "series_label",
) -> pd.DataFrame:
    """
    Summarize pairwise layer comparisons with above/below/equal counts and ratios.

    This mirrors the annotation shown in the pairwise figure panels.
    """
    required = GROUP_KEYS + [label_col, "result", "repaired_accuracy"]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    base = runs[required].copy()
    base = base[base[label_col].isin(labels)]
    base["repaired_accuracy_success"] = base["repaired_accuracy"].where(
        base["result"].astype(str).str.lower().eq("success")
    )
    base["pair_id"] = base.groupby(label_col, sort=False).cumcount()

    pairwise = (
        base.pivot(index="pair_id", columns=label_col, values="repaired_accuracy_success")
        .reindex(columns=labels)
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for a, b in combinations(labels, 2):
        sub = pairwise[[a, b]].dropna(subset=[a, b]).copy()
        n_above = int((sub[b] > sub[a]).sum())
        n_below = int((sub[b] < sub[a]).sum())
        n_equal = int((sub[b] == sub[a]).sum())
        total = n_above + n_below + n_equal

        rows.append(
            {
                "x": _label_display_name(a),
                "y": _label_display_name(b),
                "x win": n_below,
                "y win": n_above,
                "equal": n_equal,
            }
        )

    return pd.DataFrame(rows)


def make_success_count_table(
    runs: pd.DataFrame,
    labels: list[str],
    x_col: str,
    label_col: str = "approach",
) -> pd.DataFrame:
    """
    Build a success-count table where rows are labels and columns are x-axis values.
    """
    required = [label_col, x_col, "result"]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    base = runs[required].copy()
    base = base[base[label_col].isin(labels)]
    base["is_success"] = base["result"].astype(str).str.lower().eq("success")

    counts = (
        base.pivot_table(
            index=label_col,
            columns=x_col,
            values="is_success",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=labels)
        .fillna(0)
        .astype(int)
    )

    try:
        counts.columns = pd.Index(sorted(counts.columns, key=lambda v: float(v)))
    except Exception:
        counts.columns = pd.Index(sorted(counts.columns, key=str))
    counts.index = [_label_display_name(a) for a in counts.index]
    counts.columns = [str(v) for v in counts.columns]
    return counts



model_name_map = {
    0: "mnist 9x100",
    1: "mnist_256x4",
    2: "mnist_256x6",
    3: "mnist_conv",
    4: "cifar10_cnn1",
    5: "cifar10_cnn2",
    6: "gtsrb_cnn",
}


def _x_axis_config(perturbation: str) -> tuple[str, str]:
    if perturbation.lower() == "nonzero":
        return "ndims", "#DIM"
    return "buggy_points_num", "#P"

experiment_map = {
    0: (0, "Local Robustness", "nonzero", 1, ''),
    1: (0, "Local Robustness", "all", 1, ''),
    2: (1, "Local Robustness", "nonzero", 1, ''),
    3: (1, "Local Robustness", "all", 1, ''),
    4: (4, "Corruption + Perturbation", "all", 1, ''),
    5: (6, "Adversarial + Perturbation", "all", 1, ''),
    6: (0, "Local Robustness", "nonzero", 1, 'lastlayer,layer14,layer12,layer10'),
    7: (0, "Local Robustness", "all", 1, 'lastlayer,layer14,layer12,layer10'),
    8: (1, "Local Robustness", "nonzero", 1, 'lastlayer,layer7,layer5,layer3'),
    9: (1, "Local Robustness", "all", 1, 'lastlayer,layer7,layer5,layer3'),
    10: (4, "Corruption + Perturbation", "all", 1, 'lastlayer,layer5,layer2'),
    11: (6, "Adversarial + Perturbation", "all", 1, 'lastlayer,layer5,layer2'),
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate success-count and pairwise tables from repair CSV results.")
    parser.add_argument("--root", type=Path, default=default_repair_root(), help="repair_result root path")
    parser.add_argument("--exp_id", type=int, default=3)
    parser.add_argument("--ndims", type=int, default=None)
    parser.add_argument("--buggy-points-nums", type=str, default=None, help="comma-separated N values, e.g. 1,2,3,4,5,6,8,10,12,14")
    parser.add_argument("--out-dir", type=Path, default=Path("result_analysis/tables"))
    parser.add_argument(
        "--ours-threshold",
        type=float,
        default=100,
        help="Use only merged OURS CSVs from merged_<threshold>pct (for example 1, 10, 20, 100).",
    )
    parser.add_argument(
        "--ours-merge-strategy",
        choices=["threshold", "greedy_best_accuracy"],
        default="threshold",
        # default="greedy_best_accuracy",
        help="Choose which merged OURS CSV family to use when OURS is compared.",
    )
    parser.add_argument(
        "--make-success-count",
        action="store_true",
        help="Deprecated alias. Success-count tables are generated by default.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Also generate the pairwise-win table.",
    )
    parser.add_argument(
        "--series-labels",
        type=str,
        default=None,
        help="Comma-separated labels to compare, e.g. lastlayer,layer14,layer12,layer10.",
    )
    args = parser.parse_args()

    model_id, repair_task, perturbation, inp_eps, series_label_str = experiment_map[args.exp_id]
    model_name = model_name_map.get(model_id)
    series_labels = _parse_label_list(args.series_labels) or _parse_label_list(series_label_str)
    buggy_points_nums = _parse_int_list(args.buggy_points_nums)

    runs = load_all_csv_rows(root=args.root)
    if runs.empty:
        raise ValueError("No CSV rows found.")

    use_merged_ours = _should_use_merged_ours(series_labels)
    if use_merged_ours:
        runs = _filter_ours_rows(runs, args.ours_threshold, merge_strategy=args.ours_merge_strategy)
        runs = _label_merged_ours_rows(runs, args.ours_threshold, merge_strategy=args.ours_merge_strategy)

    runs = _filter_runs(
        runs=runs,
        model_name=model_name,
        repair_task=repair_task,
        perturbation=perturbation,
        inp_eps=str(inp_eps),
        ndims=args.ndims,
        buggy_points_nums=buggy_points_nums,
    )
    if runs.empty:
        raise ValueError("No rows left after filters.")

    out_dir = _nested_output_dir(args.out_dir, model_name, repair_task)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairwise_out_dir = _nested_pairwise_output_dir(args.out_dir, model_name, repair_task)
    pairwise_out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{repair_task}_{perturbation}_eps{inp_eps}"
    if buggy_points_nums is not None:
        ns = "-".join(str(n) for n in buggy_points_nums)
        prefix += f"_N{ns}"
    if args.ndims is not None:
        prefix += f"_ndims{args.ndims}"
    if series_labels is not None:
        prefix += "_series" + "-".join(series_labels)
    prefix += _merge_strategy_name_suffix(args.ours_merge_strategy, args.ours_threshold if use_merged_ours else None)
    prefix = prefix.replace(" ", "_").replace("/", "_")

    x_col, x_label = _x_axis_config(perturbation)
    if series_labels is not None and "series_label" in runs.columns:
        count_labels = series_labels
        count_label_col = "series_label"
    else:
        count_labels = _approaches_for_perturbation(perturbation, model_name=model_name)
        count_label_col = "approach"

    success_counts = make_success_count_table(runs, count_labels, x_col=x_col, label_col=count_label_col)
    count_out_csv = out_dir / f"{prefix}_success_count.csv"
    count_out_tex = out_dir / f"{prefix}_success_count.tex"
    success_counts.to_csv(count_out_csv)
    count_out_tex.write_text(
    success_counts.to_latex(
            index=True,
            escape=False,
            caption=f"Success counts by approach and {x_label}.",
            label="tab:success_count",
        ),
        encoding="utf-8",
    )

    print(f"Saved: {count_out_csv}")
    print(f"Saved: {count_out_tex}")

    if series_labels is not None and "series_label" in runs.columns:
        relation_df = make_pairwise_relation_summary_table(runs, series_labels, label_col="series_label")
        relation_out_csv = pairwise_out_dir / f"{prefix}_pairwise_relation_summary.csv"
        relation_out_tex = pairwise_out_dir / f"{prefix}_pairwise_relation_summary.tex"
        relation_df.to_csv(relation_out_csv, index=False)
        relation_out_tex.write_text(
            relation_df.to_latex(
                index=False,
                escape=False,
                caption="Pairwise relation summary: above/below/equal counts and ratios for each layer pair.",
                label="tab:pairwise_relation_summary",
                float_format=lambda x: f"{x:.3f}",
            ),
            encoding="utf-8",
        )

        print(f"Saved: {relation_out_csv}")
        print(f"Saved: {relation_out_tex}")

    if args.pairwise:
        if series_labels is not None and "series_label" in runs.columns:
            pairwise = make_pairwise_win_table_by_label(runs, series_labels, label_col="series_label")
        else:
            pairwise = make_pairwise_win_table(runs, _pairwise_approaches())

        out_csv = pairwise_out_dir / f"{prefix}_pairwise_win.csv"
        out_tex = pairwise_out_dir / f"{prefix}_pairwise_win.tex"

        pairwise.to_csv(out_csv)
        out_tex.write_text(
            pairwise.to_latex(
                index=True,
                escape=False,
                caption="Pairwise wins: row succeeded while column failed (or missing), counted per run instance.",
                label="tab:pairwise_win",
            ),
            encoding="utf-8",
        )

        print(f"Saved: {out_csv}")
        print(f"Saved: {out_tex}")


def generate_tables_for_experiment(
    exp_id: int,
    root: Path | None = None,
    out_dir: Path | None = None,
    ndims: int | None = None,
    buggy_points_nums: list[int] | None = None,
    ours_threshold: float | None = None,
    ours_merge_strategy: str = "threshold",
    series_labels: list[str] | None = None,
    pairwise: bool = False,
) -> list[Path]:
    """
    Generate the default success-count table for one experiment.

    If `pairwise` is True, also write the pairwise-win table.
    """
    root = default_repair_root() if root is None else Path(root)
    base_out_dir = Path("result_analysis/tables") if out_dir is None else Path(out_dir)

    if exp_id not in experiment_map:
        raise ValueError(f"Unknown exp_id: {exp_id}")

    model_id, repair_task, perturbation, inp_eps, series_label_str = experiment_map[exp_id]
    model_name = model_name_map.get(model_id)
    series_labels = series_labels or _parse_label_list(series_label_str)

    runs = load_all_csv_rows(root=root)
    if runs.empty:
        raise ValueError("No CSV rows found.")

    use_merged_ours = _should_use_merged_ours(series_labels)
    if use_merged_ours:
        runs = _filter_ours_rows(runs, ours_threshold, merge_strategy=ours_merge_strategy)
        runs = _label_merged_ours_rows(runs, ours_threshold, merge_strategy=ours_merge_strategy)

    runs = _filter_runs(
        runs=runs,
        model_name=model_name,
        repair_task=repair_task,
        perturbation=perturbation,
        inp_eps=str(inp_eps),
        ndims=ndims,
        buggy_points_nums=buggy_points_nums,
    )
    if runs.empty:
        raise ValueError("No rows left after filters.")

    out_dir = _nested_output_dir(base_out_dir, model_name, repair_task)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_{repair_task}_{perturbation}_eps{inp_eps}"
    if buggy_points_nums is not None:
        ns = "-".join(str(n) for n in buggy_points_nums)
        prefix += f"_N{ns}"
    if ndims is not None:
        prefix += f"_ndims{ndims}"
    if series_labels is not None:
        prefix += "_series" + "-".join(series_labels)
    prefix += _merge_strategy_name_suffix(ours_merge_strategy, ours_threshold if use_merged_ours else None)
    prefix = prefix.replace(" ", "_").replace("/", "_")

    saved_paths: list[Path] = []

    x_col, x_label = _x_axis_config(perturbation)
    if series_labels is not None and "series_label" in runs.columns:
        count_labels = series_labels
        count_label_col = "series_label"
    else:
        count_labels = _approaches_for_perturbation(perturbation, model_name=model_name)
        count_label_col = "approach"

    success_counts = make_success_count_table(runs, count_labels, x_col=x_col, label_col=count_label_col)
    count_out_csv = out_dir / f"{prefix}_success_count.csv"
    count_out_tex = out_dir / f"{prefix}_success_count.tex"
    success_counts.to_csv(count_out_csv)
    count_out_tex.write_text(
    success_counts.to_latex(
            index=True,
            escape=False,
            caption=f"Success counts by approach and {x_label}.",
            label="tab:success_count",
        ),
        encoding="utf-8",
    )
    saved_paths.extend([count_out_csv, count_out_tex])

    relation_df = None
    if series_labels is not None and "series_label" in runs.columns:
        relation_df = make_pairwise_relation_summary_table(runs, series_labels, label_col="series_label")
        pairwise_out_dir = _nested_pairwise_output_dir(base_out_dir, model_name, repair_task)
        pairwise_out_dir.mkdir(parents=True, exist_ok=True)
        relation_out_csv = pairwise_out_dir / f"{prefix}_pairwise_relation_summary.csv"
        relation_out_tex = pairwise_out_dir / f"{prefix}_pairwise_relation_summary.tex"
        relation_df.to_csv(relation_out_csv, index=False)
        relation_out_tex.write_text(
            relation_df.to_latex(
                index=False,
                escape=False,
                caption="Pairwise relation summary: above/below/equal counts and ratios for each layer pair.",
                label="tab:pairwise_relation_summary",
                float_format=lambda x: f"{x:.3f}",
            ),
            encoding="utf-8",
        )
        saved_paths.extend([relation_out_csv, relation_out_tex])

    if pairwise:
        if series_labels is not None and "series_label" in runs.columns:
            pairwise_df = make_pairwise_win_table_by_label(runs, series_labels, label_col="series_label")
        else:
            pairwise_df = make_pairwise_win_table(runs, _pairwise_approaches())

        pairwise_out_dir = _nested_pairwise_output_dir(base_out_dir, model_name, repair_task)
        pairwise_out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = pairwise_out_dir / f"{prefix}_pairwise_win.csv"
        out_tex = pairwise_out_dir / f"{prefix}_pairwise_win.tex"
        pairwise_df.to_csv(out_csv)
        out_tex.write_text(
            pairwise_df.to_latex(
                index=True,
                escape=False,
                caption="Pairwise wins: row succeeded while column failed (or missing), counted per run instance.",
                label="tab:pairwise_win",
            ),
            encoding="utf-8",
        )
        saved_paths.extend([out_csv, out_tex])

    return saved_paths


def generate_all_tables(
    root: Path | None = None,
    out_dir: Path | None = None,
    ndims: int | None = None,
    buggy_points_nums: list[int] | None = None,
    ours_threshold: float | None = None,
    ours_merge_strategy: str = "threshold",
    series_labels_by_exp: dict[int, list[str]] | None = None,
    pairwise: bool = False,
) -> list[Path]:
    """
    Generate tables for every configured experiment.

    This is the no-CLI batch entrypoint for producing all default tables.
    """
    saved: list[Path] = []
    for exp_id in sorted(experiment_map):
        exp_series_labels = None
        if series_labels_by_exp is not None:
            exp_series_labels = series_labels_by_exp.get(exp_id)
        saved.extend(
            generate_tables_for_experiment(
                exp_id=exp_id,
                root=root,
                out_dir=out_dir,
                ndims=ndims,
                buggy_points_nums=buggy_points_nums,
                ours_threshold=ours_threshold,
                ours_merge_strategy=ours_merge_strategy,
                series_labels=exp_series_labels,
                pairwise=pairwise,
            )
        )
    return saved


if __name__ == "__main__":
    main()
