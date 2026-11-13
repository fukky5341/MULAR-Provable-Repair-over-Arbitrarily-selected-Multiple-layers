from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    from .policy_merge import replace_layerwise_comparison_rows
except ImportError:  # Direct script execution.
    from policy_merge import replace_layerwise_comparison_rows

APPROACHES = ("aprnn", "prorepair", "prepared", "lastlayer", "ours", "FL")
APPROACHES_ALL = ("aprnn", "prorepair", "prepared", "lastlayer", "ours", "FL")
# Parse the canonical prefix and ignore any trailing method-specific suffix
# (e.g. "_ARACHNE0.01_pr3").
FILENAME_RE = re.compile(r"eps(?P<eps>[^_]+)(?:_nonzero)?_N(?P<n>\d+)(?:_ndims(?P<ndims>\d+))?(?:_.+)?$")

DISPLAY_NAME_MAP = {
    "aprnn": "APRNN",
    "prorepair": "ProRepair",
    "prepared": "PREPARED",
    "lastlayer": "LLR",
    "ours": "MULAR",
    "FL": "FL"
}

def default_repair_root() -> Path:
    """Return repair_result dir from this repository layout."""
    return Path(__file__).resolve().parents[1] / "repair_result"


def collect_csv_paths(root: Path | None = None, approach: str | None = None) -> list[Path]:
    """Collect CSV files under repair_result/csv."""
    root = default_repair_root() if root is None else Path(root)
    csv_root = root / "csv"
    if not csv_root.exists():
        return []

    candidates = sorted(p for p in csv_root.rglob("*.csv") if p.is_file())
    if approach is None:
        return candidates

    prefix = f"{approach}_"
    return [
        p for p in candidates
        if any(parent.name == approach or parent.name.startswith(prefix) for parent in p.parents)
    ]


def parse_result_filename(path: Path) -> dict[str, str | int | None]:
    """Parse eps/N/ndims from CSV file stem."""
    match = FILENAME_RE.fullmatch(path.stem)
    if not match:
        return {"inp_eps": None, "buggy_points_num": None, "ndims": None}

    return {
        "inp_eps": match.group("eps"),
        "buggy_points_num": int(match.group("n")),
        "ndims": int(match.group("ndims")) if match.group("ndims") else None,
    }


def _infer_approach_from_dirname(dirname: str) -> str:
    for approach in APPROACHES_ALL:
        if dirname == approach or dirname.startswith(f"{approach}_"):
            return approach
    return dirname.split("_", 1)[0]


def _infer_method_dir(csv_path: Path) -> Path | None:
    for parent in csv_path.parents:
        if _infer_approach_from_dirname(parent.name) in APPROACHES_ALL:
            return parent
    return None


def _infer_start_layer(csv_path: Path) -> int | None:
    for parent in csv_path.parents:
        match = re.fullmatch(r"layer(\d+)", parent.name)
        if match:
            return int(match.group(1))
    return None


def _attach_file_metadata(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    out = df.copy()
    file_meta = parse_result_filename(csv_path)
    method_dir = _infer_method_dir(csv_path)
    approach = _infer_approach_from_dirname(method_dir.name) if method_dir is not None else _infer_approach_from_dirname(csv_path.parent.name)
    start_layer = _infer_start_layer(csv_path)

    def set_or_fill(column: str, value) -> None:
        if column not in out.columns:
            out[column] = value
        elif value is not None:
            out[column] = out[column].fillna(value)

    out["source_csv"] = str(csv_path)
    set_or_fill("approach", approach)
    set_or_fill("inp_eps", file_meta["inp_eps"])
    set_or_fill("buggy_points_num", file_meta["buggy_points_num"])
    set_or_fill("ndims", file_meta["ndims"])
    set_or_fill("start_layer", start_layer)

    # Ensure canonical names exist even if log parser used older/newer keys.
    if "network" in out.columns:
        if "model_name" not in out.columns:
            out["model_name"] = out["network"]
        else:
            out["model_name"] = out["model_name"].fillna(out["network"])
    if "perturbation_pick" in out.columns:
        if "perturbation" not in out.columns:
            out["perturbation"] = out["perturbation_pick"]
        else:
            out["perturbation"] = out["perturbation"].fillna(out["perturbation_pick"])

    if "start_layer_label" not in out.columns:
        out["start_layer_label"] = None
    if approach == "ours":
        if "layer_set_label" in out.columns:
            out["start_layer_label"] = out["start_layer_label"].fillna(out["layer_set_label"])
        elif start_layer is not None:
            out["start_layer_label"] = out["start_layer_label"].fillna(f"layer{start_layer}")
    elif approach == "lastlayer":
        out["start_layer_label"] = out["start_layer_label"].fillna("lastlayer")

    if "series_label" not in out.columns:
        out["series_label"] = None
    if approach == "ours" and "layer_set_label" in out.columns:
        out["series_label"] = out["series_label"].fillna(out["layer_set_label"])
    elif approach == "FL" and "variant" in out.columns:
        out["series_label"] = out["series_label"].fillna(out["variant"])
    else:
        out["series_label"] = out["series_label"].fillna(approach)

    return out


def load_all_csv_rows(root: Path | None = None, approach: str | None = None) -> pd.DataFrame:
    """Load all run-level rows from matching CSV files."""
    paths = collect_csv_paths(root=root, approach=approach)
    if not paths:
        return pd.DataFrame()

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        frames.append(_attach_file_metadata(df, path))
    return pd.concat(frames, ignore_index=True)


def _gen_columns(df: pd.DataFrame, suffix: str) -> list[str]:
    return sorted(col for col in df.columns if col.startswith("gen_dist_") and col.endswith(suffix))


def _aggregate_one_group(group: pd.DataFrame) -> pd.Series:
    total_num = len(group)
    success_df = group[group["result"] == "success"]
    success_num = len(success_df)

    result: dict[str, float | int | str | None] = {
        "total_num": int(total_num),
        "success_num": int(success_num),
        "success_rate": (success_num / total_num) if total_num else 0.0,
        "avg_total_repair_time_sec_success": success_df["total_repair_time_sec"].mean() if success_num else None,
        "avg_total_steps_success": success_df["total_steps"].mean() if success_num else None,
        "original_accuracy": group["original_accuracy"].dropna().iloc[0] if group["original_accuracy"].notna().any() else None,
        "avg_repaired_accuracy_success": success_df["repaired_accuracy"].mean() if success_num else None,
    }

    for col in _gen_columns(group, "_orig"):
        result[f"avg_{col}_success"] = success_df[col].mean() if success_num and col in success_df.columns else None
    for col in _gen_columns(group, "_repaired"):
        result[f"avg_{col}_success"] = success_df[col].mean() if success_num and col in success_df.columns else None

    return pd.Series(result)


def aggregate_results_from_csvs(
    root: Path | None = None,
    *,
    merge_ours_policy: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Aggregate repair results from CSV files.

    Group key:
      (model_name, repair_task, perturbation, inp_eps, buggy_points_num, ndims)

    Returns:
      dict[approach] = aggregated DataFrame for that approach.
    """
    all_rows = load_all_csv_rows(root=root)
    if all_rows.empty:
        return {}
    if merge_ours_policy:
        all_rows, _ = replace_layerwise_comparison_rows(all_rows)

    required = ["model_name", "repair_task", "perturbation", "inp_eps", "buggy_points_num", "ndims", "approach", "result"]
    missing = [c for c in required if c not in all_rows.columns]
    if missing:
        raise ValueError(f"Missing required columns for aggregation: {missing}")

    by_approach: dict[str, pd.DataFrame] = {}
    group_keys = ["model_name", "repair_task", "perturbation", "inp_eps", "buggy_points_num", "ndims"]

    for approach, df_app in all_rows.groupby("approach", dropna=False):
        agg = df_app.groupby(group_keys, dropna=False).apply(_aggregate_one_group).reset_index()
        agg = agg.sort_values(group_keys).reset_index(drop=True)
        by_approach[str(approach)] = agg

    return by_approach


def compare_approaches(
    aggregated_by_approach: dict[str, pd.DataFrame] | None = None,
    root: Path | None = None,
    approaches: tuple[str, ...] = APPROACHES,
) -> pd.DataFrame:
    """
    For each set of (model_name, repair_task, perturbation, inp_eps, buggy_points_num, ndims),
    compare selected summary metrics side-by-side across approaches.

    Returns columns:
    - model_name
    - repair_task
    - perturbation
    - inp_eps
    - buggy_points_num
    - ndims
    - success_num_{approach}
    - success_rate_{approach}
    - avg_total_repair_time_sec_success_{approach}
    - original_accuracy_{approach}
    - avg_repaired_accuracy_success_{approach}
    """
    if aggregated_by_approach is None:
        aggregated_by_approach = aggregate_results_from_csvs(root=root)

    if not aggregated_by_approach:
        return pd.DataFrame()

    group_keys = ["model_name", "repair_task", "perturbation", "inp_eps", "buggy_points_num", "ndims"]
    metrics = [
        "success_num",
        "success_rate",
        "avg_total_repair_time_sec_success",
        "original_accuracy",
        "avg_repaired_accuracy_success",
    ]

    merged: pd.DataFrame | None = None
    for approach in approaches:
        df = aggregated_by_approach.get(approach)
        if df is None or df.empty:
            continue

        missing = [c for c in group_keys + metrics if c not in df.columns]
        if missing:
            raise ValueError(f"Approach '{approach}' is missing required columns: {missing}")

        cols = group_keys + metrics
        renamed = df[cols].copy().rename(columns={m: f"{m}_{approach}" for m in metrics})

        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on=group_keys, how="outer")

    if merged is None:
        return pd.DataFrame(columns=group_keys)

    return merged.sort_values(group_keys).reset_index(drop=True)


def main():
    root = default_repair_root()
    aggregated = aggregate_results_from_csvs(root=root)

    compared = compare_approaches(aggregated_by_approach=aggregated)

    return aggregated, compared


if __name__ == "__main__":
    main()
