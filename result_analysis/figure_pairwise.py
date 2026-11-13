'''
keep cmments, do not change the content of comments, but you can change the formatting of comments if needed

figure1
-----------
diagonal comparison
    x-axis and y-axis represent approaches a and b, respectively
    each plot represents a single repair task identified by (model name, repair task, perturbation pick, eps, N, run index)
    comparing accuracy difference, if a is more accurate than b, the point is above the diagonal, otherwise below
    assign different shapes to different ndims or N

    each figure corresponds to a single (model name, repair task, perturbation pick, eps) combination, and includes all runs with different N and ndims for that combination
'''

from __future__ import annotations

import argparse
import math
import re
import sys
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from result_analysis.analysis import APPROACHES, APPROACHES_ALL, DISPLAY_NAME_MAP, default_repair_root, load_all_csv_rows

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
    0: (0, "Local Robustness", "nonzero", 1),
    1: (0, "Local Robustness", "all", 1),
    2: (1, "Local Robustness", "nonzero", 1),
    3: (1, "Local Robustness", "all", 1),
    4: (4, "Corruption + Perturbation", "all", 1),
    5: (6, "Adversarial + Perturbation", "all", 1),
}

GROUP_KEYS = ["model_name", "repair_task", "perturbation", "inp_eps", "buggy_points_num", "ndims", "run_index"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]

FONT_SIZE = 14
MERGED_DIR_RE = re.compile(r"^merged_(?P<threshold>[^/]+)$")
GREEDY_MERGED_DIR = "merged_greedy_best_accuracy"

def _display_name(approach: str) -> str:
    return DISPLAY_NAME_MAP.get(approach, approach)


def _normalize_text(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(value)).lower()


def _text_matches(value: object, target: str) -> bool:
    value_norm = _normalize_text(value)
    target_norm = _normalize_text(target)
    if value_norm == target_norm:
        return True
    return value_norm.replace("and", "") == target_norm.replace("and", "")


def _approaches_for_perturbation(perturbation: str) -> list[str]:
    if perturbation.lower() == "nonzero":
        return list(APPROACHES_ALL)
    return list(APPROACHES)


def _parse_label_list(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _series_labels_for_exp(exp_id: int) -> list[str] | None:
    raw = expid_layers_map.get(exp_id)
    if raw is None:
        return None
    if isinstance(raw, str):
        return _parse_label_list(raw)
    if isinstance(raw, tuple) and len(raw) == 1:
        return _parse_label_list(raw[0])
    return list(raw)


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
    merged_dir = csv_path.parent
    ours_dir = csv_path.parent.parent

    if merged_dir is None or ours_dir is None:
        return False
    if not ours_dir.name.startswith("ours_"):
        return False

    match = MERGED_DIR_RE.fullmatch(merged_dir.name)
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
    model_name: str,
    repair_task: str,
    perturbation: str,
    inp_eps: str,
) -> pd.DataFrame:
    out = runs.copy()
    out = out[out["model_name"].map(lambda value: _text_matches(value, model_name))]
    out = out[out["repair_task"].map(lambda value: _text_matches(value, repair_task))]
    out = out[out["perturbation"].map(lambda value: _text_matches(value, perturbation))]
    out = out[out["inp_eps"].astype(str) == str(inp_eps)]
    return out


def _prepare_pairwise_frame(
    runs: pd.DataFrame,
    labels: list[str],
    label_col: str,
    marker_key: str,
) -> pd.DataFrame:
    required = GROUP_KEYS + [label_col, "result", "repaired_accuracy"]
    missing = [c for c in required if c not in runs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if marker_key not in runs.columns:
        raise ValueError(f"Missing required column: {marker_key}")

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
    meta = base.groupby("pair_id", sort=False)[marker_key].first().reset_index()
    pairwise = pairwise.merge(meta, on="pair_id", how="left")
    return pairwise


def make_pairwise_figure(
    root: Path | None,
    exp_id: int,
    selected_pairs: list[tuple[str, str]] | None = None,
    shape_mode: str = "by_key",
    same_marker: str = "o",
    ours_threshold: float | None = 100,
    ours_merge_strategy: str = "threshold",
    comparison_kind: str = "auto",
    save_path: Path | None = None,
) -> plt.Figure:
    if exp_id not in experiment_map:
        raise ValueError(f"Unknown exp_id={exp_id}. Available: {sorted(experiment_map.keys())}")

    model_id, repair_task, perturbation, inp_eps = experiment_map[exp_id]
    model_name = model_name_map[model_id]
    layer_labels = _series_labels_for_exp(exp_id)
    if comparison_kind == "auto":
        use_layer_series = layer_labels is not None
    elif comparison_kind == "layers":
        use_layer_series = True
    elif comparison_kind == "approaches":
        use_layer_series = False
    else:
        raise ValueError(f"Unknown comparison_kind={comparison_kind}")

    if use_layer_series:
        if layer_labels is None:
            raise ValueError(f"exp_id={exp_id} does not define layer-series labels.")
        compare_labels = layer_labels
        compare_col = "series_label"
    else:
        compare_labels = _approaches_for_perturbation(perturbation)
        compare_col = "approach"

    runs = load_all_csv_rows(root=root)
    if runs.empty:
        raise ValueError("No CSV rows found.")

    if not use_layer_series:
        runs = _filter_ours_rows(runs, ours_threshold, merge_strategy=ours_merge_strategy)
        runs = _label_merged_ours_rows(runs, ours_threshold, merge_strategy=ours_merge_strategy)

    runs = _filter_runs(
        runs=runs,
        model_name=model_name,
        repair_task=repair_task,
        perturbation=perturbation,
        inp_eps=str(inp_eps),
    )
    if runs.empty:
        raise ValueError("No rows left after filters.")

    if selected_pairs is None:
        pairs = list(combinations(compare_labels, 2))
    else:
        pairs = selected_pairs
        unknown = sorted(
            {
                name
                for pair in pairs
                for name in pair
                if name not in compare_labels
            }
        )
        if unknown:
            raise ValueError(
                f"Unknown approach(es) for this experiment: {unknown}. "
                f"Available: {compare_labels}"
            )
    if not pairs:
        raise ValueError("No approach pairs to compare.")
    n_panels = len(pairs)
    ncols = 1 if n_panels == 1 else 3
    nrows = math.ceil(n_panels / ncols)
    fig_w = 5.2 if n_panels == 1 else 6 * ncols
    fig_h = 4.2 if n_panels == 1 else 5 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    marker_key = "ndims" if perturbation.lower() == "nonzero" else "buggy_points_num"
    pairwise = _prepare_pairwise_frame(runs, compare_labels, compare_col, marker_key=marker_key)
    marker_vals = [v for v in sorted(pairwise[marker_key].unique()) if v != "__NA__"]
    marker_map = {v: MARKERS[i % len(MARKERS)] for i, v in enumerate(marker_vals)}
    default_marker = same_marker

    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]

        sub = pairwise[[a, b, marker_key]].dropna(subset=[a, b]).copy()
        if sub.empty:
            ax.set_title(
                f"{_display_name(a)} vs {_display_name(b)} (n=0)",
                fontsize=FONT_SIZE,
            )
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(_display_name(a), fontsize=FONT_SIZE)
            ax.set_ylabel(_display_name(b), fontsize=FONT_SIZE)
            ax.tick_params(axis="both", labelsize=FONT_SIZE)
            ax.grid(alpha=0.3)
            continue

        if shape_mode == "same":
            ax.scatter(sub[a], sub[b], marker=default_marker, s=45, alpha=0.8)
        else:
            for mval, group in sub.groupby(marker_key, dropna=False):
                marker = marker_map.get(mval, default_marker)
                label = f"{marker_key}={mval}" if mval != "__NA__" else marker_key
                ax.scatter(group[a], group[b], marker=marker, s=45, alpha=0.8, label=label)

        mins = min(float(sub[a].min()), float(sub[b].min()))
        maxs = max(float(sub[a].max()), float(sub[b].max()))
        lo = min(0.0, mins - 0.01)
        hi = max(1.0, maxs + 0.01)

        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(_display_name(a), fontsize=FONT_SIZE)
        ax.set_ylabel(_display_name(b), fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)

        n_above = int((sub[b] > sub[a]).sum())
        n_below = int((sub[b] < sub[a]).sum())
        n_equal = int((sub[b] == sub[a]).sum())
        total = n_above + n_below + n_equal

        # ax.set_title(f"{_display_name(a)} vs {_display_name(b)} (n={len(sub)})")
        ax.text(
            0.97,
            0.03,
            f"above: {n_above} / below: {n_below} / equal: {n_equal} (n={total})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FONT_SIZE,
        )
        ax.grid(alpha=0.3)

    # Hide unused axes.
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    if shape_mode == "same":
        # legend_handles.append(
        #     Line2D(
        #         [0],
        #         [0],
        #         marker=default_marker,
        #         linestyle="None",
        #         markersize=7,
        #         markerfacecolor="#4C78A8",
        #         markeredgecolor="#4C78A8",
        #         alpha=0.8,
        #     )
        # )
        # legend_labels.append("all points")
        pass
    else:
        for v in marker_vals:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[v],
                    linestyle="None",
                    markersize=7,
                    markerfacecolor="#4C78A8",
                    markeredgecolor="#4C78A8",
                    alpha=0.8,
                )
            )
            legend_labels.append(f"{marker_key}={v}")

    # legend_handles.append(Line2D([0], [0], linestyle="--", color="gray", linewidth=1))
    # legend_labels.append("diagonal (equal accuracy)")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(6, len(legend_handles)),
            frameon=False,
            fontsize=FONT_SIZE,
        )

    # fig.suptitle(
    #     f"Pairwise Diagonal Comparison\n{model_name} | {repair_task} | {perturbation} | eps={inp_eps}",
    #     fontsize=14,
    # )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return fig


comparison_map = {
    0: ("prepared", 'lastlayer'),
    1: ("prepared", "ours"),
    2: ("lastlayer", "ours"),
}

expid_layers_map = {
    0: "lastlayer,layer14,layer12,layer10",
    1: "lastlayer,layer14,layer12,layer10",
    2: "lastlayer,layer7,layer5,layer3",
    3: "lastlayer,layer7,layer5,layer3",
    4: "lastlayer,layer5,layer2",
    5: "lastlayer,layer5,layer2",
}



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pairwise diagonal comparison plots from repair CSV results.")
    parser.add_argument("--root", type=Path, default=default_repair_root(), help="repair_result root path")
    parser.add_argument("--exp_id", type=int, default=4, help=f"experiment id in {sorted(experiment_map.keys())}")
    compare_group = parser.add_mutually_exclusive_group()
    compare_group.add_argument(
        "--compare",
        nargs=2,
        metavar=("APPROACH_A", "APPROACH_B"),
        default=None,
        help="compare only one pair, e.g. --compare FL ours",
    )
    compare_group.add_argument(
        "--compare-id",
        type=int,
        default=None,
        help=f"pair id from comparison_map: {comparison_map}",
    )
    compare_group.add_argument(
        "--compare-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help=f"list of pair ids from comparison_map, e.g. --compare-ids 0 1 2. Available: {comparison_map}",
    )
    parser.add_argument(
        "--shape-mode",
        choices=["by_key", "same"],
        default="same",
        help="marker style mode: by_key=shape by ndims/N, same=single shape for all points",
    )
    parser.add_argument(
        "--same-marker",
        type=str,
        default="o",
        help="marker used when --shape-mode same (e.g. o, s, ^, D, x, +)",
    )
    parser.add_argument(
        "--ours-threshold",
        type=float,
        default=100,
        help="Use merged OURS results from merged_<threshold>pct (for example 1, 10, 20, 100).",
    )
    parser.add_argument(
        "--ours-merge-strategy",
        choices=["threshold", "greedy_best_accuracy"],
        default="threshold",
        # default="greedy_best_accuracy",
        help="Choose which merged OURS CSV family to use when OURS is compared.",
    )
    parser.add_argument(
        "--comparison-kind",
        choices=["auto", "approaches", "layers"],
        default="approaches",
        # default="layers",
        help="Choose baseline approach comparisons or layer-series comparisons. auto uses layers when exp_id is in expid_layers_map.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("result_analysis/figures/pairwise"), help="directory to save figures")
    parser.add_argument("--show", action="store_true", help="show interactive plot")
    args = parser.parse_args()

    selected_pairs: list[tuple[str, str]] | None = None
    pair_suffix = ""
    compare_pairs_to_generate: list[tuple[str, str]] | None = None
    if args.compare is not None:
        selected_pairs = [(args.compare[0], args.compare[1])]
        pair_suffix = f"_{args.compare[0]}_vs_{args.compare[1]}"
    elif args.compare_id is not None:
        if args.compare_id not in comparison_map:
            raise ValueError(f"Unknown compare-id={args.compare_id}. Available: {sorted(comparison_map.keys())}")
        a, b = comparison_map[args.compare_id]
        selected_pairs = [(a, b)]
        pair_suffix = f"_{a}_vs_{b}"

    model_id, repair_task, perturbation, inp_eps = experiment_map[args.exp_id]
    model_name = model_name_map[model_id]
    prefix = f"{model_name}_{perturbation}_eps{inp_eps}".replace(" ", "_").replace("/", "_")
    if args.exp_id in expid_layers_map:
        prefix += "_series" + "-".join(_series_labels_for_exp(args.exp_id) or [])
    elif args.ours_merge_strategy == "greedy_best_accuracy":
        prefix += "_oursmerged_greedy_best_accuracy"
    else:
        prefix += f"_oursmerged{args.ours_threshold:g}pct"

    if selected_pairs is not None:
        compare_pairs_to_generate = selected_pairs
    else:
        if args.comparison_kind == "layers":
            compare_labels = _series_labels_for_exp(args.exp_id)
            if compare_labels is None:
                raise ValueError(f"exp_id={args.exp_id} does not define layer-series labels.")
            compare_pairs_to_generate = list(combinations(compare_labels, 2))
        else:
            if args.compare_ids is not None:
                unknown_ids = sorted({cid for cid in args.compare_ids if cid not in comparison_map})
                if unknown_ids:
                    raise ValueError(
                        f"Unknown compare-id(s)={unknown_ids}. Available: {sorted(comparison_map.keys())}"
                    )
                compare_pairs_to_generate = [comparison_map[cid] for cid in args.compare_ids]
            else:
                compare_pairs_to_generate = list(combinations(_approaches_for_perturbation(perturbation), 2))

    for a, b in compare_pairs_to_generate:
        pair_suffix = f"_{a}_vs_{b}"
        out_path = args.out_dir / f"{prefix}{pair_suffix}_diag.png"
        make_pairwise_figure(
            root=args.root,
            exp_id=args.exp_id,
            selected_pairs=[(a, b)],
            shape_mode=args.shape_mode,
            same_marker=args.same_marker,
            ours_threshold=args.ours_threshold,
            ours_merge_strategy=args.ours_merge_strategy,
            comparison_kind=args.comparison_kind,
            save_path=out_path,
        )
        print(f"Saved: {out_path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
