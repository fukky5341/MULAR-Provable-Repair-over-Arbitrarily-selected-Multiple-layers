"""Reconstruct the sequential layer-expansion policy used by our approach.

Raw experiments remain available for layer-policy studies.  For comparisons
against other approaches, this module produces one row per run by trying the
configured layer sets in order and accumulating the time until first success.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PolicyStage:
    label: str
    raw_aliases: tuple[str, ...]
    allow_lastlayer: bool = False


POLICY_STAGES: dict[str, tuple[PolicyStage, ...]] = {
    "mnist 9x100": (
        # Legacy nonzero logs use [14] as their last-layer configuration and
        # omit the newer output-layer index from every printed layer set.
        PolicyStage("layers_16", ("layers_16", "layers_14"), True),
        PolicyStage("layers_14_16", ("layers_14_16", "layers_12_14")),
        PolicyStage("layers_12_14_16", ("layers_12_14_16", "layers_10_12_14")),
        PolicyStage("layers_10_12_14_16", ("layers_10_12_14_16",)),
    ),
    # Legacy logs omit the automatically included output layer (9) from their
    # printed repaired-layer list, so both canonical and legacy aliases are accepted.
    "mnist_256x4": (
        PolicyStage("layers_9", ("layers_9",), True),
        PolicyStage("layers_7_9", ("layers_7_9", "layers_7")),
        PolicyStage("layers_5_7_9", ("layers_5_7_9", "layers_5_7")),
        PolicyStage("layers_3_5_7_9", ("layers_3_5_7_9", "layers_3_5_7")),
    ),
    "cifar10_cnn1": (
        PolicyStage("layers_7", ("layers_7",), True),
        PolicyStage("layers_5_7", ("layers_5_7",)),
        PolicyStage("layers_2_5_7", ("layers_2_5_7",)),
    ),
    "gtsrb_cnn": (
        PolicyStage("layers_7", ("layers_7",), True),
        PolicyStage("layers_5_7", ("layers_5_7",)),
        PolicyStage("layers_2_5_7", ("layers_2_5_7",)),
    ),
}

INSTANCE_KEYS = [
    "model_name",
    "repair_task",
    "perturbation",
    "inp_eps",
    "buggy_points_num",
    "ndims",
    "run_index",
    "repair_data_ids",
]

ALLOW_MISSING_POLICY_PREFIX: set[tuple[str, str]] = set()


def _numeric(value: object) -> float | None:
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def collect_policy_stage_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return raw policy rows with architecture-consistent layer-set labels."""

    selected: list[pd.DataFrame] = []
    for model, stages in POLICY_STAGES.items():
        model_rows = rows[rows["model_name"] == model]
        for stage_index, stage in enumerate(stages):
            ours = model_rows[
                (model_rows["approach"] == "ours")
                & model_rows["layer_set_label"].isin(stage.raw_aliases)
            ].copy()
            # Prefer the current canonical label over a legacy alias when both
            # are present for the same experiment instance.
            ours["_policy_source_rank"] = (
                ours["layer_set_label"].astype(str) != stage.label
            ).astype(int)
            candidates = [ours]
            if stage.allow_lastlayer:
                last = model_rows[model_rows["approach"] == "lastlayer"].copy()
                last["_policy_source_rank"] = 2
                candidates.append(last)
            stage_rows = pd.concat(candidates, ignore_index=True, sort=False)
            if stage_rows.empty:
                continue
            stage_rows["policy_stage"] = stage_index
            stage_rows["policy_layer_set"] = stage.label
            stage_rows["layer_set_label"] = stage.label
            stage_rows["evaluation_series"] = stage.label
            # Prefer an explicit OURS stage over the standalone last-layer log
            # when both represent the same run.
            available_keys = [key for key in INSTANCE_KEYS if key in stage_rows.columns]
            stage_rows = stage_rows.sort_values("_policy_source_rank").drop_duplicates(
                subset=available_keys, keep="first"
            )
            selected.append(stage_rows.drop(columns="_policy_source_rank"))
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True, sort=False)


def _merge_instance(group: pd.DataFrame) -> pd.Series:
    model = str(group.iloc[0]["model_name"])
    perturbation = str(group.iloc[0]["perturbation"])
    stages = POLICY_STAGES[model]
    by_stage = {
        int(stage): stage_rows.iloc[0]
        for stage, stage_rows in group.groupby("policy_stage", sort=True)
    }
    cumulative_time = 0.0
    cumulative_steps = 0.0
    time_known = True
    steps_known = True
    attempted: list[str] = []
    attempted_input_construction_times: list[float] = []
    chosen: pd.Series | None = None
    final_result = "incomplete"
    missing_stage: str | None = None
    first_available_stage = min(by_stage)
    allow_missing_prefix = (
        first_available_stage > 0
        and (model, perturbation) in ALLOW_MISSING_POLICY_PREFIX
    )
    start_stage = first_available_stage if allow_missing_prefix else 0

    for stage_index, stage in enumerate(stages[start_stage:], start=start_stage):
        row = by_stage.get(stage_index)
        if row is None:
            missing_stage = stage.label
            break
        chosen = row.copy()
        attempted.append(stage.label)
        runtime = _numeric(row.get("total_repair_time_sec"))
        steps = _numeric(row.get("total_steps"))
        if runtime is None:
            time_known = False
        else:
            cumulative_time += runtime
        input_construction_time = _numeric(
            row.get("input_construction_time_sec")
        )
        if input_construction_time is not None:
            attempted_input_construction_times.append(input_construction_time)
        if steps is None:
            steps_known = False
        else:
            cumulative_steps += steps

        status = str(row.get("result", "incomplete")).lower()
        final_result = status
        if status == "success" or status == "incomplete":
            break

    if chosen is None:
        # A later-stage raw result exists, but the mandatory first attempt does not.
        chosen = group.sort_values("policy_stage").iloc[0].copy()
        final_result = "incomplete"

    out = chosen.copy()
    out["approach"] = "ours"
    out["variant"] = "sequential_layer_expansion"
    out["evaluation_series"] = "ours"
    out["result"] = final_result
    requires_input_construction_time = len(attempted) > 1
    included_input_construction_time = (
        attempted_input_construction_times[0]
        if requires_input_construction_time and attempted_input_construction_times
        else 0.0
    )
    cumulative_time += included_input_construction_time
    out["total_repair_time_sec"] = cumulative_time if time_known else pd.NA
    out["included_input_construction_time_sec"] = (
        included_input_construction_time
    )
    out["input_construction_included"] = bool(
        requires_input_construction_time and attempted_input_construction_times
    )
    out["input_construction_time_missing"] = bool(
        requires_input_construction_time and not attempted_input_construction_times
    )
    out["total_steps"] = int(cumulative_steps) if steps_known else pd.NA
    out["attempt_count"] = len(attempted)
    out["attempted_layer_sets"] = " -> ".join(attempted)
    out["policy_layer_sequence"] = " -> ".join(stage.label for stage in stages)
    out["missing_policy_prefix"] = (
        " -> ".join(stage.label for stage in stages[:start_stage])
        if allow_missing_prefix
        else pd.NA
    )
    out["successful_layer_set"] = (
        str(chosen.get("policy_layer_set")) if final_result == "success" else pd.NA
    )
    out["policy_complete"] = not allow_missing_prefix and (
        final_result == "success"
        or (missing_stage is None and len(attempted) == len(stages))
    )
    if missing_stage is not None:
        out["result"] = "incomplete"
        out["failure_reason"] = f"missing_policy_stage:{missing_stage}"
        out["repaired_accuracy"] = pd.NA
        out["accuracy_drop"] = pd.NA
        out["successful_layer_set"] = pd.NA
    return out


def merge_ours_policy_results(rows: pd.DataFrame) -> pd.DataFrame:
    """Produce one sequential-policy result per experiment run."""

    stages = collect_policy_stage_rows(rows)
    if stages.empty:
        return stages
    keys = [key for key in INSTANCE_KEYS if key in stages.columns]
    merged = [
        _merge_instance(group)
        for _, group in stages.groupby(keys, dropna=False, sort=True)
    ]
    return pd.DataFrame(merged).sort_values(keys).reset_index(drop=True)


def collect_last_layer_baseline(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one normalized last-layer baseline row per run."""

    stages = collect_policy_stage_rows(rows)
    if stages.empty:
        return stages
    last = stages[stages["policy_stage"] == 0].copy()
    last["approach"] = "lastlayer"
    last["variant"] = "lastlayer"
    last["evaluation_series"] = "lastlayer"
    return last


def replace_layerwise_comparison_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace raw OURS/last-layer rows with normalized comparison records.

    Returns ``(comparison_rows, merged_ours_rows)``.  Other approaches, including
    FL variants, are left unchanged so callers can apply their own RQ filters.
    """

    merged_ours = merge_ours_policy_results(rows)
    lastlayer = collect_last_layer_baseline(rows)
    other = rows[~rows["approach"].isin(["ours", "lastlayer"])].copy()
    comparison = pd.concat([other, lastlayer, merged_ours], ignore_index=True, sort=False)
    return comparison, merged_ours
