import pandas as pd

from result_analysis.policy_merge import merge_ours_policy_results, replace_layerwise_comparison_rows


def _row(
    layer: str | None,
    result: str,
    runtime: float,
    *,
    approach: str = "ours",
    accuracy=0.9,
    input_time=pd.NA,
):
    return {
        "model_name": "mnist 9x100",
        "repair_task": "LocalRobustness",
        "perturbation": "all",
        "inp_eps": "1",
        "buggy_points_num": 1,
        "ndims": pd.NA,
        "run_index": 1,
        "repair_data_ids": "8",
        "approach": approach,
        "variant": layer or "lastlayer",
        "layer_set_label": layer,
        "result": result,
        "failure_reason": "repair_failed" if result == "failed" else pd.NA,
        "total_repair_time_sec": runtime,
        "input_construction_time_sec": input_time,
        "total_steps": 2,
        "original_accuracy": 0.95,
        "repaired_accuracy": accuracy if result == "success" else pd.NA,
        "accuracy_drop": 0.95 - accuracy if result == "success" else pd.NA,
    }


def test_policy_stops_at_first_success_and_sums_attempts() -> None:
    rows = pd.DataFrame(
        [
            _row("layers_16", "failed", 3.0),
            _row("layers_14_16", "success", 7.0, accuracy=0.92),
            _row("layers_12_14_16", "success", 100.0, accuracy=0.94),
            _row("layers_10_12_14_16", "success", 100.0, accuracy=0.945),
        ]
    )
    merged = merge_ours_policy_results(rows).iloc[0]
    assert merged["result"] == "success"
    assert merged["successful_layer_set"] == "layers_14_16"
    assert merged["attempted_layer_sets"] == "layers_16 -> layers_14_16"
    assert merged["attempt_count"] == 2
    assert merged["total_repair_time_sec"] == 10.0
    assert merged["total_steps"] == 4
    assert merged["repaired_accuracy"] == 0.92


def test_policy_does_not_charge_unattempted_layers() -> None:
    rows = pd.DataFrame(
        [
            _row("layers_16", "success", 3.0),
            _row("layers_14_16", "success", 7.0),
        ]
    )
    merged = merge_ours_policy_results(rows).iloc[0]
    assert merged["attempt_count"] == 1
    assert merged["total_repair_time_sec"] == 3.0


def test_policy_charges_input_construction_once_for_multiple_layer_stages() -> None:
    rows = pd.DataFrame(
        [
            _row("layers_16", "failed", 3.0, input_time=5.0),
            _row("layers_14_16", "failed", 7.0, input_time=5.0),
            _row("layers_12_14_16", "success", 11.0, input_time=5.0),
        ]
    )
    merged = merge_ours_policy_results(rows).iloc[0]
    assert merged["attempt_count"] == 3
    assert merged["total_repair_time_sec"] == 26.0
    assert merged["included_input_construction_time_sec"] == 5.0
    assert bool(merged["input_construction_included"])


def test_missing_required_stage_after_failure_is_incomplete() -> None:
    rows = pd.DataFrame(
        [
            _row("layers_16", "failed", 3.0),
            _row("layers_12_14_16", "success", 7.0),
        ]
    )
    merged = merge_ours_policy_results(rows).iloc[0]
    assert merged["result"] == "incomplete"
    assert merged["failure_reason"] == "missing_policy_stage:layers_14_16"
    assert merged["attempt_count"] == 1
    assert merged["total_repair_time_sec"] == 3.0


def test_legacy_mnist256_labels_include_output_layer() -> None:
    base = _row(None, "failed", 2.0, approach="lastlayer")
    base["model_name"] = "mnist_256x4"
    expanded = _row("layers_7", "success", 5.0)
    expanded["model_name"] = "mnist_256x4"
    merged = merge_ours_policy_results(pd.DataFrame([base, expanded])).iloc[0]
    assert merged["successful_layer_set"] == "layers_7_9"
    assert merged["attempted_layer_sets"] == "layers_9 -> layers_7_9"
    assert merged["total_repair_time_sec"] == 7.0


def test_comparison_view_contains_only_one_ours_row() -> None:
    raw = pd.DataFrame(
        [
            _row("layers_16", "failed", 3.0),
            _row("layers_14_16", "success", 7.0),
        ]
    )
    prepared = _row(None, "success", 1.0, approach="prepared")
    raw = pd.concat([raw, pd.DataFrame([prepared])], ignore_index=True)
    comparison, merged = replace_layerwise_comparison_rows(raw)
    assert len(merged) == 1
    assert len(comparison[comparison["approach"] == "ours"]) == 1
    assert len(comparison[comparison["approach"] == "prepared"]) == 1


def test_legacy_mnist9_nonzero_maps_layer14_to_last_layer() -> None:
    row = _row("layers_14", "success", 4.0)
    row["perturbation"] = "nonzero"
    row["ndims"] = 5
    merged = merge_ours_policy_results(pd.DataFrame([row])).iloc[0]
    assert merged["result"] == "success"
    assert merged["successful_layer_set"] == "layers_16"
    assert pd.isna(merged["missing_policy_prefix"])
    assert bool(merged["policy_complete"])
