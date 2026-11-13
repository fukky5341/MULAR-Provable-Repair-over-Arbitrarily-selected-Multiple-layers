from pathlib import Path

import pandas as pd

from result_analysis.collect_results import parse_txt_log


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_current_per_run_log_uses_header_run_and_summary_accuracy(tmp_path: Path) -> None:
    root = tmp_path / "repair_result"
    path = _write(
        root
        / "ours_target"
        / "layers_12_14_16"
        / "mnist 9x100"
        / "LocalRobustness"
        / "all"
        / "search_exponential_b1_g2"
        / "eps1_N4"
        / "run7.txt",
        """=== Repair Test Log ===
Run: 7/20
Network: mnist 9x100
Repair task: LocalRobustness
Perturbation pick: all
Modifiable layers: [12, 14, 16]
=== Repair run 1/20 ===
Data ids for this repair run: [1, 2, 3, 4]
Input construction time for this run's 4 selected target instances: 6.25 seconds
Step 2: Successfully repaired the DNN!
Total repair time: 12.5 seconds
=== Summary of Repair Results ===
Run 7: Original accuracy = 0.98, Repaired accuracy = 0.97
""",
    )
    row = parse_txt_log(path, root).iloc[0]
    assert row["run_index"] == 7
    assert row["run_total"] == 20
    assert row["result"] == "success"
    assert row["total_repair_time_sec"] == 12.5
    assert row["input_construction_time_sec"] == 6.25
    assert row["repaired_accuracy"] == 0.97
    assert row["layer_set_label"] == "layers_12_14_16"


def test_search_configuration_is_read_from_log_without_search_folder(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repair_result"
    path = _write(
        root
        / "ours_target"
        / "layers_8_10_12"
        / "mnist 9x100"
        / "LocalRobustness"
        / "all"
        / "eps1_N4"
        / "run1.txt",
        """=== Repair Test Log ===
Run: 1/1
Network: mnist 9x100
Repair task: LocalRobustness
Perturbation pick: all
Modifiable layers: [8, 10, 12]
Subspace expansion search method: exponential
Subspace search batch size: 1
Subspace search growth factor: 2
Repair failed.
""",
    )
    row = parse_txt_log(path, root).iloc[0]
    assert row["search_strategy"] == "exponential"
    assert row["search_batch_size"] == 1
    assert row["search_growth_factor"] == 2.0


def test_legacy_summary_does_not_change_last_run_status(tmp_path: Path) -> None:
    root = tmp_path / "repair_result"
    path = _write(
        root / "ours" / "mnist_256x4" / "LocalRobustness" / "all_5" / "eps1_N1.txt",
        """=== Repair run 1/2 ===
Repair failed because no candidate was found.
=== Repair run 2/2 ===
Successfully repaired the DNN!
Total repair time: 3.0 seconds
=== Summary of Repair Results ===
Run 1: Repair failed
Run 2: Original accuracy = 0.9, Repaired accuracy = 0.8
""",
    )
    rows = parse_txt_log(path, root)
    assert list(rows["result"]) == ["failed", "success"]
    assert rows.iloc[1]["repaired_accuracy"] == 0.8


def test_abrupt_log_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "repair_result"
    path = _write(
        root / "prepared" / "cifar10_cnn1" / "LocalRobustness" / "all" / "eps1_N10" / "run3.txt",
        """Run: 3/10
=== Repair run 3/10 ===
Data ids for this repair run: [3]
Starting Gurobi optimization...
""",
    )
    row = parse_txt_log(path, root).iloc[0]
    assert row["result"] == "incomplete"
    assert row["failure_reason"] == "log_ended_before_result"


def test_configured_memory_limit_does_not_mark_success_as_failed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repair_result"
    path = _write(
        root / "prepared" / "cifar10_cnn1" / "LocalRobustness" / "all"
        / "eps1_N6" / "run1.txt",
        """Run: 1/1
Gurobi soft memory limit: 11.47 GB
Successfully repaired the DNN!
Run 1: Original accuracy = 0.63, Repaired accuracy = 0.30
""",
    )
    row = parse_txt_log(path, root).iloc[0]
    assert row["result"] == "success"
    assert pd.isna(row["failure_reason"])


def test_reached_memory_limit_is_still_a_failure(tmp_path: Path) -> None:
    root = tmp_path / "repair_result"
    path = _write(
        root / "prepared" / "cifar10_cnn1" / "LocalRobustness" / "all"
        / "eps1_N8" / "run1.txt",
        """Run: 1/1
Gurobi soft memory limit: 11.47 GB
LP solver failed: Gurobi soft memory limit reached (11.4703 GB).
Repair failed. No repaired DNN to evaluate.
""",
    )
    row = parse_txt_log(path, root).iloc[0]
    assert row["result"] == "failed"
    assert row["failure_reason"] == "memory_limit"
