import argparse
import ast
import re
from pathlib import Path

import pandas as pd

RUN_HEADER_RE = re.compile(r"=== Repair run (\d+)/(\d+) ===")
FLOAT_RE = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
INT_RE = r"(\d+)"


def _parse_metadata(text: str) -> dict:
    metadata = {}

    patterns = {
        "repair_task": r"Repair task:\s*(.+)",
        "repair_mode": r"Repair mode:\s*(.+)",
        "num_runs_declared": r"Number of runs:\s*(\d+)",
        "network": r"Network:\s*(.+)",
        "target_label": r"Target label:\s*(.+)",
        "num_v_polytopes": r"Number of v-polytopes:\s*(\d+)",
        "eps": rf"Perturbation distance \(eps\):\s*{FLOAT_RE}",
        "perturbation_pick": r"Perturbation pick:\s*(.+)",
        "perturbation_ndim": r"Perturbation ndim:\s*(\d+)",
        "modifiable_range": rf"Modifiable range:\s*{FLOAT_RE}",
        "first_layer_to_repair": r"First layer to repair:\s*(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).strip()
        if key in {"num_runs_declared", "num_v_polytopes", "perturbation_ndim", "first_layer_to_repair"}:
            metadata[key] = int(value)
        elif key in {"eps", "modifiable_range"}:
            metadata[key] = float(value)
        else:
            metadata[key] = value

    return metadata


def _split_run_blocks(text: str) -> list[tuple[int, int, str]]:
    matches = list(RUN_HEADER_RE.finditer(text))
    blocks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        run_index = int(match.group(1))
        run_total = int(match.group(2))
        blocks.append((run_index, run_total, text[start:end]))
    return blocks


def _extract_float(block: str, pattern: str) -> float | None:
    match = re.search(pattern, block)
    if not match:
        return None
    return float(match.group(1))


def _extract_int(block: str, pattern: str) -> int | None:
    match = re.search(pattern, block)
    if not match:
        return None
    return int(match.group(1))


def _extract_data_ids(block: str) -> list[int] | None:
    match = re.search(r"Data ids for repair regions in this run:\s*(\[[^\]]*\])", block)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    except (ValueError, SyntaxError):
        return None
    return None


def _parse_generalization(block: str) -> dict[str, tuple[float | None, float | None]]:
    results: dict[str, tuple[float | None, float | None]] = {}
    gen_pattern = re.compile(
        rf"Repaired DNN accuracy on the generalization dataset \(dist={FLOAT_RE}\):\s*\n"
        rf"{FLOAT_RE}\s*\([^\)]*\)\s*->\s*{FLOAT_RE}\s*\([^\)]*\)",
        re.MULTILINE,
    )

    for match in gen_pattern.finditer(block):
        dist = f"{float(match.group(1)):.10g}"
        before = float(match.group(2))
        after = float(match.group(3))
        results[dist] = (before, after)

    return results


def _extract_total_steps(block: str, repair_mode: str | None, failed: bool) -> int | None:
    mode = (repair_mode or "").strip().lower()

    # lastlayer: both success and failure runs contain explicit step patterns
    if mode == "last_layer_ibp":
        completed = _extract_int(block, rf"Repair completed in\s*{INT_RE}\s*steps\.")
        if completed is not None:
            return completed
        not_completed = _extract_int(block, rf"Repair not completed after\s*{INT_RE}\s*steps\.")
        if not_completed is not None:
            return not_completed

    # prepared: exactly one repair stage for each run
    if mode == "prepared":
        return 1

    # ours (lp_ibp): successful runs contain "Total steps: X", failed runs should be NaN
    if mode == "lp_ibp":
        if failed:
            return None
        return _extract_int(block, rf"Total steps:\s*{INT_RE}")

    # Fallback for unknown/new modes.
    return _extract_int(block, rf"Total steps:\s*{INT_RE}")


def _is_failed_run(block: str, repair_mode: str | None) -> bool:
    mode = (repair_mode or "").strip().lower()

    if mode == "last_layer_ibp":
        return re.search(r"Repair not completed after\s*\d+\s*steps\.", block) is not None

    # prepared / lp_ibp (ours) and fallback behavior.
    return "Repair failed. No repaired DNN to evaluate." in block


def _parse_run_block(run_index: int, run_total: int, block: str, repair_mode: str | None) -> dict:
    '''
    repair result:
        - for last_layer_ibp, failed if it contains "Repair not completed after {int} steps.", success otherwise.
        - for prepared and ours, keep current logic of checking "Repair failed. No repaired DNN to evaluate." in block.
    '''
    failed = _is_failed_run(block, repair_mode=repair_mode)

    row = {
        "run_index": run_index,
        "run_total": run_total,
        "result": "failed" if failed else "success",
        "total_repair_time_sec": _extract_float(block, rf"Total repair time:\s*{FLOAT_RE}\s*seconds"),
        "total_steps": _extract_total_steps(block, repair_mode=repair_mode, failed=failed),
        "original_accuracy": _extract_float(
            block,
            rf"Original DNN accuracy on the evaluation dataset:\s*{FLOAT_RE}",
        ),
        "repaired_accuracy": _extract_float(
            block,
            rf"Repaired DNN accuracy on the evaluation dataset:\s*{FLOAT_RE}",
        ),
    }

    data_ids = _extract_data_ids(block)
    if data_ids is not None:
        row["repair_data_ids"] = ",".join(str(x) for x in data_ids)

    generalization = _parse_generalization(block)
    for dist, (before, after) in generalization.items():
        row[f"gen_dist_{dist}_orig"] = before
        row[f"gen_dist_{dist}_repaired"] = after

    if generalization:
        after_vals = [x[1] for x in generalization.values() if x[1] is not None]
        before_vals = [x[0] for x in generalization.values() if x[0] is not None]
        if before_vals:
            row["gen_orig_mean"] = sum(before_vals) / len(before_vals)
        if after_vals:
            row["gen_repaired_mean"] = sum(after_vals) / len(after_vals)
            row["gen_repaired_min"] = min(after_vals)
            row["gen_repaired_max"] = max(after_vals)

    return row


def parse_log_file(log_path: Path) -> pd.DataFrame:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metadata = _parse_metadata(text)
    blocks = _split_run_blocks(text)

    rows = []
    for run_index, run_total, block in blocks:
        row = _parse_run_block(run_index, run_total, block, repair_mode=metadata.get("repair_mode"))
        row["source_log"] = str(log_path)
        row.update(metadata)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("run_index").reset_index(drop=True)
    return df


def _build_output_path(log_path: Path, input_root: Path, output_root: Path) -> Path:
    relative = log_path.relative_to(input_root)
    # Examples:
    # prepared/mnist 9x100/LocalCounterexample/all/eps1_N1.txt
    # -> csv/prepared_mnist 9x100_LocalCounterexample_all/eps1_N1.csv
    # ours/mnist 9x100/LocalCounterexample/all_3/eps1_N1.txt
    # -> csv/ours_mnist 9x100_LocalCounterexample_all/layer3/eps1_N1.csv
    if len(relative.parts) < 5:
        out_dir = output_root / "misc"
    else:
        approach, model_name, repair_task, perturbation_folder = relative.parts[:4]
        if approach == "ours":
            perturbation, start_layer = perturbation_folder.rsplit("_", 1) if "_" in perturbation_folder else (perturbation_folder, None)
            out_dir = output_root / f"{approach}_{model_name}_{repair_task}_{perturbation}"
            if start_layer is not None:
                out_dir = out_dir / f"layer{start_layer}"
        else:
            out_dir = output_root / f"{approach}_{model_name}_{repair_task}_{perturbation_folder}"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{log_path.stem}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse legacy/current repair logs into canonical summary CSV files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("repair_result"),
        help="Root directory that contains repair log txt files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("repair_result/csv"),
        help="Directory where summary CSVs will be written.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="Deprecated; all supported .txt and .log result files are collected.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Do not replace the generated CSV tree.")
    args = parser.parse_args()

    try:
        from .collect_results import collect_results
    except ImportError:  # Direct script execution.
        from collect_results import collect_results

    written = collect_results(args.input_root, args.output_root, clean=not args.no_clean)
    print(f"Wrote {len(written)} canonical CSV files.")


if __name__ == "__main__":
    main()
