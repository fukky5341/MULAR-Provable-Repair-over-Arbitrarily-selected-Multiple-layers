import argparse
import re
from pathlib import Path

import pandas as pd

FLOAT_RE = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

PROREPAIR_RUN_HEADER_RE = re.compile(r"\*{10}\s*Run\s*(\d+)/(\d+)\s*\*{10}")
APRNN_RUN_HEADER_RE = re.compile(r"\*{6}\s*Run\s*(\d+)/(\d+)\s*\*{6}")
FILE_NAME_RE = re.compile(r"eps(?P<eps>[^_]+)_nonzero_N(?P<n>\d+)_ndims(?P<ndims>\d+)$")


def _extract_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _extract_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def _parse_filename_metadata(log_path: Path) -> dict:
    match = FILE_NAME_RE.fullmatch(log_path.stem)
    if not match:
        return {"eps": None, "buggy_points_num": None, "ndims_from_name": None}
    return {
        "eps": match.group("eps"),
        "buggy_points_num": int(match.group("n")),
        "ndims_from_name": int(match.group("ndims")),
    }


def _infer_approach_from_path(log_path: Path) -> str | None:
    for part in log_path.parts:
        if part in {"prorepair", "aprnn"}:
            return part
    return None


def _parse_metadata(text: str, log_path: Path) -> dict:
    file_meta = _parse_filename_metadata(log_path)

    approach = _infer_approach_from_path(log_path)

    metadata = {
        "repair_mode": approach,
        "repair_task": None,
        "network": None,
        "num_runs_declared": None,
        "perturbation_pick": "nonzero",
        "perturbation_ndim": None,
        "eps": None,
        "buggy_points_num": file_meta["buggy_points_num"],
    }

    task = re.search(r"Repair task\s+(.+)", text)
    if task:
        metadata["repair_task"] = task.group(1).strip()

    net = re.search(r"Repair net\s+(.+)", text)
    if net:
        metadata["network"] = net.group(1).strip()

    runs = re.search(r"Repair num runs\s+(\d+)", text)
    if runs:
        metadata["num_runs_declared"] = int(runs.group(1))

    ndims = re.search(r"Repair ndims\s+(\d+)", text)
    if ndims:
        metadata["perturbation_ndim"] = int(ndims.group(1))
    elif file_meta["ndims_from_name"] is not None:
        metadata["perturbation_ndim"] = file_meta["ndims_from_name"]

    eps = re.search(rf"Repair eps\s+{FLOAT_RE}", text)
    if eps:
        metadata["eps"] = float(eps.group(1))

    return metadata


def _split_run_blocks(text: str, approach: str) -> list[tuple[int, int, str]]:
    header_re = PROREPAIR_RUN_HEADER_RE if approach == "prorepair" else APRNN_RUN_HEADER_RE
    matches = list(header_re.finditer(text))

    blocks: list[tuple[int, int, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        run_index = int(match.group(1))
        run_total = int(match.group(2))
        blocks.append((run_index, run_total, text[start:end]))
    return blocks


def _parse_generalization(block: str, approach: str) -> dict[str, tuple[float | None, float | None]]:
    results: dict[str, tuple[float | None, float | None]] = {}

    if approach == "prorepair":
        pattern = re.compile(
            rf"Repaired DNN accuracy on the generalization dataset \(dist={FLOAT_RE}\):\s*\n"
            rf"{FLOAT_RE}\s*\([^\)]*\)\s*->\s*{FLOAT_RE}\s*\([^\)]*\)",
            re.MULTILINE,
        )
        for m in pattern.finditer(block):
            dist = f"{float(m.group(1)):.10g}"
            orig = float(m.group(2))
            repaired = float(m.group(3))
            results[dist] = (orig, repaired)
    else:
        pattern = re.compile(
            rf"Gen distance \({FLOAT_RE}\) Drawdown:\s*\n\s*"
            rf"{FLOAT_RE}%\s*\([^\)]*\)\s*->\s*{FLOAT_RE}%\s*\([^\)]*\)",
            re.MULTILINE,
        )
        for m in pattern.finditer(block):
            dist = f"{float(m.group(1)):.10g}"
            orig = float(m.group(2)) / 100.0
            repaired = float(m.group(3)) / 100.0
            results[dist] = (orig, repaired)

    return results


def _parse_prorepair_run(run_index: int, run_total: int, block: str) -> dict:
    success = "RESULT: Successfully repaired!" in block
    failed = "RESULT: Failed to repair." in block or not success

    original_acc_pct = _extract_float(block, rf"Before repair, accuracy on test set:\s*{FLOAT_RE}")
    repaired_acc_pct = _extract_float(block, rf"After repair, accuracy on test set:\s*{FLOAT_RE}")

    row = {
        "run_index": run_index,
        "run_total": run_total,
        "result": "failed" if failed else "success",
        "total_repair_time_sec": _extract_float(block, rf"Repair time\s*{FLOAT_RE}s"),
        "total_steps": None,
        "original_accuracy": (original_acc_pct / 100.0) if original_acc_pct is not None else None,
        "repaired_accuracy": (repaired_acc_pct / 100.0) if repaired_acc_pct is not None else None,
    }

    gen = _parse_generalization(block, "prorepair")
    for dist, (orig, repaired) in gen.items():
        row[f"gen_dist_{dist}_orig"] = orig
        row[f"gen_dist_{dist}_repaired"] = repaired

    return row


def _parse_aprnn_run(run_index: int, run_total: int, block: str) -> dict:
    success = "RESULT: APRNN successfully repaired." in block
    failed = not success

    original_acc_pct = _extract_float(block, rf"APRNN Drawdown:\s*{FLOAT_RE}%\s*\({FLOAT_RE}%\s*->\s*{FLOAT_RE}%\)")
    # group(1)=drawdown, group(2)=orig, group(3)=repaired if match exists
    drawdown_match = re.search(rf"APRNN Drawdown:\s*{FLOAT_RE}%\s*\({FLOAT_RE}%\s*->\s*{FLOAT_RE}%\)\.", block)
    orig = None
    repaired = None
    if drawdown_match:
        orig = float(drawdown_match.group(2)) / 100.0
        repaired = float(drawdown_match.group(3)) / 100.0

    row = {
        "run_index": run_index,
        "run_total": run_total,
        "result": "failed" if failed else "success",
        "total_repair_time_sec": _extract_float(block, rf"APRNN Time:\s*{FLOAT_RE}\s*seconds\."),
        "total_steps": None,
        "original_accuracy": orig,
        "repaired_accuracy": repaired,
    }

    gen = _parse_generalization(block, "aprnn")
    for dist, (orig_v, repaired_v) in gen.items():
        row[f"gen_dist_{dist}_orig"] = orig_v
        row[f"gen_dist_{dist}_repaired"] = repaired_v

    return row


def parse_log_file(log_path: Path) -> pd.DataFrame:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    approach = _infer_approach_from_path(log_path)
    if approach not in {"prorepair", "aprnn"}:
        return pd.DataFrame()

    metadata = _parse_metadata(text, log_path)
    blocks = _split_run_blocks(text, approach)

    rows = []
    for run_index, run_total, block in blocks:
        row = _parse_prorepair_run(run_index, run_total, block) if approach == "prorepair" else _parse_aprnn_run(run_index, run_total, block)
        row["source_log"] = str(log_path)
        row.update(metadata)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("run_index").reset_index(drop=True)


def _build_output_path(log_path: Path, input_root: Path, output_root: Path) -> Path:
    relative = log_path.relative_to(input_root)
    # Expected: {approach}/{model_name}/{repair_task}/{file}.log
    if len(relative.parts) < 4:
        out_dir = output_root / "misc"
    else:
        approach, model_name, repair_task = relative.parts[:3]
        perturbation = "nonzero"
        out_dir = output_root / f"{approach}_{model_name}_{repair_task}_{perturbation}"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{log_path.stem}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse prorepair/aprnn logs and generate summary CSV files.")
    parser.add_argument("--input-root", type=Path, default=Path("repair_result"))
    parser.add_argument("--output-root", type=Path, default=Path("repair_result/csv"))
    parser.add_argument("--pattern", type=str, default="*.log")
    args = parser.parse_args()

    log_files = sorted(
        p for p in args.input_root.rglob(args.pattern)
        if p.is_file() and _infer_approach_from_path(p) in {"prorepair", "aprnn"}
    )

    written = 0
    skipped = 0
    for log_path in log_files:
        df = parse_log_file(log_path)
        if df.empty:
            skipped += 1
            continue
        out_path = _build_output_path(log_path, args.input_root, args.output_root)
        df.to_csv(out_path, index=False)
        written += 1

    print(f"Processed {len(log_files)} log files. Wrote {written} CSVs, skipped {skipped} files.")


if __name__ == "__main__":
    main()
