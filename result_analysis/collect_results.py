"""Collect legacy and current repair logs into a canonical CSV layout.

The project has used two log layouts:

* one text file containing every run (legacy), and
* one ``runN.txt`` file per run inside an ``eps*_N*`` directory (current).

This module deliberately reads the information printed by the program as well as
the directory structure.  The resulting CSVs therefore remain usable if either
layout is present in ``repair_result``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RUN_HEADING_RE = re.compile(r"^=== Repair run (\d+)/(\d+) ===\s*$", re.MULTILINE)
CONFIG_RE = re.compile(
    r"^eps(?P<eps>[^_]+)(?:_nonzero)?_N(?P<n>\d+)(?:(?:_(?P<ndims_a>\d+)dims)|(?:_ndims(?P<ndims_b>\d+)))?(?P<suffix>.*)$"
)


def _match(text: str, pattern: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _number(text: str, label: str) -> float | None:
    value = _match(
        text,
        rf"^{re.escape(label)}:\s*({FLOAT})(?:\s+(?:seconds?|secs?|%))?\s*$",
        re.MULTILINE,
    )
    return float(value) if value is not None else None


def _integer(text: str, label: str) -> int | None:
    value = _match(text, rf"^{re.escape(label)}:\s*(\d+)\s*$", re.MULTILINE)
    return int(value) if value is not None else None


def _int_list(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item) for item in re.findall(r"-?\d+", value))


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().replace("/", "-")
    return re.sub(r"\s+", "_", text)


def _config_metadata(config_name: str) -> dict[str, Any]:
    match = CONFIG_RE.match(config_name)
    if not match:
        return {"config_name": config_name}
    result: dict[str, Any] = {
        "config_name": config_name,
        "inp_eps": match.group("eps"),
        "buggy_points_num": int(match.group("n")),
        "ndims": int(match.group("ndims_a") or match.group("ndims_b"))
        if (match.group("ndims_a") or match.group("ndims_b"))
        else pd.NA,
    }
    suffix = match.group("suffix") or ""
    pareto = re.search(r"(?:^|_)pr(\d+)(?:_|$)", suffix)
    if pareto:
        result["pareto_rounds"] = int(pareto.group(1))
    order = re.search(r"_order_(.+?)(?:_neg_|_seed|$)", suffix)
    if order:
        result["arachne_layer_order"] = order.group(1)
    negative = re.search(r"_neg_(.+?)(?:_seed|$)", suffix)
    if negative:
        result["arachne_negative_source"] = negative.group(1)
    parameter_mode = re.search(r"_params_(.+?)(?:_order_|_neg_|_seed|$)", suffix)
    if parameter_mode:
        result["arachne_parameter_selection"] = parameter_mode.group(1)
    return result


def _relative_parts(path: Path, result_root: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(result_root).parts
    except ValueError:
        return path.parts


def infer_path_metadata(path: Path, result_root: Path) -> dict[str, Any]:
    """Infer experiment metadata without relying on a fixed path depth."""

    parts = _relative_parts(path, result_root)
    if not parts:
        return {}
    top = parts[0]
    per_run = bool(re.fullmatch(r"run\d+\.txt", path.name))
    config_name = path.parent.name if per_run else path.stem
    result: dict[str, Any] = {"source_log": str(path), **_config_metadata(config_name)}

    # Defaults are subsequently refined from the log header.
    result.update(
        {
            "approach": top,
            "variant": "default",
            "model_name": pd.NA,
            "repair_task": pd.NA,
            "perturbation": pd.NA,
            "search_strategy": pd.NA,
        }
    )
    if top == "ours_target" and len(parts) >= 6:
        result.update(
            approach="ours",
            variant=parts[1],
            model_name=parts[2],
            repair_task=parts[3],
            perturbation=parts[4],
        )
        if parts[5].startswith("search_"):
            result["search_strategy"] = parts[5]
        layers = _int_list(parts[1])
        if layers:
            result["modifiable_layers"] = ",".join(map(str, layers))
            result["layer_set_label"] = "layers_" + "_".join(map(str, layers))
    elif top == "FL" and len(parts) >= 6:
        result.update(
            approach="FL",
            variant=_slug(parts[1]),
            model_name=parts[2],
            repair_task=parts[3],
            perturbation=parts[4],
        )
        if parts[5].startswith("search_"):
            result["search_strategy"] = parts[5]
        if "random" in parts[1].lower():
            result.setdefault("arachne_parameter_selection", "random_matched")
        else:
            result.setdefault("arachne_parameter_selection", "original")
    elif (
        top == "ours"
        and len(parts) >= 7
        and parts[1] == "sequential_reset"
    ):
        result.update(
            approach="ours",
            variant="sequential_reset",
            model_name=parts[3],
            repair_task=parts[4],
            perturbation=parts[5],
        )
    elif top in {"ours", "prepared", "lastlayer", "aprnn", "prorepair"} and len(parts) >= 4:
        perturbation = "nonzero" if path.suffix == ".log" else parts[3]
        result.update(model_name=parts[1], repair_task=parts[2], perturbation=perturbation)
        if top == "ours":
            result["approach"] = "ours"
        elif top == "lastlayer":
            result["variant"] = "lastlayer"
        if top == "ours" and "_" in parts[3]:
            perturbation, start = parts[3].rsplit("_", 1)
            if start.isdigit():
                result["perturbation"] = perturbation
                result["variant"] = f"start_layer_{start}"
                result["start_layer"] = int(start)
    return result


def _header_metadata(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = {
        "model_name": "Network",
        "repair_task": "Repair task",
        "repair_mode": "Repair mode",
        "perturbation": "Perturbation pick",
        "repair_layer_selection": "Repair layer selection",
        "arachne_layer_order": "Arachne layer order",
        "arachne_negative_source": "Arachne negative source",
        "arachne_parameter_selection": "Arachne parameter selection",
    }
    for key, label in fields.items():
        value = _match(text, rf"^{re.escape(label)}:\s*(.+?)\s*$", re.MULTILINE)
        if value is not None:
            result[key] = value
    result["search_strategy"] = _match(
        text,
        r"^(?:Subspace expansion search method|Subspace search strategy):\s*(.+?)\s*$",
        re.MULTILINE,
    )
    result["search_batch_size"] = _integer(text, "Subspace search batch size")
    result["search_growth_factor"] = _number(
        text, "Subspace search growth factor"
    )
    result["start_time"] = _match(text, r"^Start time:\s*(.+?)\s*$", re.MULTILINE)
    result["perturbation_distance"] = _number(text, "Perturbation distance")
    result["num_v_polys"] = _integer(text, "Number of v-polytopes")
    run = re.search(r"^Run:\s*(\d+)/(\d+)\s*$", text, re.MULTILINE)
    if run:
        result["header_run_index"] = int(run.group(1))
        result["header_run_total"] = int(run.group(2))
    layers = _match(
        text,
        r"^(?:Modifiable layers|Repaired layer indices):\s*(\[[^\]]*\])\s*$",
        re.MULTILINE,
    )
    parsed_layers = _int_list(layers)
    if parsed_layers:
        result["modifiable_layers"] = ",".join(map(str, parsed_layers))
        result["layer_set_label"] = "layers_" + "_".join(map(str, parsed_layers))
        result["start_layer"] = min(parsed_layers)
    return result


def _failure_reason(text: str) -> str | None:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "soft memory limit reached",
            "memory limit reached",
            "memory limit exceeded",
            "out of memory",
        )
    ):
        return "memory_limit"
    if "gurobi" in lowered and any(
        marker in lowered
        for marker in ("time limit reached", "timelimit reached", "timed out")
    ):
        return "gurobi_time_limit"
    if any(
        marker in lowered
        for marker in ("time limit reached", "timed out", "timeout")
    ):
        return "time_limit"
    if "keyboardinterrupt" in lowered or "received signal" in lowered or "terminated by signal" in lowered:
        return "interrupted"
    if "repair failed" in lowered or "failed to repair" in lowered:
        return "repair_failed"
    if "exception" in lowered or "traceback" in lowered:
        return "exception"
    return None


def _run_record(text: str, run_index: int, run_total: int) -> dict[str, Any]:
    data_ids = _match(
        text,
        r"^Data ids for (?:this repair run|repair regions in this run):\s*(\[[^\]]*\])\s*$",
        re.MULTILINE,
    )
    ids = _int_list(data_ids)
    input_construction_time = _match(
        text,
        rf"^Input construction time(?: .*?)?:\s*({FLOAT})"
        rf"(?:\s+(?:seconds?|secs?))?\s*$",
        re.MULTILINE,
    )
    reason = _failure_reason(text)
    repaired_accuracy = _number(text, "Repaired accuracy")
    success = "successfully repaired the dnn" in text.lower() or repaired_accuracy is not None
    if success:
        status = "success"
        reason = None
    elif reason is not None:
        status = "failed"
    else:
        status = "incomplete"
        reason = "log_ended_before_result"

    record: dict[str, Any] = {
        "run_index": run_index,
        "run_total": run_total,
        "result": status,
        "failure_reason": reason,
        "repair_data_ids": ",".join(map(str, ids)) if ids else pd.NA,
        "repair_data_count": len(ids) if ids else pd.NA,
        "total_repair_time_sec": _number(text, "Total repair time"),
        "input_construction_time_sec": (
            float(input_construction_time)
            if input_construction_time is not None else pd.NA
        ),
        "total_steps": _integer(text, "Total steps"),
        "original_accuracy": _number(text, "Original accuracy"),
        "repaired_accuracy": repaired_accuracy,
        "repair_rate": _number(text, "Repair rate"),
        "safety_rate": _number(text, "Safety rate"),
        "drawdown": _number(text, "Drawdown"),
        "break_rate": _number(text, "Break rate"),
    }
    if record["original_accuracy"] is not None and repaired_accuracy is not None:
        record["accuracy_drop"] = record["original_accuracy"] - repaired_accuracy
    else:
        record["accuracy_drop"] = pd.NA
    return record


def parse_txt_log(path: Path, result_root: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    path_metadata = infer_path_metadata(path, result_root)
    header = _header_metadata(text)
    headings = list(RUN_HEADING_RE.finditer(text))
    records: list[dict[str, Any]] = []
    if headings:
        for position, heading in enumerate(headings):
            end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
            summary = re.search(r"^=== Summary of Repair Results ===\s*$", text[heading.end() : end], re.MULTILINE)
            if summary:
                end = heading.end() + summary.start()
            block = text[heading.start() : end]
            index = int(heading.group(1))
            total = int(heading.group(2))
            # Current one-file-per-run logs also print Run: N/M in their header.
            if len(headings) == 1 and "header_run_index" in header:
                index = int(header["header_run_index"])
                total = int(header["header_run_total"])
            records.append(_run_record(block, index, total))
    else:
        index = int(header.get("header_run_index", 1))
        total = int(header.get("header_run_total", 1))
        records.append(_run_record(text, index, total))

    # Accuracy is commonly emitted only in a summary after all run blocks.
    accuracy_by_run = {
        int(match.group(1)): (float(match.group(2)), float(match.group(3)))
        for match in re.finditer(
            rf"^Run\s+(\d+):\s*Original accuracy\s*=\s*({FLOAT}),\s*"
            rf"Repaired accuracy\s*=\s*({FLOAT})\s*$",
            text,
            re.MULTILINE,
        )
    }
    for record in records:
        accuracy = accuracy_by_run.get(int(record["run_index"]))
        if accuracy is not None:
            record["original_accuracy"], record["repaired_accuracy"] = accuracy
            record["accuracy_drop"] = accuracy[0] - accuracy[1]

    common = {**path_metadata, **{key: value for key, value in header.items() if not key.startswith("header_")}}
    for record in records:
        record.update(common)
        if record.get("repair_data_count") is pd.NA and path_metadata.get("buggy_points_num") is not None:
            record["repair_data_count"] = path_metadata["buggy_points_num"]
    return pd.DataFrame(records)


def parse_other_log(path: Path, result_root: Path) -> pd.DataFrame:
    """Normalize APRNN/PRDNN logs parsed by their format-specific parser."""

    try:
        from .log_to_csv_others import parse_log_file
    except ImportError:  # Direct script execution.
        from log_to_csv_others import parse_log_file

    frame = parse_log_file(path)
    if frame.empty:
        return frame
    metadata = infer_path_metadata(path, result_root)
    for key, value in metadata.items():
        frame[key] = value
    if "network" in frame.columns:
        frame["model_name"] = frame["network"].fillna(frame["model_name"])
    if "perturbation_pick" in frame.columns:
        frame["perturbation"] = frame["perturbation_pick"].fillna(frame["perturbation"])
    if "perturbation_ndim" in frame.columns:
        frame["ndims"] = frame["perturbation_ndim"].fillna(frame.get("ndims"))
    frame["failure_reason"] = frame["result"].map(
        lambda value: "repair_failed" if value == "failed" else pd.NA
    )
    frame["accuracy_drop"] = frame["original_accuracy"] - frame["repaired_accuracy"]
    return frame


def _output_path(row: pd.Series, output_root: Path) -> Path:
    return (
        output_root
        / _slug(row.get("approach"))
        / _slug(row.get("variant"))
        / _slug(row.get("model_name"))
        / _slug(row.get("repair_task"))
        / _slug(row.get("perturbation"))
        / f"{_slug(row.get('config_name'))}.csv"
    )


def collect_results(
    result_root: str | Path,
    output_root: str | Path | None = None,
    *,
    clean: bool = True,
) -> list[Path]:
    """Parse every repair log and write grouped canonical CSV files."""

    result_root = Path(result_root).resolve()
    output_root = Path(output_root).resolve() if output_root else result_root / "csv"
    if clean and output_root.exists():
        marker = output_root / ".canonical-repair-results"
        default_output = result_root / "csv"
        if output_root != default_output and not marker.exists():
            raise ValueError(
                f"Refusing to replace unmarked output directory: {output_root}. "
                "Use --no-clean or choose an empty directory."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / ".canonical-repair-results").write_text(
        "Generated by result_analysis.collect_results; safe to rebuild.\n",
        encoding="utf-8",
    )
    grouped: dict[Path, list[pd.DataFrame]] = defaultdict(list)
    for path in sorted(result_root.rglob("*.txt")):
        if output_root in path.parents:
            continue
        frame = parse_txt_log(path, result_root)
        if frame.empty:
            continue
        grouped[_output_path(frame.iloc[0], output_root)].append(frame)
    for path in sorted(result_root.rglob("*.log")):
        if output_root in path.parents or not any(part in {"aprnn", "prorepair"} for part in path.parts):
            continue
        frame = parse_other_log(path, result_root)
        if frame.empty:
            continue
        grouped[_output_path(frame.iloc[0], output_root)].append(frame)

    written: list[Path] = []
    for output_path, frames in grouped.items():
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["_status_rank"] = combined["result"].map({"success": 2, "failed": 1, "incomplete": 0}).fillna(0)
        combined = combined.sort_values(["run_index", "_status_rank"]).drop_duplicates(
            subset=["run_index"], keep="last"
        )
        combined = combined.drop(columns="_status_rank").sort_values("run_index")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        written.append(output_path)
    return sorted(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="repair_result", help="Root containing raw result logs")
    parser.add_argument("--output", default=None, help="Canonical CSV root (default: INPUT/csv)")
    parser.add_argument("--no-clean", action="store_true", help="Do not replace the existing CSV tree")
    args = parser.parse_args()
    written = collect_results(args.input, args.output, clean=not args.no_clean)
    print(f"Wrote {len(written)} canonical CSV files.")


if __name__ == "__main__":
    main()
