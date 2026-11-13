"""Summarize the lightweight smoke-test results into three simple tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .analysis import load_all_csv_rows
    from .collect_results import collect_results
except ImportError:  # Direct script execution.
    from analysis import load_all_csv_rows
    from collect_results import collect_results


APPROACHES = {
    "APRNN": ("aprnn", "/aprnn/"),
    "ProRepair": ("prorepair", "/prorepair/"),
    "PREPARED": ("prepared", "/prepared/"),
    "LLR": ("ours", "/ours_target/layers_16/"),
    "MULAR": ("ours", "/ours/sequential_reset/"),
}


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    values = [headers] + [
        ["--" if pd.isna(value) else str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [max(len(row[index]) for row in values) for index in range(len(headers))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(values[0]), separator, *map(render, values[1:])])


def build_smoke_tables(
    rows: pd.DataFrame,
    *,
    approaches: list[str],
    num_runs: int,
    ndims: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build success, runtime, and accuracy-drop tables for one smoke test."""

    selected = rows.copy()
    for column in ("buggy_points_num", "ndims", "run_index"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected[
        (selected["model_name"] == "mnist 9x100")
        & (selected["perturbation"] == "nonzero")
        & (selected["buggy_points_num"] == 1)
        & (selected["ndims"] == ndims)
        & (selected["run_index"].between(1, num_runs))
    ]

    success_records: list[dict[str, object]] = []
    time_records: list[dict[str, object]] = []
    accuracy_records: list[dict[str, object]] = []
    for display_name in approaches:
        internal_name, source_fragment = APPROACHES[display_name]
        approach_rows = selected[selected["approach"] == internal_name].copy()
        if "source_log" in approach_rows.columns:
            normalized_sources = approach_rows["source_log"].astype(str).str.replace(
                "\\", "/", regex=False
            )
            matching_source = normalized_sources.str.contains(
                source_fragment, regex=False
            )
            if matching_source.any():
                approach_rows = approach_rows[matching_source]
        approach_rows = approach_rows.sort_values("source_log").drop_duplicates(
            subset="run_index", keep="last"
        )
        successful = approach_rows[approach_rows["result"] == "success"]
        timed = successful[
            pd.to_numeric(
                successful["total_repair_time_sec"], errors="coerce"
            ).notna()
        ].copy()
        timed["total_repair_time_sec"] = pd.to_numeric(
            timed["total_repair_time_sec"], errors="coerce"
        )
        accurate = successful[
            pd.to_numeric(successful["accuracy_drop"], errors="coerce").notna()
        ].copy()
        accurate["accuracy_drop"] = pd.to_numeric(
            accurate["accuracy_drop"], errors="coerce"
        )

        total = len(approach_rows)
        successes = len(successful)
        success_records.append(
            {
                "Approach": display_name,
                "Successes": successes,
                "Runs": total,
                "Success rate (%)": round(100.0 * successes / total, 1)
                if total
                else pd.NA,
            }
        )
        time_records.append(
            {
                "Approach": display_name,
                "Successful runs with time": len(timed),
                "Mean repair time (s)": round(
                    float(timed["total_repair_time_sec"].mean()), 2
                )
                if not timed.empty
                else pd.NA,
            }
        )
        accuracy_records.append(
            {
                "Approach": display_name,
                "Successful runs with accuracy": len(accurate),
                "Mean accuracy drop": round(
                    float(accurate["accuracy_drop"].mean()), 4
                )
                if not accurate.empty
                else pd.NA,
            }
        )

    return (
        pd.DataFrame(success_records),
        pd.DataFrame(time_records),
        pd.DataFrame(accuracy_records),
    )


def generate_smoke_summary(
    result_root: Path,
    *,
    approaches: list[str],
    num_runs: int,
    ndims: int,
) -> tuple[Path, str]:
    """Parse smoke-test logs, write CSV/Markdown tables, and return the report."""

    collect_results(result_root, clean=True)
    rows = load_all_csv_rows(result_root)
    if rows.empty:
        raise RuntimeError(f"No result rows found under {result_root}")
    tables = build_smoke_tables(
        rows,
        approaches=approaches,
        num_runs=num_runs,
        ndims=ndims,
    )
    output = result_root / "smoke_test_summary"
    output.mkdir(parents=True, exist_ok=True)
    filenames = ("success_counts.csv", "repair_time.csv", "accuracy_drop.csv")
    headings = ("Success counts", "Repair time", "Accuracy drop")
    sections = [
        "# Small Experiment: Aggregated Results",
        "",
        f"MNIST-F1, N=1, #Dim={ndims}, requested runs per approach={num_runs}.",
    ]
    for table, filename, heading in zip(tables, filenames, headings):
        table.to_csv(output / filename, index=False)
        sections.extend(["", f"## {heading}", "", _markdown_table(table)])
    report = "\n".join(sections) + "\n"
    report_path = output / "README.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--ndims", type=int, default=5)
    parser.add_argument(
        "--approaches",
        nargs="+",
        choices=tuple(APPROACHES),
        default=list(APPROACHES),
    )
    args = parser.parse_args()
    report_path, report = generate_smoke_summary(
        args.result_root,
        approaches=args.approaches,
        num_runs=args.num_runs,
        ndims=args.ndims,
    )
    print(report)
    print(f"Tables written to {report_path.parent}")


if __name__ == "__main__":
    main()
