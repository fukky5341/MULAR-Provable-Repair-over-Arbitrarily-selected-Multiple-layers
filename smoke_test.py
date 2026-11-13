#!/usr/bin/env python3
"""Run a small MNIST-F1 experiment with every evaluated repair approach."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from result_analysis.smoke_summary import generate_smoke_summary


REPOSITORY_ROOT = Path(__file__).resolve().parent


def approach_commands(num_runs: int, ndims: int) -> dict[str, list[str]]:
    common = [
        "--exe-id",
        "0",
        "--num-runs",
        str(num_runs),
        "--num-v-polys",
        "1",
        "--perturbation-ndim",
        str(ndims),
    ]
    return {
        "APRNN": [
            sys.executable,
            "test_aprnn.py",
            "--net-id",
            "0",
            "--num-runs",
            str(num_runs),
            "--ndims",
            str(ndims),
        ],
        "ProRepair": [sys.executable, "test_prorepair.py", *common],
        "PREPARED": [sys.executable, "test_prepared.py", *common],
        "LLR": [sys.executable, "test_lastlayer.py", *common],
        "MULAR": [sys.executable, "test.py", *common],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="number of repair instances per approach (default: %(default)s)",
    )
    parser.add_argument(
        "--ndims",
        type=int,
        default=5,
        help="number of perturbed input dimensions (default: %(default)s)",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        choices=("APRNN", "ProRepair", "PREPARED", "LLR", "MULAR"),
        help="optional subset of approaches; by default all are run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without running them",
    )
    args = parser.parse_args()
    if args.num_runs < 1:
        parser.error("--num-runs must be at least 1")
    if args.ndims < 1:
        parser.error("--ndims must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    commands = approach_commands(args.num_runs, args.ndims)
    selected = args.approaches or list(commands)
    total_started = time.monotonic()

    print(
        f"Small experiment: MNIST-F1, {args.num_runs} instances, "
        f"N=1, #Dim={args.ndims}",
        flush=True,
    )
    for position, approach in enumerate(selected, start=1):
        command = commands[approach]
        rendered = " ".join(command)
        print(f"\n[{position}/{len(selected)}] {approach}: {rendered}", flush=True)
        if args.dry_run:
            continue
        started = time.monotonic()
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        print(
            f"[{approach}] completed in {time.monotonic() - started:.1f}s",
            flush=True,
        )

    if args.dry_run:
        print("\nDry run completed; no experiments were executed.", flush=True)
    else:
        report_path, report = generate_smoke_summary(
            REPOSITORY_ROOT / "result",
            approaches=selected,
            num_runs=args.num_runs,
            ndims=args.ndims,
        )
        print(
            f"\nSmall experiment completed successfully in "
            f"{time.monotonic() - total_started:.1f}s. Results are in result/.",
            flush=True,
        )
        print(f"\n{report}", flush=True)
        print(f"Tables written to {report_path.parent}", flush=True)


if __name__ == "__main__":
    main()
