import gc
import argparse
import subprocess
import sys
from pathlib import Path
from repair.args import RepairTask
from repair.logging import completed_external_runs_exist

script = Path(__file__).parent / "APRNN" / "eval_7_aprnn.py"

net_map = {
    0: "mnist 9x100",
    1: "mnist_256x4",
}

# repair task
repair_task_map = {
    0: RepairTask.LocalRobustness,
}
repair_task_id = 0

'''
max ndims for mnist: 28*28 = 784
max ndims for cifar10: 3*32*32 = 3072
'''

parser = argparse.ArgumentParser(description="Run APRNN eval_7 across a range of ndims.")
parser.add_argument(
    "--net-id",
    type=int,
    default=0,
    choices=sorted(net_map.keys()),
    help="Key into net_map selecting which network to evaluate.",
)
parser.add_argument(
    "--num-runs",
    type=int,
    default=20,
    help="Number of repair instances per configuration (default: %(default)s).",
)
parser.add_argument(
    "--ndims",
    type=int,
    nargs="+",
    default=None,
    help="Optional perturbed-dimension values; defaults to the full benchmark grid.",
)
args = parser.parse_args()

ndims_list = args.ndims or [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

for ndims in ndims_list:
    result_path = (
        Path(__file__).parent
        / "result"
        / "aprnn"
        / net_map[args.net_id]
        / repair_task_map[repair_task_id].value
        / f"eps1_nonzero_N1_ndims{ndims}.log"
    )
    if completed_external_runs_exist(
        result_path,
        args.num_runs,
        r"^\*{6}\s*Run\s+(\d+)/(\d+)\s*\*{6}\s*$",
        r"^RESULT: APRNN (?:successfully repaired|failed to repair)\.\s*$",
    ):
        print(f"Skipping completed result: {result_path}", flush=True)
        continue

    cmd = [
        sys.executable,
        str(script),
        "--dataset", "MNIST",
        "--net", net_map[args.net_id],
        "--repair_task", repair_task_map[repair_task_id].value,
        "--num_runs", str(args.num_runs),
        "--eps", "1",  # value/255
        "--pick", "nonzero",  # 'all' or 'nonzero'
        "--ndims", str(ndims),
        "--device", "cpu",
    ]

    subprocess.run(cmd, check=True)
    gc.collect()
