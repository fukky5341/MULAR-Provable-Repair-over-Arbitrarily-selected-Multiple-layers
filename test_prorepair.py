import gc
import argparse
import subprocess
import sys
from pathlib import Path
from repair.args import RepairTask
from repair.logging import completed_external_runs_exist

script = Path(__file__).parent / "ProRepair" / "repair_rob.py"

exe_map = {
    0: ("mnist 9x100", 'nonzero'),
    1: ("mnist 9x100", 'all'),
    2: ("mnist_256x4", 'nonzero'),
    3: ("mnist_256x4", 'all'),
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

parser = argparse.ArgumentParser(description="Run ProRepair with an experiment setting from exe_map.")
parser.add_argument(
    "--exe-id",
    type=int,
    default=1,
    choices=sorted(exe_map.keys()),
    help="Key into exe_map selecting the model/pick combination.",
)
parser.add_argument(
    "--num-runs",
    type=int,
    default=20,
    help="Number of repair instances per configuration (default: %(default)s).",
)
parser.add_argument(
    "--num-v-polys",
    type=int,
    nargs="+",
    default=None,
    help="Optional numbers of repair regions; defaults to the benchmark grid.",
)
parser.add_argument(
    "--perturbation-ndim",
    type=int,
    nargs="+",
    default=None,
    help="Optional perturbed-dimension values; defaults to the benchmark grid.",
)
args = parser.parse_args()

N_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]
ndims_list = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

model_name, pick = exe_map[args.exe_id]  # 'all' or 'nonzero'

if pick == 'all':
    N_values = args.num_v_polys or N_list
    ndims_values = args.perturbation_ndim or [10]
else:
    N_values = args.num_v_polys or [1]
    ndims_values = args.perturbation_ndim or ndims_list

for N in N_values:
    for ndims in ndims_values:
        result_path = (
            Path(__file__).parent
            / "result"
            / "prorepair"
            / model_name
            / repair_task_map[repair_task_id].value
            / f"eps1_{pick}_N{N}_ndims{ndims}.log"
        )
        if completed_external_runs_exist(
            result_path,
            args.num_runs,
            r"^\*{10}\s*Run\s+(\d+)/(\d+)\s*\*{10}\s*$",
            r"^RESULT: (?:Successfully repaired!|Failed to repair\.)\s*$",
        ):
            print(f"Skipping completed result: {result_path}", flush=True)
            continue

        cmd = [
            sys.executable,
            str(script),
            "--dataset", "MNIST",
            "--model", model_name,
            "--repair_task", repair_task_map[repair_task_id].value,
            "--num_runs", str(args.num_runs),
            "--N", str(N),
            "--eps", "1",  # value/255
            "--pick", pick,
            "--ndims", str(ndims),
            "--device", "cpu",
            "--dtype", "float32",
        ]

        subprocess.run(cmd, check=True)
        gc.collect()
