import argparse
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


SUPERVISOR_INTERVAL_SECONDS = 60
SUPERVISED_CHILD_FLAG = "--supervised-child"


def _cgroup_file(filename):
    candidates = [Path("/sys/fs/cgroup") / filename]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _hierarchy, controllers, relative_path = line.split(":", 2)
            if not controllers:
                candidates.insert(
                    0,
                    Path("/sys/fs/cgroup")
                    / relative_path.lstrip("/")
                    / filename,
                )
    except (OSError, ValueError):
        pass
    return next((path for path in candidates if path.is_file()), None)


def _read_cgroup_events():
    path = _cgroup_file("memory.events")
    if path is None:
        return {}
    try:
        return {
            key: int(value)
            for key, value in (
                line.split(maxsplit=1) for line in path.read_text().splitlines()
            )
        }
    except (OSError, ValueError):
        return {}


def _read_bytes(path):
    if path is None:
        return None
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _process_rss_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _format_gib(value):
    return "unknown" if value is None else f"{value / 1024 ** 3:.2f} GiB"


def supervise_test_prepared():
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        SUPERVISED_CHILD_FLAG,
        *sys.argv[1:],
    ]
    events_before = _read_cgroup_events()
    started_at = time.time()
    print(
        "test_prepared supervisor started; child termination and cgroup OOM "
        "events will be reported.",
        flush=True,
    )
    child = subprocess.Popen(command)

    def forward_termination(signum, _frame):
        signal_name = signal.Signals(signum).name
        print(
            f"test_prepared supervisor received {signal_name} ({signum}); "
            "forwarding it to the child.",
            flush=True,
        )
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward_termination)
    signal.signal(signal.SIGINT, forward_termination)

    while True:
        try:
            return_code = child.wait(timeout=SUPERVISOR_INTERVAL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            rss = _process_rss_bytes(child.pid)
            cgroup_current = _read_bytes(_cgroup_file("memory.current"))
            cgroup_max = _read_bytes(_cgroup_file("memory.max"))
            events = _read_cgroup_events()
            elapsed = time.time() - started_at
            print(
                f"Supervisor heartbeat: elapsed={elapsed:.0f}s, "
                f"child_rss={_format_gib(rss)}, "
                f"cgroup_memory={_format_gib(cgroup_current)}/"
                f"{_format_gib(cgroup_max)}, "
                f"oom_kill={events.get('oom_kill', 'unknown')}",
                flush=True,
            )

    events_after = _read_cgroup_events()
    oom_kills = (
        events_after.get("oom_kill", 0) - events_before.get("oom_kill", 0)
    )
    elapsed = time.time() - started_at
    if oom_kills > 0:
        reason = f"cgroup OOM kill detected (oom_kill increased by {oom_kills})"
    elif return_code < 0:
        signal_number = -return_code
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = "unknown signal"
        reason = f"child killed by {signal_name} ({signal_number})"
    elif return_code == 0:
        reason = "child completed normally"
    elif 128 < return_code < 256:
        signal_number = return_code - 128
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = "unknown signal"
        reason = (
            f"child exited after {signal_name} ({signal_number}); "
            f"status={return_code}"
        )
    else:
        reason = f"child exited with status {return_code}"
    print(
        f"test_prepared supervisor: {reason}; elapsed={elapsed:.1f}s.",
        flush=True,
    )
    return 128 - return_code if return_code < 0 else return_code


if __name__ == "__main__" and SUPERVISED_CHILD_FLAG not in sys.argv:
    raise SystemExit(supervise_test_prepared())


from PREPARED.repair import repair_prepared
from input_space.dataset import Dataset
from repair.args import RepairArgs, RepairMode, RepairTask
from repair.logging import logging


active_args = None


def handle_termination_signal(signum, _frame):
    signal_name = signal.Signals(signum).name
    message = (
        f"test_prepared received {signal_name} ({signum}); "
        "terminating before completion."
    )
    if active_args is not None:
        logging(message, logfile=active_args.logfile)
    else:
        print(message, flush=True)
    raise SystemExit(128 + signum)


# experiment map (model, dataset, perturbation_pick, repair_task, num_runs)
exe_map = {
    0: ( "mnist 9x100", Dataset.MNIST, 'nonzero', RepairTask.LocalRobustness, 20 ),
    1: ( "mnist 9x100", Dataset.MNIST, 'all', RepairTask.LocalRobustness, 20 ),
    2: ( "mnist_256x4", Dataset.MNIST, 'nonzero', RepairTask.LocalRobustness, 20 ),
    3: ( "mnist_256x4", Dataset.MNIST, 'all', RepairTask.LocalRobustness, 20 ),
    4: ( "cifar10_cnn1", Dataset.CIFAR10, 'all', RepairTask.LocalRobustness, 10 ),
    5: ( "gtsrb_cnn", Dataset.GTSRB, 'all', RepairTask.LocalRobustness, 10 ),
}

# ndims_list
ndims_map = {
    0: [5, 6, 7, 8, 9, 10, 12, 14],
    1: [10],
    2: [5, 6, 7, 8, 9, 10, 12, 14],
    3: [10],
    4: [10],
    5: [10]
}

# N_list
N_list_map = {
    0: [1],
    1: [1, 2, 3, 4, 5, 6, 8, 10, 12, 14],
    2: [1],
    3: [1, 2, 3, 4, 5, 6, 8, 10, 12, 14],
    4: [12, 14, 16, 18, 20],
    5: [4, 6, 8, 10, 12]
}

if __name__ == "__main__":
    start_time = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"Starting time: {start_time}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(SUPERVISED_CHILD_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exe-id", type=int, default=4, choices=sorted(exe_map.keys()))
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Override the benchmark's number of repair instances.",
    )
    parser.add_argument(
        "--num-v-polys",
        type=int,
        nargs="+",
        default=None,
        help="Override the benchmark's repair-region counts.",
    )
    parser.add_argument(
        "--perturbation-ndim",
        type=int,
        nargs="+",
        default=None,
        help="Override the benchmark's perturbed-dimension values.",
    )
    parser.add_argument(
        "--gurobi-soft-mem-limit-gb",
        type=float,
        default=None,
        help=(
            "Gurobi soft memory limit in GB. By default, use 75%% of the "
            "smallest detected physical, cgroup, or Slurm memory limit. "
            "Set to 0 to disable."
        ),
    )
    cli_args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_termination_signal)
    signal.signal(signal.SIGINT, handle_termination_signal)

    model_name, dataset, pick, repair_task, num_runs = exe_map[cli_args.exe_id]
    if cli_args.num_runs is not None:
        num_runs = cli_args.num_runs
    num_v_polys_list = cli_args.num_v_polys or (
        N_list_map[cli_args.exe_id] if pick == 'all' else [1]
    )
    perturbation_ndim_list = cli_args.perturbation_ndim or (
        ndims_map[cli_args.exe_id] if pick == 'nonzero' else [10]
    )

    # current time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = f"log/test_log_{timestamp}.txt"

    for num_v_polys in num_v_polys_list:
        for perturbation_ndim in perturbation_ndim_list:
            args = RepairArgs(
                repair_task=repair_task,
                model_name=model_name,
                dataset=dataset,
                logfile=logfile,
            )

            # repair settings
            args.repair_mode = RepairMode.PREPARED
            args.target_label = None
            args.num_runs = num_runs
            args.inp_eps = 1.0
            args.num_v_polys = num_v_polys
            args.perturbation_pick = pick
            args.perturbation_ndim = perturbation_ndim
            args.modifiable_range = 0.5
            args.gurobi_soft_mem_limit_gb = cli_args.gurobi_soft_mem_limit_gb

            # PREPARED repairs the network suffix beginning at this layer.
            repair_start_layer_idx_map = {
                "mnist 9x100": 12,
                "mnist_256x4": 7,
                "cifar10_cnn1": 5,
                "gtsrb_cnn": 5,
            }
            args.repair_start_layer_idx = repair_start_layer_idx_map[model_name]

            active_args = args
            try:
                repair_prepared(args)
            except Exception as exc:
                logging(
                    f"test_prepared terminated with {type(exc).__name__}: "
                    f"{exc}",
                    logfile=args.logfile,
                )
                raise
            finally:
                active_args = None
            gc.collect()
