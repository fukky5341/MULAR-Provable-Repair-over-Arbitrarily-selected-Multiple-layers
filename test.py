import argparse
import time
import gc

from repair.repair import repair_test
from FL.util import FLmode
from input_space.dataset import Dataset
from repair.args import RepairArgs, RepairMode, RepairTask
from LPsolver.solver import ObjectiveType


# experiment map (model, dataset, perturbation_pick, repair_task, num_runs, set_of_repairable_layers)
exe_map = {
    0: ( "mnist 9x100", Dataset.MNIST, 'nonzero', RepairTask.LocalRobustness, 20, "m9" ),
    1: ( "mnist 9x100", Dataset.MNIST, 'all', RepairTask.LocalRobustness, 20, "m9" ),
    2: ( "mnist_256x4", Dataset.MNIST, 'nonzero', RepairTask.LocalRobustness, 20, "m4" ),
    3: ( "mnist_256x4", Dataset.MNIST, 'all', RepairTask.LocalRobustness, 20, "m4" ),
    4: ( "cifar10_cnn1", Dataset.CIFAR10, 'all', RepairTask.LocalRobustness, 10, "cg" ),
    5: ( "gtsrb_cnn", Dataset.GTSRB, 'all', RepairTask.LocalRobustness, 10, "cg" ),
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

# repairable layers and max subrepair loops (ours_target)
repairable_layers_and_max_loops = {
    "last_1": ([], 10),
    "last_2": ([], 5),
    "last_3": ([], 4),
    "last_4": ([], 3),
    "m9_2_16": ([2, 16], 3),
    "m9_10_14": ([10, 14], 4),
    "m9_10_16": ([10, 16], 3),
    "m9_12_16": ([12, 16], 4),
    "m9_14_16": ([14, 16], 5),
    "m9_16": ([16, 16], 10),
    "m4_3_9": ([3, 9], 3),
    "m4_5_9": ([5, 9], 4),
    "m4_7_9": ([7, 9], 5),
    "m4_9": ([9, 9], 10),
    "cg_2_5": ([2, 5], 5),
    "cg_5_7": ([5, 7], 5),
    "cg_2_7": ([2, 7], 4),
    "cg_7": ([7, 7], 10),
}

# Affine-layer indices for architecture-independent last_N configurations.
affine_layers_by_model = {
    "mnist 9x100": [0, 2, 4, 6, 8, 10, 12, 14, 16],
    "mnist_256x4": [1, 3, 5, 7, 9],
    "cifar10_cnn1": [0, 2, 5, 7],
    "gtsrb_cnn": [0, 2, 5, 7],
}

# set of repairable layers and max subrepair loops (ours)
set_of_repairable_layers_and_max_loops = {
    "m9": [([16],10), ([14,16],5), ([12,14,16],4), ([10,12,14,16],3)],
    "m4": [([9],10), ([7,9],5), ([5,7,9],4), ([3,5,7,9],3)],
    "cg": [([7],10), ([5,7],5), ([2,5,7],4)]
}

# discontinuous layers
discontinuous_layers = {
    "m9_disc_0": ([8,12,16], 4),  #  [0,2,4,6,8,10,12,14,16]
    "m9_disc_1": ([6,10,14], 4),
    "m9_disc_2": ([10,12,14], 4),
    "m9_disc_3": ([8,10,12], 4),
}


def resolve_automatic_layer_config(model_name, config_name):
    """Resolve a last_N alias to exact affine indices for one architecture."""

    if not config_name.startswith("last_"):
        raise ValueError(f"Not an automatic layer configuration: {config_name}")
    layer_count = int(config_name.removeprefix("last_"))
    affine_layers = affine_layers_by_model[model_name]
    if layer_count < 1 or layer_count > len(affine_layers):
        raise ValueError(
            f"Configuration {config_name} requests {layer_count} layers, but "
            f"{model_name} has only {len(affine_layers)} affine layers."
        )
    _unused_layers, max_subrepair_loops = \
        repairable_layers_and_max_loops[config_name]
    return affine_layers[-layer_count:], max_subrepair_loops


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exe-id", type=int, default=1, choices=sorted(exe_map.keys()))
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

    # repair settings
    parser.add_argument("--target_label", type=int, default=None)
    parser.add_argument("--inp_eps", type=float, default=1.0)  # would be divided by 255 for normalization
    parser.add_argument(
        "--repair-start-layer-idx",
        type=int,
        default=None,
        help="Override the first index in the target repair layer range.",
    )
    parser.add_argument(
        "--repair-last-layer-idx",
        type=int,
        default=None,
        help="Override the last index in the target repair layer range.",
    )
    parser.add_argument(
        "--repair-layer-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Repair exactly these affine layer indices in the given order. "
            "This enables non-contiguous layer sets and cannot be combined "
            "with start/last overrides or --repair-layer-config."
        ),
    )
    parser.add_argument(
        "--repair-layer-config",
        choices=sorted(
            set(repairable_layers_and_max_loops) | set(discontinuous_layers)
        ),
        default=None,
        help=(
            "Use a named layer configuration. last_1 through last_4 "
            "automatically select the corresponding number of final affine "
            "layers for the network chosen by --exe-id. Entries in "
            "discontinuous_layers select exactly the listed layers."
        ),
    )
    parser.add_argument(
        "--sequential-layer-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Try the predefined layer sets in order and advance only after "
            "failure. Each candidate starts from the original unmodified "
            "network. This is the default when no explicit layer selection "
            "is supplied; use --no-sequential-layer-candidates for the "
            "legacy default start/last range."
        ),
    )

    # optimization
    parser.add_argument("--lambda_strength_dist", type=float, default=1000)
    parser.add_argument("--lambda_strength_rate", type=float, default=1)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--modifiable_range", type=float, default=0.5)
    parser.add_argument(
        "--max-subrepair-loops",
        type=int,
        default=None,
        help=(
            "Maximum number of complete traversals of the modifiable layer "
            "set. In sequential mode, this overrides the configured limit "
            "for every candidate; otherwise it defaults to the named "
            "configuration value or 5."
        ),
    )
    parser.add_argument(
        "--subspace-search-batch-size",
        type=int,
        default=1,
        help="Number of candidate radii evaluated in each subspace search pass.",
    )
    parser.add_argument(
        "--subspace-search-method",
        choices=("binary", "exponential"),
        default="exponential",
        help=(
            "Search strategy used to expand certified subspaces. Shrinking "
            "always uses binary search."
        ),
    )
    parser.add_argument(
        "--subspace-search-growth-factor",
        type=float,
        default=2.0,
        help="Multiplicative candidate growth used by exponential search.",
    )

    # FL
    parser.add_argument("--perform_FL", type=bool, default=False)
    parser.add_argument(
        "--fl-mode",
        choices=tuple(mode.value for mode in FLmode),
        default=FLmode.ARACHNE.value,
        help=(
            "Fault-localization method. original_arachne localizes weights "
            "once across all configured layers, orders those layers by "
            "selected-neuron score, and reuses the fixed masks during repair."
        ),
    )
    parser.add_argument(
        "--arachne-layer-order",
        choices=("max_neuron_score", "selected_parameter_count"),
        default="max_neuron_score",
        help=(
            "How original_arachne orders selected layers. "
            "selected_parameter_count uses "
            "|W_arachne intersect W_layer|."
        ),
    )
    parser.add_argument(
        "--arachne-negative-source",
        choices=("region", "test"),
        default="region",
        help=(
            "Negative samples used by original_arachne: representative "
            "region points or concrete misclassified test-set samples."
        ),
    )
    parser.add_argument(
        "--arachne-negative-sample-count",
        type=int,
        default=200,
        help=(
            "Maximum number of misclassified test samples used when "
            "--arachne-negative-source=test."
        ),
    )
    parser.add_argument(
        "--arachne-parameter-selection",
        choices=("original", "random_matched", "random_layer_matched"),
        default="original",
        help=(
            "Use Original-Arachne-selected weights, or uniformly sample the "
            "same total number globally (random_matched), or preserve "
            "Arachne's selected count separately in every layer "
            "(random_layer_matched)."
        ),
    )
    parser.add_argument(
        "--arachne-random-seed",
        type=int,
        default=0,
        help=(
            "Base random seed for random parameter selection. "
            "The repair run index is added to this seed."
        ),
    )
    parser.add_argument("--fl_k_ratio", type=float, default=0.01)
    parser.add_argument("--pareto_round", type=int, default=3)

    args = parser.parse_args()
    range_override_used = (
        args.repair_start_layer_idx is not None
        or args.repair_last_layer_idx is not None
    )
    if args.repair_layer_indices is not None and (
            range_override_used or args.repair_layer_config is not None):
        parser.error(
            "--repair-layer-indices cannot be combined with start/last layer "
            "overrides or --repair-layer-config."
        )
    if args.repair_layer_config is not None and range_override_used:
        parser.error(
            "--repair-layer-config cannot be combined with start/last layer "
            "overrides."
        )
    explicit_layer_selection = (
        args.repair_layer_indices is not None
        or args.repair_layer_config is not None
        or range_override_used
    )
    if explicit_layer_selection:
        args.sequential_layer_candidates = False
    return args



# run the repair test
if __name__ == "__main__":

    # current time
    start_time = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"Starting time: {start_time}", flush=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = f"log/test_log_{timestamp}.txt"

    # cli
    cli_args = parse_args()

    (
        model_name,
        dataset,
        pick,
        repair_task,
        num_runs,
        layer_candidate_key,
    ) = exe_map[cli_args.exe_id]
    if cli_args.num_runs is not None:
        num_runs = cli_args.num_runs
    num_v_polys_list = cli_args.num_v_polys or (
        N_list_map[cli_args.exe_id] if pick == 'all' else [1]
    )
    perturbation_ndim_list = cli_args.perturbation_ndim or (
        ndims_map[cli_args.exe_id] if pick == 'nonzero' else [10]
    )

    for vpolys_num in num_v_polys_list:
        for perturbation_ndim in perturbation_ndim_list:

            args = RepairArgs(repair_task=repair_task,
                            model_name=model_name, 
                            dataset=dataset, 
                            num_runs=num_runs,
                            logfile=logfile)
            
            args.debug = False

            # repair settings
            args.repair_mode = RepairMode.LP_ibp
            args.target_label = cli_args.target_label
            args.inp_eps = cli_args.inp_eps
            args.num_v_polys = vpolys_num
            args.perturbation_pick = pick
            args.perturbation_ndim = perturbation_ndim
            configured_max_subrepair_loops = None

            # layer to repair
            if args.repair_mode == RepairMode.LP_hard or args.repair_mode == RepairMode.LP_ibp:
                start_last_layer_idx_map = {
                    "mnist 9x100": (14, 16),  # [0,2,4,6,8,10,12,14,16]
                    "mnist_256x4": (7, 9),  # [1,3,5,7,9]
                    "cifar10_cnn1": (5, 7),  # conv: [0,2], linear: [5,7]
                    "gtsrb_cnn": (5, 7),  # conv: [0,2], linear: [5,7]
                }
                if cli_args.sequential_layer_candidates:
                    candidate_configs = \
                        set_of_repairable_layers_and_max_loops[
                            layer_candidate_key
                        ]
                    args.repair_layer_candidates = [
                        list(layer_set)
                        for layer_set, _max_loops in candidate_configs
                    ]
                    args.repair_layer_candidate_max_loops = [
                        max_loops
                        for _layer_set, max_loops in candidate_configs
                    ]
                    args.repair_start_layer_idx = min(
                        min(layer_set)
                        for layer_set in args.repair_layer_candidates
                    )
                    args.repair_last_layer_idx = max(
                        max(layer_set)
                        for layer_set in args.repair_layer_candidates
                    )
                elif (
                    cli_args.repair_layer_config is not None
                    and cli_args.repair_layer_config.startswith("last_")
                ):
                    layer_indices, configured_max_subrepair_loops = \
                        resolve_automatic_layer_config(
                            model_name, cli_args.repair_layer_config
                        )
                    args.repair_layer_indices = list(layer_indices)
                    args.repair_start_layer_idx = min(layer_indices)
                    args.repair_last_layer_idx = max(layer_indices)
                elif cli_args.repair_layer_config in discontinuous_layers:
                    layer_indices, configured_max_subrepair_loops = \
                        discontinuous_layers[cli_args.repair_layer_config]
                    args.repair_layer_indices = list(layer_indices)
                    args.repair_start_layer_idx = min(layer_indices)
                    args.repair_last_layer_idx = max(layer_indices)
                elif cli_args.repair_layer_config in repairable_layers_and_max_loops:
                    layer_range, configured_max_subrepair_loops = \
                        repairable_layers_and_max_loops[
                            cli_args.repair_layer_config
                        ]
                    args.repair_start_layer_idx, args.repair_last_layer_idx = \
                        layer_range
                elif cli_args.repair_layer_indices is not None:
                    args.repair_layer_indices = list(
                        cli_args.repair_layer_indices
                    )
                    args.repair_start_layer_idx = min(args.repair_layer_indices)
                    args.repair_last_layer_idx = max(args.repair_layer_indices)
                else:
                    args.repair_start_layer_idx, args.repair_last_layer_idx = \
                        start_last_layer_idx_map[model_name]
                    if cli_args.repair_start_layer_idx is not None:
                        args.repair_start_layer_idx = \
                            cli_args.repair_start_layer_idx
                    if cli_args.repair_last_layer_idx is not None:
                        args.repair_last_layer_idx = \
                            cli_args.repair_last_layer_idx
            else:
                raise ValueError(f"Unsupported repair mode: {args.repair_mode}")
            if cli_args.max_subrepair_loops is not None:
                args.max_subrepair_loops = cli_args.max_subrepair_loops
                if args.repair_layer_candidates is not None:
                    args.repair_layer_candidate_max_loops = [
                        cli_args.max_subrepair_loops
                        for _candidate in args.repair_layer_candidates
                    ]
            elif configured_max_subrepair_loops is not None:
                args.max_subrepair_loops = configured_max_subrepair_loops
            elif args.repair_layer_candidate_max_loops is not None:
                args.max_subrepair_loops = \
                    args.repair_layer_candidate_max_loops[0]
            else:
                args.max_subrepair_loops = 5

            # optimization settings
            args.obj_regularization = "l2"  # fixed to l2
            args.obj_type = ObjectiveType.POLYTOPE_DISTANCE
            args.lambda_strength_dist = cli_args.lambda_strength_dist
            args.lambda_strength_rate = cli_args.lambda_strength_rate
            args.lambda_reg = cli_args.lambda_reg
            args.modifiable_range = cli_args.modifiable_range
            args.subspace_search_batch_size = cli_args.subspace_search_batch_size
            args.subspace_search_method = cli_args.subspace_search_method
            args.subspace_search_growth_factor = \
                cli_args.subspace_search_growth_factor
            args.use_priority_weight = False

            # FL parameters
            args.perform_FL = cli_args.perform_FL
            args.flmode = FLmode(cli_args.fl_mode)
            args.arachne_layer_order = cli_args.arachne_layer_order
            args.arachne_negative_source = cli_args.arachne_negative_source
            args.arachne_negative_sample_count = \
                cli_args.arachne_negative_sample_count
            args.arachne_parameter_selection = \
                cli_args.arachne_parameter_selection
            args.arachne_random_seed = cli_args.arachne_random_seed
            args.fl_k_ratio = cli_args.fl_k_ratio
            args.pareto_round = cli_args.pareto_round

            repair_test(args)
            
            gc.collect()
