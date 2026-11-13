from enum import Enum
from typing import Optional
from dataclasses import dataclass
import torch

import sytorch as st

from LPsolver.solver import ObjectiveType
from FL.util import FLmode
from input_space.dataset import Dataset


class RepairMode(Enum):
    LP_hard = "lp_hard"
    LP_ibp = "lp_ibp"
    LP_SEQUENTIAL = "lp_sequential"
    BARRIER = "barrier"
    PREPARED = "prepared"
    LastLayer_ibp = "last_layer_ibp"


class RepairTask(Enum):
    LocalRobustness = "Local Robustness"  # local robustness for given safe center point and violating radius


# decide to use flattened input or not based on the dataset
input_shape_map = {
    "acasxu": (True, (5,)),
    "mnist 9x100": (True, (784,)),
    "mnist_256x4": (True, (784,)),
    "mnist_256x6": (True, (784,)),
    "mnist_conv": (False, (1, 28, 28)),
    "cifar10_cnn1": (False, (3, 32, 32)),
    "cifar10_cnn2": (False, (3, 32, 32)),
    "gtsrb_cnn": (False, (3, 32, 32)),
}

@dataclass
class RepairArgs:
    # experiment mode (to be provided)
    repair_task: RepairTask

    # network (to be provided)
    model_name: str

    # dataset (to be provided)
    dataset: Dataset

    # acasxu net key
    acasxu_net_key: Optional[tuple[int, int]] = None  # (aprev, tau)

    # Set parameters
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = st.float64

    # input shape
    @property
    def input_flatten(self):
        return input_shape_map.get(self.model_name, (True,))[0]

    @property
    def input_shape(self):
        return input_shape_map.get(self.model_name, (784,))[1]
    
    # number of runs
    num_runs: int = 1

    # repair settings
    target_label: int = None
    perturbation_pick: str = 'all'  # nonzero
    perturbation_ndim: int = 10  # number of dimensions to perturb (for nonzero pick)
    inp_eps: float = 0.005
    num_v_polys: int = 1
    modifiable_range: float = 0.5
    subspace_search_batch_size: int = 1
    subspace_search_method: str = "exponential"
    subspace_search_growth_factor: float = 2.0

    repair_start_layer_idx: int = 8
    repair_last_layer_idx: int = 12
    # When provided, repair exactly these affine layers in this order instead
    # of selecting every affine layer in the start/last range.
    repair_layer_indices: Optional[list[int]] = None
    # Ordered candidate layer sets. Each candidate is attempted only after the
    # previous one fails, starting again from the original network and regions.
    repair_layer_candidates: Optional[list[list[int]]] = None
    # Optional per-candidate traversal limits, aligned with the layer sets.
    repair_layer_candidate_max_loops: Optional[list[int]] = None

    repair_mode: RepairMode = RepairMode.LP_ibp

    # FL settings
    perform_FL: bool = False
    flmode: FLmode = FLmode.ARACHNE
    fl_k_ratio: float = 0.1  # topk
    pareto_round: int = 3
    arachne_layer_order: str = "max_neuron_score"
    arachne_negative_source: str = "region"
    arachne_negative_sample_count: int = 200
    arachne_parameter_selection: str = "original"
    arachne_random_seed: int = 0

    # log file
    logfile: str = "repair_test_log.txt"
    debug: bool = False

    # optimization settings
    # objective
    # regularization: "l1" or "l2"
    obj_regularization: str = "l2"
    obj_type: ObjectiveType = ObjectiveType.POLYTOPE_DISTANCE
    use_priority_weight: bool = False

    # strength of distance term in objective
    lambda_strength_dist: float = 1000
    lambda_strength_rate: float = 10
    lambda_reg: float = 1.0
    obj_weight_positive: float = 1000

    # Gurobi memory limit in GB. None lets PREPARED derive a conservative
    # limit from the host/cgroup/Slurm allocation; 0 disables the soft limit.
    gurobi_soft_mem_limit_gb: float | None = None

    # maximum number of complete traversals of the modifiable layer set
    max_subrepair_loops: int = 4

    def __post_init__(self):
        if self.model_name == "acasxu" \
            and self.acasxu_net_key is None: 
            raise ValueError("acasxu_net_key must be provided for acasxu model.")

        if self.repair_task == RepairTask.LocalRobustness and self.inp_eps == None:
            raise ValueError("inp_eps must be provided for Local Robustness repair experiment.")
