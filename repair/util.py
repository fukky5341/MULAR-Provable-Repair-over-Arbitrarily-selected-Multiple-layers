from dataclasses import dataclass
from typing import Optional
from enum import Enum
import torch


class SpecType(Enum):
    ALL = "all"
    ANY = "any"


@dataclass
class Spec:
    '''
    specification is defined:
        c_i = y_t - y_i > min_threshold for i != t
        C = [c_1, c_2, ..., c_{t-1}, c_{t}, c_{t+1}, ..., c_n]  # shape: (num_constraints, num_outputs)
    '''
    C: torch.Tensor  # (num_constraints, num_outputs)
    target_label: Optional[int] = None

    min_threshold: float = 0.0
    # e.g., (mnist, cifar) 'safe if c_i >= 0 for all i except t', then min_threshold = 0 and C = [[-1, 1, 0, ..., 0], [0, 1, -1, ..., 0], ...] (num_classes-1, num_classes) 
    # e.f., (acasxu) 'safe if COC >= 1500', then min_threshold = 1500 and C = [[0, 0, 0, 0, 1]]

    spec_type: SpecType = SpecType.ALL

    def check_violation(self, objs, tol=1e-6):
        where_violated = (objs < -tol).nonzero(as_tuple=True)[0]
        is_violated = len(where_violated) > 0
        return is_violated, where_violated
    
    def violation_loss(self, y):
        vals = self.C @ y
        return vals.min()
    

class SpecList:
    def __init__(self, specs, specs_type):
        self.specs: list[Spec] = specs
        self.specs_type = specs_type  # 'all' or 'any'


class RepairStatus(Enum):
    REPAIRED = "repaired"
    PROCEEDING = "proceeding"
    FAILED = "failed"