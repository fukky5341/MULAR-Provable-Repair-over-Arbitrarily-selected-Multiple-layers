from dataclasses import dataclass
import numpy as np
import torch
from .models import *
from repair.util import SpecType, Spec, SpecList

RHO, THETA, PSI, V_OWN, V_INT = list(range(5))
LB = [0.0, -3.141593, -3.141593, 100.0, 0.0]
UB = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]

COC, WL, WR, SL, SR = list(range(5))


@dataclass
class PropertySpec:
    model_keys: list | tuple
    input_polytopes: np.ndarray  # shape: (num_polytopes, 5, 2)

    def __call__(self, output):
        return self.output_constraints(output)

    def output_constraints(self, output):
        raise NotImplementedError

    def input_bounds(self, normalize_input=None, device=None, dtype=None):
        """
        Return input polytopes, optionally normalized.
        Output shape: (num_polytopes, 5, 2)
        """
        boxes = self.input_polytopes.clone()
        if normalize_input is not None:
            normed = []
            for box in boxes:
                nb = normalize_input(box.T).T  # shape: (5, 2)
                if dtype is not None:
                    nb = nb.to(dtype)
                if device is not None:
                    nb = nb.to(device)
                normed.append(nb)
            return torch.stack(normed, dim=0)
        else:
            out = boxes
            if dtype is not None:
                out = out.to(dtype)
            if device is not None:
                out = out.to(device)
            return out

    def get_input_polytopes(self, normalize_input=None, device=None, dtype=None):
        return self.input_bounds(normalize_input, device, dtype)

    def check_output_spec(self, output):
        return self.output_constraints(output)
    
    def output_spec(self, device, dtype):
        """
        Return Spec
        """
        raise NotImplementedError


class Property_1(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(),
            input_polytopes=torch.tensor([[
                [55947.691, UB[RHO]],
                [LB[THETA], UB[THETA]],
                [LB[PSI],   UB[PSI]],
                [1145.,     UB[V_OWN]],
                [LB[V_INT], 60.],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        return output[..., COC] <= 1500.
    
    def output_spec(self, device, dtype):
        C = torch.zeros((1, 5), device=device, dtype=dtype)
        C[0, COC] = -1.
        return Spec(C=C, min_threshold=-1500., spec_type=SpecType.ALL)


class Property_2(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(),
            input_polytopes=torch.tensor([[
                [55947.691, UB[RHO]],
                [LB[THETA], UB[THETA]],
                [LB[PSI],   UB[PSI]],
                [1145.,     UB[V_OWN]],
                [LB[V_INT], 60.],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if COC is not maximal
        return output.argmax(-1) != COC
    
    def output_spec(self, device, dtype):
        others = torch.tensor([WL, WR, SL, SR], device=device, dtype=dtype)
        C = torch.zeros((len(others), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others):
            C[i, idx] = 1.
            C[i, COC] = -1.
        return Spec(C=C, min_threshold=0., spec_type=SpecType.ANY)


class Property_3(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(lambda keys: keys not in [(1, 7), (1, 8), (1, 9)]),
            input_polytopes=torch.tensor([[
                [1500., 1800.],
                [-0.06, 0.06],
                [3.10,  UB[PSI]],
                [980.,  UB[V_OWN]],
                [960.,  UB[V_INT]],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if COC is not minimal
        return output.argmin(-1) != COC

    def output_spec(self, device, dtype):
        others = torch.tensor([WL, WR, SL, SR], device=device, dtype=dtype)
        C = torch.zeros((len(others), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others):
            C[i, idx] = -1.
            C[i, COC] = 1.
        return Spec(C=C, min_threshold=0., spec_type=SpecType.ANY)


class Property_4(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(lambda keys: keys not in [(1, 7), (1, 8), (1, 9)]),
            input_polytopes=torch.tensor([[
                [1500., 1800.],
                [-0.06, 0.06],
                [0.,    0.],
                [1000., UB[V_OWN]],
                [700.,  800.],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if COC is not minimal
        return output.argmin(-1) != COC
    
    def output_spec(self, device, dtype):
        others = torch.tensor([WL, WR, SL, SR], device=device, dtype=dtype)
        C = torch.zeros((len(others), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others):
            C[i, idx] = -1.
            C[i, COC] = 1.
        return Spec(C=C, min_threshold=0., spec_type=SpecType.ANY)


class Property_5(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(lambda keys: keys == (1, 1)),
            input_polytopes=torch.tensor([[
                [250.,      400.],
                [0.2,       0.4],
                [-3.141592, -3.141592 + 0.005],
                [100.,      400.],
                [0.,        400.],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if SR is minimal
        return output.argmin(-1) == SR
    
    def output_spec(self, device, dtype):
        others = torch.tensor([WL, WR, SL, COC], device=device, dtype=dtype)
        C = torch.zeros((len(others), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others):
            C[i, idx] = 1.
            C[i, SR] = -1.
        return Spec(C=C, min_threshold=0., spec_type=SpecType.ALL)

class Property_6(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(lambda keys: keys == (1, 1)),
            input_polytopes=torch.tensor([
                [
                    [12000.,    62000.],
                    [0.7,       3.141592],
                    [-3.141592, -3.141592 + 0.005],
                    [100.,      1200.],
                    [0.,        1200.],
                ],
                [
                    [12000.,    62000.],
                    [-3.141592, -0.7],
                    [-3.141592, -3.141592 + 0.005],
                    [100.,      1200.],
                    [0.,        1200.],
                ]
            ], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if COC is minimal
        return output.argmin(-1) == COC
    
    def output_spec(self, device, dtype):
        others = torch.tensor([WL, WR, SL, SR], device=device, dtype=dtype)
        C = torch.zeros((len(others), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others):
            C[i, idx] = 1.
            C[i, COC] = -1.
        return Spec(C=C, min_threshold=0., spec_type=SpecType.ALL)


class Property_7(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(lambda keys: keys == (1, 9)),
            input_polytopes=torch.tensor([[
                [0.,        60760.],
                [-3.141592, 3.141592],
                [-3.141592, 3.141592],
                [100.,      1200.],
                [0.,        1200.],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if strong left is not minimal and strong right is not minimal
        label = output.argmin(-1)
        return (label != SL) & (label != SR)
    
    def output_spec(self, device, dtype):
        # safe if strong left is not minimal
        others1 = torch.tensor([WL, WR, SR, COC], device=device, dtype=dtype)
        C1 = torch.zeros((len(others1), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others1):
            C1[i, idx] = -1.
            C1[i, SL] = 1.
        spec1 = Spec(C=C1, min_threshold=0., spec_type=SpecType.ANY)
        # safe if strong right is not minimal
        others2 = torch.tensor([WL, WR, SL, COC], device=device, dtype=dtype)
        C2 = torch.zeros((len(others2), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others2):
            C2[i, idx] = -1.
            C2[i, SR] = 1.
        spec2 = Spec(C=C2, min_threshold=0., spec_type=SpecType.ANY)
        return SpecList(specs=[spec1, spec2], specs_type=SpecType.ALL)


class Property_8(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(lambda keys: keys == (2, 9)),
            input_polytopes=torch.tensor([[
                [0.,        60760.],
                [-3.141592, -0.75 * 3.141592],
                [-0.1,      0.1],
                [600.,      UB[V_OWN]],
                [600.,      UB[V_INT]],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if WL is minimal or COC is minimal
        label = output.argmin(-1)
        if isinstance(output, torch.Tensor):
            valid = torch.tensor([COC, WL], device=label.device)
            return torch.isin(label, valid)
        else:
            return np.isin(label, np.array([COC, WL]))
        
    def output_spec(self, device, dtype):
        # safe if WL is minimal
        others1 = torch.tensor([WR, SL, SR, COC], device=device, dtype=dtype)
        C1 = torch.zeros((len(others1), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others1):
            C1[i, idx] = 1.
            C1[i, WL] = -1.
        spec1 = Spec(C=C1, min_threshold=0., spec_type=SpecType.ALL)
        # safe if COC is minimal
        others2 = torch.tensor([WL, WR, SL, SR], device=device, dtype=dtype)
        C2 = torch.zeros((len(others2), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others2):
            C2[i, idx] = 1.
            C2[i, COC] = -1.
        spec2 = Spec(C=C2, min_threshold=0., spec_type=SpecType.ALL)
        return SpecList(specs=[spec1, spec2], specs_type=SpecType.ANY)


class Property_9(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(lambda keys: keys == (3, 3)),
            input_polytopes=torch.tensor([[
                [2000.,     7000.],
                [-0.4,      -0.14],
                [-3.141592, -3.141592 + 0.01],
                [100.,      150.],
                [0.,        150.],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if SL is minimal
        return output.argmin(-1) == SL
    
    def output_spec(self, device, dtype):
        others = torch.tensor([WL, WR, SR, COC], device=device, dtype=dtype)
        C = torch.zeros((len(others), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others):
            C[i, idx] = 1.
            C[i, SL] = -1.
        return Spec(C=C, min_threshold=0., spec_type=SpecType.ALL)


class Property_10(PropertySpec):
    def __init__(self, device, dtype):
        super().__init__(
            model_keys=all_model_keys(lambda keys: keys == (4, 5)),
            input_polytopes=torch.tensor([[
                [36000.,    60760.],
                [0.7,       3.141592],
                [-3.141592, -3.141592 + 0.01],
                [900.,      1200.],
                [600.,      1200.],
            ]], dtype=dtype, device=device)
        )

    def output_constraints(self, output):
        # safe if COC is minimal
        return output.argmin(-1) == COC

    def output_spec(self, device, dtype):
        others = torch.tensor([WL, WR, SL, SR], device=device, dtype=dtype)
        C = torch.zeros((len(others), 5), device=device, dtype=dtype)
        for i, idx in enumerate(others):
            C[i, idx] = 1.
            C[i, COC] = -1.
        return Spec(C=C, min_threshold=0., spec_type=SpecType.ALL)

DICT = {
    1: Property_1,
    2: Property_2,
    3: Property_3,
    4: Property_4,
    5: Property_5,
    6: Property_6,
    7: Property_7,
    8: Property_8,
    9: Property_9,
    10: Property_10,
}


def property(no, device, dtype) -> PropertySpec:
    return DICT[no](device=device, dtype=dtype)


def applicable_properties_for_model(model_key, device, dtype):
    props = []
    for no in sorted(DICT.keys()):
        prop = property(no, device=device, dtype=dtype)
        if model_key in prop.model_keys:
            props.append((no, prop))
    return props