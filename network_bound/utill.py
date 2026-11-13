import torch
from repair.util import SpecType

# =======================================================
# for classification specification C f(x) >= 0
# =======================================================
def filter_violated_spec(C, out_lb, out_ub):
    """
    Filter out already-satisfied rows of C using independent output bounds.

    Args:
        C:
            shape (out_dim,) or (num_specs, out_dim)
        out_lb, out_ub:
            output lower/upper bounds, shape (out_dim,)

    Returns:
        filtered_C, keep_mask, satisfied_mask
    """
    if C.dim() == 1:
        C = C.unsqueeze(0)

    C_pos = torch.clamp(C, min=0)
    C_neg = torch.clamp(C, max=0)

    # lower bound on C f(x)
    spec_lb = (C_pos * out_lb.unsqueeze(0)).sum(dim=1) + (C_neg * out_ub.unsqueeze(0)).sum(dim=1)
    satisfied_mask = spec_lb >= 0  # shape (num_specs,)

    violated_mask = ~satisfied_mask
    # filtered_C = C[violated_mask]

    return violated_mask, spec_lb

def update_violated_mask(violated_mask, objs):
    '''
    violated_mask: shape (num_specs,)
        - True: violated
        - False: satisfied
    objs: shape (num_violated_specs_previously,)
    returns updated violated_mask: shape (num_specs,)
    '''
    pre_violated_indices = violated_mask.nonzero(as_tuple=True)[0]
    for i, idx in enumerate(pre_violated_indices):
        if objs[i] >= 0:
            violated_mask[idx] = False
    return violated_mask


# =======================================================
# for complicated specification (e.g., acasxu)
# =======================================================
def filter_important_spec(spec, out_lb, out_ub):
    '''
    spec: Spec or SpecList
        - C
        - min_threshold
        - spec_type ('all' or 'any')
    
    out_lb, out_ub: shape (out_dim,)
    
    we check
        - for 'all' spec: collect violated dimensions where C f(x) < min_threshold
        - for 'any' spec: collect the dimension with the smallest violation (argmax(C f(x)))

    returns:
        important_C_list: list of C (shape (num_important_specs, out_dim))
    '''
    is_speclist = isinstance(spec, list)
    if is_speclist:  # spec is SpecList
        specs = spec
        specs_type = spec.specs_type
    else:
        specs = [spec]
        specs_type = SpecType.ALL
    
    important_C_masks = []
    violated_scores = []
    for spec in specs:
        C = spec.C
        min_threshold = spec.min_threshold

        C_pos = torch.clamp(C, min=0)
        C_neg = torch.clamp(C, max=0)

        spec_lb = (C_pos * out_lb.unsqueeze(0)).sum(dim=1) + (C_neg * out_ub.unsqueeze(0)).sum(dim=1)

        if spec.spec_type == SpecType.ALL:
            violated_score = spec_lb - min_threshold
            worst_violation_score, _ = torch.min(violated_score, dim=0)
            violated_mask = violated_score < 0
            if violated_mask.any():
                violated_scores.append(worst_violation_score)
                important_C_masks.append(violated_mask)
            else:
                violated_scores.append(None)
                important_C_masks.append(None)
        else:
            assert spec.spec_type == SpecType.ANY
            if (spec_lb < min_threshold).any():
                violation_score = spec_lb - min_threshold
                smallest_violation_score, smallest_violation_idx = torch.max(violation_score, dim=0)
                violated_scores.append(smallest_violation_score)
                important_C_masks.append(torch.tensor([smallest_violation_idx]))
            else:
                violated_scores.append(None)
                important_C_masks.append(None)
    
    if important_C_masks == [None] * len(important_C_masks):
        return None

    if specs_type == SpecType.ALL:
        # keep all violated dimensions
        return_C_list = []
        for violated_mask, _spec in zip(important_C_masks, specs):
            if violated_mask is not None:
                return_C_list.append(_spec.C[violated_mask])
        return return_C_list
    else:
        assert specs_type == SpecType.ANY
        # pick one with the smallest violation (closest to being satisfied)

        best_score = None
        for score in violated_scores:
            if score is not None:
                if best_score is None or score < best_score:
                    best_score = score

        return_C_list = []
        for score, C_mask, _spec in zip(violated_scores, important_C_masks, specs):
            if score == best_score:
                # if there are multiple specs with the same best score, we can pick all of them
                return_C_list.append(_spec.C[C_mask])
        return return_C_list