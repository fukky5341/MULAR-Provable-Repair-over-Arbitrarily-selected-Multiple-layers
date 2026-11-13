from __future__ import annotations

from typing import Literal

import numpy as np

import scipy.stats


def _compute_rank_based_crowding_distance(ranks: np.ndarray) -> np.ndarray:
    (n_observations, n_obj) = ranks.shape
    order = np.argsort(ranks, axis=0)
    order_inv = np.zeros_like(order[:, 0], dtype=int)
    dists = np.zeros(n_observations)
    for m in range(n_obj):
        sorted_ranks = ranks[:, m][order[:, m]]
        order_inv[order[:, m]] = np.arange(n_observations)
        scale = sorted_ranks[-1] - sorted_ranks[0]
        crowding_dists = np.hstack([np.inf, sorted_ranks[2:] - sorted_ranks[:-2], np.inf]) / scale
        dists += crowding_dists[order_inv]

    # crowding dist is better when it is larger
    return scipy.stats.rankdata(-dists).astype(int)


def _tie_break_by_method(
    nd_masks: list[list[int]], ranks: np.ndarray, avg_ranks: np.ndarray | None = None
) -> np.ndarray:
    """
    Tie-break the non-domination ranks (, but we cannot guarantee no duplications)

    Args:
        nd_masks (list[list[int]]):
            The indices of observations in each non-domination rank.
        ranks (np.ndarray):
            The ranks of each objective in each observation.
            The shape is (n_observations, n_obj).
        avg_ranks (np.ndarray | None):
            The average rank + small deviation by the best rank of objectives in each observation.
            The shape is (n_observations, ).

    Returns:
        tie_broken_nd_ranks (np.ndarray):
            The each non-dominated rank will be tie-broken
            so that we can sort identically (but we may get duplications).
            The shape is (n_observations, ) and the array is a permutation of zero to n_observations - 1.

    Reference for avg_rank:
        Paper Title:
            Techniques for Highly Multiobjective Optimisation: Some Nondominated Points are Better than Others
        One sentence summary:
            Average ranking strategy is effective to tie-break in some evolution strategies methods.
        Authors:
            David Come and Joshua Knowles
        URL:
            https://arxiv.org/pdf/0908.3025.pdf

    Reference for crowding distance:
        Paper Title:
            A fast and elitist multiobjective genetic algorithm: NSGA-II
        One sentence summary:
            Consider the proximity to neighbors.
        Authors:
            K. Deb et al.
        URL:
            http://vision.ucsd.edu/~sagarwal/nsga2.pdf
    """
    n_checked = 0
    (size, _) = ranks.shape
    tie_broken_nd_ranks = np.zeros(size, dtype=int)

    for mask in nd_masks:
        if avg_ranks is not None:
            tie_break_ranks = scipy.stats.rankdata(avg_ranks[mask]).astype(int)
        else:  # Use crowding distance
            tie_break_ranks = _compute_rank_based_crowding_distance(ranks=ranks[mask])

        # -1 to start tie_broken_nd_ranks from zero
        tie_broken_nd_ranks[mask] = tie_break_ranks + n_checked - 1
        n_checked += len(mask)

    return tie_broken_nd_ranks


def _tie_break(
    costs: np.ndarray, nd_ranks: np.ndarray, tie_break: Literal["crowding_distance", "avg_rank"]
) -> np.ndarray:
    methods = ["crowding_distance", "avg_rank"]
    ranks = scipy.stats.rankdata(costs, axis=0)
    masks: list[list[int]] = [[] for _ in range(nd_ranks.max() + 1)]
    for idx, nd_rank in enumerate(nd_ranks):
        masks[nd_rank].append(idx)

    if tie_break == methods[0]:
        return _tie_break_by_method(nd_masks=masks, ranks=ranks)
    elif tie_break == methods[1]:
        # min_ranks_factor plays a role when we tie-break same average ranks
        min_ranks_factor = np.min(ranks, axis=-1) / (nd_ranks.size**2 + 1)
        avg_ranks = np.mean(ranks, axis=-1) + min_ranks_factor
        return _tie_break_by_method(nd_masks=masks, ranks=ranks, avg_ranks=avg_ranks)
    else:
        raise ValueError(f"tie_break method must be in {methods}, but got {tie_break}")
