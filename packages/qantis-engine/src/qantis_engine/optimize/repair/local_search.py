"""Classical local search operators applied after quantum sampling.

Two-opt and pairwise-swap improve a feasible candidate without changing its
feasibility. They are deliberately simple — the goal of QANTIS-Optimize is
to demonstrate that classical-local-search-on-quantum-incumbent beats
classical-local-search-from-cold-start under fixed time budgets.

References:
    Lin, Kernighan, "An Effective Heuristic Algorithm for the TSP",
        Operations Research 21:498 (1973).
    Croes, "A method for solving traveling-salesman problems",
        Operations Research 6:791 (1958).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def two_opt(
    tour: NDArray[np.int_],
    cost: Callable[[NDArray[np.int_]], float],
    max_iters: int = 1000,
) -> NDArray[np.int_]:
    """2-opt local search on a tour permutation.

    Repeatedly reverses subtours that decrease total cost. Terminates on
    no-improving-move or after ``max_iters`` iterations. ``cost(tour)`` must
    be a callable that returns the cost of the supplied permutation.
    """
    n = len(tour)
    best = tour.copy()
    best_c = cost(best)
    for _ in range(max_iters):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                cand = best.copy()
                cand[i:j] = cand[i:j][::-1]
                c = cost(cand)
                if c + 1e-12 < best_c:
                    best = cand
                    best_c = c
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def swap_search(
    assignment: NDArray[np.int_],
    cost: Callable[[NDArray[np.int_]], float],
    max_iters: int = 1000,
) -> NDArray[np.int_]:
    """Pairwise-swap local search on an integer-vector assignment.

    Tries every swap of two positions; accepts the first improving move and
    restarts. Useful for assignment and resource-allocation problems where
    a permutation-style 2-opt does not apply (the codomain is a flat label
    set, not a tour).
    """
    n = len(assignment)
    best = assignment.copy()
    best_c = cost(best)
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if best[i] == best[j]:
                    continue
                cand = best.copy()
                cand[i], cand[j] = cand[j], cand[i]
                c = cost(cand)
                if c + 1e-12 < best_c:
                    best = cand
                    best_c = c
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best
