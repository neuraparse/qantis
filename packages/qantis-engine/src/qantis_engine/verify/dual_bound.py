"""Dual / lower-bound estimators for assignment problems.

For assignment / matching problems the LP relaxation has the integrality
property: the LP optimum is achieved at an integer vertex of the
doubly-stochastic polytope, so the LP lower bound equals the optimum. This
gives us a "Gurobi-free" optimality certificate for fair benchmarking.

For more general MIPs (chance-constrained, capacitated) we provide a
Lagrangian-dual lower bound via subgradient ascent — useful as a sanity
check for the QANTIS-Optimize hybrid B&B incumbent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


@dataclass
class DualBoundReport:
    """Result of a dual / lower-bound computation."""

    lower_bound: float
    method: str
    is_tight: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def lp_assignment_lower_bound(
    cost_matrix: NDArray[np.float64],
) -> DualBoundReport:
    """Compute the LP relaxation lower bound for the assignment problem.

    Because the assignment polytope has the integrality property, the LP
    optimum equals the IP optimum. We therefore return a *tight* bound,
    which the benchmark harness uses to compute incumbent-suboptimality gaps
    deterministically.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    rows, cols = linear_sum_assignment(cost_matrix)
    lb = float(sum(cost_matrix[r, c] for r, c in zip(rows, cols, strict=False)))
    return DualBoundReport(
        lower_bound=lb,
        method="lp_relaxation_with_integrality",
        is_tight=True,
        metadata={"n_rows": int(cost_matrix.shape[0]), "n_cols": int(cost_matrix.shape[1])},
    )


def lagrangian_dual_lower_bound(
    cost: NDArray[np.float64],
    coupling: NDArray[np.float64],
    rhs: NDArray[np.float64],
    n_iters: int = 64,
    step: float = 0.1,
) -> DualBoundReport:
    """Subgradient-ascent Lagrangian lower bound for ``min c.x s.t. A x = b, x in {0,1}``."""
    mu = np.zeros(rhs.size, dtype=np.float64)
    best_dual = -float("inf")
    for it in range(n_iters):
        modified = cost + coupling.T @ mu
        x = (modified < 0).astype(np.float64)
        residual = coupling @ x - rhs
        dual_val = float(cost @ x + mu @ residual)
        if dual_val > best_dual:
            best_dual = dual_val
        mu = mu + step * residual / max(1.0, float(it + 1))
    return DualBoundReport(
        lower_bound=best_dual,
        method="lagrangian_subgradient",
        is_tight=False,
        metadata={"n_iters": n_iters, "step": step},
    )
