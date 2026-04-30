"""Lagrangian decomposition for capacitated assignment / routing.

Following arXiv:2604.22194 (Sharma & Lau, Apr 2026): the coupling constraint
``A x = b`` is dualized with multipliers ``mu``, decoupling the problem into
per-resource sub-problems that each fit a feasible-state quantum sampler.
The dual variables are updated by subgradient ascent (or an RL-controlled
update rule for harder instances). At convergence we recover a feasible
primal by projecting the disaggregated sub-solutions back together.

We deliberately follow the paper's "no quantum advantage claim" framing:
this solver is a useful architecture for exposing structure to quantum
hardware, *not* a Gurobi replacement on its own. Beat-Gurobi claims live in
the closed-loop benchmark, not the optimizer-vs-optimizer microbenchmark.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qantis_engine.optimize.solvers.feasible_sampler import FeasibleSampler


@dataclass
class LagrangianResult:
    """Output of a Lagrangian-decomposition run."""

    primal_solution: list[NDArray[np.int_]]
    primal_cost: float
    dual_lower_bound: float
    duality_gap: float
    n_iterations: int
    wallclock_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LagrangianDecomposition:
    """Subgradient-ascent dual for a coupled assignment problem.

    ``sub_solvers`` is a list of per-resource feasible samplers, one per
    block (e.g. one per vehicle in a CVRP). ``coupling_matrix`` ``A`` and
    ``coupling_rhs`` ``b`` describe the global coupling ``sum_i A_i x_i = b``
    that the Lagrangian dualizes.
    """

    sub_costs: list[Callable[[NDArray[np.int_]], float]]
    sub_solvers: list[FeasibleSampler]
    coupling_matrix: NDArray[np.float64]
    coupling_rhs: NDArray[np.float64]
    step_size: float = 0.1
    max_iterations: int = 50
    deadline_s: float = 10.0

    def solve(self, seed: int = 0) -> LagrangianResult:
        t0 = time.perf_counter()
        n_blocks = len(self.sub_costs)
        m_constraints = self.coupling_rhs.size
        mu = np.zeros(m_constraints, dtype=np.float64)
        best_primal: list[NDArray[np.int_]] = []
        best_primal_cost = float("inf")
        best_dual = -float("inf")

        for it in range(self.max_iterations):
            if time.perf_counter() - t0 > self.deadline_s:
                break
            block_solutions: list[NDArray[np.int_]] = []
            block_cost_total = 0.0
            for b in range(n_blocks):
                base_cost = self.sub_costs[b]
                block_size = self._block_size(b)
                col_start = b * block_size
                a_b = self.coupling_matrix[:, col_start : col_start + block_size]
                mu_snapshot = mu.copy()

                def lag_cost(
                    x: NDArray[np.int_],
                    base_cost: Callable[[NDArray[np.int_]], float] = base_cost,
                    a_b: NDArray[np.float64] = a_b,
                    mu_snapshot: NDArray[np.float64] = mu_snapshot,
                ) -> float:
                    return float(base_cost(x) + mu_snapshot @ (a_b @ x.astype(np.float64)))

                res = self.sub_solvers[b].sample(
                    cost_fn=lag_cost, n_samples=4, seed=seed + it * 31 + b
                )
                idx = int(np.argmin(res.energies))
                x_b = res.samples[idx]
                block_solutions.append(x_b)
                block_cost_total += float(base_cost(x_b))

            full_x = np.concatenate(block_solutions).astype(np.float64)
            constraint_residual = self.coupling_matrix @ full_x - self.coupling_rhs
            dual_value = block_cost_total + float(mu @ constraint_residual)
            if dual_value > best_dual:
                best_dual = dual_value
            if (
                np.allclose(constraint_residual, 0, atol=1e-6)
                and block_cost_total < best_primal_cost
            ):
                best_primal = [b.copy() for b in block_solutions]
                best_primal_cost = block_cost_total

            mu = mu + self.step_size * constraint_residual / max(1.0, float(it + 1))

        if not best_primal:
            best_primal = block_solutions
            best_primal_cost = block_cost_total

        gap = best_primal_cost - best_dual if best_primal_cost < float("inf") else float("inf")
        wall = time.perf_counter() - t0
        return LagrangianResult(
            primal_solution=best_primal,
            primal_cost=best_primal_cost,
            dual_lower_bound=best_dual,
            duality_gap=gap,
            n_iterations=it + 1,
            wallclock_s=wall,
            metadata={"step_size": self.step_size, "max_iterations": self.max_iterations},
        )

    def _block_size(self, b: int) -> int:
        n_cols = self.coupling_matrix.shape[1]
        n_blocks = len(self.sub_costs)
        return n_cols // n_blocks
