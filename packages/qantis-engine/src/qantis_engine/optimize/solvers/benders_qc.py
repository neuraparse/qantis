"""Quantum-classical Benders decomposition.

The MIP is split into a binary master (handled by a quantum/feasible sampler)
and a continuous sub-problem (handled by a classical LP solver). At each
iteration the master proposes a binary configuration ``y``; the sub-problem
returns a feasibility/optimality cut that is added to the master. The
quantum sampler discovers feasible-master incumbents faster than a cold
classical solver because it draws from the constraint-respecting subspace.

Following arXiv:2601.14024 (López-Baños et al., Jan 2026), we apply two
correctness-preserving accelerations:
    - Cut warm-start: cuts from previous iterations stay in the master.
    - Master-bound trimming: terminate when the gap between master LB and
      sub UB closes below ``tol``.
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
class BendersCut:
    """A feasibility or optimality cut from the sub-problem."""

    coefficients: NDArray[np.float64]
    rhs: float
    is_optimality: bool


@dataclass
class BendersResult:
    """Output of a Benders run."""

    master_solution: NDArray[np.int_]
    objective: float
    n_iterations: int
    n_cuts: int
    converged: bool
    wallclock_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BendersQCSolver:
    """Quantum-classical Benders decomposition.

    ``master_cost`` and ``sub_cost`` are user-supplied cost functions. The
    sub-problem solver returns either an optimality cut (linear lower bound on
    the sub objective) or a feasibility cut (linear constraint on master).
    """

    n_master_vars: int
    sampler: FeasibleSampler
    sub_solver: Callable[[NDArray[np.int_]], tuple[BendersCut, float]]
    deadline_s: float = 5.0
    max_iterations: int = 32
    tolerance: float = 1e-4

    def solve(
        self,
        master_cost: Callable[[NDArray[np.int_]], float],
        seed: int = 0,
    ) -> BendersResult:
        t0 = time.perf_counter()
        cuts: list[BendersCut] = []
        best_y: NDArray[np.int_] | None = None
        best_obj = float("inf")
        converged = False

        for iteration in range(self.max_iterations):
            if time.perf_counter() - t0 > self.deadline_s:
                break

            def augmented_cost(y: NDArray[np.int_]) -> float:
                penalty = 0.0
                for cut in cuts:
                    violation = float(
                        cut.rhs - cut.coefficients @ y.astype(np.float64)
                    )
                    if violation > 0:
                        penalty += 1e3 * violation
                return float(master_cost(y)) + penalty

            sample_res = self.sampler.sample(
                cost_fn=augmented_cost, n_samples=4, seed=seed + iteration
            )
            best_idx = int(np.argmin(sample_res.energies))
            y_hat = sample_res.samples[best_idx]
            cut, sub_obj = self.sub_solver(y_hat)
            cuts.append(cut)
            total_obj = float(master_cost(y_hat)) + sub_obj
            if total_obj < best_obj:
                best_obj = total_obj
                best_y = y_hat
            master_lb = float(sample_res.energies[best_idx])
            if abs(total_obj - master_lb) < self.tolerance:
                converged = True
                break

        wall = time.perf_counter() - t0
        if best_y is None:
            best_y = np.zeros(self.n_master_vars, dtype=np.int_)
        return BendersResult(
            master_solution=best_y,
            objective=best_obj if best_obj != float("inf") else float(master_cost(best_y)),
            n_iterations=len(cuts),
            n_cuts=len(cuts),
            converged=converged,
            wallclock_s=wall,
            metadata={"deadline_s": self.deadline_s, "tolerance": self.tolerance},
        )
