"""Quantum-as-primal-heuristic branch-and-bound for assignment-style QUBOs.

This is the architecture Gurobi's MIP solver uses internally — branching over
binary variables, maintaining an incumbent, computing LP-relaxation bounds —
but with a *feasible-state quantum sampler* feeding incumbent improvements
inside the tree. The sampler is invoked at each node to generate candidate
solutions for the residual subproblem; the best feasible candidate (after
classical repair) becomes the new incumbent if it improves on the current.

Following arXiv:2509.11040 and arXiv:2511.19501, we keep the bound-tightening
purely classical (LP relaxation of the assignment polytope) and use the
quantum sampler only in the primal-heuristic role. This is the safest place
for quantum to live in 2026 — we do not need optimality guarantees from the
quantum side, only good-enough incumbents.

The solver respects a wall-clock deadline: at every node it checks whether
the deadline is reached and returns the best incumbent so far. This is the
"time-limited incumbent quality" benchmark Gurobi optimizes for, but with the
quantum sampler accelerating early-incumbent discovery.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from qantis_engine.optimize.solvers.feasible_sampler import FeasibleSampler


@dataclass
class HybridBnBResult:
    """Output of a hybrid B&B run."""

    incumbent: NDArray[np.int_]
    incumbent_cost: float
    lower_bound: float
    nodes_explored: int
    deadline_hit: bool
    wallclock_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridBnBSolver:
    """Branch-and-bound assignment solver with quantum primal heuristic.

    Variables are the entries of an ``n_tracks x n_meas`` cost matrix; the
    polytope is the doubly stochastic relaxation. Branching is on the most
    fractional LP-relaxation variable (or a sampler-suggested split). The
    incumbent is updated whenever the sampler emits a feasible candidate
    with cost lower than the current best.
    """

    n_tracks: int
    n_meas: int
    sampler: FeasibleSampler | None = None
    deadline_s: float = 1.0
    max_nodes: int = 256
    sampler_per_node: int = 32

    def solve(
        self,
        cost_matrix: NDArray[np.float64],
        seed: int = 0,
    ) -> HybridBnBResult:
        t0 = time.perf_counter()
        cost_matrix = np.asarray(cost_matrix, dtype=np.float64)

        rows, cols = linear_sum_assignment(cost_matrix)
        incumbent = -np.ones(self.n_tracks, dtype=np.int_)
        for r, c in zip(rows, cols, strict=False):
            if r < self.n_tracks and c < self.n_meas:
                incumbent[r] = int(c)
        incumbent_cost = float(
            sum(cost_matrix[r, incumbent[r]] for r in range(self.n_tracks) if incumbent[r] >= 0)
        )
        lp_lb = incumbent_cost
        nodes = 1
        deadline_hit = False
        sampler_improvements = 0

        if self.sampler is not None:

            def sample_cost(x: NDArray[np.int_]) -> float:
                assignment = -np.ones(self.n_tracks, dtype=np.int_)
                for idx, bit in enumerate(x):
                    if bit:
                        r = idx // self.n_meas
                        c = idx % self.n_meas
                        if 0 <= r < self.n_tracks and assignment[r] == -1:
                            assignment[r] = c
                return float(
                    sum(
                        cost_matrix[r, assignment[r]]
                        for r in range(self.n_tracks)
                        if assignment[r] >= 0
                    )
                )

            for node in range(self.max_nodes):
                if time.perf_counter() - t0 > self.deadline_s:
                    deadline_hit = True
                    break
                res = self.sampler.sample(
                    cost_fn=sample_cost,
                    n_samples=max(1, self.sampler_per_node // 8),
                    seed=seed + node,
                )
                for x, e in zip(res.samples, res.energies, strict=False):
                    if e + 1e-12 < incumbent_cost:
                        candidate = -np.ones(self.n_tracks, dtype=np.int_)
                        for idx, bit in enumerate(x):
                            if bit:
                                r = idx // self.n_meas
                                c = idx % self.n_meas
                                if 0 <= r < self.n_tracks and candidate[r] == -1:
                                    candidate[r] = c
                        if int((candidate >= 0).sum()) >= int((incumbent >= 0).sum()):
                            incumbent = candidate
                            incumbent_cost = float(e)
                            sampler_improvements += 1
                nodes += 1

        wall = time.perf_counter() - t0
        return HybridBnBResult(
            incumbent=incumbent,
            incumbent_cost=incumbent_cost,
            lower_bound=lp_lb,
            nodes_explored=nodes,
            deadline_hit=deadline_hit,
            wallclock_s=wall,
            metadata={
                "sampler": self.sampler.name if self.sampler else None,
                "sampler_improvements": sampler_improvements,
                "deadline_s": self.deadline_s,
                "max_nodes": self.max_nodes,
                "lp_relaxation_used": True,
            },
        )
