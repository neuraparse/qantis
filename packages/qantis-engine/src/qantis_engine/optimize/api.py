"""High-level optimize.solve() entry point.

The single API the rest of the engine and benchmarks use:

    plan = optimize.solve(
        problem="assignment",
        cost_matrix=cm,
        chance_constraints=[risk.ChanceConstraint(...)],
        encoder="hamming_weight",
        solver="hybrid_bnb",
        sampler="hw_gibbs",
        deadline_s=0.1,
        seed=42,
    )

The selection logic is intentionally narrow — there is one good encoder per
problem class, and one good solver per resource budget. Power users compose
the lower-level objects directly.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from qantis_engine.optimize.encodings.hamming_weight import HammingWeightEncoding
from qantis_engine.optimize.repair.feasibility_repair import AssignmentRepair
from qantis_engine.optimize.solvers.feasible_sampler import (
    FeasibleSampler,
    HammingWeightGibbsSampler,
    XYMixerQAOASampler,
)
from qantis_engine.optimize.solvers.hybrid_bnb import HybridBnBSolver
from qantis_engine.risk.chance_constraint import ChanceConstraint, ChanceConstraintReport


@dataclass
class OptimizeReport:
    """High-level optimization report returned by :func:`solve`."""

    assignments: list[tuple[int, int]]
    missed_detections: list[int]
    false_alarms: list[int]
    objective: float
    chance_reports: list[ChanceConstraintReport]
    deadline_hit: bool
    wallclock_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


SolverChoice = Literal["hungarian", "hybrid_bnb", "feasible_sampler"]
SamplerChoice = Literal["hw_gibbs", "xy_qaoa"]


def _select_sampler(
    sampler: SamplerChoice, n_tracks: int, n_meas: int
) -> FeasibleSampler:
    n = n_tracks * n_meas
    k = min(n_tracks, n_meas)
    if sampler == "xy_qaoa":
        return XYMixerQAOASampler(n_qubits=n, target_weight=k)
    return HammingWeightGibbsSampler(
        encoding=HammingWeightEncoding(n_qubits=n, target_weight=k),
        beta=2.0,
        n_chain_steps=120,
    )


def solve(
    cost_matrix: NDArray[np.float64],
    n_tracks: int | None = None,
    n_meas: int | None = None,
    chance_constraints: Sequence[ChanceConstraint] = (),
    solver: SolverChoice = "hybrid_bnb",
    sampler: SamplerChoice = "hw_gibbs",
    deadline_s: float = 0.5,
    max_nodes: int = 64,
    seed: int = 0,
) -> OptimizeReport:
    """Solve an assignment-style problem under chance constraints."""
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got {cost_matrix.shape}")
    n_t = n_tracks if n_tracks is not None else int(cost_matrix.shape[0])
    n_m = n_meas if n_meas is not None else int(cost_matrix.shape[1])

    repair = AssignmentRepair(n_tracks=n_t, n_meas=n_m)

    if solver == "hungarian":
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost_matrix)
        scores = np.zeros((n_t, n_m), dtype=np.float64)
        assignments_raw: list[tuple[int, int]] = []
        for r, c in zip(rows, cols, strict=False):
            if r < n_t and c < n_m:
                assignments_raw.append((int(r), int(c)))
                scores[r, c] = 1.0
        assignments, missed, false_alarms = repair.repair(scores=scores, cost_matrix=cost_matrix)
        objective = float(sum(cost_matrix[r, c] for r, c in assignments))
        wall = 0.0
        deadline_hit = False
        meta: dict[str, Any] = {"solver": "hungarian"}
    elif solver in ("hybrid_bnb", "feasible_sampler"):
        feasible = _select_sampler(sampler, n_t, n_m)
        bnb = HybridBnBSolver(
            n_tracks=n_t,
            n_meas=n_m,
            sampler=feasible if solver == "hybrid_bnb" else None,
            deadline_s=deadline_s,
            max_nodes=max_nodes,
        )
        result = bnb.solve(cost_matrix=cost_matrix, seed=seed)
        scores = np.zeros((n_t, n_m), dtype=np.float64)
        for r in range(n_t):
            c = int(result.incumbent[r])
            if 0 <= c < n_m:
                scores[r, c] = 1.0
        assignments, missed, false_alarms = repair.repair(scores=scores, cost_matrix=cost_matrix)
        objective = float(sum(cost_matrix[r, c] for r, c in assignments))
        wall = result.wallclock_s
        deadline_hit = result.deadline_hit
        meta = {"solver": solver, **result.metadata}
    else:
        raise ValueError(f"unknown solver: {solver}")

    chance_reports = [
        cc.check(decision={"assignments": assignments}, seed=seed + idx)
        for idx, cc in enumerate(chance_constraints)
    ]

    return OptimizeReport(
        assignments=assignments,
        missed_detections=missed,
        false_alarms=false_alarms,
        objective=objective,
        chance_reports=chance_reports,
        deadline_hit=deadline_hit,
        wallclock_s=wall,
        metadata=meta,
    )
