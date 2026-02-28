"""Global Nearest Neighbor (GNN) solver for MTDA.

Implements the Global Nearest Neighbor assignment algorithm, which greedily
assigns tracks to measurements in order of increasing cost. GNN is the
simplest data association method and serves as a fast classical baseline.

Algorithm:
    1. Sort all (track, measurement) pairs by assignment cost.
    2. Greedily assign the lowest-cost pair, excluding already-assigned
       tracks and measurements.
    3. Unassigned tracks become missed detections; unassigned measurements
       become false alarms.

GNN is suboptimal for ambiguous situations (crossing targets, dense clutter)
because greedy decisions cannot be revised. It serves as a lower-bound
baseline against which quantum solvers are compared.

Academic References:
    Blackman, "Multiple Target Tracking with Radar Applications", Artech
        House, 1986 -- GNN as the simplest multi-target association method.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995 -- GNN limitations and comparison
        with optimal assignment methods.

Quantum Comparison (2026):
    Global Nearest Neighbor (Blackman 1986): Simplest greedy assignment,
    O(n^2) per frame. Provides the lower bound baseline for tracking
    quality. All quantum solvers (annealing, QAOA, FPC-QAOA) must exceed
    GNN accuracy to justify quantum overhead. GNN fails in dense clutter
    and crossing scenarios where the greedy decision is suboptimal --
    precisely the scenarios where QUBO optimization shows advantage.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
import numpy as np
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

@dataclass
class GNNSolver(MTDASolver):
    """Global Nearest Neighbor (greedy) assignment solver.

    Greedy assignment by ascending cost. O(NM log(NM)) complexity due to
    sorting. Suboptimal for ambiguous scenarios but very fast.

    References:
        Blackman, "Multiple Target Tracking with Radar Applications", 1986.
    """
    gate_threshold: float = 9.21

    @property
    def name(self) -> str:
        return "GNN"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()
        n_tracks = qubo_result.variables.n_tracks
        n_meas = qubo_result.variables.n_measurements
        variables = qubo_result.variables

        cost_matrix = np.zeros((n_tracks, n_meas))
        for i in range(n_tracks):
            for j in range(n_meas):
                idx = variables.var_index(i, j)
                cost_matrix[i, j] = qubo_result.Q.get((idx, idx), 1e6)

        assignments = []
        used_tracks = set()
        used_meas = set()

        # Greedy assignment: sort by cost, assign lowest-cost pairs first (Blackman 1986)
        flat_indices = np.argsort(cost_matrix, axis=None)
        for flat_idx in flat_indices:
            i, j = divmod(int(flat_idx), n_meas)
            if i in used_tracks or j in used_meas:
                continue
            if cost_matrix[i, j] > 1e5:
                break
            assignments.append((i, j))
            used_tracks.add(i)
            used_meas.add(j)

        missed = [i for i in range(n_tracks) if i not in used_tracks]
        false_alarms = [j for j in range(n_meas) if j not in used_meas]
        objective = sum(cost_matrix[i, j] for i, j in assignments)
        elapsed = time.perf_counter() - t0

        return SolverResult(
            assignments=assignments,
            missed_detections=missed,
            false_alarms=false_alarms,
            objective_value=float(objective),
            solve_time_s=elapsed,
            solver_name=self.name,
        )
