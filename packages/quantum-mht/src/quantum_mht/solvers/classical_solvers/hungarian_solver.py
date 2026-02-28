"""Hungarian algorithm (optimal 2D assignment) for MTDA.

Implements the Hungarian algorithm for finding the minimum-cost assignment
in a bipartite graph (tracks to measurements). This is the optimal classical
baseline for single-frame data association.

The algorithm guarantees the globally optimal assignment in O(n^3) time,
where n = max(N_tracks, N_measurements). Non-square cost matrices are
padded with high-cost dummy entries.

Academic References:
    Kuhn, "The Hungarian Method for the Assignment Problem", Naval Research
        Logistics Quarterly 2(1-2):83-97, 1955 -- original algorithm.
    Munkres, "Algorithms for the Assignment and Transportation Problems",
        J. SIAM 5(1):32-38, 1957 -- polynomial-time implementation (O(n^3)).
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995 --
        Hungarian algorithm as optimal 2D assignment baseline for MTDA.

Quantum Comparison (2026):
    The Hungarian algorithm (Kuhn, 1955; Munkres, 1957) provides the
    optimal O(n^3) solution for single-frame 2D assignment -- the classical
    gold standard. However, it cannot handle:
    - Multi-frame association (MHT) -- NP-hard generalization
    - Missed detections and false alarms as optimization variables
    Quantum annealing (Ihara 2025, Sci. Rep. 15:24294) and QAOA
    (Farhi et al. 2014; FPC-QAOA arXiv:2512.21181 Dec 2025) solve
    the QUBO formulation that naturally encodes these extensions.
    Hungarian serves as the classical lower bound for single-frame
    association accuracy in our benchmark suite.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

@dataclass
class HungarianSolver(MTDASolver):
    """Optimal 2D assignment using the Hungarian algorithm.

    Uses scipy.optimize.linear_sum_assignment which implements the
    Kuhn-Munkres algorithm in O(n^3) time complexity.

    References:
        Kuhn, "The Hungarian Method", Naval Res. Logistics 2(1-2):83-97, 1955.
    """

    @property
    def name(self) -> str:
        return "Hungarian"

    def solve(self, qubo_result: Any) -> SolverResult:
        return self.solve_from_cost_matrix(
            qubo_result.variables.n_tracks,
            qubo_result.variables.n_measurements,
            qubo_result,
        )

    def solve_from_cost_matrix(self, n_tracks: int, n_meas: int, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()
        variables = qubo_result.variables

        # Build cost matrix from QUBO diagonal for the assignment search.
        # Q[(idx,idx)] = original_cost[i,j] - 2*penalty due to row+column constraint
        # encoding. The uniform -2*penalty shift preserves relative ordering so
        # linear_sum_assignment still finds the globally optimal assignment.
        cost_matrix = np.zeros((n_tracks, n_meas))
        for i in range(n_tracks):
            for j in range(n_meas):
                idx = variables.var_index(i, j)
                cost_matrix[i, j] = qubo_result.Q.get((idx, idx), 1e6)

        # Pad matrix if not square (required by Hungarian algorithm, Kuhn 1955)
        max_dim = max(n_tracks, n_meas)
        padded = np.full((max_dim, max_dim), 1e6)
        padded[:n_tracks, :n_meas] = cost_matrix

        # Kuhn-Munkres O(n^3) optimal assignment (Kuhn 1955)
        row_ind, col_ind = linear_sum_assignment(padded)
        assignments = [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if r < n_tracks and c < n_meas]
        assigned_tracks = {a[0] for a in assignments}
        assigned_meas = {a[1] for a in assignments}
        missed = [i for i in range(n_tracks) if i not in assigned_tracks]
        false_alarms = [j for j in range(n_meas) if j not in assigned_meas]

        # Evaluate the full QUBO objective for the feasible Hungarian solution.
        # This includes assignment, missed detection, and false alarm variables so
        # the value is directly comparable to QAOA's QUBO objective evaluation.
        # For a feasible solution all off-diagonal penalty terms sum to zero, so
        # we only need to sum the diagonal entries for active (=1) variables.
        active_vars: dict[int, int] = {}
        for i, j in assignments:
            active_vars[variables.var_index(i, j)] = 1
        for i in missed:
            if getattr(variables, "include_missed", False):
                active_vars[variables.var_index(i, -1)] = 1
        for j in false_alarms:
            if getattr(variables, "include_false_alarm", False):
                active_vars[variables.var_index(-1, j)] = 1

        objective = sum(
            qubo_result.Q.get((k, k), 0.0) for k in active_vars
        )
        elapsed = time.perf_counter() - t0

        return SolverResult(
            assignments=assignments,
            missed_detections=missed,
            false_alarms=false_alarms,
            objective_value=float(objective),
            solve_time_s=elapsed,
            solver_name=self.name,
        )
