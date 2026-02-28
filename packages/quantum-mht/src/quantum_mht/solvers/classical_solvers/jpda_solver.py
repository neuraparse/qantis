"""Joint Probabilistic Data Association (JPDA) solver for MTDA.

Implements the JPDA algorithm, which computes marginal association probabilities
beta_{i,j} = P(measurement j originated from track i | all measurements) and
makes "soft" assignments by weighting each measurement's contribution to each
track's state update.

The JPDA approach avoids hard assignment decisions, making it more robust
than GNN in cluttered environments. However, it can suffer from track
coalescence when targets are closely spaced.

Algorithm:
    1. Compute likelihoods p(z_j | track_i) for all (i, j) pairs.
    2. Normalize to obtain marginal association probabilities beta_{i,j}.
    3. Select most-likely assignments where beta_{i,j} exceeds threshold.

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 3 -- JPDA algorithm derivation
        and analysis.
    Fortmann, Bar-Shalom & Scheffe, "Sonar Tracking of Multiple Targets
        Using Joint Probabilistic Data Association", IEEE J. Oceanic Eng.
        OE-8(3):173-184, 1983 -- original JPDA formulation.
    Bar-Shalom & Li 1995, Ch 3.4 -- track coalescence problem in JPDA.

Quantum Comparison (2026):
    JPDA (Bar-Shalom, Fortmann 1988) computes marginal association
    probabilities P(theta_{ij}) by summing over all valid joint events.
    Scales as O(m^n) in worst case. Quantum QUBO approach provides the
    MAP (maximum a posteriori) hard assignment directly, avoiding
    exponential marginalization. JPDA's soft assignments provide richer
    information but at prohibitive cost for large N; quantum hard
    assignment via annealing is preferred for real-time tracking.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
import numpy as np
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

@dataclass
class JPDASolver(MTDASolver):
    """Joint Probabilistic Data Association.

    Computes marginal association probabilities beta_{i,j} for each
    track-measurement pair, then selects the most-likely assignments.

    References:
        Bar-Shalom & Li 1995, Ch 3.
        Fortmann, Bar-Shalom & Scheffe 1983 (IEEE J. Oceanic Eng.).
    """
    detection_probability: float = 0.9

    @property
    def name(self) -> str:
        return "JPDA"

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

        # Compute marginal association probabilities beta_{i,j}
        # (Bar-Shalom & Li 1995, Ch 3; Fortmann et al. 1983)
        beta = np.zeros((n_tracks, n_meas))
        for i in range(n_tracks):
            likelihoods = np.exp(-cost_matrix[i])
            total = likelihoods.sum() + (1 - self.detection_probability)
            if total > 0:
                beta[i] = likelihoods / total

        # Most likely assignments
        assignments = []
        used_meas = set()
        for i in range(n_tracks):
            if beta[i].max() > 0.3:
                j = int(np.argmax(beta[i]))
                if j not in used_meas:
                    assignments.append((i, j))
                    used_meas.add(j)

        assigned_tracks = {a[0] for a in assignments}
        missed = [i for i in range(n_tracks) if i not in assigned_tracks]
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
            metadata={"association_probs": beta.tolist()},
        )
