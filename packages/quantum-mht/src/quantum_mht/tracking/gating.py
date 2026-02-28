"""Chi-squared measurement gating using Mahalanobis distance.

Implements statistical gating to prune infeasible track-measurement associations
before QUBO construction. A measurement z_j passes the gate for track i if its
Mahalanobis distance satisfies:

    d^2(z_j, track_i) = (z_j - H*x_i)^T * S_i^{-1} * (z_j - H*x_i) <= threshold

where S_i = H*P_i*H^T + R is the innovation covariance and the threshold is
chosen from the chi-squared distribution at the desired confidence level.

Common Thresholds (2D measurements):
    95% confidence: 5.99
    99% confidence: 9.21  (default)
    99.9% confidence: 13.82

Gating reduces the number of QUBO variables by eliminating impossible
associations, directly reducing qubit requirements for quantum solvers.

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 2.4 -- chi-squared gating,
        Mahalanobis distance, and gate threshold selection.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 3 -- gating in MHT systems.

Quantum Pipeline Role (2026):
    Chi-squared gating is a CRITICAL classical preprocessing step that
    directly impacts quantum solver feasibility. By pruning unlikely
    track-measurement pairs BEFORE QUBO construction, gating reduces
    the number of binary variables from N*M (all pairs) to approximately
    3-5 candidates per track. For N=10 tracks and M=15 measurements:
    - Without gating: 150 QUBO variables -> ~450 physical qubits (Zephyr)
    - With gating: ~40 QUBO variables -> ~120 physical qubits (Zephyr)
    This reduction makes problems feasible on D-Wave Advantage2 (4400+ qubits,
    Zephyr 20-way connectivity, May 2025 GA) that would otherwise require
    LeapHybrid decomposition.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class GatingFilter:
    """Chi-squared measurement gating (Bar-Shalom & Li 1995, Ch 2.4).

    Uses Mahalanobis distance with chi-squared threshold to prune infeasible
    track-measurement associations. Reduces QUBO variable count for quantum solvers.

    References:
        Bar-Shalom & Li 1995, Ch 2.4 -- gating theory and threshold tables.
    """
    # Chi-squared 99% threshold for 2D measurements (Bar-Shalom & Li 1995, Ch 2.4)
    gate_threshold: float = 9.21
    dim: int = 2

    def gate(self, predicted_state: NDArray[np.float64], measurements: NDArray[np.float64], innovation_cov: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return boolean mask of measurements that pass gating."""
        S_inv = np.linalg.inv(innovation_cov)
        mask = np.zeros(len(measurements), dtype=bool)
        for j, z in enumerate(measurements):
            innovation = z - predicted_state
            d2 = float(innovation @ S_inv @ innovation)
            mask[j] = d2 <= self.gate_threshold
        return mask

    def mahalanobis_distances(self, predicted_state: NDArray[np.float64], measurements: NDArray[np.float64], innovation_cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Mahalanobis distances (Bar-Shalom & Li 1995, Ch 2.4).

        d^2 = (z - H*x)^T * S^{-1} * (z - H*x)
        """
        S_inv = np.linalg.inv(innovation_cov)
        distances = np.zeros(len(measurements))
        for j, z in enumerate(measurements):
            inn = z - predicted_state
            distances[j] = float(inn @ S_inv @ inn)
        return distances
