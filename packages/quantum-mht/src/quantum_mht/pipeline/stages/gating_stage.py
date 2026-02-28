"""Gating stage of the MHT tracking pipeline.

Applies chi-squared statistical gating to compute a boolean mask indicating
which track-measurement pairs are feasible associations. Measurements outside
a track's validation gate (Mahalanobis distance exceeds chi-squared threshold)
are excluded from the QUBO formulation, directly reducing qubit count.

This is Stage 2 of the predict -> gate -> QUBO -> solve -> update pipeline.

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 2.4 -- chi-squared gating,
        validation gate, and Mahalanobis distance.
    Stollenwerk et al., arXiv:2110.08346, 2021 -- gating to reduce QUBO size.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from quantum_mht.tracking.gating import GatingFilter

@dataclass
class GatingStage:
    gating_filter: GatingFilter

    def process(self, predicted_positions: NDArray, measurements: NDArray, innovation_covs: NDArray) -> NDArray[np.bool_]:
        """Return gating mask (n_tracks, n_meas).

        Uses chi-squared gating (Bar-Shalom & Li 1995, Ch 2.4) to identify
        feasible track-measurement pairs before QUBO construction.
        """
        n_tracks = len(predicted_positions)
        n_meas = len(measurements)
        mask = np.zeros((n_tracks, n_meas), dtype=bool)
        for i in range(n_tracks):
            mask[i] = self.gating_filter.gate(predicted_positions[i], measurements, innovation_covs[i])
        return mask
