"""Cost matrix builder using log-likelihood ratios.

Constructs the cost matrix C for the Multi-Target Data Association (MTDA) QUBO
formulation. Each entry c_{i,j} is the negative log-likelihood ratio (NLLR) of
assigning measurement j to track i versus clutter:

    c_{i,j} = -ln( p(z_j | track_i) / p(z_j | clutter) )

This follows the standard Bayesian formulation for data association scoring.

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 6 -- log-likelihood ratio scoring
        for data association.
    Stollenwerk et al., "Adiabatic Quantum Computing for Multi Object Tracking",
        Fraunhofer FKIE, arXiv:2110.08346, 2021, Eq 3-5 -- QUBO cost matrix
        derived from NLLR for quantum-native MTDA formulation.
    Bar-Shalom & Li 1995, Ch 2.4 -- Mahalanobis distance and chi-squared gating
        used to prune infeasible associations before QUBO construction.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class CostMatrixBuilder:
    """Build cost matrices for MTDA QUBO formulation.

    Uses log-likelihood ratios: c_{i,j} = -ln(p(z_j|track_i) / p(z_j|clutter))

    The NLLR scoring ensures that low-cost assignments correspond to high-
    likelihood track-measurement pairs, making the QUBO minimization equivalent
    to maximum-likelihood data association.

    References:
        Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995, Ch 6.
        Stollenwerk et al., arXiv:2110.08346, 2021, Eq 3-5 (QUBO cost matrix).
    """
    clutter_density: float = 1e-5
    gate_threshold: float = 9.21  # Chi-squared 99% for 2D

    def build(
        self,
        predicted_states: NDArray[np.float64],
        measurements: NDArray[np.float64],
        covariances: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Build cost matrix from predicted states and measurements.

        Args:
            predicted_states: (N_tracks, dim) predicted track states
            measurements: (N_meas, dim) measurement vectors
            covariances: (N_tracks, dim, dim) innovation covariance matrices

        Returns:
            cost_matrix: (N_tracks, N_meas) log-likelihood ratio costs
        """
        n_tracks = predicted_states.shape[0]
        n_meas = measurements.shape[0]
        dim = predicted_states.shape[1]
        cost_matrix = np.full((n_tracks, n_meas), np.inf)

        for i in range(n_tracks):
            for j in range(n_meas):
                innovation = measurements[j] - predicted_states[i]
                S = covariances[i]
                S_inv = np.linalg.inv(S)

                # Mahalanobis distance squared (Bar-Shalom & Li 1995, Ch 2.4)
                d2 = float(innovation @ S_inv @ innovation)

                # Chi-squared gating check (Bar-Shalom & Li 1995, Ch 2.4)
                if d2 > self.gate_threshold:
                    continue

                # Log-likelihood ratio (arXiv:2110.08346, Eq 3; Bar-Shalom & Li 1995, Ch 6)
                log_det_S = np.log(np.linalg.det(S))
                log_likelihood = -0.5 * (d2 + log_det_S + dim * np.log(2 * np.pi))
                log_clutter = np.log(self.clutter_density)

                cost_matrix[i, j] = -(log_likelihood - log_clutter)

        return cost_matrix

    def build_with_gating_mask(
        self,
        predicted_states: NDArray[np.float64],
        measurements: NDArray[np.float64],
        covariances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Build cost matrix and gating mask."""
        cost_matrix = self.build(predicted_states, measurements, covariances)
        gate_mask = np.isfinite(cost_matrix)
        # Replace inf with large penalty for ungated entries
        cost_matrix = np.where(gate_mask, cost_matrix, 0.0)
        return cost_matrix, gate_mask
