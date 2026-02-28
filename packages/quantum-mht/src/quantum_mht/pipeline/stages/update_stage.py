"""Update stage: Kalman measurement update for assigned tracks.

Applies the Kalman filter measurement-update step for each track that was
assigned a measurement in the association stage:

    K_k = P_{k|k-1} * H^T * S_k^{-1}
    x_{k|k} = x_{k|k-1} + K_k * (z_k - H * x_{k|k-1})
    P_{k|k} = (I - K_k * H) * P_{k|k-1}

This is Stage 4 of the predict -> gate -> QUBO -> solve -> update pipeline.

Academic References:
    Kalman, "A New Approach to Linear Filtering and Prediction Problems",
        J. Basic Engineering 82(1):35-45, 1960 -- Kalman update equations.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995, Ch 1
        -- measurement update in tracking context.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from quantum_mht.tracking.track import Track
from quantum_mht.tracking.kalman_filter import KalmanFilter

@dataclass
class UpdateStage:
    kalman_filter: KalmanFilter

    def process(self, tracks: list[Track], assignments: list[tuple[int, int]], measurements: NDArray[np.float64]) -> None:
        """Apply Kalman update to assigned tracks (Kalman 1960)."""
        for track_idx, meas_idx in assignments:
            if track_idx < len(tracks) and meas_idx < len(measurements):
                new_state, new_cov = self.kalman_filter.update(
                    tracks[track_idx].state, tracks[track_idx].covariance, measurements[meas_idx],
                )
                tracks[track_idx].update_state(new_state, new_cov)
                tracks[track_idx].record_hit()
