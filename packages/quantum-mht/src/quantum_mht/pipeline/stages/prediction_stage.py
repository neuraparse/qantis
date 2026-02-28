"""Prediction stage of the MHT tracking pipeline.

Implements the Kalman filter time-update (predict) step for all active tracks.
This propagates each track's state estimate and covariance forward by one time
step using the constant-velocity (or other configured) motion model.

    x_{k|k-1} = F * x_{k-1|k-1}
    P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q

This is Stage 1 of the predict -> gate -> QUBO -> solve -> update pipeline
(Stollenwerk et al., arXiv:2110.08346, 2021).

Academic References:
    Kalman, "A New Approach to Linear Filtering and Prediction Problems",
        J. Basic Engineering 82(1):35-45, 1960 -- Kalman predict step.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995, Ch 1
        -- prediction in multi-target tracking context.
"""
from __future__ import annotations
from dataclasses import dataclass
from quantum_mht.tracking.track import Track
from quantum_mht.tracking.kalman_filter import KalmanFilter

@dataclass
class PredictionStage:
    kalman_filter: KalmanFilter

    def process(self, tracks: list[Track]) -> list[Track]:
        """Apply Kalman predict to all tracks (Kalman 1960)."""
        for track in tracks:
            pred_state, pred_cov = self.kalman_filter.predict(track.state, track.covariance)
            track.update_state(pred_state, pred_cov)
        return tracks
