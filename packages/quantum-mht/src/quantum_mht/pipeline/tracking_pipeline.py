"""Main tracking pipeline: predict -> gate -> QUBO -> solve -> update.

Implements the full MHT tracking pipeline as a sequential processing chain.
Each measurement scan is processed through the following stages:

    1. Predict: Propagate track states forward via Kalman filter (Kalman 1960).
    2. Gate: Chi-squared gating to prune infeasible associations
       (Bar-Shalom & Li 1995, Ch 2.4).
    3. QUBO Build: Construct the MTDA QUBO from cost matrix and constraints
       (Stollenwerk et al., arXiv:2110.08346, 2021).
    4. Solve: Solve QUBO via quantum (annealing/QAOA) or classical solver.
    5. Update: Kalman update for assigned tracks (Kalman 1960).
    6. Track Management: M/N confirmation and deletion
       (Blackman & Popoli 1999, Ch 4.3).
    7. Track Initiation: Create new tentative tracks from unassigned
       measurements (false alarms that may be new targets).

This pipeline architecture follows the standard MHT processing loop described
in Stollenwerk et al. (arXiv:2110.08346) adapted for quantum-native MTDA.

Academic References:
    Stollenwerk et al., "Adiabatic Quantum Computing for Multi Object Tracking",
        Fraunhofer FKIE, arXiv:2110.08346, 2021 -- pipeline architecture for
        quantum-native MTDA (predict -> gate -> QUBO -> solve -> update).
    Kalman, "A New Approach to Linear Filtering and Prediction Problems",
        J. Basic Engineering 82(1):35-45, 1960 -- predict and update steps.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995 -- gating,
        cost matrix, and association framework.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 4 -- track lifecycle management.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import logging
import numpy as np
from quantum_mht.tracking.track_manager import TrackManager
from quantum_mht.tracking.kalman_filter import KalmanFilter
from quantum_mht.tracking.gating import GatingFilter
from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult
from quantum_mht.fusion.measurement import MeasurementScan

logger = logging.getLogger(__name__)

@dataclass
class PipelineScanResult:
    """Result from processing one measurement scan."""
    solver_result: SolverResult
    num_tracks: int
    num_measurements: int
    num_confirmed: int
    timestamp: float = 0.0

@dataclass
class TrackingPipeline:
    """Main MHT tracking pipeline.

    Orchestrates the full predict -> gate -> QUBO -> solve -> update cycle
    (arXiv:2110.08346). Accepts any MTDASolver implementation (quantum or
    classical) via the solver parameter.
    """
    solver: MTDASolver | None = None
    track_manager: TrackManager = field(default_factory=TrackManager)
    kalman_filter: KalmanFilter = field(default_factory=KalmanFilter)
    gating_filter: GatingFilter = field(default_factory=GatingFilter)
    qubo_builder: MTDAQuboBuilder = field(default_factory=MTDAQuboBuilder)

    def process_scan(self, scan: MeasurementScan) -> PipelineScanResult:
        """Process one measurement scan through the full pipeline."""
        if self.solver is None:
            raise ValueError("Solver not configured")

        # 1. Predict: Kalman time update (Kalman 1960)
        active_tracks = self.track_manager.active_tracks
        for track in active_tracks:
            pred_state, pred_cov = self.kalman_filter.predict(track.state, track.covariance)
            track.update_state(pred_state, pred_cov)

        # 2. Build QUBO (arXiv:2110.08346: cost matrix + constraint penalties)
        positions = scan.positions
        if len(active_tracks) == 0 or len(positions) == 0:
            # Initialize tracks from measurements
            for m in scan.measurements:
                state = np.zeros(self.kalman_filter.dim_state)
                state[:len(m.position)] = m.position
                cov = np.eye(self.kalman_filter.dim_state) * 10.0
                self.track_manager.create_track(state, cov)
            return PipelineScanResult(
                solver_result=SolverResult([], [], [], 0.0, 0.0, "init"),
                num_tracks=len(self.track_manager.active_tracks),
                num_measurements=len(positions),
                num_confirmed=len(self.track_manager.confirmed_tracks),
                timestamp=scan.timestamp,
            )

        predicted_states, pred_covs = self.track_manager.get_predicted_states()
        # Innovation covariance: S = H*P*H' + R (Kalman 1960; Bar-Shalom & Li 1995, Ch 1)
        meas_dim = positions.shape[1] if len(positions.shape) > 1 else 2
        innovation_covs = pred_covs + np.eye(meas_dim) * self.kalman_filter.measurement_noise

        # Build cost matrix with gating (Bar-Shalom & Li 1995, Ch 2.4 + Ch 6)
        cost_matrix, gate_mask = self.qubo_builder.cost_builder.build_with_gating_mask(
            predicted_states, positions, innovation_covs,
        )
        qubo_result = self.qubo_builder.build_from_cost_matrix(cost_matrix, gate_mask)

        # 3. Solve QUBO (quantum or classical solver)
        solver_result = self.solver.solve(qubo_result)

        # 4. Kalman measurement update for assigned tracks (Kalman 1960)
        for track_idx, meas_idx in solver_result.assignments:
            if track_idx < len(active_tracks):
                track = active_tracks[track_idx]
                measurement = positions[meas_idx]
                new_state, new_cov = self.kalman_filter.update(track.state, track.covariance, measurement)
                track.update_state(new_state, new_cov)
                track.record_hit()

        # 5. Handle missed detections
        for track_idx in solver_result.missed_detections:
            if track_idx < len(active_tracks):
                active_tracks[track_idx].record_miss()

        # 6. Initialize new tracks from false alarms
        for meas_idx in solver_result.false_alarms:
            if meas_idx < len(positions):
                state = np.zeros(self.kalman_filter.dim_state)
                state[:len(positions[meas_idx])] = positions[meas_idx]
                cov = np.eye(self.kalman_filter.dim_state) * 10.0
                self.track_manager.create_track(state, cov)

        # 7. Track lifecycle: M/N confirmation and deletion (Blackman & Popoli 1999, Ch 4.3)
        self.track_manager.update_lifecycle()

        return PipelineScanResult(
            solver_result=solver_result,
            num_tracks=len(self.track_manager.active_tracks),
            num_measurements=len(positions),
            num_confirmed=len(self.track_manager.confirmed_tracks),
            timestamp=scan.timestamp,
        )
