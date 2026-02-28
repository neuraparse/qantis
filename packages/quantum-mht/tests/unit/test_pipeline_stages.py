"""Tests for quantum_mht.pipeline.stages module.

Validates the individual pipeline stages: PredictionStage, GatingStage,
AssociationStage, UpdateStage, and ManagementStage.
"""

import numpy as np
import pytest

from quantum_mht.pipeline.stages.prediction_stage import PredictionStage
from quantum_mht.pipeline.stages.gating_stage import GatingStage
from quantum_mht.pipeline.stages.update_stage import UpdateStage
from quantum_mht.pipeline.stages.management_stage import ManagementStage
from quantum_mht.tracking.kalman_filter import KalmanFilter
from quantum_mht.tracking.gating import GatingFilter
from quantum_mht.tracking.track import Track, TrackStatus
from quantum_mht.tracking.track_manager import TrackManager


class TestPredictionStage:
    """Test PredictionStage (Kalman predict for all tracks)."""

    def test_returns_list_of_tracks(self) -> None:
        """PredictionStage.process should return the list of tracks."""
        kf = KalmanFilter(dt=1.0, process_noise=0.1)
        stage = PredictionStage(kalman_filter=kf)

        t = Track(track_id=0, state=np.array([0.0, 1.0, 0.0, 1.0]), covariance=np.eye(4))
        result = stage.process([t])

        assert isinstance(result, list)
        assert len(result) == 1

    def test_updates_track_states(self) -> None:
        """After prediction, track states should be updated."""
        kf = KalmanFilter(dt=1.0, process_noise=0.1)
        stage = PredictionStage(kalman_filter=kf)

        initial_state = np.array([0.0, 1.0, 0.0, 1.0])
        t = Track(track_id=0, state=initial_state.copy(), covariance=np.eye(4))

        stage.process([t])

        # State should have changed (position advances by velocity * dt)
        assert not np.array_equal(t.state, initial_state)
        # x should have moved: x + vx * dt = 0 + 1 * 1 = 1
        assert np.isclose(t.state[0], 1.0)
        assert np.isclose(t.state[2], 1.0)

    def test_multiple_tracks(self) -> None:
        """PredictionStage should process multiple tracks."""
        kf = KalmanFilter(dt=1.0)
        stage = PredictionStage(kalman_filter=kf)

        tracks = [
            Track(track_id=0, state=np.array([0.0, 2.0, 0.0, 3.0]), covariance=np.eye(4)),
            Track(track_id=1, state=np.array([10.0, -1.0, 5.0, 0.0]), covariance=np.eye(4)),
        ]
        result = stage.process(tracks)
        assert len(result) == 2

        # Track 0: x = 0 + 2*1 = 2, y = 0 + 3*1 = 3
        assert np.isclose(result[0].state[0], 2.0)
        assert np.isclose(result[0].state[2], 3.0)

        # Track 1: x = 10 + (-1)*1 = 9, y = 5 + 0*1 = 5
        assert np.isclose(result[1].state[0], 9.0)
        assert np.isclose(result[1].state[2], 5.0)

    def test_empty_tracks_list(self) -> None:
        """PredictionStage with no tracks should return empty list."""
        kf = KalmanFilter()
        stage = PredictionStage(kalman_filter=kf)
        result = stage.process([])
        assert result == []

    def test_covariance_updated(self) -> None:
        """After prediction, track covariance should be updated (increased)."""
        kf = KalmanFilter(dt=1.0, process_noise=1.0)
        stage = PredictionStage(kalman_filter=kf)

        initial_cov = np.eye(4) * 0.5
        t = Track(track_id=0, state=np.zeros(4), covariance=initial_cov.copy())

        stage.process([t])

        # Covariance should have increased due to process noise
        assert np.trace(t.covariance) > np.trace(initial_cov)


class TestGatingStage:
    """Test GatingStage (chi-squared gating for all track-measurement pairs)."""

    def test_returns_boolean_mask(self) -> None:
        """GatingStage.process should return a boolean mask (n_tracks, n_meas)."""
        gf = GatingFilter(gate_threshold=10.0)
        stage = GatingStage(gating_filter=gf)

        predicted_positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        measurements = np.array([[0.1, 0.1], [50.0, 50.0], [10.1, 10.1]])
        innovation_covs = np.array([np.eye(2), np.eye(2)])

        mask = stage.process(predicted_positions, measurements, innovation_covs)
        assert mask.shape == (2, 3)
        assert mask.dtype == bool

    def test_close_measurements_gated_in(self) -> None:
        """Measurements close to predicted positions should pass gating."""
        gf = GatingFilter(gate_threshold=10.0)
        stage = GatingStage(gating_filter=gf)

        predicted_positions = np.array([[0.0, 0.0]])
        measurements = np.array([[0.5, 0.5]])  # d^2 = 0.5 < 10
        innovation_covs = np.array([np.eye(2)])

        mask = stage.process(predicted_positions, measurements, innovation_covs)
        assert mask[0, 0] == True

    def test_far_measurements_gated_out(self) -> None:
        """Measurements far from predicted positions should fail gating."""
        gf = GatingFilter(gate_threshold=10.0)
        stage = GatingStage(gating_filter=gf)

        predicted_positions = np.array([[0.0, 0.0]])
        measurements = np.array([[100.0, 100.0]])  # d^2 = 20000 >> 10
        innovation_covs = np.array([np.eye(2)])

        mask = stage.process(predicted_positions, measurements, innovation_covs)
        assert mask[0, 0] == False


class TestUpdateStage:
    """Test UpdateStage (Kalman measurement update for assigned tracks)."""

    def test_updates_assigned_tracks(self) -> None:
        """UpdateStage should modify states of assigned tracks."""
        kf = KalmanFilter(measurement_noise=1.0)
        stage = UpdateStage(kalman_filter=kf)

        t = Track(track_id=0, state=np.array([0.0, 0.0, 0.0, 0.0]), covariance=np.eye(4) * 10.0)
        measurements = np.array([[5.0, 5.0]])
        assignments = [(0, 0)]  # track 0 -> measurement 0

        old_state = t.state.copy()
        stage.process([t], assignments, measurements)

        # State should have moved toward the measurement
        assert not np.array_equal(t.state, old_state)
        # Position components should be closer to measurement
        assert abs(t.state[0] - 5.0) < abs(old_state[0] - 5.0)

    def test_records_hit(self) -> None:
        """Assigned tracks should have a hit recorded."""
        kf = KalmanFilter()
        stage = UpdateStage(kalman_filter=kf)

        t = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4) * 5.0)
        initial_hits = t.hits
        measurements = np.array([[1.0, 1.0]])
        stage.process([t], [(0, 0)], measurements)

        assert t.hits == initial_hits + 1

    def test_unassigned_tracks_unchanged(self) -> None:
        """Tracks not in assignments should not be modified."""
        kf = KalmanFilter()
        stage = UpdateStage(kalman_filter=kf)

        t0 = Track(track_id=0, state=np.array([0.0, 0.0, 0.0, 0.0]), covariance=np.eye(4))
        t1 = Track(track_id=1, state=np.array([10.0, 0.0, 10.0, 0.0]), covariance=np.eye(4))
        t1_old_state = t1.state.copy()
        t1_old_hits = t1.hits

        measurements = np.array([[0.5, 0.5]])
        # Only track 0 is assigned
        stage.process([t0, t1], [(0, 0)], measurements)

        np.testing.assert_array_equal(t1.state, t1_old_state)
        assert t1.hits == t1_old_hits

    def test_empty_assignments(self) -> None:
        """With no assignments, no tracks should be updated."""
        kf = KalmanFilter()
        stage = UpdateStage(kalman_filter=kf)

        t = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        old_state = t.state.copy()
        old_hits = t.hits

        stage.process([t], [], np.empty((0, 2)))

        np.testing.assert_array_equal(t.state, old_state)
        assert t.hits == old_hits


class TestManagementStage:
    """Test ManagementStage (missed detections and lifecycle updates)."""

    def test_records_misses_for_missed_tracks(self) -> None:
        """ManagementStage should record misses for specified track indices."""
        manager = TrackManager(confirm_hits=3, delete_misses=5)
        t = manager.create_track(np.zeros(4), np.eye(4))
        initial_misses = t.misses

        stage = ManagementStage(track_manager=manager)
        stage.process(missed_track_indices=[0])

        assert t.misses == initial_misses + 1

    def test_applies_lifecycle_rules(self) -> None:
        """ManagementStage should apply M/N confirmation and deletion rules."""
        manager = TrackManager(confirm_hits=2, delete_misses=2)
        t = manager.create_track(np.zeros(4), np.eye(4))

        # Record a hit to confirm
        t.record_hit()  # hits=2

        stage = ManagementStage(track_manager=manager)
        stage.process(missed_track_indices=[])  # triggers update_lifecycle

        assert t.status == TrackStatus.CONFIRMED

    def test_deletes_track_after_misses(self) -> None:
        """Track with enough misses should be deleted after management stage."""
        manager = TrackManager(confirm_hits=3, delete_misses=2)
        t = manager.create_track(np.zeros(4), np.eye(4))

        stage = ManagementStage(track_manager=manager)

        # Two consecutive misses
        stage.process(missed_track_indices=[0])
        stage.process(missed_track_indices=[0])

        assert t.status == TrackStatus.DELETED

    def test_empty_missed_indices(self) -> None:
        """ManagementStage with no missed indices should just run lifecycle."""
        manager = TrackManager()
        manager.create_track(np.zeros(4), np.eye(4))

        stage = ManagementStage(track_manager=manager)
        # Should not raise
        stage.process(missed_track_indices=[])

    def test_out_of_range_index_handled(self) -> None:
        """Missed index beyond track list should be handled gracefully."""
        manager = TrackManager()
        manager.create_track(np.zeros(4), np.eye(4))

        stage = ManagementStage(track_manager=manager)
        # Index 5 is out of range (only 1 track), should not crash
        stage.process(missed_track_indices=[5])
