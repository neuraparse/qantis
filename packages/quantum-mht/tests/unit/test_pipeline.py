"""Tests for MHT tracking pipeline."""

import numpy as np
import pytest

from quantum_mht.tracking.track import Track, TrackStatus
from quantum_mht.tracking.track_manager import TrackManager
from quantum_mht.tracking.kalman_filter import KalmanFilter
from quantum_mht.tracking.gating import GatingFilter
from quantum_mht.fusion.measurement import Measurement, MeasurementScan


class TestTrack:
    def test_creation(self) -> None:
        state = np.array([1.0, 0.5, 2.0, 0.3])
        cov = np.eye(4)
        track = Track(track_id=0, state=state, covariance=cov)
        assert track.status == TrackStatus.TENTATIVE
        assert track.hits == 1
        assert track.misses == 0

    def test_position_velocity(self) -> None:
        state = np.array([1.0, 0.5, 2.0, 0.3])
        track = Track(track_id=0, state=state, covariance=np.eye(4))
        assert np.allclose(track.position, [1.0, 0.5])
        assert np.allclose(track.velocity, [2.0, 0.3])

    def test_hit_miss(self) -> None:
        track = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        track.record_hit()
        assert track.hits == 2
        assert track.misses == 0
        track.record_miss()
        assert track.misses == 1


class TestTrackManager:
    def test_create_track(self) -> None:
        manager = TrackManager()
        track = manager.create_track(np.zeros(4), np.eye(4))
        assert track.track_id == 0
        assert len(manager.active_tracks) == 1

    def test_lifecycle(self) -> None:
        manager = TrackManager(confirm_hits=2, delete_misses=3)
        track = manager.create_track(np.zeros(4), np.eye(4))
        assert track.status == TrackStatus.TENTATIVE

        track.record_hit()
        manager.update_lifecycle()
        assert track.status == TrackStatus.CONFIRMED

        for _ in range(3):
            track.record_miss()
        manager.update_lifecycle()
        assert track.status == TrackStatus.DELETED


class TestKalmanFilter:
    def test_predict(self) -> None:
        kf = KalmanFilter(dt=1.0)
        state = np.array([0.0, 1.0, 0.0, 1.0])
        cov = np.eye(4)
        pred_state, pred_cov = kf.predict(state, cov)
        assert np.isclose(pred_state[0], 1.0)  # x + vx*dt
        assert np.isclose(pred_state[2], 1.0)  # y + vy*dt

    def test_update(self) -> None:
        kf = KalmanFilter(dt=1.0)
        state = np.array([0.0, 1.0, 0.0, 1.0])
        cov = np.eye(4) * 10
        measurement = np.array([0.5, 0.5])
        new_state, new_cov = kf.update(state, cov, measurement)
        # Updated position should move toward measurement
        assert abs(new_state[0] - 0.5) < abs(state[0] - 0.5)


class TestGatingFilter:
    def test_gate_close_measurements(self) -> None:
        gating = GatingFilter(gate_threshold=10.0)
        predicted = np.array([0.0, 0.0])
        measurements = np.array([[0.1, 0.1], [100.0, 100.0]])
        S = np.eye(2)
        mask = gating.gate(predicted, measurements, S)
        assert mask[0] == True
        assert mask[1] == False


class TestMeasurementScan:
    def test_positions(self) -> None:
        m1 = Measurement(position=np.array([1.0, 2.0]), covariance=np.eye(2))
        m2 = Measurement(position=np.array([3.0, 4.0]), covariance=np.eye(2))
        scan = MeasurementScan(measurements=[m1, m2])
        assert scan.positions.shape == (2, 2)
        assert len(scan) == 2
