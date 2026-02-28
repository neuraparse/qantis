"""Tests for quantum_mht.tracking.track and track_manager lifecycle logic.

Validates the Track lifecycle state machine (TENTATIVE -> CONFIRMED -> DELETED)
and the TrackManager M/N confirmation logic (Blackman & Popoli 1999, Ch 4.3).
"""

import numpy as np
import pytest

from quantum_mht.tracking.track import Track, TrackStatus
from quantum_mht.tracking.track_manager import TrackManager


class TestTrackStatusEnum:
    """Test TrackStatus enum values."""

    def test_tentative_exists(self) -> None:
        assert hasattr(TrackStatus, "TENTATIVE")

    def test_confirmed_exists(self) -> None:
        assert hasattr(TrackStatus, "CONFIRMED")

    def test_deleted_exists(self) -> None:
        assert hasattr(TrackStatus, "DELETED")

    def test_enum_values_distinct(self) -> None:
        """All three status values should be distinct."""
        values = {TrackStatus.TENTATIVE.value, TrackStatus.CONFIRMED.value, TrackStatus.DELETED.value}
        assert len(values) == 3

    def test_enum_member_count(self) -> None:
        """TrackStatus should have exactly three members."""
        assert len(TrackStatus) == 3


class TestTrackCreation:
    """Test Track instantiation and initial state."""

    def test_creation_with_state_and_covariance(self) -> None:
        """Track should be created with given state vector and covariance."""
        state = np.array([10.0, 1.0, 20.0, 0.5])
        cov = np.eye(4) * 2.0
        track = Track(track_id=0, state=state, covariance=cov)

        np.testing.assert_array_equal(track.state, state)
        np.testing.assert_array_equal(track.covariance, cov)
        assert track.track_id == 0

    def test_initial_status_is_tentative(self) -> None:
        """New tracks should start with TENTATIVE status."""
        track = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        assert track.status == TrackStatus.TENTATIVE

    def test_initial_hits_and_misses(self) -> None:
        """New tracks should start with hits=1, misses=0, age=1."""
        track = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        assert track.hits == 1
        assert track.misses == 0
        assert track.age == 1


class TestTrackPositionVelocity:
    """Test Track position and velocity properties."""

    def test_position_returns_first_half(self) -> None:
        """position should return the first half of the state vector."""
        state = np.array([10.0, 1.0, 20.0, 0.5])
        track = Track(track_id=0, state=state, covariance=np.eye(4))
        np.testing.assert_array_equal(track.position, [10.0, 1.0])

    def test_velocity_returns_second_half(self) -> None:
        """velocity should return the second half of the state vector."""
        state = np.array([10.0, 1.0, 20.0, 0.5])
        track = Track(track_id=0, state=state, covariance=np.eye(4))
        np.testing.assert_array_equal(track.velocity, [20.0, 0.5])

    def test_position_6d_state(self) -> None:
        """position with 6D state [x, vx, ax, y, vy, ay] returns first 3."""
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        track = Track(track_id=0, state=state, covariance=np.eye(6))
        np.testing.assert_array_equal(track.position, [1.0, 2.0, 3.0])

    def test_velocity_6d_state(self) -> None:
        """velocity with 6D state returns second half."""
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        track = Track(track_id=0, state=state, covariance=np.eye(6))
        np.testing.assert_array_equal(track.velocity, [4.0, 5.0, 6.0])


class TestTrackRecordHitMiss:
    """Test hit/miss recording."""

    def test_record_hit_increments_hits(self) -> None:
        track = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        assert track.hits == 1
        track.record_hit()
        assert track.hits == 2
        track.record_hit()
        assert track.hits == 3

    def test_record_hit_resets_misses(self) -> None:
        """record_hit should reset the consecutive miss counter to 0."""
        track = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        track.record_miss()
        track.record_miss()
        assert track.misses == 2
        track.record_hit()
        assert track.misses == 0

    def test_record_hit_increments_age(self) -> None:
        track = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        assert track.age == 1
        track.record_hit()
        assert track.age == 2

    def test_record_miss_increments_misses(self) -> None:
        track = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        assert track.misses == 0
        track.record_miss()
        assert track.misses == 1
        track.record_miss()
        assert track.misses == 2

    def test_record_miss_increments_age(self) -> None:
        track = Track(track_id=0, state=np.zeros(4), covariance=np.eye(4))
        track.record_miss()
        assert track.age == 2


class TestTrackManagerMNConfirmation:
    """Test M/N confirmation logic (Blackman & Popoli 1999, Ch 4.3)."""

    def test_confirmation_with_3_hits(self) -> None:
        """Track with confirm_hits=3 should become CONFIRMED after 3 hits."""
        manager = TrackManager(confirm_hits=3, confirm_window=5, delete_misses=5)
        track = manager.create_track(np.zeros(4), np.eye(4))
        assert track.status == TrackStatus.TENTATIVE

        # Track starts with hits=1, need 2 more
        track.record_hit()  # hits=2
        manager.update_lifecycle()
        assert track.status == TrackStatus.TENTATIVE

        track.record_hit()  # hits=3
        manager.update_lifecycle()
        assert track.status == TrackStatus.CONFIRMED

    def test_deletion_with_too_many_misses(self) -> None:
        """Track with enough consecutive misses should become DELETED."""
        manager = TrackManager(confirm_hits=3, confirm_window=5, delete_misses=3)
        track = manager.create_track(np.zeros(4), np.eye(4))

        for _ in range(3):
            track.record_miss()
        manager.update_lifecycle()

        assert track.status == TrackStatus.DELETED

    def test_confirmed_track_can_be_deleted(self) -> None:
        """Even confirmed tracks should be deleted after enough misses."""
        manager = TrackManager(confirm_hits=2, delete_misses=3)
        track = manager.create_track(np.zeros(4), np.eye(4))

        # Confirm the track
        track.record_hit()  # hits=2
        manager.update_lifecycle()
        assert track.status == TrackStatus.CONFIRMED

        # Now miss enough times
        for _ in range(3):
            track.record_miss()
        manager.update_lifecycle()
        assert track.status == TrackStatus.DELETED

    def test_hit_resets_miss_counter(self) -> None:
        """A hit should reset consecutive misses, preventing deletion."""
        manager = TrackManager(confirm_hits=5, delete_misses=3)
        track = manager.create_track(np.zeros(4), np.eye(4))

        track.record_miss()
        track.record_miss()  # 2 misses, about to be deleted
        track.record_hit()   # resets misses to 0
        manager.update_lifecycle()
        assert track.status != TrackStatus.DELETED


class TestTrackManagerActiveTracks:
    """Test active_tracks filtering."""

    def test_active_tracks_excludes_deleted(self) -> None:
        """active_tracks should not include DELETED tracks."""
        manager = TrackManager(confirm_hits=2, delete_misses=2)
        t1 = manager.create_track(np.zeros(4), np.eye(4))
        t2 = manager.create_track(np.ones(4), np.eye(4))

        # Delete t1
        t1.record_miss()
        t1.record_miss()
        manager.update_lifecycle()
        assert t1.status == TrackStatus.DELETED

        active = manager.active_tracks
        assert len(active) == 1
        assert active[0].track_id == t2.track_id

    def test_active_tracks_includes_tentative_and_confirmed(self) -> None:
        """active_tracks should include both TENTATIVE and CONFIRMED tracks."""
        manager = TrackManager(confirm_hits=2)
        t1 = manager.create_track(np.zeros(4), np.eye(4))  # tentative
        t2 = manager.create_track(np.ones(4), np.eye(4))
        t2.record_hit()  # hits=2
        manager.update_lifecycle()
        assert t2.status == TrackStatus.CONFIRMED

        active = manager.active_tracks
        assert len(active) == 2


class TestTrackManagerConfirmedTracks:
    """Test confirmed_tracks filtering."""

    def test_confirmed_tracks_only_confirmed(self) -> None:
        """confirmed_tracks should return only CONFIRMED tracks."""
        manager = TrackManager(confirm_hits=2, delete_misses=3)
        t1 = manager.create_track(np.zeros(4), np.eye(4))  # tentative
        t2 = manager.create_track(np.ones(4), np.eye(4))
        t2.record_hit()  # hits=2 -> confirmed
        manager.update_lifecycle()

        confirmed = manager.confirmed_tracks
        assert len(confirmed) == 1
        assert confirmed[0].track_id == t2.track_id

    def test_no_confirmed_initially(self) -> None:
        """No tracks should be confirmed immediately after creation."""
        manager = TrackManager(confirm_hits=3)
        manager.create_track(np.zeros(4), np.eye(4))
        manager.create_track(np.ones(4), np.eye(4))
        assert len(manager.confirmed_tracks) == 0


class TestTrackManagerGetPredictedStates:
    """Test get_predicted_states method."""

    def test_returns_correct_shape(self) -> None:
        """get_predicted_states should return arrays with correct shape."""
        manager = TrackManager()
        state1 = np.array([1.0, 0.5, 2.0, 0.3])
        state2 = np.array([3.0, 0.1, 4.0, 0.2])
        manager.create_track(state1, np.eye(4))
        manager.create_track(state2, np.eye(4))

        states, covs = manager.get_predicted_states()
        # position is first half of state (2 elements for 4D state)
        assert states.shape == (2, 2)
        assert covs.shape == (2, 2, 2)

    def test_empty_manager_returns_empty(self) -> None:
        """Empty manager should return empty arrays."""
        manager = TrackManager()
        states, covs = manager.get_predicted_states()
        assert states.shape == (0, 0)
        assert covs.shape == (0, 0, 0)

    def test_excludes_deleted_tracks(self) -> None:
        """get_predicted_states should only include active tracks."""
        manager = TrackManager(delete_misses=1)
        t1 = manager.create_track(np.array([1.0, 0.5, 2.0, 0.3]), np.eye(4))
        t2 = manager.create_track(np.array([3.0, 0.1, 4.0, 0.2]), np.eye(4))

        # Delete t1
        t1.record_miss()
        manager.update_lifecycle()

        states, covs = manager.get_predicted_states()
        assert states.shape == (1, 2)
