"""Tests for quantum_mht.tracking.gating module.

Validates the GatingFilter that implements chi-squared measurement gating
using Mahalanobis distance (Bar-Shalom & Li 1995, Ch 2.4).
"""

import numpy as np
import pytest

from quantum_mht.tracking.gating import GatingFilter


class TestGatingFilterCreation:
    """Test GatingFilter instantiation and defaults."""

    def test_default_threshold(self) -> None:
        """Default gate_threshold should be 9.21 (chi-squared 99% for 2D)."""
        gf = GatingFilter()
        assert gf.gate_threshold == 9.21

    def test_default_dim(self) -> None:
        """Default measurement dimension should be 2."""
        gf = GatingFilter()
        assert gf.dim == 2

    def test_custom_threshold(self) -> None:
        """Should accept a custom gate_threshold."""
        gf = GatingFilter(gate_threshold=5.99)
        assert gf.gate_threshold == 5.99


class TestMahalanobisDistances:
    """Test mahalanobis_distances computation."""

    def test_returns_correct_shape(self) -> None:
        """Should return array with one distance per measurement."""
        gf = GatingFilter()
        predicted = np.array([0.0, 0.0])
        measurements = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        S = np.eye(2)

        distances = gf.mahalanobis_distances(predicted, measurements, S)
        assert distances.shape == (3,)

    def test_zero_distance_at_predicted_position(self) -> None:
        """Measurement at predicted position should have distance approx 0."""
        gf = GatingFilter()
        predicted = np.array([5.0, 3.0])
        measurements = np.array([[5.0, 3.0]])
        S = np.eye(2)

        distances = gf.mahalanobis_distances(predicted, measurements, S)
        assert np.isclose(distances[0], 0.0, atol=1e-12)

    def test_known_distance_identity_covariance(self) -> None:
        """With identity covariance, Mahalanobis distance = Euclidean distance squared."""
        gf = GatingFilter()
        predicted = np.array([0.0, 0.0])
        measurements = np.array([[3.0, 4.0]])
        S = np.eye(2)

        distances = gf.mahalanobis_distances(predicted, measurements, S)
        # d^2 = 3^2 + 4^2 = 25
        assert np.isclose(distances[0], 25.0)

    def test_far_measurement_large_distance(self) -> None:
        """Far-away measurement should have a large Mahalanobis distance."""
        gf = GatingFilter()
        predicted = np.array([0.0, 0.0])
        measurements = np.array([[100.0, 100.0]])
        S = np.eye(2)

        distances = gf.mahalanobis_distances(predicted, measurements, S)
        assert distances[0] > 1000.0

    def test_scaled_covariance(self) -> None:
        """Larger covariance should produce smaller Mahalanobis distances."""
        gf = GatingFilter()
        predicted = np.array([0.0, 0.0])
        measurements = np.array([[3.0, 4.0]])

        S_small = np.eye(2) * 1.0
        S_large = np.eye(2) * 100.0

        d_small = gf.mahalanobis_distances(predicted, measurements, S_small)
        d_large = gf.mahalanobis_distances(predicted, measurements, S_large)

        assert d_large[0] < d_small[0]

    def test_multiple_measurements(self) -> None:
        """Should correctly compute distances for multiple measurements."""
        gf = GatingFilter()
        predicted = np.array([0.0, 0.0])
        measurements = np.array([
            [1.0, 0.0],  # d^2 = 1
            [0.0, 2.0],  # d^2 = 4
            [3.0, 4.0],  # d^2 = 25
        ])
        S = np.eye(2)

        distances = gf.mahalanobis_distances(predicted, measurements, S)
        np.testing.assert_allclose(distances, [1.0, 4.0, 25.0])


class TestGate:
    """Test gate method (boolean mask generation)."""

    def test_returns_boolean_mask(self) -> None:
        """gate should return a boolean array."""
        gf = GatingFilter(gate_threshold=10.0)
        predicted = np.array([0.0, 0.0])
        measurements = np.array([[1.0, 1.0]])
        S = np.eye(2)

        mask = gf.gate(predicted, measurements, S)
        assert mask.dtype == bool
        assert mask.shape == (1,)

    def test_accepts_close_measurements(self) -> None:
        """Close measurements (within gate) should be accepted (True)."""
        gf = GatingFilter(gate_threshold=10.0)
        predicted = np.array([0.0, 0.0])
        # d^2 = 0.01 + 0.01 = 0.02 << 10.0
        measurements = np.array([[0.1, 0.1]])
        S = np.eye(2)

        mask = gf.gate(predicted, measurements, S)
        assert mask[0] is np.bool_(True)

    def test_rejects_far_measurements(self) -> None:
        """Far measurements (outside gate) should be rejected (False)."""
        gf = GatingFilter(gate_threshold=10.0)
        predicted = np.array([0.0, 0.0])
        # d^2 = 100 + 100 = 200 >> 10.0
        measurements = np.array([[10.0, 10.0]])
        S = np.eye(2)

        mask = gf.gate(predicted, measurements, S)
        assert mask[0] is np.bool_(False)

    def test_mixed_close_and_far(self) -> None:
        """Should correctly gate a mix of close and far measurements."""
        gf = GatingFilter(gate_threshold=10.0)
        predicted = np.array([0.0, 0.0])
        measurements = np.array([
            [0.5, 0.5],    # d^2 = 0.5, accepted
            [100.0, 0.0],  # d^2 = 10000, rejected
            [1.0, 1.0],    # d^2 = 2, accepted
            [5.0, 5.0],    # d^2 = 50, rejected
        ])
        S = np.eye(2)

        mask = gf.gate(predicted, measurements, S)
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == True
        assert mask[3] == False

    def test_boundary_measurement(self) -> None:
        """Measurement exactly at the threshold boundary should be accepted (<=)."""
        threshold = 9.21
        gf = GatingFilter(gate_threshold=threshold)
        predicted = np.array([0.0, 0.0])
        # d^2 = x^2 + 0 = threshold => x = sqrt(threshold)
        x_at_boundary = np.sqrt(threshold)
        measurements = np.array([[x_at_boundary, 0.0]])
        S = np.eye(2)

        mask = gf.gate(predicted, measurements, S)
        assert mask[0] == True

    @pytest.mark.parametrize("threshold", [1.0, 5.99, 9.21, 13.82, 50.0])
    def test_gate_with_various_thresholds(self, threshold: float) -> None:
        """Gate should respect different threshold values."""
        gf = GatingFilter(gate_threshold=threshold)
        predicted = np.array([0.0, 0.0])
        S = np.eye(2)

        # Measurement at distance slightly below and above threshold
        r_below = np.sqrt(threshold * 0.5)
        r_above = np.sqrt(threshold * 2.0)
        measurements = np.array([[r_below, 0.0], [r_above, 0.0]])

        mask = gf.gate(predicted, measurements, S)
        assert mask[0] == True   # below threshold
        assert mask[1] == False  # above threshold
