"""Tests for quantum_mht.fusion module.

Validates the Measurement, MeasurementScan, LinearSensor,
CovarianceIntersection, and FusionEngine classes.
"""

import numpy as np
import pytest

from quantum_mht.fusion.measurement import Measurement, MeasurementScan
from quantum_mht.fusion.sensor_model import LinearSensor
from quantum_mht.fusion.covariance_intersection import CovarianceIntersection
from quantum_mht.fusion.fusion_engine import FusionEngine


class TestMeasurement:
    """Test Measurement data class."""

    def test_creation(self) -> None:
        """Measurement should store position, covariance, and metadata."""
        pos = np.array([1.0, 2.0])
        cov = np.eye(2) * 0.5
        m = Measurement(position=pos, covariance=cov, sensor_id=1, timestamp=10.0)

        np.testing.assert_array_equal(m.position, pos)
        np.testing.assert_array_equal(m.covariance, cov)
        assert m.sensor_id == 1
        assert m.timestamp == 10.0

    def test_default_values(self) -> None:
        """Measurement defaults: sensor_id=0, timestamp=0.0, confidence=1.0."""
        m = Measurement(position=np.zeros(2), covariance=np.eye(2))
        assert m.sensor_id == 0
        assert m.timestamp == 0.0
        assert m.confidence == 1.0

    def test_custom_confidence(self) -> None:
        """Measurement should accept custom confidence."""
        m = Measurement(position=np.zeros(2), covariance=np.eye(2), confidence=0.8)
        assert m.confidence == 0.8


class TestMeasurementScan:
    """Test MeasurementScan collection."""

    def test_positions_returns_array(self) -> None:
        """positions property should return an (N, 2) array of measurement positions."""
        m1 = Measurement(position=np.array([1.0, 2.0]), covariance=np.eye(2))
        m2 = Measurement(position=np.array([3.0, 4.0]), covariance=np.eye(2))
        m3 = Measurement(position=np.array([5.0, 6.0]), covariance=np.eye(2))
        scan = MeasurementScan(measurements=[m1, m2, m3])

        positions = scan.positions
        assert positions.shape == (3, 2)
        np.testing.assert_array_equal(positions[0], [1.0, 2.0])
        np.testing.assert_array_equal(positions[2], [5.0, 6.0])

    def test_len_returns_count(self) -> None:
        """__len__ should return the number of measurements in the scan."""
        measurements = [
            Measurement(position=np.zeros(2), covariance=np.eye(2))
            for _ in range(5)
        ]
        scan = MeasurementScan(measurements=measurements)
        assert len(scan) == 5

    def test_empty_scan(self) -> None:
        """Empty scan should return empty positions and length 0."""
        scan = MeasurementScan(measurements=[])
        assert len(scan) == 0
        assert scan.positions.shape == (0, 2)

    def test_covariances_property(self) -> None:
        """covariances should return an (N, 2, 2) array."""
        m1 = Measurement(position=np.zeros(2), covariance=np.eye(2) * 1.0)
        m2 = Measurement(position=np.zeros(2), covariance=np.eye(2) * 2.0)
        scan = MeasurementScan(measurements=[m1, m2])

        covs = scan.covariances
        assert covs.shape == (2, 2, 2)

    def test_timestamp(self) -> None:
        """MeasurementScan should store timestamp."""
        scan = MeasurementScan(measurements=[], timestamp=42.0)
        assert scan.timestamp == 42.0


class TestLinearSensor:
    """Test LinearSensor measurement generation."""

    def test_creation_defaults(self) -> None:
        """LinearSensor should have sensible defaults."""
        sensor = LinearSensor()
        assert sensor.detection_probability == 0.9
        assert sensor.clutter_rate == 0.1
        assert sensor.measurement_noise == 1.0
        assert sensor.dim == 2

    def test_custom_parameters(self) -> None:
        """LinearSensor should accept custom parameters."""
        sensor = LinearSensor(
            sensor_id=5,
            detection_probability=0.95,
            measurement_noise=0.5,
            dim=3,
        )
        assert sensor.sensor_id == 5
        assert sensor.detection_probability == 0.95
        assert sensor.dim == 3

    def test_generate_measurement_returns_array_or_none(self) -> None:
        """generate_measurement should return a 2D array or None."""
        rng = np.random.default_rng(42)
        sensor = LinearSensor(detection_probability=1.0)  # guaranteed detection
        true_state = np.array([10.0, 1.0, 20.0, 0.5])

        result = sensor.generate_measurement(true_state, rng)
        assert result is not None
        assert result.shape == (2,)

    def test_generate_measurement_with_no_detection(self) -> None:
        """With detection_probability=0, should always return None."""
        rng = np.random.default_rng(42)
        sensor = LinearSensor(detection_probability=0.0)
        true_state = np.array([10.0, 1.0, 20.0, 0.5])

        for _ in range(10):
            result = sensor.generate_measurement(true_state, rng)
            assert result is None

    def test_generate_measurement_adds_noise(self) -> None:
        """Generated measurement should differ from true position due to noise."""
        rng = np.random.default_rng(42)
        sensor = LinearSensor(detection_probability=1.0, measurement_noise=5.0)
        true_state = np.array([10.0, 0.0, 20.0, 0.0])

        result = sensor.generate_measurement(true_state, rng)
        assert result is not None
        # Very unlikely to be exactly at true position with noise=5.0
        assert not np.allclose(result, [10.0, 20.0])

    def test_measurement_covariance_shape(self) -> None:
        """measurement_covariance should return a (dim, dim) matrix."""
        sensor = LinearSensor(dim=2, measurement_noise=3.0)
        cov = sensor.measurement_covariance()
        assert cov.shape == (2, 2)
        expected = np.eye(2) * 9.0  # noise^2
        np.testing.assert_array_equal(cov, expected)

    def test_measurement_covariance_3d(self) -> None:
        """measurement_covariance should work for 3D sensors."""
        sensor = LinearSensor(dim=3, measurement_noise=2.0)
        cov = sensor.measurement_covariance()
        assert cov.shape == (3, 3)
        expected = np.eye(3) * 4.0
        np.testing.assert_array_equal(cov, expected)


class TestCovarianceIntersection:
    """Test Covariance Intersection fusion algorithm."""

    def test_fuse_produces_valid_result(self) -> None:
        """CI fuse should produce a valid fused mean and covariance."""
        ci = CovarianceIntersection()
        mean1 = np.array([1.0, 2.0])
        cov1 = np.eye(2) * 2.0
        mean2 = np.array([3.0, 4.0])
        cov2 = np.eye(2) * 3.0

        fused_mean, fused_cov = ci.fuse(mean1, cov1, mean2, cov2, omega=0.5)

        assert fused_mean.shape == (2,)
        assert fused_cov.shape == (2, 2)

    def test_fuse_with_omega_half_is_weighted_average(self) -> None:
        """With omega=0.5 and equal covariances, fused mean should be the average."""
        ci = CovarianceIntersection()
        mean1 = np.array([0.0, 0.0])
        cov1 = np.eye(2)
        mean2 = np.array([4.0, 4.0])
        cov2 = np.eye(2)

        fused_mean, _ = ci.fuse(mean1, cov1, mean2, cov2, omega=0.5)
        # With equal weights and equal covariances: fused = (0+4)/2 = 2
        np.testing.assert_allclose(fused_mean, [2.0, 2.0], atol=1e-10)

    def test_fuse_covariance_is_positive_definite(self) -> None:
        """Fused covariance should be positive definite."""
        ci = CovarianceIntersection()
        rng = np.random.default_rng(42)

        # Generate random positive definite covariances
        A = rng.random((2, 2))
        cov1 = A @ A.T + np.eye(2)
        B = rng.random((2, 2))
        cov2 = B @ B.T + np.eye(2)

        mean1 = rng.random(2) * 10
        mean2 = rng.random(2) * 10

        _, fused_cov = ci.fuse(mean1, cov1, mean2, cov2, omega=0.5)

        eigenvalues = np.linalg.eigvalsh(fused_cov)
        assert np.all(eigenvalues > 0), f"Fused covariance not positive definite: {eigenvalues}"

    def test_fuse_omega_zero(self) -> None:
        """omega=0 should weight entirely toward estimate 2."""
        ci = CovarianceIntersection()
        mean1 = np.array([0.0, 0.0])
        cov1 = np.eye(2) * 5.0
        mean2 = np.array([10.0, 10.0])
        cov2 = np.eye(2) * 5.0

        # omega=0.01 (close to 0, avoid singular matrix with exactly 0)
        fused_mean, _ = ci.fuse(mean1, cov1, mean2, cov2, omega=0.01)
        # Should be very close to mean2
        assert np.linalg.norm(fused_mean - mean2) < np.linalg.norm(fused_mean - mean1)

    def test_fuse_omega_one(self) -> None:
        """omega=1.0 should weight entirely toward estimate 1."""
        ci = CovarianceIntersection()
        mean1 = np.array([10.0, 10.0])
        cov1 = np.eye(2) * 5.0
        mean2 = np.array([0.0, 0.0])
        cov2 = np.eye(2) * 5.0

        # omega=0.99 (close to 1)
        fused_mean, _ = ci.fuse(mean1, cov1, mean2, cov2, omega=0.99)
        assert np.linalg.norm(fused_mean - mean1) < np.linalg.norm(fused_mean - mean2)

    def test_optimal_omega_returns_value_in_range(self) -> None:
        """optimal_omega should return a value in [0, 1]."""
        ci = CovarianceIntersection()
        cov1 = np.eye(2) * 2.0
        cov2 = np.eye(2) * 3.0

        omega = ci.optimal_omega(cov1, cov2)
        assert 0.0 <= omega <= 1.0

    def test_optimal_omega_favors_tighter_estimate(self) -> None:
        """optimal_omega should weight more toward the tighter (smaller cov) estimate."""
        ci = CovarianceIntersection()
        cov1 = np.eye(2) * 1.0   # tighter
        cov2 = np.eye(2) * 100.0  # much looser

        omega = ci.optimal_omega(cov1, cov2)
        # omega weights estimate 1, so should be > 0.5 (favor tighter)
        assert omega > 0.5


class TestFusionEngine:
    """Test centralized FusionEngine."""

    def test_fuse_scans_merges_measurements(self) -> None:
        """fuse_scans should merge all measurements from multiple scans."""
        engine = FusionEngine()

        m1 = Measurement(position=np.array([1.0, 2.0]), covariance=np.eye(2), sensor_id=0)
        m2 = Measurement(position=np.array([3.0, 4.0]), covariance=np.eye(2), sensor_id=0)
        scan1 = MeasurementScan(measurements=[m1, m2], timestamp=1.0)

        m3 = Measurement(position=np.array([5.0, 6.0]), covariance=np.eye(2), sensor_id=1)
        scan2 = MeasurementScan(measurements=[m3], timestamp=1.0)

        fused = engine.fuse_scans([scan1, scan2])

        assert len(fused) == 3
        assert fused.timestamp == 1.0

    def test_fuse_scans_preserves_all_positions(self) -> None:
        """All original positions should appear in the fused scan."""
        engine = FusionEngine()

        positions = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        scan1 = MeasurementScan(measurements=[
            Measurement(position=np.array(positions[0]), covariance=np.eye(2)),
            Measurement(position=np.array(positions[1]), covariance=np.eye(2)),
        ], timestamp=5.0)
        scan2 = MeasurementScan(measurements=[
            Measurement(position=np.array(positions[2]), covariance=np.eye(2)),
            Measurement(position=np.array(positions[3]), covariance=np.eye(2)),
        ], timestamp=5.0)

        fused = engine.fuse_scans([scan1, scan2])
        fused_positions = fused.positions

        assert fused_positions.shape == (4, 2)
        for p in positions:
            found = any(np.allclose(fused_positions[i], p) for i in range(4))
            assert found, f"Position {p} not found in fused scan"

    def test_fuse_single_scan(self) -> None:
        """Fusing a single scan should return an equivalent scan."""
        engine = FusionEngine()
        m = Measurement(position=np.array([1.0, 2.0]), covariance=np.eye(2))
        scan = MeasurementScan(measurements=[m], timestamp=3.0)

        fused = engine.fuse_scans([scan])
        assert len(fused) == 1
        np.testing.assert_array_equal(fused.positions[0], [1.0, 2.0])

    def test_fuse_preserves_sensor_ids(self) -> None:
        """Fused scan should preserve sensor_id from original measurements."""
        engine = FusionEngine()

        m1 = Measurement(position=np.zeros(2), covariance=np.eye(2), sensor_id=0)
        m2 = Measurement(position=np.ones(2), covariance=np.eye(2), sensor_id=1)
        scan1 = MeasurementScan(measurements=[m1])
        scan2 = MeasurementScan(measurements=[m2])

        fused = engine.fuse_scans([scan1, scan2])
        sensor_ids = {m.sensor_id for m in fused.measurements}
        assert sensor_ids == {0, 1}
