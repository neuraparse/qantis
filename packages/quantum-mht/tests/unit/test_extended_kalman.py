"""Tests for quantum_mht.tracking.extended_kalman module.

Validates the ExtendedKalmanFilter (EKF) that extends the standard Kalman
filter to handle nonlinear dynamics via first-order Taylor linearization
(Bar-Shalom & Li 1995, Ch 5).
"""

import numpy as np
import pytest

from quantum_mht.tracking.extended_kalman import ExtendedKalmanFilter
from quantum_mht.tracking.kalman_filter import KalmanFilter


class TestEKFCreation:
    """Test ExtendedKalmanFilter instantiation and defaults."""

    def test_default_parameters(self) -> None:
        """EKF should initialize with dim_state=4, dim_meas=2."""
        ekf = ExtendedKalmanFilter()
        assert ekf.dim_state == 4
        assert ekf.dim_meas == 2
        assert ekf.process_noise == 1.0
        assert ekf.measurement_noise == 1.0

    def test_custom_parameters(self) -> None:
        """EKF should accept custom dimensions and noise parameters."""
        ekf = ExtendedKalmanFilter(
            dim_state=6,
            dim_meas=3,
            process_noise=0.5,
            measurement_noise=2.0,
        )
        assert ekf.dim_state == 6
        assert ekf.dim_meas == 3
        assert ekf.Q.shape == (6, 6)
        assert ekf.R.shape == (3, 3)

    def test_process_noise_matrix(self) -> None:
        """Q matrix should be identity * process_noise."""
        ekf = ExtendedKalmanFilter(dim_state=4, process_noise=0.5)
        expected = np.eye(4) * 0.5
        assert np.allclose(ekf.Q, expected)

    def test_measurement_noise_matrix(self) -> None:
        """R matrix should be identity * measurement_noise."""
        ekf = ExtendedKalmanFilter(dim_meas=2, measurement_noise=3.0)
        expected = np.eye(2) * 3.0
        assert np.allclose(ekf.R, expected)

    def test_no_dynamics_fn_by_default(self) -> None:
        """By default, dynamics_fn and jacobian_fn should be None."""
        ekf = ExtendedKalmanFilter()
        assert ekf.dynamics_fn is None
        assert ekf.jacobian_fn is None
        assert ekf.measurement_fn is None
        assert ekf.measurement_jacobian_fn is None


class TestEKFLinearEquivalence:
    """EKF with default (linear/identity) dynamics should match KalmanFilter."""

    def test_predict_matches_kf_identity(self) -> None:
        """EKF with no dynamics_fn should behave like KF with identity F.

        When dynamics_fn is None, EKF uses x_predicted = x (identity).
        KF uses F = [[1, dt, 0, 0], ...], so with dt=0 the state is unchanged.
        We compare with identical process noise matrices.
        """
        # For a fair comparison, we need EKF identity behavior:
        # EKF: predicted_state = state, F = I
        # KF with dt=0: F = I (since off-diagonal dt terms become 0)
        ekf = ExtendedKalmanFilter(dim_state=4, dim_meas=2, process_noise=1.0)
        kf = KalmanFilter(dim_state=4, dim_meas=2, dt=0.0, process_noise=1.0)

        state = np.array([1.0, 0.5, 2.0, 0.3])
        cov = np.eye(4) * 2.0

        ekf_state, ekf_cov = ekf.predict(state, cov)
        kf_state, kf_cov = kf.predict(state, cov)

        # Both should produce: predicted_state = state, predicted_cov = cov + Q
        np.testing.assert_allclose(ekf_state, kf_state, atol=1e-10)
        np.testing.assert_allclose(ekf_cov, kf_cov, atol=1e-10)

    def test_update_matches_kf_with_matching_H(self) -> None:
        """EKF update should match KF update when using the same measurement model.

        Note: EKF default H = [[1,0,0,0],[0,1,0,0]] (extracts state[0:2]),
        while KF default H = [[1,0,0,0],[0,0,1,0]] (extracts x, y positions).
        To compare fairly, we supply matching measurement functions to the EKF.
        """
        kf = KalmanFilter(dim_state=4, dim_meas=2, measurement_noise=1.0)

        # Make EKF use the same H as KF
        def kf_measurement_fn(x: np.ndarray) -> np.ndarray:
            return kf.H @ x

        def kf_measurement_jacobian_fn(x: np.ndarray) -> np.ndarray:
            return kf.H

        ekf = ExtendedKalmanFilter(
            dim_state=4, dim_meas=2, measurement_noise=1.0,
            measurement_fn=kf_measurement_fn,
            measurement_jacobian_fn=kf_measurement_jacobian_fn,
        )

        state = np.array([1.0, 0.5, 2.0, 0.3])
        cov = np.eye(4) * 5.0
        measurement = np.array([1.5, 2.5])

        ekf_state, ekf_cov = ekf.update(state, cov, measurement)
        kf_state, kf_cov = kf.update(state, cov, measurement)

        np.testing.assert_allclose(ekf_state, kf_state, atol=1e-10)
        np.testing.assert_allclose(ekf_cov, kf_cov, atol=1e-10)


class TestEKFPredict:
    """Test EKF prediction step."""

    def test_predict_with_identity_preserves_state(self) -> None:
        """Without dynamics_fn, predict should not change the state (identity propagation)."""
        ekf = ExtendedKalmanFilter(process_noise=0.0)
        state = np.array([1.0, 0.5, 2.0, 0.3])
        cov = np.eye(4)

        pred_state, pred_cov = ekf.predict(state, cov)
        np.testing.assert_allclose(pred_state, state)

    def test_predict_changes_state_with_dynamics(self) -> None:
        """EKF predict with custom dynamics should change state."""
        def cv_dynamics(x: np.ndarray) -> np.ndarray:
            """Constant velocity: x_new = x + vx*dt, y_new = y + vy*dt."""
            dt = 1.0
            new = x.copy()
            new[0] += x[1] * dt
            new[2] += x[3] * dt
            return new

        def cv_jacobian(x: np.ndarray) -> np.ndarray:
            dt = 1.0
            F = np.eye(4)
            F[0, 1] = dt
            F[2, 3] = dt
            return F

        ekf = ExtendedKalmanFilter(
            dynamics_fn=cv_dynamics,
            jacobian_fn=cv_jacobian,
            process_noise=0.1,
        )

        state = np.array([0.0, 1.0, 0.0, 2.0])
        cov = np.eye(4)

        pred_state, pred_cov = ekf.predict(state, cov)

        # After CV propagation: x=0+1*1=1, y=0+2*1=2
        assert np.isclose(pred_state[0], 1.0)
        assert np.isclose(pred_state[2], 2.0)
        # Velocities unchanged
        assert np.isclose(pred_state[1], 1.0)
        assert np.isclose(pred_state[3], 2.0)

    def test_predict_increases_covariance(self) -> None:
        """Prediction should increase covariance (adds process noise Q)."""
        ekf = ExtendedKalmanFilter(process_noise=1.0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        cov = np.eye(4) * 0.5

        _, pred_cov = ekf.predict(state, cov)

        # P_pred = F*P*F^T + Q = I*P*I + Q = P + Q
        # trace(P_pred) > trace(P)
        assert np.trace(pred_cov) > np.trace(cov)


class TestEKFUpdate:
    """Test EKF measurement update step."""

    def test_update_moves_state_toward_measurement(self) -> None:
        """After update, observed state components should move toward measurement.

        EKF default H = [[1,0,0,0],[0,1,0,0]], so it observes state[0] and state[1].
        The measurement [5.0, 5.0] should pull state[0] and state[1] toward 5.
        """
        ekf = ExtendedKalmanFilter(measurement_noise=1.0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        cov = np.eye(4) * 10.0
        measurement = np.array([5.0, 5.0])

        updated_state, _ = ekf.update(state, cov, measurement)

        # state[0] and state[1] are observed -> should move toward measurement
        assert updated_state[0] > 0.0  # moved toward 5.0
        assert updated_state[1] > 0.0  # moved toward 5.0

    def test_update_reduces_covariance(self) -> None:
        """Measurement update should reduce covariance uncertainty."""
        ekf = ExtendedKalmanFilter(measurement_noise=1.0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        cov = np.eye(4) * 10.0
        measurement = np.array([1.0, 1.0])

        _, updated_cov = ekf.update(state, cov, measurement)

        # Updated covariance trace should be less than prior
        assert np.trace(updated_cov) < np.trace(cov)

    def test_update_with_exact_measurement(self) -> None:
        """If measurement exactly matches predicted observation, innovation is zero.

        EKF default: z_pred = state[:dim_meas] = state[:2].
        So measurement must equal state[:2] for zero innovation.
        """
        ekf = ExtendedKalmanFilter(measurement_noise=1.0)
        state = np.array([3.0, 0.0, 4.0, 0.0])
        cov = np.eye(4) * 5.0
        # Measurement matches state[:2] = [3.0, 0.0]
        measurement = np.array([3.0, 0.0])

        updated_state, _ = ekf.update(state, cov, measurement)

        # No innovation -> state should stay very close to original
        np.testing.assert_allclose(updated_state, state, atol=1e-10)


class TestEKFWithCustomDynamics:
    """Test EKF with nonlinear dynamics (constant turn model)."""

    def test_constant_turn_dynamics(self) -> None:
        """EKF with constant turn dynamics should produce valid state propagation."""
        omega = 0.1  # turn rate rad/s
        dt = 1.0

        def ct_dynamics(x: np.ndarray) -> np.ndarray:
            """Constant turn model: coordinated turn with rate omega."""
            sw = np.sin(omega * dt)
            cw = np.cos(omega * dt)
            new = np.zeros_like(x)
            new[0] = x[0] + sw / omega * x[1] - (1 - cw) / omega * x[3]
            new[1] = cw * x[1] - sw * x[3]
            new[2] = x[2] + (1 - cw) / omega * x[1] + sw / omega * x[3]
            new[3] = sw * x[1] + cw * x[3]
            return new

        def ct_jacobian(x: np.ndarray) -> np.ndarray:
            sw = np.sin(omega * dt)
            cw = np.cos(omega * dt)
            return np.array([
                [1, sw / omega, 0, -(1 - cw) / omega],
                [0, cw, 0, -sw],
                [0, (1 - cw) / omega, 1, sw / omega],
                [0, sw, 0, cw],
            ])

        ekf = ExtendedKalmanFilter(
            dynamics_fn=ct_dynamics,
            jacobian_fn=ct_jacobian,
            process_noise=0.01,
        )

        # Target moving east at 10 m/s
        state = np.array([0.0, 10.0, 0.0, 0.0])
        cov = np.eye(4) * 0.1

        pred_state, pred_cov = ekf.predict(state, cov)

        # State should have changed
        assert not np.allclose(pred_state, state)
        # Predicted covariance should be positive definite
        eigenvalues = np.linalg.eigvalsh(pred_cov)
        assert np.all(eigenvalues > 0)
        # State dimension should be preserved
        assert pred_state.shape == state.shape
        assert pred_cov.shape == cov.shape
