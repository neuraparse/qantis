"""Extended Kalman Filter (EKF) for nonlinear dynamics.

Extends the standard Kalman filter to handle nonlinear state dynamics and/or
nonlinear measurement models by linearizing about the current state estimate
using first-order Taylor expansion (Jacobian matrices).

EKF Predict:
    x_{k|k-1} = f(x_{k-1|k-1})
    P_{k|k-1} = F_k * P_{k-1|k-1} * F_k^T + Q
    where F_k = df/dx |_{x_{k-1|k-1}} is the dynamics Jacobian.

EKF Update:
    y_k = z_k - h(x_{k|k-1})
    S_k = H_k * P_{k|k-1} * H_k^T + R
    K_k = P_{k|k-1} * H_k^T * S_k^{-1}
    where H_k = dh/dx |_{x_{k|k-1}} is the measurement Jacobian.

This is essential for tracking targets with nonlinear dynamics (e.g.,
constant-turn models) or nonlinear measurement models (e.g., range-bearing
sensors).

Academic References:
    Kalman, "A New Approach to Linear Filtering and Prediction Problems",
        J. Basic Engineering 82(1):35-45, 1960 -- linear Kalman filter
        foundation extended here.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 5 -- EKF for target tracking
        with nonlinear dynamics.
    Li & Jilkov, "Survey of Maneuvering Target Tracking Part I: Dynamic
        Models", IEEE TAES 39(4):1333-1364, 2003 -- nonlinear motion models
        (CT, coordinated turn) requiring EKF.

Quantum Context (2026):
    EKF handles nonlinear dynamics (coordinated turn, acceleration) via
    first-order Taylor linearization. Quantum EKF research is nascent --
    arXiv:2404.04554 (2024) covers only the linear case. For tracking-scale
    nonlinear problems (4-8D state), classical EKF/UKF remains practical
    and preferred. The quantum pipeline applies quantum advantage to the
    association step (QUBO solver), not the prediction/update steps.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray

@dataclass
class ExtendedKalmanFilter:
    """Extended Kalman filter with nonlinear dynamics.

    Uses user-supplied dynamics/measurement functions and their Jacobians
    for first-order linearization. Falls back to identity (linear) if
    no functions are provided.

    References:
        Bar-Shalom & Li 1995, Ch 5 -- EKF for nonlinear tracking.
        Li & Jilkov 2003 (IEEE TAES) -- nonlinear motion models.
    """
    dim_state: int = 4
    dim_meas: int = 2
    process_noise: float = 1.0
    measurement_noise: float = 1.0
    dynamics_fn: Callable[[NDArray], NDArray] | None = None
    jacobian_fn: Callable[[NDArray], NDArray] | None = None
    measurement_fn: Callable[[NDArray], NDArray] | None = None
    measurement_jacobian_fn: Callable[[NDArray], NDArray] | None = None

    def __post_init__(self) -> None:
        self.Q = np.eye(self.dim_state) * self.process_noise
        self.R = np.eye(self.dim_meas) * self.measurement_noise

    def predict(self, state: NDArray[np.float64], covariance: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """EKF predict: propagate through nonlinear dynamics f(x) with Jacobian F."""
        if self.dynamics_fn is None:
            predicted_state = state
        else:
            predicted_state = self.dynamics_fn(state)
        if self.jacobian_fn is None:
            F = np.eye(self.dim_state)
        else:
            F = self.jacobian_fn(state)
        predicted_cov = F @ covariance @ F.T + self.Q
        return predicted_state, predicted_cov

    def update(self, state: NDArray[np.float64], covariance: NDArray[np.float64], measurement: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """EKF update: fuse measurement through nonlinear model h(x) with Jacobian H."""
        if self.measurement_fn is None:
            z_pred = state[:self.dim_meas]
        else:
            z_pred = self.measurement_fn(state)
        if self.measurement_jacobian_fn is None:
            H = np.zeros((self.dim_meas, self.dim_state))
            for i in range(min(self.dim_meas, self.dim_state)):
                H[i, i] = 1.0
        else:
            H = self.measurement_jacobian_fn(state)
        innovation = measurement - z_pred
        S = H @ covariance @ H.T + self.R
        K = covariance @ H.T @ np.linalg.inv(S)
        updated_state = state + K @ innovation
        updated_cov = (np.eye(self.dim_state) - K @ H) @ covariance
        return updated_state, updated_cov
