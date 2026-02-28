"""Linear Kalman filter for track state estimation.

Implements the standard discrete-time Kalman filter with the predict-update
cycle for linear Gaussian state estimation.

Predict Step:
    x_{k|k-1} = F * x_{k-1|k-1}
    P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q

Update Step:
    innovation: y_k = z_k - H * x_{k|k-1}
    innovation covariance: S_k = H * P_{k|k-1} * H^T + R
    Kalman gain: K_k = P_{k|k-1} * H^T * S_k^{-1}
    updated state: x_{k|k} = x_{k|k-1} + K_k * y_k
    updated covariance: P_{k|k} = (I - K_k * H) * P_{k|k-1}

The default state model is constant-velocity (CV) with 4D state [x, vx, y, vy]
and 2D position measurements [x, y].

Academic References:
    Kalman, "A New Approach to Linear Filtering and Prediction Problems",
        J. Basic Engineering 82(1):35-45, 1960 -- original Kalman filter.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 1 -- Kalman filter in tracking.
    Li & Jilkov, "Survey of Maneuvering Target Tracking Part I: Dynamic
        Models", IEEE TAES 39(4):1333-1364, 2003 -- CV model definition.

Quantum Context (2026):
    Quantum Kalman filter (arXiv:2404.04554, 2024) provides O(kappa polylog(n/eps))
    complexity via block encoding, offering exponential speedup over classical
    O(n^3) for large state dimensions (n>100). However, our tracking problem
    uses small state vectors (4D: [x, vx, y, vy] or 6D with acceleration),
    where classical Kalman remains optimal. The quantum advantage applies
    to the DATA ASSOCIATION step (QUBO/annealing), not the state estimation.
    Classical Kalman is preserved as the SOTA estimator for tracking-scale
    problems in our hybrid quantum-classical pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class KalmanFilter:
    """Standard linear Kalman filter (Kalman 1960).

    Implements the predict-update cycle for linear Gaussian state estimation.
    Default: constant-velocity (CV) model with 4D state and 2D measurements.

    References:
        Kalman, J. Basic Engineering 82(1):35-45, 1960.
        Bar-Shalom & Li 1995, Ch 1.
    """
    dim_state: int = 4  # [x, vx, y, vy] -- CV model (Li & Jilkov 2003)
    dim_meas: int = 2   # [x, y] -- position-only measurements
    dt: float = 1.0
    process_noise: float = 1.0
    measurement_noise: float = 1.0

    def __post_init__(self) -> None:
        self.F = np.eye(self.dim_state)
        if self.dim_state >= 4 and self.dim_meas >= 2:
            self.F[0, 1] = self.dt
            self.F[2, 3] = self.dt
        self.H = np.zeros((self.dim_meas, self.dim_state))
        self.H[0, 0] = 1.0
        if self.dim_meas >= 2 and self.dim_state >= 3:
            self.H[1, 2] = 1.0
        self.Q = np.eye(self.dim_state) * self.process_noise
        self.R = np.eye(self.dim_meas) * self.measurement_noise

    def predict(self, state: NDArray[np.float64], covariance: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Kalman predict step (Kalman 1960): x_{k|k-1} = F*x, P_{k|k-1} = F*P*F^T + Q."""
        predicted_state = self.F @ state
        predicted_cov = self.F @ covariance @ self.F.T + self.Q
        return predicted_state, predicted_cov

    def update(self, state: NDArray[np.float64], covariance: NDArray[np.float64], measurement: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Kalman update step (Kalman 1960): fuse measurement with prediction."""
        innovation = measurement - self.H @ state          # y_k = z_k - H*x_{k|k-1}
        S = self.H @ covariance @ self.H.T + self.R        # Innovation covariance S_k
        K = covariance @ self.H.T @ np.linalg.inv(S)       # Kalman gain K_k
        updated_state = state + K @ innovation              # x_{k|k} = x_{k|k-1} + K*y
        updated_cov = (np.eye(self.dim_state) - K @ self.H) @ covariance  # Joseph form simplification
        return updated_state, updated_cov

    def innovation_covariance(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute innovation covariance S = H*P*H^T + R (used for gating)."""
        return self.H @ covariance @ self.H.T + self.R
