"""Covariance Intersection (CI) for distributed sensor fusion.

Implements the Covariance Intersection algorithm for fusing state estimates
from multiple sensors when the cross-correlations between their estimation
errors are unknown. CI produces a consistent (non-divergent) fused estimate
regardless of the unknown correlation structure.

The CI fusion rule:
    P_fused^{-1} = omega * P_1^{-1} + (1-omega) * P_2^{-1}
    x_fused = P_fused * (omega * P_1^{-1} * x_1 + (1-omega) * P_2^{-1} * x_2)

where omega in [0, 1] is optimized to minimize tr(P_fused) or det(P_fused).

CI is more conservative than naive covariance fusion (which assumes
independence) but is guaranteed to be consistent even when the actual
cross-correlations are non-zero and unknown.

Academic References:
    Julier & Uhlmann, "A Non-divergent Estimation Algorithm in the Presence
        of Unknown Correlations", Proc. American Control Conference, 1997
        -- original Covariance Intersection algorithm.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995, Ch 8
        -- distributed fusion context.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 10.4 -- CI in multi-platform tracking.

Quantum Context (2026):
    Covariance Intersection (Julier & Uhlmann, 1997) provides conservative
    fusion without requiring cross-correlation knowledge -- critical for
    distributed drone swarms with asynchronous sensor updates. The CI
    optimization (minimizing det(P_fused) over omega in [0,1]) is solved
    classically in O(1) per fusion step.

    Quantum distributed fusion is an OPEN research problem as of 2026.
    Distributed Quantum Approximate Optimization (DQAOA) has been proposed
    but is not yet practical for real-time tracking. Classical CI is
    preserved as the mathematically guaranteed conservative approach.

    Quantum sensing integration: Q-CTRL Ironstone Opal (TIME Best
    Invention 2025) achieves 111x accuracy improvement in GPS-denied
    navigation via quantum magnetometry. When quantum sensors are used,
    CI fusion combines their high-fidelity measurements with classical
    IMU data, reducing the measurement noise covariance R and improving
    gating + association quality downstream.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class CovarianceIntersection:
    """CI-based distributed sensor fusion (Julier & Uhlmann 1997).

    Fuses two state estimates with unknown cross-correlations using the
    Covariance Intersection algorithm. Guarantees consistency regardless
    of the true correlation structure.

    References:
        Julier & Uhlmann 1997 (American Control Conference).
    """

    def fuse(self, mean1: NDArray, cov1: NDArray, mean2: NDArray, cov2: NDArray, omega: float = 0.5) -> tuple[NDArray, NDArray]:
        """Fuse two estimates using Covariance Intersection (Julier & Uhlmann 1997).

        P_fused^{-1} = omega*P1^{-1} + (1-omega)*P2^{-1}
        x_fused = P_fused * (omega*P1^{-1}*x1 + (1-omega)*P2^{-1}*x2)
        """
        P1_inv = np.linalg.inv(cov1)
        P2_inv = np.linalg.inv(cov2)
        P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
        P_fused = np.linalg.inv(P_fused_inv)
        x_fused = P_fused @ (omega * P1_inv @ mean1 + (1 - omega) * P2_inv @ mean2)
        return x_fused, P_fused

    def optimal_omega(self, cov1: NDArray, cov2: NDArray, num_steps: int = 100) -> float:
        """Find omega that minimizes trace of fused covariance.

        Grid search over omega in [0, 1] to minimize tr(P_fused).
        See Julier & Uhlmann 1997 for the optimization criterion.
        """
        best_omega = 0.5
        best_trace = float("inf")
        for i in range(num_steps + 1):
            w = i / num_steps
            try:
                _, P = self.fuse(np.zeros(len(cov1)), cov1, np.zeros(len(cov1)), cov2, w)
                t = np.trace(P)
                if t < best_trace:
                    best_trace = t
                    best_omega = w
            except np.linalg.LinAlgError:
                continue
        return best_omega
