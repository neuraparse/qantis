"""Measurement data structures for multi-target tracking.

Defines the measurement and measurement scan data structures used throughout
the tracking pipeline. Measurements are the fundamental input to the MTDA
QUBO formulation.

Each measurement contains:
    - position: observed target location (z_j in the QUBO formulation)
    - covariance: measurement uncertainty (R_j, used for gating and cost)
    - sensor_id: originating sensor (for multi-sensor fusion)
    - timestamp: scan time (for temporal processing)

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 1 -- measurement models and
        scan-based processing framework.
    Stollenwerk et al., arXiv:2110.08346, 2021 -- measurements as input
        to the MTDA QUBO cost matrix.

Quantum Pipeline Role (2026):
    Measurement and MeasurementScan are the atomic data structures that
    flow through the entire hybrid quantum-classical pipeline:
      Sensor -> Measurement -> MeasurementScan -> Gating -> CostMatrix -> QUBO -> Solver
    Each measurement's position z_j and covariance R_j are used to compute
    the log-likelihood ratio c_{i,j} in the QUBO cost matrix. The number
    of measurements M in a scan directly determines QUBO size (N*M + N + M
    binary variables), making measurement-level clutter filtering critical
    for keeping problems within QPU capacity.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

@dataclass
class Measurement:
    """Single sensor measurement (Bar-Shalom & Li 1995, Ch 1).

    Represents one observation z_j from a sensor at a specific time.
    """
    position: NDArray[np.float64]
    covariance: NDArray[np.float64]
    sensor_id: int = 0
    timestamp: float = 0.0
    confidence: float = 1.0

@dataclass
class MeasurementScan:
    """Collection of measurements at a single time step.

    A scan represents all measurements received at one discrete time step.
    This is the atomic input to the tracking pipeline's association stage
    (Bar-Shalom & Li 1995, Ch 1).
    """
    measurements: list[Measurement]
    timestamp: float = 0.0

    @property
    def positions(self) -> NDArray[np.float64]:
        if not self.measurements:
            return np.empty((0, 2))
        return np.array([m.position for m in self.measurements])

    @property
    def covariances(self) -> NDArray[np.float64]:
        if not self.measurements:
            return np.empty((0, 2, 2))
        return np.array([m.covariance for m in self.measurements])

    def __len__(self) -> int:
        return len(self.measurements)
