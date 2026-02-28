"""Abstract sensor model for multi-sensor tracking.

Defines the sensor interface for measurement generation in multi-target
tracking simulations. Each sensor has detection probability, clutter rate,
and measurement noise characteristics that directly affect the MTDA cost
matrix and QUBO formulation.

The sensor model determines the measurement likelihood p(z_j | track_i)
used in the log-likelihood ratio computation (Bar-Shalom & Li 1995, Ch 6).

Multi-Sensor Configurations:
    - Homogeneous: all sensors have identical parameters.
    - Heterogeneous: sensors with different detection probabilities, noise
      levels, or measurement dimensions (e.g., radar + optical).

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 1 -- measurement models and
        sensor characteristics for multi-target tracking.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 2 -- sensor models and detection probability.

Quantum Sensing Context (2026):
    Classical sensor models (radar, optical, IMU) define the measurement
    noise covariance R that feeds into the QUBO cost matrix via the
    log-likelihood ratio c_{i,j} = (z_j - H*x_i)^T * S^{-1} * (z_j - H*x_i).
    Quantum sensors (Q-CTRL Ironstone Opal quantum magnetometer, 2025)
    provide dramatically lower R (111x accuracy improvement in GPS-denied
    navigation), which tightens gating thresholds and reduces the number
    of QUBO variables -- making quantum annealing more feasible on QPU.
    The SensorModel interface abstracts over classical and quantum sensor
    payloads identically; only the noise parameters differ.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class SensorModel(abc.ABC):
    """Abstract sensor model for measurement generation.

    Sensor parameters affect the MTDA cost matrix:
    - detection_probability (P_D): probability of detecting a target in FOV
    - clutter_rate (lambda_c): expected false alarms per scan
    - measurement_noise (sigma): measurement accuracy

    References:
        Bar-Shalom & Li 1995, Ch 1 -- sensor-dependent measurement models.
    """
    sensor_id: int = 0
    detection_probability: float = 0.9
    clutter_rate: float = 0.1
    measurement_noise: float = 1.0

    @abc.abstractmethod
    def generate_measurement(self, true_state: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64] | None: ...

    @abc.abstractmethod
    def measurement_covariance(self) -> NDArray[np.float64]: ...

@dataclass
class LinearSensor(SensorModel):
    """Linear sensor that observes position directly.

    Generates noisy position measurements z = H*x + v, where v ~ N(0, R).
    This is the simplest measurement model (Bar-Shalom & Li 1995, Ch 1).
    """
    dim: int = 2

    def generate_measurement(self, true_state: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64] | None:
        if rng.random() > self.detection_probability:
            return None
        noise = rng.normal(0, self.measurement_noise, size=self.dim)
        return true_state[:self.dim] + noise

    def measurement_covariance(self) -> NDArray[np.float64]:
        return np.eye(self.dim) * self.measurement_noise ** 2
