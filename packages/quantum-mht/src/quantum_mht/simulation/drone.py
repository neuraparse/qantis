"""Individual drone agent for surveillance.

Models a single UAV surveillance platform with position, velocity, sensor
payload, and field-of-view (FOV) constraints. Each drone can observe targets
within its FOV radius and generate noisy measurements via its sensor model.

The drone's motion model is simple constant-velocity (no control input),
suitable for pre-planned patrol patterns. For reactive UAV control, a
higher-fidelity dynamics model would be needed.

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 8 -- multi-platform sensor
        management and observation geometry.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 10 -- airborne sensor platform modeling.

Quantum Benchmark Context (2026):
    UAV dynamics simplified to constant-velocity for proof-of-concept.
    Production tracking uses 6DOF models, but QUBO complexity is
    independent of dynamics model -- depends only on track-measurement
    count. The drone platform serves as the sensor carrier; its motion
    model does not affect the quantum solver's performance characteristics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from quantum_mht.fusion.sensor_model import LinearSensor

@dataclass
class Drone:
    """Simulated surveillance drone with sensor payload.

    Models a UAV platform with configurable FOV and linear sensor.
    """
    drone_id: int
    position: NDArray[np.float64]
    sensor: LinearSensor = field(default_factory=LinearSensor)
    fov_radius: float = 50.0
    velocity: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.velocity is None:
            self.velocity = np.zeros(2)

    def step(self, dt: float) -> None:
        self.position = self.position + self.velocity * dt

    def can_observe(self, target_position: NDArray[np.float64]) -> bool:
        distance = np.linalg.norm(target_position - self.position)
        return float(distance) <= self.fov_radius

    def observe(self, target_state: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64] | None:
        if not self.can_observe(target_state[:2]):
            return None
        return self.sensor.generate_measurement(target_state, rng)
