"""Motion models: Constant Velocity (CV), Constant Turn (CT), Constant Acceleration (CA).

Implements the three standard target motion models from the Li & Jilkov (2003)
taxonomy. Each model defines a state transition matrix F(dt) and stochastic
propagation with additive Gaussian process noise.

Model Summary (Li & Jilkov 2003, Table I):
    CV (Constant Velocity):
        State: [x, vx, y, vy] (4D)
        F: nearly constant velocity, linear dynamics
        Use case: targets in uniform motion

    CT (Constant Turn):
        State: [x, vx, y, vy] (4D, turn rate as parameter)
        F: coordinated turn with fixed omega (rad/s)
        Use case: maneuvering targets with known turn rate

    CA (Constant Acceleration):
        State: [x, vx, ax, y, vy, ay] (6D)
        F: uniform acceleration in both axes
        Use case: accelerating/decelerating targets

Academic References:
    Li & Jilkov, "Survey of Maneuvering Target Tracking Part I: Dynamic
        Models", IEEE TAES 39(4):1333-1364, 2003, Table I -- complete
        taxonomy of motion models with transition matrices.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995, Ch 5
        -- motion models in the context of Kalman filtering.

Benchmark Context (2026):
    CV/CT/CA motion models (Li & Jilkov, IEEE TAES 39(4), 2003) remain
    the standard for multi-target tracking simulation. These classical
    dynamics generate realistic target trajectories for benchmarking
    quantum vs classical data association solvers. The QUBO formulation
    complexity is INDEPENDENT of the dynamics model -- it depends only
    on the number of tracks and measurements at each frame.
"""
from __future__ import annotations
from dataclasses import dataclass
import abc
import numpy as np
from numpy.typing import NDArray

class MotionModel(abc.ABC):
    @abc.abstractmethod
    def propagate(self, state: NDArray[np.float64], dt: float, rng: np.random.Generator) -> NDArray[np.float64]: ...

    @abc.abstractmethod
    def transition_matrix(self, dt: float) -> NDArray[np.float64]: ...

@dataclass
class ConstantVelocity(MotionModel):
    """Constant velocity (CV) motion model (Li & Jilkov 2003, Table I).

    4D state [x, vx, y, vy] with linear transition F and additive noise.
    """
    process_noise: float = 0.1

    def propagate(self, state: NDArray[np.float64], dt: float, rng: np.random.Generator) -> NDArray[np.float64]:
        F = self.transition_matrix(dt)
        noise = rng.normal(0, self.process_noise, size=len(state))
        return F @ state + noise

    def transition_matrix(self, dt: float) -> NDArray[np.float64]:
        return np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])

@dataclass
class ConstantTurn(MotionModel):
    """Constant turn rate (CT) motion model (Li & Jilkov 2003, Table I).

    4D state [x, vx, y, vy] with coordinated turn dynamics. Turn rate omega
    is a fixed parameter (not estimated). For omega -> 0, reduces to CV.
    """
    turn_rate: float = 0.1  # rad/s
    process_noise: float = 0.1

    def propagate(self, state: NDArray[np.float64], dt: float, rng: np.random.Generator) -> NDArray[np.float64]:
        F = self.transition_matrix(dt)
        noise = rng.normal(0, self.process_noise, size=len(state))
        return F @ state + noise

    def transition_matrix(self, dt: float) -> NDArray[np.float64]:
        w = self.turn_rate
        if abs(w) < 1e-6:
            return ConstantVelocity().transition_matrix(dt)
        sw = np.sin(w * dt)
        cw = np.cos(w * dt)
        return np.array([
            [1, sw / w, 0, -(1 - cw) / w],
            [0, cw, 0, -sw],
            [0, (1 - cw) / w, 1, sw / w],
            [0, sw, 0, cw],
        ])

@dataclass
class ConstantAcceleration(MotionModel):
    """Constant acceleration (CA) motion model (Li & Jilkov 2003, Table I).

    6D state [x, vx, ax, y, vy, ay] with uniform acceleration dynamics.
    """
    process_noise: float = 0.1

    def propagate(self, state: NDArray[np.float64], dt: float, rng: np.random.Generator) -> NDArray[np.float64]:
        F = self.transition_matrix(dt)
        noise = rng.normal(0, self.process_noise, size=len(state))
        return F @ state + noise

    def transition_matrix(self, dt: float) -> NDArray[np.float64]:
        return np.array([
            [1, dt, 0.5 * dt ** 2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt ** 2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ])
