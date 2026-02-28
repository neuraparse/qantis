"""Ground target with configurable motion models.

Models a single ground target whose trajectory is governed by a pluggable
motion model (CV, CT, or CA). The target maintains its state history for
ground-truth evaluation against tracker output.

Motion models follow the taxonomy in Li & Jilkov (2003):
    - ConstantVelocity (CV): linear, uniform motion
    - ConstantTurn (CT): coordinated turn with fixed turn rate
    - ConstantAcceleration (CA): uniform acceleration

Academic References:
    Li & Jilkov, "Survey of Maneuvering Target Tracking Part I: Dynamic
        Models", IEEE TAES 39(4):1333-1364, 2003 -- target motion model
        taxonomy (CV, CT, CA) and state transition matrices.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995 --
        target dynamics in tracking context.

Quantum Benchmark Role (2026):
    Target objects generate ground-truth trajectories for evaluating
    quantum vs classical data association accuracy. The number of active
    targets (N) directly determines QUBO variable count (N*M + N + M),
    making target density the primary driver of quantum solver difficulty.
    Crossing trajectories and closely-spaced targets create the hardest
    association scenarios where quantum advantage is most likely.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from quantum_mht.simulation.dynamics import MotionModel, ConstantVelocity

@dataclass
class Target:
    """Simulated ground target with pluggable motion model.

    Default: ConstantVelocity (CV) from Li & Jilkov 2003.
    """
    target_id: int
    initial_state: NDArray[np.float64]
    motion_model: MotionModel = field(default_factory=ConstantVelocity)
    _state: NDArray[np.float64] | None = None
    _trajectory: list[NDArray[np.float64]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._state = self.initial_state.copy()
        self._trajectory.append(self._state.copy())

    @property
    def state(self) -> NDArray[np.float64]:
        return self._state

    @property
    def position(self) -> NDArray[np.float64]:
        return self._state[:len(self._state) // 2]

    def step(self, dt: float, rng: np.random.Generator) -> NDArray[np.float64]:
        self._state = self.motion_model.propagate(self._state, dt, rng)
        self._trajectory.append(self._state.copy())
        return self._state

    @property
    def trajectory(self) -> NDArray[np.float64]:
        return np.array(self._trajectory)
