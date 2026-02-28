"""Abstract sensor protocol for POMDP observation sources.

Academic References:
    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    -- The observation function O(s', a, o) = P(o|s', a) defines the
       sensor model in the POMDP. This interface abstracts physical
       sensors into the discrete observation space required by the
       POMDP formulation.

    Q-CTRL Ironstone Opal (2025), "111x accuracy improvement in GPS-denied
    navigation, 4m accuracy over 700km airborne trials".
    -- Quantum-enhanced sensors (cold-atom accelerometers, atomic clocks)
       provide lower-noise observations. TIME Best Invention 2025.
       The SensorInterface protocol accommodates both classical and
       quantum sensor implementations.

    arXiv:2507.18606, Sec III.C - The observation tensor O[a, s', o]
    used in the quantum circuit U_2 is populated from sensor models
    conforming to this interface.
"""
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

@runtime_checkable
class SensorInterface(Protocol):
    """Protocol for sensors that provide observations to the POMDP.

    Maps physical sensor readings to the discrete observation space
    O(s', a, o) per Kaelbling et al. (1998). Supports both classical
    and quantum-enhanced sensor implementations.
    """
    @property
    def observation_dim(self) -> int: ...

    def get_observation(self) -> NDArray[np.float64]: ...

    def observation_to_index(self, observation: NDArray[np.float64]) -> int: ...

    def observation_probability(self, state: int, observation: int) -> float: ...

class DiscreteSensor:
    """Simple discrete sensor implementation."""
    def __init__(self, observation_matrix: NDArray[np.float64]) -> None:
        self._obs_matrix = observation_matrix  # (n_states, n_observations)

    @property
    def observation_dim(self) -> int:
        return self._obs_matrix.shape[1]

    def get_observation(self) -> NDArray[np.float64]:
        return np.zeros(self.observation_dim)

    def observation_to_index(self, observation: NDArray[np.float64]) -> int:
        return int(np.argmax(observation))

    def observation_probability(self, state: int, observation: int) -> float:
        return float(self._obs_matrix[state, observation])
