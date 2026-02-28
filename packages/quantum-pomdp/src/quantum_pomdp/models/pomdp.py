"""POMDP model definition following the (S, A, Omega, T, O, R, gamma) formalization.

Academic References:
    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially Observable
    Stochastic Domains", Artificial Intelligence 101:99-134 (1998).
    -- Canonical POMDP formal definition used throughout this module. The
       7-tuple (S, A, Omega, T, O, R, gamma) below follows their notation.

    arXiv:2507.18606 - "Hybrid quantum-classical algorithm for near-optimal
    planning in POMDPs" (Jul 2025), Sec II.
    -- Quantum POMDP formulation: maps classical POMDP components to quantum
       circuit unitaries (T -> U_1, O -> U_2, R -> U_3). The qubit register
       allocation (k1..k5) is derived from POMDP dimensions.

    Qiskit v2.3 (Jan 2026): SamplerV2, EstimatorV2 primitives used for
    circuit execution on IBM Heron R3 (156 qubits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class POMDPModel:
    """Complete POMDP specification: (S, A, Omega, T, O, R, gamma).

    Follows the formal definition from Kaelbling, Littman, Cassandra,
    "Planning and Acting in Partially Observable Stochastic Domains",
    Artificial Intelligence 101:99-134 (1998).

    The model components map directly to quantum circuit unitaries
    (arXiv:2507.18606, Sec II):
    - T -> U_1 (transition dynamics, Sec III.B)
    - O -> U_2 (sensor model, Sec III.C)
    - R -> U_3 (reward function, Sec III.D)

    Attributes:
        num_states: |S| - number of discrete states.
        num_actions: |A| - number of available actions.
        num_observations: |Omega| - number of possible observations.
        transition_tensor: T[a, s, s'] = P(s'|s, a), shape (A, S, S).
        observation_tensor: O[a, s', o] = P(o|s', a), shape (A, S, Omega).
        reward_matrix: R[s, a] = E[r|s, a], shape (S, A).
        discount_factor: gamma in [0, 1).
        initial_belief: Optional initial belief distribution, shape (S,).
    """

    num_states: int
    num_actions: int
    num_observations: int
    transition_tensor: NDArray[np.float64]
    observation_tensor: NDArray[np.float64]
    reward_matrix: NDArray[np.float64]
    discount_factor: float = 0.95
    initial_belief: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate probability constraints and tensor shapes."""
        # Validate transition tensor shape
        expected_t = (self.num_actions, self.num_states, self.num_states)
        if self.transition_tensor.shape != expected_t:
            msg = f"Transition tensor shape {self.transition_tensor.shape} != expected {expected_t}"
            raise ValueError(msg)

        # Validate observation tensor shape
        expected_o = (self.num_actions, self.num_states, self.num_observations)
        if self.observation_tensor.shape != expected_o:
            msg = f"Observation tensor shape {self.observation_tensor.shape} != expected {expected_o}"
            raise ValueError(msg)

        # Validate reward matrix shape
        expected_r = (self.num_states, self.num_actions)
        if self.reward_matrix.shape != expected_r:
            msg = f"Reward matrix shape {self.reward_matrix.shape} != expected {expected_r}"
            raise ValueError(msg)

        # Validate transition probabilities sum to 1
        for a in range(self.num_actions):
            row_sums = self.transition_tensor[a].sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                msg = f"Transition probabilities for action {a} don't sum to 1: {row_sums}"
                raise ValueError(msg)

        # Validate observation probabilities sum to 1
        for a in range(self.num_actions):
            row_sums = self.observation_tensor[a].sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                msg = f"Observation probabilities for action {a} don't sum to 1: {row_sums}"
                raise ValueError(msg)

        # Validate discount factor
        if not 0.0 <= self.discount_factor < 1.0:
            msg = f"Discount factor must be in [0, 1), got {self.discount_factor}"
            raise ValueError(msg)

        # Validate initial belief if provided
        if self.initial_belief is not None:
            if self.initial_belief.shape != (self.num_states,):
                msg = f"Initial belief shape {self.initial_belief.shape} != ({self.num_states},)"
                raise ValueError(msg)
            if not np.isclose(self.initial_belief.sum(), 1.0, atol=1e-6):
                msg = f"Initial belief doesn't sum to 1: {self.initial_belief.sum()}"
                raise ValueError(msg)

    @property
    def state_qubits(self) -> int:
        """Qubits needed to encode state space: ceil(log2(|S|))."""
        return max(1, int(np.ceil(np.log2(self.num_states))))

    @property
    def action_qubits(self) -> int:
        """Qubits needed to encode action space: ceil(log2(|A|))."""
        return max(1, int(np.ceil(np.log2(self.num_actions))))

    @property
    def observation_qubits(self) -> int:
        """Qubits needed to encode observation space: ceil(log2(|Omega|))."""
        return max(1, int(np.ceil(np.log2(self.num_observations))))

    @property
    def total_circuit_qubits(self) -> int:
        """Total qubits for QBRL circuit: k1 + k2 + k3 + k4 + k5.

        From arXiv:2507.18606 Figure 3 and Table 1 (qubit allocation):
        k1: current state S_t       -- ceil(log2(|S|)) qubits
        k2: action A_t              -- ceil(log2(|A|)) qubits
        k3: next state S_{t+1}      -- ceil(log2(|S|)) qubits
        k4: observation O_{t+1}     -- ceil(log2(|Omega|)) qubits
        k5: reward R_{t+1}          -- default 4 bits precision
        """
        reward_bits = 4
        return (
            self.state_qubits
            + self.action_qubits
            + self.state_qubits
            + self.observation_qubits
            + reward_bits
        )

    def get_default_belief(self) -> NDArray[np.float64]:
        """Return initial belief or uniform distribution."""
        if self.initial_belief is not None:
            return self.initial_belief.copy()
        return np.ones(self.num_states) / self.num_states

    def expected_reward(self, belief: NDArray[np.float64], action: int) -> float:
        """Compute expected reward E[r|b, a] = sum_s b(s) * R(s, a)."""
        return float(belief @ self.reward_matrix[:, action])

    def observation_probability(
        self, belief: NDArray[np.float64], action: int, observation: int
    ) -> float:
        """Compute P(o|b, a) = sum_{s'} P(o|s', a) * sum_s P(s'|s, a) * b(s).

        This is the evidence probability used in quantum rejection sampling
        (arXiv:2507.18606 Sec III). Low P(o|b,a) benefits most from the
        quadratic speedup: O(P(e)^{-1/2}) vs classical O(P(e)^{-1}).
        """
        predicted = self.transition_tensor[action].T @ belief
        return float(self.observation_tensor[action, :, observation] @ predicted)
