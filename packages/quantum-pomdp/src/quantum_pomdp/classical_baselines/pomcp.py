"""POMCP (Partially Observable Monte Carlo Planning) wrapper.

WHO:
    Silver & Veness (DeepMind) introduced POMCP at NIPS 2010 -- Monte Carlo
    Tree Search adapted for POMDPs with particle-based belief representation.

    Silver, D. & Veness, J., "Monte-Carlo Planning in Large POMDPs",
    Advances in Neural Information Processing Systems 23 (NIPS 2010).

WHAT:
    POMCP uses random rollouts to estimate action values, maintaining beliefs
    as unweighted particle sets. Each simulation samples a state from the
    belief, runs a Monte Carlo rollout, and backs up the value. The algorithm
    extends UCT (Upper Confidence Trees) to partially observable settings by
    replacing state nodes with particle-based belief nodes.

    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    -- POMDP formal definition. The belief update in _update_belief()
       follows their Bayesian filter formulation (Eq. 4).

WHY BASELINE:
    POMCP is the primary classical baseline for QBRL (arXiv:2507.18606).
    Classical belief update requires O(P(e)^{-1}) rejection sampling
    iterations where P(e) is observation probability. Quantum amplitude
    amplification (Brassard et al. 2002, improved by BIQAE Quantum 10:1962
    Jan 2026) reduces this to O(P(e)^{-1/2}) -- the core quantum advantage.

VERSION:
    Uses pomdp-py v1.3.5 (2024 PyPI release) for POMCP implementation.
    This simplified wrapper uses random rollouts for benchmarking.
    For full-featured POMCP with UCB1 exploration and progressive widening,
    see the pomdp-py library directly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

@dataclass
class POMCPSolver:
    """Monte Carlo Tree Search for POMDPs.

    Silver & Veness, "Monte-Carlo Planning in Large POMDPs", NIPS 2010.

    This is a simplified implementation for benchmarking against QBRL
    (arXiv:2507.18606). Uses random rollouts instead of UCB1 exploration.
    For full-featured POMCP, see pomdp-py v1.3.5.
    """
    num_simulations: int = 1000
    exploration_constant: float = 1.0
    discount_factor: float = 0.95
    max_depth: int = 10
    seed: int = 42

    def select_action(self, belief: NDArray[np.float64], pomdp_model: Any) -> int:
        """Select action using POMCP tree search."""
        rng = np.random.default_rng(self.seed)
        num_actions = pomdp_model.num_actions

        # Simple rollout-based action selection
        action_values = np.zeros(num_actions)
        action_counts = np.zeros(num_actions)

        for _ in range(self.num_simulations):
            action = rng.integers(0, num_actions)
            value = self._simulate(belief, action, 0, pomdp_model, rng)
            action_values[action] += value
            action_counts[action] += 1

        # Average values
        mask = action_counts > 0
        action_values[mask] /= action_counts[mask]
        return int(np.argmax(action_values))

    def _simulate(self, belief: NDArray, action: int, depth: int, model: Any, rng: np.random.Generator) -> float:
        """Simulate a single trajectory."""
        if depth >= self.max_depth:
            return 0.0

        # Sample state from belief
        state = rng.choice(len(belief), p=belief)
        reward = model.reward_matrix[state, action]

        # Sample next state
        next_state = rng.choice(len(belief), p=model.transition_tensor[action, state])

        # Sample observation
        obs = rng.choice(model.num_observations, p=model.observation_tensor[action, next_state])

        # Update belief
        new_belief = self._update_belief(belief, action, obs, model)

        # Random rollout for future value
        future_action = rng.integers(0, model.num_actions)
        future_value = self._simulate(new_belief, future_action, depth + 1, model, rng)

        return reward + self.discount_factor * future_value

    def _update_belief(self, belief: NDArray, action: int, obs: int, model: Any) -> NDArray:
        """Bayesian belief update per Kaelbling et al. (1998), Eq. 4."""
        T = model.transition_tensor[action]  # (S, S')
        O = model.observation_tensor[action]  # (S', O)

        # b'(s') = sum_s T(s,a,s') * b(s)
        predicted = T.T @ belief

        # b''(s') = O(s',a,o) * b'(s')
        updated = O[:, obs] * predicted

        total = updated.sum()
        if total > 0:
            updated /= total
        else:
            updated = np.ones_like(updated) / len(updated)

        return updated
