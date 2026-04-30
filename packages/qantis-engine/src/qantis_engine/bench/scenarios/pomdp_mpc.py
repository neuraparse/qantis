"""POMDP-MPC scenario generator.

Short-horizon belief-state MPC under partial observability — generalisation
of the Tiger-style POMDP to multi-state action spaces. We use the existing
``quantum_pomdp.models.pomdp`` POMDPModel where available and fall back to
a small inline POMDP for environments without it.

The benchmark looks at deadline-bounded replanning: at each step the
pipeline must produce an action within ``deadline_ms`` regardless of
horizon. POMCP / DESPOT baselines win at long deadlines; QANTIS-Infer +
QANTIS-Optimize is supposed to win at short deadlines because the belief
update converges in fewer samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class PomdpMpcScenario:
    """A small POMDP-MPC scenario."""

    transition_matrix: NDArray[np.float64]
    observation_matrix: NDArray[np.float64]
    reward_matrix: NDArray[np.float64]
    initial_belief: NDArray[np.float64]
    horizon: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_pomdp_mpc(
    n_states: int = 4,
    n_actions: int = 3,
    n_observations: int = 4,
    horizon: int = 10,
    rare_obs_p: float = 0.05,
    seed: int = 0,
) -> PomdpMpcScenario:
    """Generate a small POMDP-MPC instance with a rare-observation regime.

    The observation matrix has ``rare_obs_p`` mass on the rare observation
    aligned with state 0, so QANTIS-Infer's amplitude amplification path is
    in its sweet spot.
    """
    rng = np.random.default_rng(seed)
    transition = np.zeros((n_actions, n_states, n_states), dtype=np.float64)
    for a in range(n_actions):
        for s in range(n_states):
            row = rng.dirichlet(np.ones(n_states))
            transition[a, s] = row

    observation = np.full((n_states, n_observations), rare_obs_p)
    for s in range(n_states):
        observation[s, s % n_observations] += 1 - rare_obs_p * n_observations
        observation[s] = np.clip(observation[s], 1e-6, None)
        observation[s] /= observation[s].sum()

    reward = rng.normal(0.0, 1.0, size=(n_actions, n_states))
    initial = np.ones(n_states) / n_states

    return PomdpMpcScenario(
        transition_matrix=transition,
        observation_matrix=observation,
        reward_matrix=reward,
        initial_belief=initial,
        horizon=horizon,
        seed=seed,
        metadata={
            "n_states": n_states,
            "n_actions": n_actions,
            "n_observations": n_observations,
            "rare_obs_p": rare_obs_p,
        },
    )
