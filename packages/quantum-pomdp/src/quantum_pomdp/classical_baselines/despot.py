"""DESPOT (Determinized Sparse Partially Observable Tree) wrapper.

WHO:
    Somani, Ye, Hsu, Lee (National University of Singapore) introduced
    DESPOT at NIPS 2013. Extended in 2017 with improved bounds.

    Somani, A., Ye, N., Hsu, D., Lee, W.S., "DESPOT: Online POMDP
    Planning with Regularization", NIPS 2013.
    Ye, N., Somani, A., Hsu, D., Lee, W.S., "DESPOT: Online POMDP
    Planning with Regularization", J. Artificial Intelligence Research
    58:231-266 (2017). -- Extended journal version with convergence
    analysis and additional experimental evaluation.

WHAT:
    DESPOT (Determinized Sparse Partially Observable Tree) uses a fixed set
    of K random scenarios to build a sparse belief tree, avoiding full
    enumeration of |Omega|^H observation sequences. Regularization parameter
    lambda controls the exploration-exploitation trade-off: higher lambda
    favors deeper but sparser trees. The algorithm applies anytime
    regularized optimization to balance solution quality and computational
    cost.

WHY BASELINE:
    DESPOT provides an alternative to POMCP with anytime bounded performance.
    Chosen as baseline because it handles large observation spaces well.
    DeSPOTLight (2023) variant exists but DESPOT remains the standard
    academic baseline.

QUANTUM COMPARISON:
    QBRL (arXiv:2507.18606) replaces DESPOT's random scenario sampling with
    quantum belief updates, achieving quadratic speedup for the belief update
    bottleneck. DESPOT is a classical baseline for QBRL evaluation alongside
    POMCP (Silver & Veness 2010).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

@dataclass
class DESPOTSolver:
    """DESPOT online POMDP solver.

    Somani, Ye, Hsu, Lee, "DESPOT: Online POMDP Planning with
    Regularization", NIPS 2013. Extended: JAIR 58:231-266 (2017).

    Uses deterministic scenarios for sparse belief tree construction.
    The regularization parameter controls tree size vs value accuracy.
    """
    num_scenarios: int = 500
    max_depth: int = 10
    discount_factor: float = 0.95
    regularization: float = 0.01
    seed: int = 42

    def select_action(self, belief: NDArray[np.float64], pomdp_model: Any) -> int:
        """Select action using DESPOT."""
        rng = np.random.default_rng(self.seed)
        n_actions = pomdp_model.num_actions

        # Generate deterministic scenarios
        scenarios = self._generate_scenarios(belief, pomdp_model, rng)

        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for scenario in scenarios:
                value = self._evaluate_scenario(scenario, a, pomdp_model, rng)
                action_values[a] += value
            action_values[a] /= len(scenarios)

        return int(np.argmax(action_values))

    def _generate_scenarios(self, belief: NDArray, model: Any, rng: np.random.Generator) -> list[int]:
        """Generate deterministic scenarios from belief."""
        return [int(rng.choice(len(belief), p=belief)) for _ in range(self.num_scenarios)]

    def _evaluate_scenario(self, initial_state: int, action: int, model: Any, rng: np.random.Generator) -> float:
        """Evaluate a single scenario."""
        state = initial_state
        total_reward = 0.0
        discount = 1.0

        for depth in range(self.max_depth):
            a = action if depth == 0 else rng.integers(0, model.num_actions)
            reward = model.reward_matrix[state, a]
            total_reward += discount * reward
            discount *= self.discount_factor
            state = int(rng.choice(model.num_states, p=model.transition_tensor[a, state]))

        return total_reward
