"""Hybrid quantum-classical POMDP pipeline.

Main orchestrator for the POMDP belief estimation and planning loop.
Combines quantum belief updates with classical action selection.

Academic References:
    arXiv:2507.18606 - "Hybrid quantum-classical algorithm for near-optimal
    planning in POMDPs" (Jul 2025).
    -- The hybrid architecture: quantum circuits handle belief updates
       (where the quadratic speedup O(P(e)^{-1/2}) applies), while
       classical computation handles action selection via lookahead tree
       search. This division exploits quantum advantage where it matters
       most (low-probability evidence conditioning) while avoiding
       quantum overhead for classical-efficient operations.

    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    -- The observe-update-plan-act loop implemented in step() follows
       the standard POMDP agent cycle from this foundational work.

    Qiskit v2.3 (Jan 2026): SamplerV2 and EstimatorV2 primitives
    used for quantum circuit execution in the belief update stage.
    Compatible with IBM Heron R3 (156 qubits).

    Cai et al., "Quantum Error Mitigation", Rev. Mod. Phys. 95,
    045005 (2023).
    -- Composable error mitigation techniques applied in the
       QuantumBeliefUpdater stage to improve measurement fidelity
       on NISQ hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging
import time

import numpy as np
from numpy.typing import NDArray

from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.models.pomdp import POMDPModel
from quantum_pomdp.algorithms.qbrl import QBRLPlanner, QBRLConfig
from quantum_pomdp.pipeline.belief_updater import QuantumBeliefUpdater, BeliefUpdateResult

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    action: int
    observation: int | None
    reward: float
    belief: BeliefState
    belief_update_result: BeliefUpdateResult | None
    planning_time_s: float
    step_index: int


@dataclass
class EpisodeResult:
    """Result from a full episode."""
    steps: list[StepResult]
    total_reward: float
    total_time_s: float
    num_steps: int

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.num_steps)


class HybridPOMDPPipeline:
    """Main pipeline orchestrating quantum-classical POMDP solving.

    Implements the hybrid architecture from arXiv:2507.18606:

    Pipeline loop (per Kaelbling et al. 1998 agent cycle):
    1. Observe environment (get observation)
    2. Update belief (quantum via arXiv:2507.18606 or classical fallback)
    3. Plan action (QBRL with H-step lookahead tree)
    4. Execute action in environment
    5. Collect reward
    """

    def __init__(
        self,
        model: POMDPModel,
        planner: QBRLPlanner | None = None,
        belief_updater: QuantumBeliefUpdater | None = None,
        initial_belief: BeliefState | None = None,
    ) -> None:
        self.model = model
        self.planner = planner or QBRLPlanner(model)
        self.belief_updater = belief_updater
        self.belief = initial_belief or BeliefState.uniform(model.num_states)
        self._step_count = 0

    def step(
        self,
        observation: int | None = None,
        true_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> StepResult:
        """Execute one step of the pipeline.

        Args:
            observation: Observation from environment (if available).
            true_state: True state for simulation (optional).
            rng: Random generator for simulation.
        """
        if rng is None:
            rng = np.random.default_rng()

        self._step_count += 1

        # 1. Select action
        t0 = time.perf_counter()
        action = self.planner.select_action(self.belief)
        planning_time = time.perf_counter() - t0

        # 2. Compute reward
        if true_state is not None:
            reward = float(self.model.reward_matrix[true_state, action])
        else:
            reward = self.model.expected_reward(self.belief.probabilities, action)

        # 3. Get observation (simulated or provided)
        if observation is None and true_state is not None:
            # Simulate observation
            next_state = int(rng.choice(
                self.model.num_states,
                p=self.model.transition_tensor[action, true_state],
            ))
            observation = int(rng.choice(
                self.model.num_observations,
                p=self.model.observation_tensor[action, next_state],
            ))

        # 4. Update belief
        update_result = None
        if observation is not None:
            if self.belief_updater is not None:
                update_result = self.belief_updater.update(
                    self.belief, action, observation,
                )
                self.belief = update_result.posterior_belief
            else:
                self.belief = self.belief.classical_update(
                    action=action,
                    observation=observation,
                    transition_tensor=self.model.transition_tensor,
                    observation_tensor=self.model.observation_tensor,
                )

        return StepResult(
            action=action,
            observation=observation,
            reward=reward,
            belief=self.belief,
            belief_update_result=update_result,
            planning_time_s=planning_time,
            step_index=self._step_count,
        )

    def run_episode(
        self,
        max_steps: int = 50,
        true_initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> EpisodeResult:
        """Run a full episode of the pipeline.

        Args:
            max_steps: Maximum number of steps.
            true_initial_state: Starting state for simulation.
            rng: Random generator.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Reset belief
        self.belief = BeliefState.uniform(self.model.num_states)
        self._step_count = 0

        state = true_initial_state
        if state is None:
            state = int(rng.choice(self.model.num_states))

        steps: list[StepResult] = []
        total_reward = 0.0
        t0 = time.perf_counter()

        for _ in range(max_steps):
            result = self.step(true_state=state, rng=rng)
            steps.append(result)
            total_reward += result.reward

            # Transition to next state
            state = int(rng.choice(
                self.model.num_states,
                p=self.model.transition_tensor[result.action, state],
            ))

        total_time = time.perf_counter() - t0

        return EpisodeResult(
            steps=steps,
            total_reward=total_reward,
            total_time_s=total_time,
            num_steps=len(steps),
        )

    def reset(self) -> None:
        """Reset the pipeline to initial state."""
        self.belief = BeliefState.uniform(self.model.num_states)
        self._step_count = 0
        self.planner.reset()
