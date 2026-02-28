"""Tests for the hybrid quantum-classical POMDP pipeline."""

import numpy as np
import pytest

from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.models.pomdp import POMDPModel
from quantum_pomdp.pipeline.belief_updater import BeliefUpdateResult, QuantumBeliefUpdater
from quantum_pomdp.pipeline.hybrid_pipeline import (
    HybridPOMDPPipeline,
    StepResult,
    EpisodeResult,
)


def _make_tiger_pomdp() -> POMDPModel:
    """Create inline tiger POMDP (2 states, 3 actions, 2 observations)."""
    listen_acc = 0.85
    T = np.zeros((3, 2, 2))
    T[0] = np.eye(2)
    T[1] = np.array([[0.5, 0.5], [0.5, 0.5]])
    T[2] = np.array([[0.5, 0.5], [0.5, 0.5]])

    O = np.zeros((3, 2, 2))
    O[0] = np.array([[listen_acc, 1 - listen_acc], [1 - listen_acc, listen_acc]])
    O[1] = np.array([[0.5, 0.5], [0.5, 0.5]])
    O[2] = np.array([[0.5, 0.5], [0.5, 0.5]])

    R = np.array([[-1.0, -100.0, 10.0], [-1.0, 10.0, -100.0]])

    return POMDPModel(
        num_states=2,
        num_actions=3,
        num_observations=2,
        transition_tensor=T,
        observation_tensor=O,
        reward_matrix=R,
        discount_factor=0.95,
    )


class TestBeliefUpdateResult:
    def test_creation_defaults(self) -> None:
        belief = BeliefState.uniform(2)
        result = BeliefUpdateResult(posterior_belief=belief)
        assert result.posterior_belief is belief
        assert result.raw_counts is None
        assert result.mitigated_counts is None
        assert result.circuit_depth == 0
        assert result.execution_time_s == 0.0
        assert result.used_quantum is False

    def test_creation_with_all_fields(self) -> None:
        belief = BeliefState(np.array([0.7, 0.3]))
        result = BeliefUpdateResult(
            posterior_belief=belief,
            raw_counts={"0": 70, "1": 30},
            mitigated_counts={"0": 72, "1": 28},
            circuit_depth=15,
            execution_time_s=0.05,
            used_quantum=True,
        )
        assert result.used_quantum is True
        assert result.circuit_depth == 15
        assert result.raw_counts["0"] == 70


class TestQuantumBeliefUpdater:
    def test_creation_classical_mode(self) -> None:
        model = _make_tiger_pomdp()
        updater = QuantumBeliefUpdater(model=model, backend=None)
        assert updater.model is model
        assert updater.backend is None

    def test_classical_update_changes_belief(self) -> None:
        model = _make_tiger_pomdp()
        updater = QuantumBeliefUpdater(model=model, backend=None)
        prior = BeliefState.uniform(2)

        result = updater.update(prior, action=0, observation=0)

        assert isinstance(result, BeliefUpdateResult)
        assert result.used_quantum is False
        posterior = result.posterior_belief
        # After hearing left with listen action, tiger-left should be more likely
        assert posterior.probabilities[0] > posterior.probabilities[1]
        assert np.isclose(posterior.probabilities.sum(), 1.0)

    def test_classical_update_preserves_normalization(self) -> None:
        model = _make_tiger_pomdp()
        updater = QuantumBeliefUpdater(model=model, backend=None)
        prior = BeliefState(np.array([0.3, 0.7]))

        result = updater.update(prior, action=0, observation=1)
        assert np.isclose(result.posterior_belief.probabilities.sum(), 1.0)

    def test_classical_update_all_observations(self) -> None:
        model = _make_tiger_pomdp()
        updater = QuantumBeliefUpdater(model=model, backend=None)
        prior = BeliefState.uniform(2)

        for obs in range(model.num_observations):
            result = updater.update(prior, action=0, observation=obs)
            assert np.isclose(result.posterior_belief.probabilities.sum(), 1.0, atol=1e-6)

    def test_classical_update_listen_then_listen(self) -> None:
        """Two consecutive listen+hear-left updates should increase tiger-left probability."""
        model = _make_tiger_pomdp()
        updater = QuantumBeliefUpdater(model=model, backend=None)
        belief = BeliefState.uniform(2)

        r1 = updater.update(belief, action=0, observation=0)
        p_after_one = r1.posterior_belief.probabilities[0]

        r2 = updater.update(r1.posterior_belief, action=0, observation=0)
        p_after_two = r2.posterior_belief.probabilities[0]

        assert p_after_two > p_after_one


class TestStepResult:
    def test_creation(self) -> None:
        belief = BeliefState.uniform(2)
        result = StepResult(
            action=0,
            observation=1,
            reward=-1.0,
            belief=belief,
            belief_update_result=None,
            planning_time_s=0.01,
            step_index=1,
        )
        assert result.action == 0
        assert result.observation == 1
        assert result.reward == -1.0
        assert result.step_index == 1
        assert result.planning_time_s == 0.01
        assert result.belief_update_result is None


class TestEpisodeResult:
    def test_creation(self) -> None:
        steps = []
        result = EpisodeResult(
            steps=steps,
            total_reward=10.0,
            total_time_s=1.5,
            num_steps=5,
        )
        assert result.total_reward == 10.0
        assert result.total_time_s == 1.5
        assert result.num_steps == 5

    def test_mean_reward(self) -> None:
        result = EpisodeResult(
            steps=[],
            total_reward=20.0,
            total_time_s=1.0,
            num_steps=4,
        )
        assert np.isclose(result.mean_reward, 5.0)

    def test_mean_reward_zero_steps(self) -> None:
        result = EpisodeResult(
            steps=[],
            total_reward=0.0,
            total_time_s=0.0,
            num_steps=0,
        )
        # Should not divide by zero
        assert result.mean_reward == 0.0


class TestHybridPOMDPPipeline:
    def test_creation(self) -> None:
        model = _make_tiger_pomdp()
        pipeline = HybridPOMDPPipeline(model=model)
        assert pipeline.model is model
        assert pipeline.belief is not None
        assert np.isclose(pipeline.belief.probabilities.sum(), 1.0)

    def test_creation_with_initial_belief(self) -> None:
        model = _make_tiger_pomdp()
        initial = BeliefState(np.array([0.8, 0.2]))
        pipeline = HybridPOMDPPipeline(model=model, initial_belief=initial)
        assert np.allclose(pipeline.belief.probabilities, [0.8, 0.2])

    def test_reset(self) -> None:
        model = _make_tiger_pomdp()
        pipeline = HybridPOMDPPipeline(model=model)
        # Run a step to change state
        pipeline.step(observation=0, rng=np.random.default_rng(42))
        # Reset should restore uniform belief
        pipeline.reset()
        assert np.allclose(pipeline.belief.probabilities, [0.5, 0.5])
        assert pipeline._step_count == 0

    def test_step_returns_step_result(self) -> None:
        model = _make_tiger_pomdp()
        pipeline = HybridPOMDPPipeline(model=model)
        result = pipeline.step(observation=0, rng=np.random.default_rng(42))
        assert isinstance(result, StepResult)
        assert 0 <= result.action < model.num_actions
        assert result.step_index == 1

    def test_step_updates_belief(self) -> None:
        model = _make_tiger_pomdp()
        pipeline = HybridPOMDPPipeline(model=model)
        initial_belief = pipeline.belief.probabilities.copy()
        pipeline.step(observation=0, rng=np.random.default_rng(42))
        # Belief should have changed after observation
        assert not np.allclose(pipeline.belief.probabilities, initial_belief)

    def test_step_with_true_state(self) -> None:
        model = _make_tiger_pomdp()
        pipeline = HybridPOMDPPipeline(model=model)
        result = pipeline.step(true_state=0, rng=np.random.default_rng(42))
        assert isinstance(result, StepResult)
        assert result.observation is not None

    def test_run_episode(self) -> None:
        model = _make_tiger_pomdp()
        pipeline = HybridPOMDPPipeline(model=model)
        episode = pipeline.run_episode(
            max_steps=5,
            true_initial_state=0,
            rng=np.random.default_rng(42),
        )
        assert isinstance(episode, EpisodeResult)
        assert episode.num_steps == 5
        assert len(episode.steps) == 5
        assert episode.total_time_s >= 0

    def test_run_episode_resets_belief(self) -> None:
        model = _make_tiger_pomdp()
        pipeline = HybridPOMDPPipeline(model=model)
        # Run episode; after run_episode, belief should have been reset at start
        pipeline.run_episode(max_steps=3, rng=np.random.default_rng(42))
        # Step count should be 3 (from the episode)
        assert pipeline._step_count == 3
