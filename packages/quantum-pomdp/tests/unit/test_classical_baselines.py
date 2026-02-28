"""Tests for classical POMDP baseline solvers (POMCP, DESPOT, PBVI)."""

import numpy as np
import pytest

from quantum_pomdp.models.pomdp import POMDPModel
from quantum_pomdp.classical_baselines.pomcp import POMCPSolver
from quantum_pomdp.classical_baselines.despot import DESPOTSolver
from quantum_pomdp.classical_baselines.pbvi import PBVISolver


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


class TestPOMCPSolver:
    def test_creation_defaults(self) -> None:
        solver = POMCPSolver()
        assert solver.num_simulations == 1000
        assert solver.exploration_constant == 1.0
        assert solver.discount_factor == 0.95
        assert solver.max_depth == 10
        assert solver.seed == 42

    def test_creation_custom(self) -> None:
        solver = POMCPSolver(
            num_simulations=500,
            exploration_constant=2.0,
            max_depth=5,
            seed=99,
        )
        assert solver.num_simulations == 500
        assert solver.exploration_constant == 2.0
        assert solver.max_depth == 5

    def test_select_action_returns_valid_index(self) -> None:
        model = _make_tiger_pomdp()
        solver = POMCPSolver(num_simulations=200, max_depth=3, seed=42)
        belief = np.array([0.5, 0.5])
        action = solver.select_action(belief, model)
        assert 0 <= action < model.num_actions

    def test_select_action_with_biased_belief(self) -> None:
        model = _make_tiger_pomdp()
        solver = POMCPSolver(num_simulations=500, max_depth=5, seed=42)
        # Strongly believe tiger is on the left
        belief = np.array([0.95, 0.05])
        action = solver.select_action(belief, model)
        assert 0 <= action < model.num_actions

    def test_select_action_deterministic_with_seed(self) -> None:
        model = _make_tiger_pomdp()
        belief = np.array([0.5, 0.5])
        s1 = POMCPSolver(num_simulations=200, max_depth=3, seed=42)
        s2 = POMCPSolver(num_simulations=200, max_depth=3, seed=42)
        assert s1.select_action(belief, model) == s2.select_action(belief, model)


class TestDESPOTSolver:
    def test_creation_defaults(self) -> None:
        solver = DESPOTSolver()
        assert solver.num_scenarios == 500
        assert solver.max_depth == 10
        assert solver.discount_factor == 0.95
        assert solver.regularization == 0.01
        assert solver.seed == 42

    def test_creation_custom(self) -> None:
        solver = DESPOTSolver(
            num_scenarios=100,
            max_depth=5,
            regularization=0.1,
            seed=123,
        )
        assert solver.num_scenarios == 100
        assert solver.regularization == 0.1

    def test_select_action_returns_valid_index(self) -> None:
        model = _make_tiger_pomdp()
        solver = DESPOTSolver(num_scenarios=100, max_depth=3, seed=42)
        belief = np.array([0.5, 0.5])
        action = solver.select_action(belief, model)
        assert 0 <= action < model.num_actions

    def test_select_action_with_biased_belief(self) -> None:
        model = _make_tiger_pomdp()
        solver = DESPOTSolver(num_scenarios=200, max_depth=5, seed=42)
        belief = np.array([0.1, 0.9])
        action = solver.select_action(belief, model)
        assert 0 <= action < model.num_actions

    def test_select_action_deterministic_with_seed(self) -> None:
        model = _make_tiger_pomdp()
        belief = np.array([0.5, 0.5])
        s1 = DESPOTSolver(num_scenarios=100, max_depth=3, seed=42)
        s2 = DESPOTSolver(num_scenarios=100, max_depth=3, seed=42)
        assert s1.select_action(belief, model) == s2.select_action(belief, model)


class TestPBVISolver:
    def test_creation_defaults(self) -> None:
        solver = PBVISolver()
        assert solver.num_belief_points == 100
        assert solver.max_iterations == 50
        assert solver.tolerance == 1e-4
        assert solver.discount_factor == 0.95

    def test_creation_custom(self) -> None:
        solver = PBVISolver(
            num_belief_points=50,
            max_iterations=20,
            tolerance=1e-3,
            discount_factor=0.9,
        )
        assert solver.num_belief_points == 50
        assert solver.discount_factor == 0.9

    def test_solve_returns_alpha_vectors(self) -> None:
        model = _make_tiger_pomdp()
        solver = PBVISolver(num_belief_points=20, max_iterations=10)
        alphas = solver.solve(model)
        assert isinstance(alphas, list)
        assert len(alphas) > 0
        for alpha in alphas:
            assert alpha.shape == (model.num_states,)

    def test_alpha_vectors_are_finite(self) -> None:
        model = _make_tiger_pomdp()
        solver = PBVISolver(num_belief_points=20, max_iterations=10)
        alphas = solver.solve(model)
        for alpha in alphas:
            assert np.all(np.isfinite(alpha))

    def test_select_action_returns_valid_index(self) -> None:
        model = _make_tiger_pomdp()
        solver = PBVISolver(num_belief_points=20, max_iterations=10)
        alphas = solver.solve(model)
        belief = np.array([0.5, 0.5])
        action = solver.select_action(belief, alphas)
        assert 0 <= action < len(alphas)

    def test_select_action_with_biased_belief(self) -> None:
        model = _make_tiger_pomdp()
        solver = PBVISolver(num_belief_points=20, max_iterations=10)
        alphas = solver.solve(model)
        belief = np.array([0.9, 0.1])
        action = solver.select_action(belief, alphas)
        assert isinstance(action, int)
        assert 0 <= action < len(alphas)

    def test_select_action_empty_alphas(self) -> None:
        solver = PBVISolver()
        belief = np.array([0.5, 0.5])
        action = solver.select_action(belief, [])
        assert action == 0

    def test_solve_convergence(self) -> None:
        """Solver with tighter tolerance should still produce valid alphas."""
        model = _make_tiger_pomdp()
        solver = PBVISolver(num_belief_points=30, max_iterations=50, tolerance=1e-6)
        alphas = solver.solve(model)
        assert len(alphas) > 0
        for alpha in alphas:
            assert np.all(np.isfinite(alpha))
