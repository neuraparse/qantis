"""Tests for the GPS-denied UAV navigation scenario."""

import numpy as np
import pytest

from quantum_pomdp.scenarios.gps_denied import create_gps_denied_pomdp
from quantum_pomdp.models.pomdp import POMDPModel


class TestCreateGPSDeniedPOMDP:
    def test_returns_pomdp_model(self) -> None:
        model = create_gps_denied_pomdp()
        assert isinstance(model, POMDPModel)

    def test_default_grid_size(self) -> None:
        model = create_gps_denied_pomdp()
        assert model.num_states == 16  # 4^2

    def test_grid_size_3(self) -> None:
        model = create_gps_denied_pomdp(grid_size=3)
        assert model.num_states == 9  # 3^2

    def test_grid_size_4(self) -> None:
        model = create_gps_denied_pomdp(grid_size=4)
        assert model.num_states == 16

    def test_grid_size_6(self) -> None:
        model = create_gps_denied_pomdp(grid_size=6)
        assert model.num_states == 36  # 6^2

    def test_state_space_equals_grid_squared(self) -> None:
        for grid_size in [2, 3, 4, 5]:
            model = create_gps_denied_pomdp(grid_size=grid_size)
            assert model.num_states == grid_size ** 2

    def test_num_actions_is_five(self) -> None:
        model = create_gps_denied_pomdp(grid_size=3)
        assert model.num_actions == 5  # N, S, E, W, hover

    def test_num_observations_equals_states(self) -> None:
        for grid_size in [3, 4, 5]:
            model = create_gps_denied_pomdp(grid_size=grid_size)
            assert model.num_observations == model.num_states

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        model = create_gps_denied_pomdp(grid_size=4)
        for a in range(model.num_actions):
            for s in range(model.num_states):
                row_sum = model.transition_tensor[a, s].sum()
                assert np.isclose(row_sum, 1.0, atol=1e-10), (
                    f"Transition row does not sum to 1 for action={a}, state={s}: {row_sum}"
                )

    def test_observation_matrix_rows_sum_to_one(self) -> None:
        model = create_gps_denied_pomdp(grid_size=4)
        for a in range(model.num_actions):
            for s in range(model.num_states):
                row_sum = model.observation_tensor[a, s].sum()
                assert np.isclose(row_sum, 1.0, atol=1e-10), (
                    f"Observation row does not sum to 1 for action={a}, state={s}: {row_sum}"
                )

    def test_transition_tensor_shape(self) -> None:
        grid_size = 4
        model = create_gps_denied_pomdp(grid_size=grid_size)
        n = grid_size ** 2
        assert model.transition_tensor.shape == (5, n, n)

    def test_observation_tensor_shape(self) -> None:
        grid_size = 4
        model = create_gps_denied_pomdp(grid_size=grid_size)
        n = grid_size ** 2
        assert model.observation_tensor.shape == (5, n, n)

    def test_reward_matrix_shape(self) -> None:
        grid_size = 4
        model = create_gps_denied_pomdp(grid_size=grid_size)
        n = grid_size ** 2
        assert model.reward_matrix.shape == (n, 5)

    def test_discount_factor(self) -> None:
        model = create_gps_denied_pomdp()
        assert model.discount_factor == 0.95

    def test_different_sensor_accuracies(self) -> None:
        for accuracy in [0.5, 0.7, 0.9]:
            model = create_gps_denied_pomdp(grid_size=3, sensor_accuracy=accuracy)
            # Correct observation should have the given accuracy
            assert np.isclose(model.observation_tensor[0, 0, 0], accuracy)

    def test_high_sensor_accuracy(self) -> None:
        model = create_gps_denied_pomdp(grid_size=3, sensor_accuracy=0.99)
        # Correct terrain match should dominate
        for a in range(model.num_actions):
            for s in range(model.num_states):
                assert model.observation_tensor[a, s, s] >= 0.99

    def test_low_sensor_accuracy(self) -> None:
        model = create_gps_denied_pomdp(grid_size=3, sensor_accuracy=0.5)
        # With accuracy 0.5, the noise spread means observations are more uniform
        model_obs = model.observation_tensor[0, 0]
        assert model_obs[0] == 0.5

    def test_hover_action_self_transition(self) -> None:
        """Hover action (4) should keep the agent in the same cell with high probability."""
        grid_size = 4
        model = create_gps_denied_pomdp(grid_size=grid_size)
        hover_action = 4
        for s in range(model.num_states):
            # Agent stays in place with probability (1 - slip) + slip = 1.0
            # because intended move is (0,0) = same cell
            assert np.isclose(model.transition_tensor[hover_action, s, s], 1.0, atol=1e-10)

    def test_goal_state_reward(self) -> None:
        """Goal state (bottom-right corner) should have large positive reward."""
        grid_size = 4
        model = create_gps_denied_pomdp(grid_size=grid_size)
        goal_state = grid_size ** 2 - 1
        # Goal reward should be 10.0 for all actions
        for a in range(model.num_actions):
            assert model.reward_matrix[goal_state, a] == 10.0

    def test_non_goal_movement_cost(self) -> None:
        """Non-goal states should have a movement cost for non-hover actions."""
        grid_size = 3
        model = create_gps_denied_pomdp(grid_size=grid_size)
        goal_state = grid_size ** 2 - 1
        for s in range(model.num_states):
            if s == goal_state:
                continue
            # Movement actions (0-3) have -0.1 cost
            for a in range(4):
                assert model.reward_matrix[s, a] == -0.1

    def test_hover_penalty(self) -> None:
        """Hover action should have a larger penalty than movement at non-goal states."""
        grid_size = 3
        model = create_gps_denied_pomdp(grid_size=grid_size)
        goal_state = grid_size ** 2 - 1
        for s in range(model.num_states):
            if s == goal_state:
                continue
            assert model.reward_matrix[s, 4] == -0.5
            assert model.reward_matrix[s, 4] < model.reward_matrix[s, 0]

    def test_transition_probabilities_non_negative(self) -> None:
        model = create_gps_denied_pomdp(grid_size=4)
        assert np.all(model.transition_tensor >= 0)

    def test_observation_probabilities_non_negative(self) -> None:
        model = create_gps_denied_pomdp(grid_size=4)
        assert np.all(model.observation_tensor >= 0)

    def test_corner_transitions(self) -> None:
        """Moving into a wall should keep the agent in place."""
        grid_size = 3
        model = create_gps_denied_pomdp(grid_size=grid_size)
        # State 0 is top-left corner (row=0, col=0)
        # Action 0 = North: should stay at state 0 (already at top)
        # Expected: T[0, 0, 0] = 1 - slip + slip = 1.0 (intended=0, same as current)
        assert np.isclose(model.transition_tensor[0, 0, 0], 1.0, atol=1e-10)
