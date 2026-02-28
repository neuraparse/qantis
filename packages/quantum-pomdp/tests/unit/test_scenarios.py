"""Tests for POMDP scenarios."""
import numpy as np
import pytest
from quantum_pomdp.scenarios.tiger_problem import create_tiger_pomdp
from quantum_pomdp.scenarios.grid_navigation import create_grid_navigation_pomdp


class TestTigerScenario:
    def test_creation(self) -> None:
        model = create_tiger_pomdp()
        assert model.num_states == 2
        assert model.num_actions == 3
        assert model.num_observations == 2

    def test_listen_accuracy(self) -> None:
        model = create_tiger_pomdp(listen_accuracy=0.9)
        assert np.isclose(model.observation_tensor[0, 0, 0], 0.9)


class TestGridNavigation:
    def test_4x4_grid(self) -> None:
        model = create_grid_navigation_pomdp(grid_size=4)
        assert model.num_states == 16
        assert model.num_actions == 5

    def test_transitions_sum_to_one(self) -> None:
        model = create_grid_navigation_pomdp(grid_size=3)
        for a in range(model.num_actions):
            for s in range(model.num_states):
                assert np.isclose(model.transition_tensor[a, s].sum(), 1.0)
