"""Tests for QBRL planner."""
import numpy as np
import pytest
from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.models.pomdp import POMDPModel
from quantum_pomdp.algorithms.lookahead_tree import BeliefNode, ActionNode, build_lookahead_tree
from quantum_pomdp.algorithms.qbrl import QBRLPlanner, QBRLConfig


class TestLookaheadTree:
    def test_belief_node_creation(self) -> None:
        belief = BeliefState.uniform(2)
        node = BeliefNode(belief=belief, depth=0)
        assert node.depth == 0
        assert node.value == 0.0

    def test_action_node_creation(self) -> None:
        node = ActionNode(action=0, depth=1)
        assert node.action == 0


class TestQBRLPlanner:
    def test_config_defaults(self) -> None:
        config = QBRLConfig()
        assert config.horizon > 0
        assert config.num_samples > 0

    def test_planner_creation(self, tiger_pomdp: POMDPModel) -> None:
        config = QBRLConfig(horizon=1, use_quantum=False)
        planner = QBRLPlanner(tiger_pomdp, config)
        assert planner is not None

    def test_classical_action_selection(self, tiger_pomdp: POMDPModel) -> None:
        config = QBRLConfig(horizon=1, use_quantum=False, num_samples=100)
        planner = QBRLPlanner(tiger_pomdp, config)
        belief = BeliefState.uniform(2)
        action = planner.select_action(belief)
        assert 0 <= action < tiger_pomdp.num_actions
