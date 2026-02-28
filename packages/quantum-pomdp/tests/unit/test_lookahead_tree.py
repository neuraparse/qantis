"""Tests for the lookahead tree used in QBRL planning."""

import numpy as np
import pytest

from quantum_pomdp.algorithms.lookahead_tree import (
    BeliefNode,
    ActionNode,
    build_lookahead_tree,
)
from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.models.pomdp import POMDPModel


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


class TestBeliefNode:
    def test_creation_defaults(self) -> None:
        belief = BeliefState.uniform(2)
        node = BeliefNode(belief=belief)
        assert node.depth == 0
        assert node.value == 0.0
        assert node.children == []
        assert node.visit_count == 0

    def test_is_leaf_when_no_children(self) -> None:
        belief = BeliefState.uniform(2)
        node = BeliefNode(belief=belief)
        assert node.is_leaf() is True

    def test_is_not_leaf_with_children(self) -> None:
        belief = BeliefState.uniform(2)
        node = BeliefNode(belief=belief)
        node.children.append(ActionNode(action=0))
        assert node.is_leaf() is False

    def test_best_action_returns_correct_index(self) -> None:
        belief = BeliefState.uniform(2)
        node = BeliefNode(belief=belief)
        a0 = ActionNode(action=0, value=1.0)
        a1 = ActionNode(action=1, value=5.0)
        a2 = ActionNode(action=2, value=3.0)
        node.children = [a0, a1, a2]
        assert node.best_action() == 1

    def test_best_action_returns_zero_when_no_children(self) -> None:
        belief = BeliefState.uniform(2)
        node = BeliefNode(belief=belief)
        assert node.best_action() == 0

    def test_best_action_with_single_child(self) -> None:
        belief = BeliefState.uniform(2)
        node = BeliefNode(belief=belief)
        node.children.append(ActionNode(action=0, value=2.5))
        assert node.best_action() == 0

    def test_best_action_tie_breaking(self) -> None:
        """When multiple actions have the same value, max returns the first."""
        belief = BeliefState.uniform(2)
        node = BeliefNode(belief=belief)
        node.children = [
            ActionNode(action=0, value=3.0),
            ActionNode(action=1, value=3.0),
        ]
        # max with index key returns first occurrence
        assert node.best_action() == 0


class TestActionNode:
    def test_creation_defaults(self) -> None:
        node = ActionNode(action=1)
        assert node.action == 1
        assert node.depth == 0
        assert node.value == 0.0
        assert node.immediate_reward == 0.0
        assert node.children == {}
        assert node.visit_count == 0

    def test_add_observation_child(self) -> None:
        action_node = ActionNode(action=0)
        child_belief = BeliefState.uniform(2)
        child_node = BeliefNode(belief=child_belief, depth=1)
        action_node.add_observation_child(0, child_node)

        assert 0 in action_node.children
        assert action_node.children[0] is child_node

    def test_add_multiple_observation_children(self) -> None:
        action_node = ActionNode(action=0)
        for obs in range(3):
            child = BeliefNode(belief=BeliefState.uniform(2), depth=1)
            action_node.add_observation_child(obs, child)
        assert len(action_node.children) == 3

    def test_overwrite_observation_child(self) -> None:
        action_node = ActionNode(action=0)
        child1 = BeliefNode(belief=BeliefState.uniform(2), depth=1, value=1.0)
        child2 = BeliefNode(belief=BeliefState.uniform(2), depth=1, value=2.0)
        action_node.add_observation_child(0, child1)
        action_node.add_observation_child(0, child2)
        assert action_node.children[0].value == 2.0


class TestBuildLookaheadTree:
    def test_horizon_1_has_action_children(self) -> None:
        model = _make_tiger_pomdp()
        root_belief = BeliefState.uniform(2)
        root = build_lookahead_tree(
            root_belief=root_belief,
            pomdp_model=model,
            horizon=1,
            num_samples=100,
            rng=np.random.default_rng(42),
        )
        assert not root.is_leaf()
        assert len(root.children) == model.num_actions

    def test_horizon_1_children_are_action_nodes(self) -> None:
        model = _make_tiger_pomdp()
        root_belief = BeliefState.uniform(2)
        root = build_lookahead_tree(
            root_belief=root_belief,
            pomdp_model=model,
            horizon=1,
            num_samples=100,
            rng=np.random.default_rng(42),
        )
        for child in root.children:
            assert isinstance(child, ActionNode)

    def test_action_nodes_have_observation_children(self) -> None:
        model = _make_tiger_pomdp()
        root_belief = BeliefState.uniform(2)
        root = build_lookahead_tree(
            root_belief=root_belief,
            pomdp_model=model,
            horizon=1,
            num_samples=200,
            rng=np.random.default_rng(42),
        )
        for action_node in root.children:
            # Each action node should have at least one observation child
            assert len(action_node.children) > 0
            for obs, belief_node in action_node.children.items():
                assert isinstance(belief_node, BeliefNode)
                assert belief_node.depth == 1

    def test_root_depth_is_zero(self) -> None:
        model = _make_tiger_pomdp()
        root = build_lookahead_tree(
            root_belief=BeliefState.uniform(2),
            pomdp_model=model,
            horizon=1,
            rng=np.random.default_rng(42),
        )
        assert root.depth == 0

    def test_horizon_2_tree_depth(self) -> None:
        model = _make_tiger_pomdp()
        root = build_lookahead_tree(
            root_belief=BeliefState.uniform(2),
            pomdp_model=model,
            horizon=2,
            num_samples=50,
            rng=np.random.default_rng(42),
        )
        # Root at depth 0, action nodes at depth 0,
        # observation children at depth 1, then action nodes from them at depth 1,
        # and their observation children at depth 2 (leaf).
        assert not root.is_leaf()
        for action_node in root.children:
            for obs, belief_node in action_node.children.items():
                assert belief_node.depth == 1
                # At horizon=2, depth-1 nodes should be expanded further
                assert len(belief_node.children) == model.num_actions
                for sub_action_node in belief_node.children:
                    for sub_obs, sub_belief_node in sub_action_node.children.items():
                        assert sub_belief_node.depth == 2
                        # At depth == horizon, nodes should be leaves
                        assert sub_belief_node.is_leaf()

    def test_root_value_is_computed(self) -> None:
        model = _make_tiger_pomdp()
        root = build_lookahead_tree(
            root_belief=BeliefState.uniform(2),
            pomdp_model=model,
            horizon=1,
            num_samples=100,
            rng=np.random.default_rng(42),
        )
        # Root value should be set (best action value)
        assert isinstance(root.value, float)

    def test_best_action_is_valid(self) -> None:
        model = _make_tiger_pomdp()
        root = build_lookahead_tree(
            root_belief=BeliefState.uniform(2),
            pomdp_model=model,
            horizon=1,
            num_samples=100,
            rng=np.random.default_rng(42),
        )
        best = root.best_action()
        assert 0 <= best < model.num_actions

    def test_listen_is_preferred_with_uniform_belief(self) -> None:
        """With uniform belief and enough samples, listen should be preferred
        over opening a door (since expected opening reward = -45)."""
        model = _make_tiger_pomdp()
        root = build_lookahead_tree(
            root_belief=BeliefState.uniform(2),
            pomdp_model=model,
            horizon=1,
            num_samples=500,
            rng=np.random.default_rng(42),
        )
        # action 0 = listen has reward -1, actions 1 and 2 have expected -45
        listen_value = root.children[0].value
        for i in range(1, len(root.children)):
            assert listen_value > root.children[i].value

    def test_deterministic_with_same_seed(self) -> None:
        model = _make_tiger_pomdp()
        belief = BeliefState.uniform(2)

        root1 = build_lookahead_tree(
            root_belief=belief, pomdp_model=model, horizon=1,
            num_samples=100, rng=np.random.default_rng(42),
        )
        root2 = build_lookahead_tree(
            root_belief=belief, pomdp_model=model, horizon=1,
            num_samples=100, rng=np.random.default_rng(42),
        )
        assert np.isclose(root1.value, root2.value)
        assert root1.best_action() == root2.best_action()
