"""Tests for Bayesian network representation of POMDP dynamics."""

import numpy as np
import pytest

from quantum_pomdp.models.bayesian_network import BayesianNetworkNode, BayesianNetwork
from quantum_pomdp.models.pomdp import POMDPModel


def _make_simple_pomdp() -> POMDPModel:
    """Create a minimal 2-state, 2-action, 2-observation POMDP."""
    T = np.zeros((2, 2, 2))
    T[0] = np.array([[0.8, 0.2], [0.3, 0.7]])
    T[1] = np.array([[0.5, 0.5], [0.5, 0.5]])

    O = np.zeros((2, 2, 2))
    O[0] = np.array([[0.9, 0.1], [0.2, 0.8]])
    O[1] = np.array([[0.5, 0.5], [0.5, 0.5]])

    R = np.array([[1.0, -1.0], [0.0, 2.0]])

    return POMDPModel(
        num_states=2,
        num_actions=2,
        num_observations=2,
        transition_tensor=T,
        observation_tensor=O,
        reward_matrix=R,
        discount_factor=0.95,
    )


class TestBayesianNetworkNode:
    def test_node_creation_defaults(self) -> None:
        node = BayesianNetworkNode(name="test", num_values=4)
        assert node.name == "test"
        assert node.num_values == 4
        assert node.parents == []
        assert node.cpt is None

    def test_node_creation_with_parents(self) -> None:
        cpt = np.array([[0.6, 0.4], [0.3, 0.7]])
        node = BayesianNetworkNode(
            name="child",
            num_values=2,
            parents=["parent_a", "parent_b"],
            cpt=cpt,
        )
        assert node.name == "child"
        assert node.num_values == 2
        assert node.parents == ["parent_a", "parent_b"]
        assert np.array_equal(node.cpt, cpt)

    def test_node_creation_with_empty_parents(self) -> None:
        node = BayesianNetworkNode(name="root", num_values=3, parents=[])
        assert len(node.parents) == 0


class TestBayesianNetwork:
    def test_from_pomdp_creates_expected_nodes(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        expected_nodes = {"S_t", "A_t", "S_t+1", "O_t+1", "R_t+1"}
        assert set(bn.nodes.keys()) == expected_nodes

    def test_from_pomdp_root_nodes_have_no_parents(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        assert bn.nodes["S_t"].parents == []
        assert bn.nodes["A_t"].parents == []

    def test_from_pomdp_child_nodes_have_correct_parents(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        assert set(bn.nodes["S_t+1"].parents) == {"S_t", "A_t"}
        assert set(bn.nodes["O_t+1"].parents) == {"S_t+1", "A_t"}
        assert set(bn.nodes["R_t+1"].parents) == {"S_t", "A_t"}

    def test_from_pomdp_node_values_match_dimensions(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        assert bn.nodes["S_t"].num_values == 2
        assert bn.nodes["A_t"].num_values == 2
        assert bn.nodes["S_t+1"].num_values == 2
        assert bn.nodes["O_t+1"].num_values == 2

    def test_topological_order_parents_before_children(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        order = bn.topological_order()
        assert len(order) == 5
        # Every node's parents must appear before it in the ordering
        index_map = {name: i for i, name in enumerate(order)}
        for name, node in bn.nodes.items():
            for parent in node.parents:
                assert index_map[parent] < index_map[name], (
                    f"Parent {parent} should appear before {name} in topological order"
                )

    def test_topological_order_contains_all_nodes(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        order = bn.topological_order()
        assert set(order) == set(bn.nodes.keys())

    def test_max_parent_count(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        # S_t+1, O_t+1, and R_t+1 each have 2 parents; S_t and A_t have 0
        assert bn.max_parent_count == 2

    def test_max_parent_count_empty_network(self) -> None:
        bn = BayesianNetwork()
        assert bn.max_parent_count == 0

    def test_sparsity_is_between_zero_and_one(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        s = bn.sparsity
        assert 0.0 <= s <= 1.0

    def test_sparsity_empty_network(self) -> None:
        bn = BayesianNetwork()
        assert bn.sparsity == 0.0

    def test_sparsity_with_sparse_tensors(self) -> None:
        """Network built from tensors with many zeros should have higher sparsity."""
        T = np.zeros((2, 2, 2))
        T[0] = np.array([[1.0, 0.0], [0.0, 1.0]])  # identity = 50% zeros
        T[1] = np.array([[1.0, 0.0], [0.0, 1.0]])

        O = np.zeros((2, 2, 2))
        O[0] = np.array([[1.0, 0.0], [0.0, 1.0]])
        O[1] = np.array([[1.0, 0.0], [0.0, 1.0]])

        R = np.array([[0.0, 0.0], [0.0, 1.0]])

        bn = BayesianNetwork.from_pomdp(
            num_states=2, num_actions=2, num_observations=2,
            transition_tensor=T, observation_tensor=O, reward_matrix=R,
        )
        assert bn.sparsity > 0.0

    def test_circuit_depth_estimate_positive(self) -> None:
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        depth = bn.circuit_depth_estimate()
        assert isinstance(depth, int)
        assert depth > 0

    def test_circuit_depth_estimate_scales_with_parents(self) -> None:
        """Circuit depth should be N * 2^M, so 5 * 2^2 = 20 for this POMDP."""
        model = _make_simple_pomdp()
        bn = BayesianNetwork.from_pomdp(
            num_states=model.num_states,
            num_actions=model.num_actions,
            num_observations=model.num_observations,
            transition_tensor=model.transition_tensor,
            observation_tensor=model.observation_tensor,
            reward_matrix=model.reward_matrix,
        )
        n = len(bn.nodes)  # 5
        m = bn.max_parent_count  # 2
        expected = n * (2 ** m)
        assert bn.circuit_depth_estimate() == expected
