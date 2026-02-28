"""H-step lookahead tree for QBRL planning (BeliefNode/ActionNode).

Academic References:
    arXiv:2507.18606, Sec IV - Belief tree search for QBRL. The H-step
    lookahead tree expands from the current belief, branching on actions
    and observations. At each belief node, the tree evaluates expected
    rewards and discounted future values. Quantum belief updates at
    the first level provide quadratic speedup via amplitude amplification.

    Silver & Veness, "Monte-Carlo Planning in Large POMDPs",
    NIPS 2010.
    -- POMCP baseline: the Monte Carlo tree search approach for POMDPs.
       QBRL's lookahead tree structure is analogous to POMCP's search
       tree, but replaces particle-based belief approximation with
       exact Bayesian updates (classical) or quantum circuit-based
       updates (quantum mode).

    Somani, Ye, Hsu, Lee, "DESPOT: Online POMDP Planning with
    Regularization", NIPS 2013.
    -- Alternative tree search using deterministic scenarios. See
       classical_baselines/despot.py for comparison implementation.

    Pineau, Gordon, Thrun, "Point-Based Value Iteration: An Anytime
    Algorithm for POMDPs", IJCAI 2003.
    -- Offline PBVI provides optimal alpha-vector policies for
       comparison with online tree search approaches.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray
from quantum_pomdp.models.belief_state import BeliefState


@dataclass
class BeliefNode:
    """Node in the belief tree representing a belief state."""
    belief: BeliefState
    depth: int = 0
    value: float = 0.0
    children: list[ActionNode] = field(default_factory=list)
    visit_count: int = 0

    def best_action(self) -> int:
        """Return the action index with highest value."""
        if not self.children:
            return 0
        return max(range(len(self.children)), key=lambda i: self.children[i].value)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class ActionNode:
    """Node representing an action in the lookahead tree."""
    action: int
    depth: int = 0
    value: float = 0.0
    immediate_reward: float = 0.0
    children: dict[int, BeliefNode] = field(default_factory=dict)
    visit_count: int = 0

    def add_observation_child(self, observation: int, belief_node: BeliefNode) -> None:
        self.children[observation] = belief_node


def build_lookahead_tree(
    root_belief: BeliefState,
    pomdp_model: Any,
    horizon: int = 2,
    num_samples: int = 50,
    discount: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BeliefNode:
    """Build an H-step lookahead tree from a root belief.

    Uses Monte Carlo sampling to estimate observation probabilities
    and builds a tree of belief-action-observation nodes (arXiv:2507.18606,
    Sec IV). Tree structure is analogous to POMCP (Silver & Veness,
    NIPS 2010) but with exact Bayesian belief updates instead of
    particle filters.

    Args:
        root_belief: Starting belief state.
        pomdp_model: The POMDP model with T, O, R.
        horizon: Planning horizon H (depth of tree).
        num_samples: Monte Carlo samples per node for obs. estimation.
        discount: Discount factor gamma.
        rng: Random number generator.

    Returns:
        Root BeliefNode with populated children.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    root = BeliefNode(belief=root_belief, depth=0)
    _expand_node(root, pomdp_model, horizon, num_samples, discount, rng)
    return root


def _expand_node(
    node: BeliefNode,
    model: Any,
    horizon: int,
    num_samples: int,
    discount: float,
    rng: np.random.Generator,
) -> None:
    """Recursively expand a belief node."""
    if node.depth >= horizon:
        return

    best_value = -np.inf

    for action in range(model.num_actions):
        action_node = ActionNode(action=action, depth=node.depth)
        reward = model.expected_reward(node.belief.probabilities, action)
        action_node.immediate_reward = reward

        # Sample observations
        observation_beliefs: dict[int, list[NDArray]] = {}
        observation_counts: dict[int, int] = {}

        for _ in range(num_samples):
            # Sample state from belief
            state = rng.choice(model.num_states, p=node.belief.probabilities)
            # Sample next state
            next_state = rng.choice(model.num_states, p=model.transition_tensor[action, state])
            # Sample observation
            obs = rng.choice(model.num_observations, p=model.observation_tensor[action, next_state])

            observation_counts[obs] = observation_counts.get(obs, 0) + 1

        # For each observed observation, compute posterior belief
        future_value = 0.0
        for obs, count in observation_counts.items():
            obs_prob = count / num_samples

            # Bayesian belief update
            new_belief = node.belief.classical_update(
                action=action,
                observation=obs,
                transition_tensor=model.transition_tensor,
                observation_tensor=model.observation_tensor,
            )

            child_node = BeliefNode(belief=new_belief, depth=node.depth + 1)
            action_node.add_observation_child(obs, child_node)

            # Recurse
            _expand_node(child_node, model, horizon, num_samples, discount, rng)
            future_value += obs_prob * child_node.value

        action_node.value = reward + discount * future_value
        node.children.append(action_node)

        if action_node.value > best_value:
            best_value = action_node.value

    node.value = best_value if best_value > -np.inf else 0.0
