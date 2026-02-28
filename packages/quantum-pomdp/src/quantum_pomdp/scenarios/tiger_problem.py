"""Classic Tiger POMDP for validation.

The Tiger problem (Kaelbling et al., 1998):
- 2 states: tiger-left, tiger-right
- 3 actions: listen, open-left, open-right
- 2 observations: hear-left, hear-right
- Listen: small cost, noisy observation
- Open correct door: large reward
- Open wrong door: large penalty

Academic References:
    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    -- The Tiger problem is the canonical POMDP benchmark, originally
       introduced by Cassandra, Kaelbling, Littman (1994) and formalized
       in this 1998 paper. It demonstrates the value of information
       gathering (listening) vs acting under uncertainty.

    arXiv:2507.18606 - Tiger POMDP used as the primary small-scale
    validation scenario for the QBRL algorithm. With |S|=2, |A|=3,
    |Omega|=2, the quantum circuit requires only k1=1, k2=2, k3=1,
    k4=1, k5=4 qubits (9 total + ancilla), making it feasible for
    both simulation and NISQ hardware execution.

    Silver & Veness, "Monte-Carlo Planning in Large POMDPs",
    NIPS 2010.
    -- Uses Tiger as a benchmark scenario for POMCP evaluation.
"""
from __future__ import annotations

import numpy as np
from quantum_pomdp.models.pomdp import POMDPModel


def create_tiger_pomdp(
    listen_accuracy: float = 0.85,
    listen_cost: float = -1.0,
    tiger_penalty: float = -100.0,
    treasure_reward: float = 10.0,
    discount_factor: float = 0.95,
) -> POMDPModel:
    """Create the classic Tiger POMDP.

    Args:
        listen_accuracy: Probability of hearing correctly (0.5 to 1.0).
        listen_cost: Cost of the listen action (negative).
        tiger_penalty: Penalty for opening the wrong door (negative).
        treasure_reward: Reward for opening the correct door (positive).
        discount_factor: Discount factor gamma.

    Returns:
        POMDPModel configured as the Tiger problem.
    """
    num_states = 2       # tiger-left (0), tiger-right (1)
    num_actions = 3      # listen (0), open-left (1), open-right (2)
    num_observations = 2 # hear-left (0), hear-right (1)

    # Transition tensor T[a, s, s']
    T = np.zeros((num_actions, num_states, num_states))
    # Listen: state doesn't change
    T[0] = np.eye(num_states)
    # Open-left/right: resets to uniform (new game)
    T[1] = np.array([[0.5, 0.5], [0.5, 0.5]])
    T[2] = np.array([[0.5, 0.5], [0.5, 0.5]])

    # Observation tensor O[a, s', o]
    O = np.zeros((num_actions, num_states, num_observations))
    # Listen: hear tiger location with listen_accuracy
    O[0, 0, 0] = listen_accuracy       # tiger-left -> hear-left
    O[0, 0, 1] = 1 - listen_accuracy   # tiger-left -> hear-right
    O[0, 1, 0] = 1 - listen_accuracy   # tiger-right -> hear-left
    O[0, 1, 1] = listen_accuracy       # tiger-right -> hear-right
    # Open actions: uniform observation (doesn't matter)
    O[1] = np.array([[0.5, 0.5], [0.5, 0.5]])
    O[2] = np.array([[0.5, 0.5], [0.5, 0.5]])

    # Reward matrix R[s, a]
    R = np.zeros((num_states, num_actions))
    # Listen: small cost regardless of state
    R[:, 0] = listen_cost
    # Open-left: tiger-left is bad, tiger-right is good
    R[0, 1] = tiger_penalty    # tiger-left, open-left -> penalty
    R[1, 1] = treasure_reward  # tiger-right, open-left -> reward
    # Open-right: opposite
    R[0, 2] = treasure_reward  # tiger-left, open-right -> reward
    R[1, 2] = tiger_penalty    # tiger-right, open-right -> penalty

    return POMDPModel(
        num_states=num_states,
        num_actions=num_actions,
        num_observations=num_observations,
        transition_tensor=T,
        observation_tensor=O,
        reward_matrix=R,
        discount_factor=discount_factor,
    )
