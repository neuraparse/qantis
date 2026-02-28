"""Grid world navigation POMDP.

A robot navigates a grid world with noisy position observations.
States: grid positions (grid_size^2)
Actions: N, S, E, W, stay (5 actions)
Observations: noisy position readings

Academic References:
    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    -- Grid navigation POMDPs are standard benchmarks in the POMDP
       literature, testing scalability of planners with state spaces
       growing as O(grid_size^2).

    arXiv:2507.18606 - Grid navigation used as a medium-scale POMDP
    benchmark for QBRL. For grid_size=4: |S|=16 requires k1=k3=4
    state qubits, total circuit width ~20+ qubits, feasible on
    IBM Heron R3 (156 qubits, Qiskit v2.3, Jan 2026).

    Silver & Veness, "Monte-Carlo Planning in Large POMDPs",
    NIPS 2010.
    -- Grid navigation benchmarks used for POMCP scaling analysis.

    Pineau, Gordon, Thrun, "Point-Based Value Iteration: An Anytime
    Algorithm for POMDPs", IJCAI 2003.
    -- PBVI evaluated on grid world navigation problems.
"""
from __future__ import annotations

import numpy as np
from quantum_pomdp.models.pomdp import POMDPModel


def create_grid_navigation_pomdp(
    grid_size: int = 4,
    sensor_accuracy: float = 0.8,
    slip_probability: float = 0.1,
    step_cost: float = -0.1,
    goal_reward: float = 10.0,
    discount_factor: float = 0.95,
) -> POMDPModel:
    """Create a grid navigation POMDP.

    Args:
        grid_size: Size of the grid (grid_size x grid_size).
        sensor_accuracy: Probability of correct position observation.
        slip_probability: Probability of staying in place instead of moving.
        step_cost: Cost per movement step.
        goal_reward: Reward for reaching the goal.
        discount_factor: Discount factor gamma.

    Returns:
        POMDPModel for grid navigation.
    """
    n_states = grid_size ** 2
    n_actions = 5  # N(0), S(1), E(2), W(3), Stay(4)
    n_observations = n_states  # Observe position (noisy)

    # Transition tensor T[a, s, s']
    T = np.zeros((n_actions, n_states, n_states))

    moves = {
        0: (-1, 0),  # North
        1: (1, 0),   # South
        2: (0, 1),   # East
        3: (0, -1),  # West
        4: (0, 0),   # Stay
    }

    for s in range(n_states):
        row, col = divmod(s, grid_size)
        for a, (dr, dc) in moves.items():
            new_row = max(0, min(grid_size - 1, row + dr))
            new_col = max(0, min(grid_size - 1, col + dc))
            intended = new_row * grid_size + new_col

            if a == 4:  # Stay action
                T[a, s, s] = 1.0
            else:
                T[a, s, intended] += 1.0 - slip_probability
                T[a, s, s] += slip_probability

    # Observation tensor O[a, s', o]
    O = np.zeros((n_actions, n_states, n_observations))
    for a in range(n_actions):
        for s in range(n_states):
            # Correct observation with sensor_accuracy
            O[a, s, s] = sensor_accuracy
            # Uniform noise over other positions
            noise_per_other = (1 - sensor_accuracy) / max(1, n_observations - 1)
            for o in range(n_observations):
                if o != s:
                    O[a, s, o] = noise_per_other

    # Reward matrix R[s, a]
    goal_state = n_states - 1  # Bottom-right corner
    R = np.full((n_states, n_actions), step_cost)
    R[:, 4] = step_cost / 2  # Stay is cheaper
    R[goal_state, :] = goal_reward

    return POMDPModel(
        num_states=n_states,
        num_actions=n_actions,
        num_observations=n_observations,
        transition_tensor=T,
        observation_tensor=O,
        reward_matrix=R,
        discount_factor=discount_factor,
    )
