"""GPS-denied UAV navigation scenario.

Academic References:
    Quantum-enhanced inertial navigation via cold-atom interferometry sensors
    fused with classical IMU data motivates quantum sensor integration for
    POMDP observation models in GPS-denied environments.

    arXiv:2507.18606 - GPS-denied navigation as a real-world application
    scenario for QBRL. The low observation accuracy (sensor_accuracy ~0.7)
    leads to low P(o|b,a), where quantum amplitude amplification provides
    maximal speedup: O(P(e)^{-1/2}) vs classical O(P(e)^{-1}).

    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    -- POMDP formulation of navigation under uncertainty.
"""
from __future__ import annotations

import numpy as np
from quantum_pomdp.models.pomdp import POMDPModel

def create_gps_denied_pomdp(grid_size: int = 4, sensor_accuracy: float = 0.7) -> POMDPModel:
    """Create GPS-denied navigation POMDP.

    Simulates a UAV navigating without GPS, relying on noisy
    terrain-matching and quantum-enhanced IMU fusion. Motivated by
    Q-CTRL Ironstone Opal's 111x GPS-denied navigation improvement (TIME
    Best Invention 2025) using quantum magnetometry + AI sensor fusion.
    Achieved 4m GPS-like accuracy over 700km airborne trials.

    States: grid_size^2 positions
    Actions: 5 (N, S, E, W, hover)
    Observations: grid_size^2 terrain signatures (noisy)

    The low sensor_accuracy (default 0.7) leads to low evidence
    probability P(o|b,a), maximizing quantum advantage via amplitude
    amplification (arXiv:2507.18606).
    """
    n_states = grid_size ** 2
    n_actions = 5  # N, S, E, W, hover
    n_observations = n_states  # terrain signature for each cell

    # Transition tensor: deterministic movement with slip probability
    slip_prob = 0.1
    T = np.zeros((n_actions, n_states, n_states))

    for s in range(n_states):
        row, col = divmod(s, grid_size)

        # Action mappings: 0=N, 1=S, 2=E, 3=W, 4=hover
        moves = {
            0: (-1, 0),  # North
            1: (1, 0),   # South
            2: (0, 1),   # East
            3: (0, -1),  # West
            4: (0, 0),   # Hover
        }

        for a, (dr, dc) in moves.items():
            new_row = max(0, min(grid_size - 1, row + dr))
            new_col = max(0, min(grid_size - 1, col + dc))
            intended = new_row * grid_size + new_col

            T[a, s, intended] += 1.0 - slip_prob
            T[a, s, s] += slip_prob  # Slip: stay in place

    # Observation tensor: terrain matching with noise
    O = np.zeros((n_actions, n_states, n_observations))
    for a in range(n_actions):
        for s in range(n_states):
            O[a, s, s] = sensor_accuracy  # Correct terrain match
            noise = (1 - sensor_accuracy) / (n_observations - 1)
            O[a, s, :] += noise
            O[a, s, s] = sensor_accuracy  # Reset correct

    # Reward: reaching goal state, penalty for hovering
    goal_state = n_states - 1  # Bottom-right corner
    R = np.full((n_states, n_actions), -0.1)  # Small movement cost
    R[:, 4] = -0.5  # Hover penalty
    R[goal_state, :] = 10.0  # Goal reward

    return POMDPModel(
        num_states=n_states,
        num_actions=n_actions,
        num_observations=n_observations,
        transition_tensor=T,
        observation_tensor=O,
        reward_matrix=R,
        discount_factor=0.95,
    )
