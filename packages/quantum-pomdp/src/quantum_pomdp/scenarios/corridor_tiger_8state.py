"""Corridor-Tiger-8: 8-cell linear-corridor POMDP for scaling evidence.

Extends the classic Kaelbling Tiger problem to |S|=8 with a graded
listen-accuracy profile so that the QCE 2026 submission can report
genuine Ronnow-Shaydulin TTS(99%) scaling slopes over |S| in {2, 4, 8}
(Shaydulin et al., Sci. Adv. 10:eadm6761, 2024).

Design choices
--------------
* |S| = 8 (3 state-qubits); |A| = 3 (listen, open-left, open-right);
  |Omega| = 2 (hear-left, hear-right). Same structural scaffold as
  `tiger_problem.py` so the Hellinger oracle used in the 2- and 4-state
  tests carries over directly.
* Graded accuracy profile: listening yields a monotonically varying
  P(hear-left|s) across cells from 0.95 (leftmost) to 0.05 (rightmost).
  This is the "multi-cell corridor" analogue of Kaelbling's Tiger --
  cells further right are more likely to emit "hear-right".
* Transition model: "listen" is identity; "open-{left,right}" reset
  to uniform over {0,...,|S|-1} -- matches the 2-state Tiger "game
  restart" convention (Kaelbling et al. 1998).
* Reward: opening door on a cell whose tiger-side matches the door
  yields ``-tiger_penalty``; opening the opposite door yields
  ``+treasure_reward``. Listening costs ``listen_cost``.

Resource estimate on IBM Heron R3 (per analyst notes 2026-04):
    * state_qubits = 3, action_qubits = 2, obs_qubits = 1, reward_bits = 2
      -> 9 logical qubits + ancilla.
    * Logical depth with UCR_Y encoding: ~320 CNOTs before AA; ~600 after
      single Grover iteration.
    * Transpiled to Heron R3 basis: ~450-550 ECR gates with one SWAP
      layer for linear subgraph routing.

References
----------
    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    Shaydulin et al., "Quantum-assisted combinatorial optimization
    with QAOA at utility scale", Sci. Adv. 10:eadm6761 (2024).
"""
from __future__ import annotations

import numpy as np

from quantum_pomdp.models.pomdp import POMDPModel


def create_corridor_tiger_pomdp(
    num_cells: int = 8,
    listen_cost: float = -1.0,
    tiger_penalty: float = -100.0,
    treasure_reward: float = 10.0,
    discount_factor: float = 0.95,
    accuracy_min: float = 0.05,
    accuracy_max: float = 0.95,
) -> POMDPModel:
    """Create an |S|=num_cells corridor Tiger POMDP.

    The corridor has a graded listen-accuracy profile that interpolates
    linearly from ``accuracy_max`` at cell 0 to ``accuracy_min`` at cell
    ``num_cells - 1``. Cells in the left half of the corridor are more
    likely to emit "hear-left"; cells in the right half emit "hear-right".
    The 2-state Tiger is the special case ``num_cells=2``.

    Parameters
    ----------
    num_cells : int
        Number of discrete cells in the corridor. Must be a power of 2
        for amplitude encoding to saturate the state register.
    listen_cost, tiger_penalty, treasure_reward, discount_factor
        Same semantics as :func:`create_tiger_pomdp`.
    accuracy_min, accuracy_max
        End-points of the P(hear-left | cell) gradient across cells.

    Returns
    -------
    POMDPModel
    """
    if num_cells < 2:
        raise ValueError("Corridor must have at least 2 cells")

    num_states = int(num_cells)
    num_actions = 3       # listen, open-left, open-right
    num_observations = 2  # hear-left, hear-right

    # Linear accuracy gradient: cell 0 = accuracy_max, cell N-1 = accuracy_min
    p_hear_left = np.linspace(accuracy_max, accuracy_min, num_states)

    # Transition tensor T[a, s, s']
    T = np.zeros((num_actions, num_states, num_states))
    T[0] = np.eye(num_states)  # listen is identity
    uniform = np.full((num_states, num_states), 1.0 / num_states)
    T[1] = uniform  # open-left: reset uniform
    T[2] = uniform  # open-right: reset uniform

    # Observation tensor O[a, s', o]
    O = np.zeros((num_actions, num_states, num_observations))
    O[0, :, 0] = p_hear_left        # P(hear-left | s, listen)
    O[0, :, 1] = 1.0 - p_hear_left  # P(hear-right | s, listen)
    open_obs = np.full((num_states, num_observations), 0.5)
    O[1] = open_obs
    O[2] = open_obs

    # Reward matrix R[s, a]
    R = np.zeros((num_states, num_actions))
    R[:, 0] = listen_cost
    # Split the corridor in half: left half -> tiger-left
    split = num_states // 2
    for s in range(num_states):
        if s < split:  # left-side cells hold the tiger
            R[s, 1] = tiger_penalty     # open-left: bad
            R[s, 2] = treasure_reward   # open-right: good
        else:          # right-side cells are safe on the left door
            R[s, 1] = treasure_reward
            R[s, 2] = tiger_penalty

    return POMDPModel(
        num_states=num_states,
        num_actions=num_actions,
        num_observations=num_observations,
        transition_tensor=T,
        observation_tensor=O,
        reward_matrix=R,
        discount_factor=discount_factor,
    )


def create_corridor_tiger_4state(
    listen_cost: float = -1.0,
    tiger_penalty: float = -100.0,
    treasure_reward: float = 10.0,
    discount_factor: float = 0.95,
) -> POMDPModel:
    """|S|=4 checkpoint case used in the existing 4-state advisor tests."""
    return create_corridor_tiger_pomdp(
        num_cells=4,
        listen_cost=listen_cost,
        tiger_penalty=tiger_penalty,
        treasure_reward=treasure_reward,
        discount_factor=discount_factor,
    )


def create_corridor_tiger_8state(
    listen_cost: float = -1.0,
    tiger_penalty: float = -100.0,
    treasure_reward: float = 10.0,
    discount_factor: float = 0.95,
) -> POMDPModel:
    """|S|=8 corridor -- the scaling-evidence case for QCE 2026."""
    return create_corridor_tiger_pomdp(
        num_cells=8,
        listen_cost=listen_cost,
        tiger_penalty=tiger_penalty,
        treasure_reward=treasure_reward,
        discount_factor=discount_factor,
    )
