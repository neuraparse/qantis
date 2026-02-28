"""Bayesian network representation of POMDP dynamics for quantum rejection sampling.

From arXiv:2507.18606: The POMDP belief update can be expressed as inference
on a Bayesian network with nodes {S_t, A_t, S_{t+1}, O_{t+1}, R_{t+1}},
enabling quantum rejection sampling with sub-quadratic speedup.

The maximum parent count M is critical for scaling:
- Circuit depth scales as O(N * 2^M)
- Speedup degrades if M is large

Academic References:
    arXiv:2507.18606, Sec II-III - BN structure for POMDP dynamics and
    compilation to quantum circuit unitaries (U_1, U_2, U_3).

    Ozols, Roetteler, Roland, "Quantum Rejection Sampling",
    ACM Trans. Computation Theory 5(3):11 (2013).
    -- The BN-to-circuit compilation enables quantum rejection sampling,
       where each CPT row becomes a controlled rotation sub-circuit.

    Moettonen, Vartiainen, Bergholm, Salomaa, PRL 93, 130502 (2004).
    -- UCR_Y decomposition used for encoding CPT rows as amplitude rotations.
       Each node with M parents requires O(2^M) controlled rotations.

    Shende, Bullock, Markov, "Synthesis of Quantum Logic Circuits",
    IEEE Trans. CAD 25(6) (2006).
    -- Lower bounds on circuit depth for arbitrary state preparation:
       the BN sparsity directly affects achievable circuit compression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class BayesianNetworkNode:
    """A node in the Bayesian network with conditional probability table."""

    name: str
    num_values: int
    parents: list[str] = field(default_factory=list)
    cpt: NDArray[np.float64] | None = None


@dataclass
class BayesianNetwork:
    """Bayesian network representation of POMDP dynamics.

    Network structure (arXiv:2507.18606, Sec II, Fig. 2):
        S_t --> S_{t+1} --> O_{t+1}
        A_t --> S_{t+1}
        A_t --> O_{t+1}
        S_t, A_t --> R_{t+1}

    This DAG structure determines the quantum circuit topology: each node's
    CPT is compiled to controlled rotations conditioned on parent registers.
    The topological order defines the circuit layer ordering.
    """

    nodes: dict[str, BayesianNetworkNode] = field(default_factory=dict)

    @classmethod
    def from_pomdp(
        cls,
        num_states: int,
        num_actions: int,
        num_observations: int,
        transition_tensor: NDArray[np.float64],
        observation_tensor: NDArray[np.float64],
        reward_matrix: NDArray[np.float64],
    ) -> BayesianNetwork:
        """Construct BN from POMDP components."""
        bn = cls()

        # S_t: current state (root node, initialized from belief)
        bn.nodes["S_t"] = BayesianNetworkNode(
            name="S_t",
            num_values=num_states,
            parents=[],
            cpt=None,  # Initialized from belief state
        )

        # A_t: action (root node, deterministic)
        bn.nodes["A_t"] = BayesianNetworkNode(
            name="A_t",
            num_values=num_actions,
            parents=[],
            cpt=None,  # Set deterministically
        )

        # S_{t+1}: next state, depends on S_t and A_t
        bn.nodes["S_t+1"] = BayesianNetworkNode(
            name="S_t+1",
            num_values=num_states,
            parents=["S_t", "A_t"],
            cpt=transition_tensor,  # T[a, s, s']
        )

        # O_{t+1}: observation, depends on S_{t+1} and A_t
        bn.nodes["O_t+1"] = BayesianNetworkNode(
            name="O_t+1",
            num_values=num_observations,
            parents=["S_t+1", "A_t"],
            cpt=observation_tensor,  # O[a, s', o]
        )

        # R_{t+1}: reward, depends on S_t and A_t
        bn.nodes["R_t+1"] = BayesianNetworkNode(
            name="R_t+1",
            num_values=1,  # Continuous, discretized for circuit
            parents=["S_t", "A_t"],
            cpt=reward_matrix,  # R[s, a]
        )

        return bn

    @property
    def max_parent_count(self) -> int:
        """M: max number of parents across all nodes.

        Critical for quantum circuit scaling: O(N * 2^M).
        """
        if not self.nodes:
            return 0
        return max(len(node.parents) for node in self.nodes.values())

    @property
    def sparsity(self) -> float:
        """Fraction of zero entries in CPTs (higher = better for quantum)."""
        total_entries = 0
        zero_entries = 0
        for node in self.nodes.values():
            if node.cpt is not None:
                total_entries += node.cpt.size
                zero_entries += np.sum(np.abs(node.cpt) < 1e-12)
        if total_entries == 0:
            return 0.0
        return float(zero_entries / total_entries)

    def topological_order(self) -> list[str]:
        """Return nodes in topological order for circuit construction."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for parent in self.nodes[name].parents:
                visit(parent)
            order.append(name)

        for name in self.nodes:
            visit(name)

        return order

    def circuit_depth_estimate(self) -> int:
        """Estimate quantum circuit depth based on BN structure.

        From arXiv:2507.18606 Sec III: depth scales as O(N * 2^M) where
        N is the number of nodes and M is the max parent count. This
        scaling follows from the UCR_Y decomposition (Moettonen et al.,
        PRL 93, 130502, 2004) applied to each CPT.
        """
        n = len(self.nodes)
        m = self.max_parent_count
        return n * (2**m)
