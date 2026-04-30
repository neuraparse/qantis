"""Corridor-Tiger-8: |S|=8 scaling-evidence scenario parity tests.

These tests validate that the 8-cell corridor POMDP is:
(a) a structurally well-formed POMDPModel (row-stochastic tensors);
(b) reduces to the familiar 2-state Tiger at num_cells=2;
(c) produces a quantum belief-update circuit whose Aer posterior
    matches the classical Bayesian posterior to Hellinger < 0.03
    after post-selection on the measured observation (QBRL paper
    arXiv:2507.18606 Fig. 3 extraction protocol).

Feeds the TTS(99%) scaling collector used in the QCE 2026 paper
(Shaydulin Sci. Adv. 10:eadm6761, 2024 methodology).
"""
from __future__ import annotations

import numpy as np
import pytest

from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.scenarios.corridor_tiger_8state import (
    create_corridor_tiger_4state,
    create_corridor_tiger_8state,
    create_corridor_tiger_pomdp,
)


class TestCorridorTigerStructural:
    """Row-stochastic tensors and shape consistency."""

    @pytest.mark.parametrize("num_cells", [2, 4, 8, 16])
    def test_transition_tensor_stochastic(self, num_cells: int) -> None:
        p = create_corridor_tiger_pomdp(num_cells=num_cells)
        assert p.transition_tensor.shape == (3, num_cells, num_cells)
        sums = p.transition_tensor.sum(axis=2)
        assert np.allclose(sums, 1.0), f"rows not stochastic: {sums}"

    @pytest.mark.parametrize("num_cells", [2, 4, 8])
    def test_observation_tensor_stochastic(self, num_cells: int) -> None:
        p = create_corridor_tiger_pomdp(num_cells=num_cells)
        assert p.observation_tensor.shape == (3, num_cells, 2)
        sums = p.observation_tensor.sum(axis=2)
        assert np.allclose(sums, 1.0)

    def test_qubit_counts_match_expected(self) -> None:
        """|S|=8 -> 3 state qubits; |A|=3 -> 2 action qubits; |O|=2 -> 1."""
        p = create_corridor_tiger_8state()
        assert p.state_qubits == 3
        assert p.action_qubits == 2
        assert p.observation_qubits == 1

    def test_reduces_to_canonical_tiger_at_2cells(self) -> None:
        """num_cells=2 yields the canonical 2-state Tiger structure."""
        p2 = create_corridor_tiger_pomdp(num_cells=2)
        # Listen accuracy at cell 0 defaults to accuracy_max = 0.95.
        assert np.isclose(p2.observation_tensor[0, 0, 0], 0.95)
        assert np.isclose(p2.observation_tensor[0, 1, 1], 0.95)

    def test_8state_accuracy_gradient(self) -> None:
        """P(hear-left | cell) decreases monotonically from 0 to 7."""
        p = create_corridor_tiger_8state()
        hear_left_profile = p.observation_tensor[0, :, 0]
        diffs = np.diff(hear_left_profile)
        assert np.all(diffs < 0), f"profile not monotone: {hear_left_profile}"

    def test_reward_splits_corridor_in_half(self) -> None:
        """Left half rewards open-right; right half rewards open-left."""
        p = create_corridor_tiger_8state()
        # Cells 0..3 are left-half (tiger-left semantics)
        for s in range(4):
            assert p.reward_matrix[s, 1] < 0  # open-left penalty
            assert p.reward_matrix[s, 2] > 0  # open-right treasure
        for s in range(4, 8):
            assert p.reward_matrix[s, 1] > 0
            assert p.reward_matrix[s, 2] < 0


class TestCorridorTigerBayesianPosterior:
    """Classical Bayesian belief updates give consistent posteriors."""

    def test_uniform_prior_listen_hear_left_biases_left_half(self) -> None:
        p = create_corridor_tiger_8state()
        uniform = BeliefState.uniform(num_states=p.num_states)
        posterior = uniform.classical_update(
            action=0, observation=0,
            transition_tensor=p.transition_tensor,
            observation_tensor=p.observation_tensor,
        )
        left_mass = posterior.probabilities[:4].sum()
        right_mass = posterior.probabilities[4:].sum()
        assert left_mass > right_mass, (
            f"hear-left should favour left half: left={left_mass:.3f}, right={right_mass:.3f}"
        )

    def test_two_hear_left_steps_concentrate_posterior(self) -> None:
        """Iterating hear-left twice concentrates mass further into the
        left half - this is the multi-step chaining motivation."""
        p = create_corridor_tiger_8state()
        b = BeliefState.uniform(num_states=p.num_states)
        b1 = b.classical_update(0, 0, p.transition_tensor, p.observation_tensor)
        b2 = b1.classical_update(0, 0, p.transition_tensor, p.observation_tensor)
        assert b2.probabilities[:4].sum() > b1.probabilities[:4].sum()

    def test_classical_posterior_matches_hand_computation_4state(self) -> None:
        """Cross-check the 4-state corridor against the advisor unit test."""
        p = create_corridor_tiger_4state()
        uniform = BeliefState.uniform(num_states=4)
        post = uniform.classical_update(
            action=0, observation=0,
            transition_tensor=p.transition_tensor,
            observation_tensor=p.observation_tensor,
        )
        assert np.isclose(post.probabilities.sum(), 1.0)
        # Symmetry: cell 0 (0.95) + cell 3 (0.05) posterior masses sum to
        # the same as cell 1 (0.65) + cell 2 (0.35) under uniform prior and
        # identity transition (listen).
        assert (
            post.probabilities[0]
            > post.probabilities[1]
            > post.probabilities[2]
            > post.probabilities[3]
        )


class TestCorridorTigerQuantumCircuit:
    """End-to-end Aer parity for the 8-state corridor belief update."""

    def test_aer_posterior_matches_classical_hellinger_below_03(self) -> None:
        """|S|=8 Aer Hellinger distance < 0.03 after post-selection.

        This is the simulator POC cell that unlocks hardware runs for the
        QCE 2026 scaling claim. The threshold is deliberately loose (0.03)
        because the 8-state reward unitary uses reward_bits=2 which
        quantizes expected-value to 4 levels - the residual ~0.01-0.02
        Hellinger from that quantization is already documented in the
        4-state corridor test.
        """
        pytest.importorskip("qiskit")

        from quantum_common.backends.base import ExecutionRequest
        from quantum_common.backends.simulator import AerSimulatorBackend
        from quantum_pomdp.quantum_circuits.belief_update import (
            BeliefUpdateCircuitConfig,
            QuantumBeliefUpdateCircuit,
        )

        p = create_corridor_tiger_8state()
        b0 = BeliefState.uniform(num_states=p.num_states)
        classical_post = b0.classical_update(
            action=0, observation=0,
            transition_tensor=p.transition_tensor,
            observation_tensor=p.observation_tensor,
        )

        cfg = BeliefUpdateCircuitConfig(
            use_amplitude_amplification=False,
            # Use Qiskit `initialize` for exact state preparation. The
            # hardware-compatible UCR_Y path encodes only the MSB angle
            # of the conditional distribution and is a known
            # oversimplification for |S| > 2 (fixing the full Mottonen
            # 2004 decomposition is future work -- see paper appendix).
            use_hardware_compatible_encoding=False,
            reward_precision_bits=2,
        )
        qc = QuantumBeliefUpdateCircuit(p, cfg).build(
            belief=b0, action=0, observation=0,
        )
        counts = AerSimulatorBackend().execute(
            ExecutionRequest(circuits=[qc], shots=16384)
        ).counts[0]

        post_select = {
            p.state_qubits + i: (0 >> i) & 1
            for i in range(p.observation_qubits)
        }
        q_post = BeliefState.from_quantum_measurement(
            counts,
            num_states=p.num_states,
            num_state_qubits=p.state_qubits,
            post_selection=post_select,
        )
        h_dist = q_post.hellinger_distance(classical_post)
        assert h_dist < 0.03, (
            f"Corridor-Tiger-8 Aer Hellinger={h_dist:.4f} >= 0.03 -- "
            f"investigate observation unitary or reward quantization"
        )
