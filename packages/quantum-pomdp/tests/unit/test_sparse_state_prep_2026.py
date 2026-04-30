"""Smoke tests for Rupprecht-Wolk sparse state preparation (2601.09388)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit", reason="qiskit required for encoder tests")
from qiskit.circuit import QuantumCircuit

from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap
from quantum_pomdp.quantum_circuits.state_encoder import BeliefStateEncoder


def _build_encoder(state_qubits: int) -> tuple[BeliefStateEncoder, QuantumCircuit]:
    reg = POMDPRegisterMap.from_dimensions(
        state_qubits=state_qubits,
        action_qubits=1,
        observation_qubits=1,
        reward_bits=0,
    )
    circuit = QuantumCircuit(
        reg.state_current, reg.action, reg.state_next,
        reg.observation, reg.reward, reg.ancilla, reg.classical,
    )
    return BeliefStateEncoder(reg), circuit


class TestSparseEncoder:
    def test_single_support_basis_state(self) -> None:
        encoder, circuit = _build_encoder(state_qubits=3)
        probs = np.zeros(8)
        probs[5] = 1.0
        k = encoder.encode_belief_sparse(probs, circuit)
        assert k == 1

    def test_k_sparse_returns_support_size(self) -> None:
        encoder, circuit = _build_encoder(state_qubits=3)
        probs = np.zeros(8)
        probs[1] = 0.5
        probs[4] = 0.5
        k = encoder.encode_belief_sparse(probs, circuit)
        assert k == 2

    def test_dense_belief_falls_back_to_ucr(self) -> None:
        encoder, circuit = _build_encoder(state_qubits=3)
        probs = np.ones(8) / 8.0
        k = encoder.encode_belief_sparse(probs, circuit)
        # Dense branch: we fall back to the UCR_Y path; k still equals
        # the full support size for reporting.
        assert k == 8

    def test_single_support_adds_gates(self) -> None:
        encoder, circuit = _build_encoder(state_qubits=2)
        probs = np.zeros(4)
        probs[3] = 1.0
        encoder.encode_belief_sparse(probs, circuit)
        # Single-support basis state: X gates on every '1' bit of the
        # target index. Circuit must contain at least one operation and
        # no measurements yet.
        ops = [instr.operation.name for instr in circuit.data]
        assert ops
        assert "x" in ops
        assert "measure" not in ops
