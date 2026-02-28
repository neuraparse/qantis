"""Tests for POMDP quantum circuits.

Requires qiskit — skipped automatically when not installed.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

from quantum_pomdp.models.belief_state import BeliefState  # noqa: E402
from quantum_pomdp.models.pomdp import POMDPModel  # noqa: E402
from quantum_pomdp.quantum_circuits.belief_update import (  # noqa: E402
    BeliefUpdateCircuitConfig,
    QuantumBeliefUpdateCircuit,
)
from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap  # noqa: E402


class TestPOMDPRegisterMap:
    def test_from_dimensions(self) -> None:
        reg = POMDPRegisterMap.from_dimensions(
            state_qubits=2, action_qubits=1, observation_qubits=1,
        )
        assert reg.state_current.size == 2
        assert reg.action.size == 1
        assert reg.state_next.size == 2
        assert reg.observation.size == 1
        assert reg.reward.size == 4
        assert reg.total_qubits > 0

    def test_tiger_registers(self, tiger_pomdp: POMDPModel) -> None:
        reg = POMDPRegisterMap.from_dimensions(
            state_qubits=tiger_pomdp.state_qubits,
            action_qubits=tiger_pomdp.action_qubits,
            observation_qubits=tiger_pomdp.observation_qubits,
        )
        assert reg.state_current.size == 1  # 2 states -> 1 qubit
        assert reg.action.size == 2  # 3 actions -> 2 qubits
        assert reg.observation.size == 1  # 2 observations -> 1 qubit


class TestQuantumBeliefUpdateCircuit:
    def test_build_without_aa(self, tiger_pomdp: POMDPModel) -> None:
        config = BeliefUpdateCircuitConfig(
            use_amplitude_amplification=False,
            include_reward_register=False,
        )
        builder = QuantumBeliefUpdateCircuit(tiger_pomdp, config)
        belief = BeliefState.uniform(2)

        circuit = builder.build_without_aa(belief, action=0)
        assert circuit is not None
        assert circuit.num_qubits > 0

    def test_build_with_observation(self, tiger_pomdp: POMDPModel) -> None:
        config = BeliefUpdateCircuitConfig(
            use_amplitude_amplification=True,
            aa_iterations=1,
            include_reward_register=False,
        )
        builder = QuantumBeliefUpdateCircuit(tiger_pomdp, config)
        belief = BeliefState.uniform(2)

        circuit = builder.build(belief, action=0, observation=0)
        assert circuit is not None

    def test_circuit_info(self, tiger_pomdp: POMDPModel) -> None:
        config = BeliefUpdateCircuitConfig(include_reward_register=False)
        builder = QuantumBeliefUpdateCircuit(tiger_pomdp, config)
        belief = BeliefState.uniform(2)

        info = builder.get_circuit_info(belief, action=0)
        assert "total_qubits" in info
        assert "circuit_depth" in info
        assert info["total_qubits"] > 0
