"""Qubit register allocation for POMDP belief update circuits.

Maps directly to the five register groups in QBRL (Figure 3 and Table 1
of arXiv:2507.18606):
  S_t (k1 qubits) | A_t (k2) | S_{t+1} (k3) | O_{t+1} (k4) | R_{t+1} (k5)
Plus ancilla qubits for multi-controlled gates and amplitude amplification.

Academic References:
    arXiv:2507.18606, Table 1 - Qubit register allocation scheme:
        k1 = ceil(log2(|S|))     -- current state register
        k2 = ceil(log2(|A|))     -- action register
        k3 = ceil(log2(|S|))     -- next state register
        k4 = ceil(log2(|Omega|)) -- observation register
        k5 = reward precision    -- reward register (default 4 bits)
    Total circuit width: k1 + k2 + k3 + k4 + k5 + ancilla.

    Qiskit v2.3 (Jan 2026): Uses QuantumRegister, AncillaRegister, and
    ClassicalRegister primitives for register management. Compatible with
    IBM Heron R3 (156 qubits) for moderately-sized POMDPs.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import AncillaRegister, ClassicalRegister, QuantumRegister


@dataclass(frozen=True)
class POMDPRegisterMap:
    """Qubit register allocation for POMDP belief update circuit.

    Register layout follows arXiv:2507.18606, Table 1 and Figure 3.
    """

    state_current: QuantumRegister  # S_t: k1 = ceil(log2(|S|)), per Table 1
    action: QuantumRegister  # A_t: k2 = ceil(log2(|A|)), per Table 1
    state_next: QuantumRegister  # S_{t+1}: k3 = ceil(log2(|S|)), per Table 1
    observation: QuantumRegister  # O_{t+1}: k4 = ceil(log2(|Omega|)), per Table 1
    reward: QuantumRegister  # R_{t+1}: k5 = reward precision bits, per Table 1
    ancilla: AncillaRegister  # Work qubits for MCX decomposition
    classical: ClassicalRegister  # Measurement outcomes

    @classmethod
    def from_dimensions(
        cls,
        state_qubits: int,
        action_qubits: int,
        observation_qubits: int,
        reward_bits: int = 4,
    ) -> POMDPRegisterMap:
        """Allocate registers based on POMDP dimensions."""
        n_ancilla = max(state_qubits, action_qubits) + 2

        return cls(
            state_current=QuantumRegister(state_qubits, name="s_t"),
            action=QuantumRegister(action_qubits, name="a_t"),
            state_next=QuantumRegister(state_qubits, name="s_next"),
            observation=QuantumRegister(observation_qubits, name="obs"),
            reward=QuantumRegister(reward_bits, name="reward"),
            ancilla=AncillaRegister(n_ancilla, name="anc"),
            classical=ClassicalRegister(
                state_qubits + observation_qubits, name="meas"
            ),
        )

    @property
    def total_qubits(self) -> int:
        """Total number of qubits across all quantum registers."""
        return sum(
            reg.size
            for reg in [
                self.state_current,
                self.action,
                self.state_next,
                self.observation,
                self.reward,
                self.ancilla,
            ]
        )

    @property
    def all_quantum_registers(self) -> list[QuantumRegister | AncillaRegister]:
        """All quantum registers in circuit order."""
        return [
            self.state_current,
            self.action,
            self.state_next,
            self.observation,
            self.reward,
            self.ancilla,
        ]
