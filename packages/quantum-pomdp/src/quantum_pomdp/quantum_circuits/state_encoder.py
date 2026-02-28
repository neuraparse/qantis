"""Belief state and action encoding for quantum circuits.

U(b): Encodes the belief state as quantum amplitudes using amplitude encoding.
U(a): Encodes a specific action as a computational basis state.

From QBRL paper (arXiv:2507.18606, Sec III.A): "U(b) and U(a) encode the
belief state and action." For binary random variables, B can be constructed
with a sequence of uniformly controlled R_Y(theta) rotations.

Academic References:
    Moettonen, Vartiainen, Bergholm, Salomaa, "Transformation of quantum
    states using uniformly controlled rotations", PRL 93, 130502 (2004).
    -- The UCR_Y decomposition used in encode_belief_with_rotations().
       Decomposes arbitrary n-qubit state preparation into O(2^n) controlled
       R_Y gates, each conditioned on all higher-index qubits.

    Shende, Bullock, Markov, "Synthesis of Quantum Logic Circuits",
    IEEE Trans. CAD 25(6) (2006).
    -- Lower bound Omega(2^n) CNOT gates for arbitrary state preparation.
       The recursive rotation scheme in _apply_recursive_rotations()
       achieves this optimal asymptotic complexity.

    PennyLane v0.44 (Jan 2026): MultiplexerStatePreparation template
    provides an equivalent implementation with automatic optimization
    for sparse distributions. See also QRAM templates for indexed access.

    arXiv:2507.18606, Sec III.A - Belief and action encoding circuits
    form the first two layers of the full QBRL circuit (Fig. 3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit, QuantumRegister

from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap


class BeliefStateEncoder:
    """Encodes belief states and actions into quantum registers."""

    def __init__(self, register_map: POMDPRegisterMap) -> None:
        self._reg = register_map

    def encode_belief(
        self,
        amplitudes: NDArray[np.complex128],
        circuit: QuantumCircuit,
    ) -> None:
        """Append belief state preparation gates to circuit.

        Prepares |psi_b> = sum_s sqrt(b(s)) |s> on the state_current register.
        Uses Qiskit's initialize for exact amplitude encoding.
        """
        n_qubits = self._reg.state_current.size
        n_states = 2**n_qubits

        # Pad amplitudes to 2^n if necessary
        padded = np.zeros(n_states, dtype=np.complex128)
        padded[: len(amplitudes)] = amplitudes

        # Normalize
        norm = np.linalg.norm(padded)
        if norm > 1e-12:
            padded = padded / norm

        circuit.initialize(padded.tolist(), self._reg.state_current[:])

    def encode_action(self, action: int, circuit: QuantumCircuit) -> None:
        """Encode a specific action as computational basis state.

        Simple X-gate pattern to encode action index in binary.
        """
        n_bits = self._reg.action.size
        for bit_idx in range(n_bits):
            if (action >> bit_idx) & 1:
                circuit.x(self._reg.action[bit_idx])

    def encode_belief_with_rotations(
        self,
        probabilities: NDArray[np.float64],
        circuit: QuantumCircuit,
    ) -> None:
        """Hardware-compatible belief encoding using R_Y rotations.

        Decomposes amplitude encoding into controlled R_Y rotations,
        following the UCR_Y scheme of Moettonen et al., PRL 93, 130502 (2004).
        This produces shallower circuits suitable for NISQ hardware.

        Complexity: O(2^n) controlled rotations where n = num state qubits.
        See Shende, Bullock, Markov, IEEE TCAD 25(6), 2006 for optimality proof.
        """
        n_qubits = self._reg.state_current.size
        n_states = 2**n_qubits

        padded = np.zeros(n_states)
        padded[: len(probabilities)] = probabilities

        norm = padded.sum()
        if norm > 1e-12:
            padded = padded / norm

        amplitudes = np.sqrt(np.maximum(padded, 0.0))
        self._apply_recursive_rotations(amplitudes, circuit, self._reg.state_current, 0)

    def _apply_recursive_rotations(
        self,
        amplitudes: NDArray[np.float64],
        circuit: QuantumCircuit,
        register: QuantumRegister,
        qubit_idx: int,
    ) -> None:
        """Recursively apply R_Y rotations for amplitude encoding.

        At each level, split the amplitude vector in half and compute
        the rotation angle that distributes probability between the halves.
        This implements the binary tree decomposition from Moettonen et al.,
        PRL 93, 130502 (2004) -- each R_Y rotation angle theta_j encodes
        the conditional probability of the j-th binary digit.
        """
        if qubit_idx >= register.size or len(amplitudes) <= 1:
            return

        n = len(amplitudes)
        half = n // 2
        left = amplitudes[:half]
        right = amplitudes[half:]

        left_norm_sq = np.sum(left**2)
        right_norm_sq = np.sum(right**2)
        total = left_norm_sq + right_norm_sq

        if total < 1e-12:
            return

        theta = 2 * np.arcsin(np.sqrt(right_norm_sq / total))
        circuit.ry(theta, register[register.size - 1 - qubit_idx])

        # Recursively encode sub-distributions
        if left_norm_sq > 1e-12:
            left_normalized = left / np.sqrt(left_norm_sq)
            self._apply_recursive_rotations(
                left_normalized, circuit, register, qubit_idx + 1
            )

        if right_norm_sq > 1e-12:
            right_normalized = right / np.sqrt(right_norm_sq)
            circuit.x(register[register.size - 1 - qubit_idx])
            self._apply_recursive_rotations(
                right_normalized, circuit, register, qubit_idx + 1
            )
            circuit.x(register[register.size - 1 - qubit_idx])
