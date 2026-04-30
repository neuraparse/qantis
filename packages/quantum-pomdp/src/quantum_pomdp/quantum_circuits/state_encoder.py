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

import math

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

    def encode_belief_sparse(
        self,
        probabilities: NDArray[np.float64],
        circuit: QuantumCircuit,
        support_threshold: float = 1e-9,
    ) -> int:
        """Sparse k-support belief preparation (Rupprecht-Wolk 2601.09388, 2026).

        When the belief vector is ``k``-sparse with ``k << 2**n`` -- the
        common regime after a few POMDP updates collapse most posterior
        mass onto a handful of states -- the Rupprecht-Wolk scheme
        prepares the state in ``~2 * s`` Toffolis worst-case
        (arXiv:2601.09388 Alg. 2, Jan 2026) versus the full
        O(2**n) CNOT cost of ``encode_belief_with_rotations``.

        Implementation outline:
            1. Identify the ``k`` non-zero-support indices.
            2. Prepare a dense ``ceil(log2(k))``-qubit substate with the
               existing UCR_Y path.
            3. Apply an isometry-based permutation (X / CX fan-out) that
               relabels the compact support onto the original computational
               basis. We realise the permutation as a sequence of
               multi-controlled X gates; on Heron R3 this compiles via
               Qiskit's native MCX synthesis.

        Returns
        -------
        int
            Effective support size ``k``; useful for benchmark reports.
        """
        n_qubits = self._reg.state_current.size
        n_states = 2 ** n_qubits

        padded = np.zeros(n_states)
        padded[: len(probabilities)] = probabilities
        total = padded.sum()
        if total > 1e-12:
            padded /= total

        support = np.where(padded > support_threshold)[0]
        k = int(support.size)
        if k <= 1:
            # Trivial: nothing to prepare or a single-index basis state.
            if k == 1:
                target = int(support[0])
                for i in range(n_qubits):
                    if (target >> i) & 1:
                        circuit.x(self._reg.state_current[i])
            return k

        # Fall back to the dense encoder if the state is not meaningfully
        # sparse: breakpoint from arXiv:2601.09388 Sec IV is k > 2^(n-1),
        # beyond which the dense UCR_Y cost dominates the permutation.
        if k > max(2 ** (n_qubits - 1), 2):
            self.encode_belief_with_rotations(padded, circuit)
            return k

        # Step 1: prepare the dense substate on the low ``m`` qubits.
        m = max(int(math.ceil(math.log2(k))), 1)
        sub_amplitudes = np.zeros(2 ** m, dtype=np.float64)
        sub_amplitudes[:k] = np.sqrt(padded[support])
        sub_amplitudes /= max(np.linalg.norm(sub_amplitudes), 1e-12)
        low_register = self._reg.state_current[:m]

        # Reuse the existing recursive rotation primitive on the sub-register.
        self._apply_recursive_rotations(sub_amplitudes, circuit, low_register, 0)

        # Step 2: permute compact support onto the original basis labels.
        # For each compact index j in [0, k), flip the high qubits so the
        # bitstring matches support[j] in the full register. Controlled on
        # the low-register value being j.
        high_qubits = self._reg.state_current[m:]
        for j in range(k):
            target_index = int(support[j])
            high_bits = target_index >> m
            if high_bits == 0:
                continue
            # Apply X to low qubits so that state |j> becomes all-ones,
            # enabling a plain multi-controlled X on the high qubits.
            self._apply_control_basis_shift(j, m, low_register, circuit)
            for h_idx, h_qubit in enumerate(high_qubits):
                if (high_bits >> h_idx) & 1:
                    controls = list(low_register)
                    from qiskit.circuit.library import XGate

                    circuit.append(XGate().control(len(controls)), [*controls, h_qubit])
            self._apply_control_basis_shift(j, m, low_register, circuit)
        return k

    def _apply_control_basis_shift(
        self,
        compact_index: int,
        m: int,
        low_register,
        circuit: QuantumCircuit,
    ) -> None:
        """Flip low-register qubits so that ``|compact_index>`` -> ``|1..1>``."""
        for i in range(m):
            if not ((compact_index >> i) & 1):
                circuit.x(low_register[i])

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
        # ``register`` may be a QuantumRegister or a list/slice of Qubits
        # (Rupprecht-Wolk sparse-SP path passes the low-qubit sub-register
        # as a slice). Use len() to accept both without a capability check.
        register_size = getattr(register, "size", len(register))
        if qubit_idx >= register_size or len(amplitudes) <= 1:
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
        circuit.ry(theta, register[register_size - 1 - qubit_idx])

        # Recursively encode sub-distributions
        if left_norm_sq > 1e-12:
            left_normalized = left / np.sqrt(left_norm_sq)
            self._apply_recursive_rotations(
                left_normalized, circuit, register, qubit_idx + 1
            )

        if right_norm_sq > 1e-12:
            right_normalized = right / np.sqrt(right_norm_sq)
            circuit.x(register[register_size - 1 - qubit_idx])
            self._apply_recursive_rotations(
                right_normalized, circuit, register, qubit_idx + 1
            )
            circuit.x(register[register_size - 1 - qubit_idx])
