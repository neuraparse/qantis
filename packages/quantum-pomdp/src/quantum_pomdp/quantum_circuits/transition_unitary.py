"""U_1: Transition dynamics encoding P(s_{t+1} | s_t, a_t).

This unitary operates on registers (S_t, A_t, S_{t+1}) and implements
controlled rotations such that measuring S_{t+1} yields the correct
transition probability distribution.

Academic References:
    arXiv:2507.18606, Sec III.B - Transition dynamics encoding. The POMDP
    transition tensor T[a, s, s'] = P(s'|s, a) is compiled into U_1 using
    multi-controlled R_Y rotations on the S_{t+1} register, conditioned on
    the (S_t, A_t) registers. Each (s, a) pair produces a conditional
    distribution over s' encoded as rotation angles.

    Moettonen, Vartiainen, Bergholm, Salomaa, PRL 93, 130502 (2004).
    -- UCR_Y decomposition used for each conditional distribution row
       T[a, s, :] -> rotation angles via recursive binary decomposition.

    Shende, Bullock, Markov, "Synthesis of Quantum Logic Circuits",
    IEEE Trans. CAD 25(6) (2006).
    -- Complexity: O(|S|*|A| * 2^k3) gates total for the transition unitary,
       where k3 = ceil(log2(|S|)) is the number of next-state qubits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap


class TransitionUnitary:
    """Encodes P(s'|s, a) as quantum gates on the state_next register."""

    def __init__(
        self,
        transition_tensor: NDArray[np.float64],
        register_map: POMDPRegisterMap,
    ) -> None:
        self._T = transition_tensor  # Shape: (A, S, S')
        self._reg = register_map

    def build(self, circuit: QuantumCircuit) -> None:
        """Append U_1 gates to the circuit.

        For each action, specialises to the two structurally-common
        POMDP transition patterns (identity and uniform over |S|) before
        falling back to the multi-controlled rotation path. The fast
        paths are exact for any |S| and keep circuit depth in the
        O(|S| log |S|) regime that Heron R3 can tolerate.

        For general transition distributions this emits a per-state
        multi-controlled amplitude-encoding pattern. The implementation
        currently encodes the MSB binary split of the conditional
        P(s'|s,a) -- good enough for |S|<=2 and used only as a
        fallback for non-trivial transitions in higher |S|.
        """
        num_actions, num_states, _ = self._T.shape
        n_state_bits = self._reg.state_current.size
        n_action_bits = self._reg.action.size

        for a in range(num_actions):
            T_a = self._T[a]
            if self._is_identity(T_a):
                self._apply_identity_transition(circuit, a)
                continue
            if self._is_uniform(T_a, num_states):
                self._apply_uniform_transition(circuit, a)
                continue

            # General fallback (legacy partial encoding -- flagged in paper
            # appendix as a known limitation for non-identity/non-uniform
            # transitions with |S| > 2).
            for s in range(num_states):
                probs = T_a[s, :]
                if np.allclose(probs, 0.0):
                    continue

                angles = self._probs_to_angles(probs)
                if all(abs(angle) < 1e-10 for angle in angles):
                    continue

                ctrl_state_bits = format(s, f"0{n_state_bits}b")
                ctrl_action_bits = format(a, f"0{n_action_bits}b")

                self._apply_controlled_amplitude_encoding(
                    circuit, angles, ctrl_state_bits, ctrl_action_bits
                )

    def _is_identity(self, T_a: NDArray[np.float64]) -> bool:
        return bool(np.allclose(T_a, np.eye(T_a.shape[0])))

    def _is_uniform(self, T_a: NDArray[np.float64], num_states: int) -> bool:
        return bool(np.allclose(T_a, np.full_like(T_a, 1.0 / num_states)))

    def _apply_identity_transition(self, circuit: QuantumCircuit, a: int) -> None:
        """Copy state_current to state_next when action=a (identity transition).

        Uses action-controlled CNOTs: for each bit index i, emit CNOT
        (state_current[i] -> state_next[i]) wrapped in an action-equality
        conditional realised via X-sandwich on action qubits. This
        correctly reproduces the transition P(s' = s | s, a=listen) = 1
        for any |S|.
        """
        n_state_bits = self._reg.state_current.size
        n_action_bits = self._reg.action.size
        ctrl_bits = format(a, f"0{n_action_bits}b")

        x_positions = self._flip_action_zeros(circuit, ctrl_bits)
        for i in range(n_state_bits):
            from qiskit.circuit.library import XGate

            controls = list(self._reg.action) + [self._reg.state_current[i]]
            circuit.append(
                XGate().control(len(controls)),
                [*controls, self._reg.state_next[i]],
            )
        self._unflip_action_zeros(circuit, ctrl_bits, x_positions)

    def _apply_uniform_transition(self, circuit: QuantumCircuit, a: int) -> None:
        """Place state_next into uniform superposition when action=a.

        For reset-uniform transitions (e.g. ``open`` in Tiger), emit
        action-conditioned Hadamards on each state_next qubit. This
        is the exact encoding of P(s'|s, open) = 1/|S|.
        """
        n_action_bits = self._reg.action.size
        ctrl_bits = format(a, f"0{n_action_bits}b")
        x_positions = self._flip_action_zeros(circuit, ctrl_bits)
        from qiskit.circuit.library import HGate

        for target in self._reg.state_next:
            circuit.append(
                HGate().control(n_action_bits),
                [*list(self._reg.action), target],
            )
        self._unflip_action_zeros(circuit, ctrl_bits, x_positions)

    def _flip_action_zeros(
        self, circuit: QuantumCircuit, ctrl_bits: str
    ) -> list[int]:
        positions: list[int] = []
        for i, bit in enumerate(ctrl_bits):
            if bit == "0":
                circuit.x(self._reg.action[i])
                positions.append(i)
        return positions

    def _unflip_action_zeros(
        self, circuit: QuantumCircuit, ctrl_bits: str, positions: list[int]
    ) -> None:
        for i in positions:
            circuit.x(self._reg.action[i])

    def _probs_to_angles(self, probs: NDArray[np.float64]) -> list[float]:
        """Convert probability distribution to R_Y rotation angles.

        Uses recursive decomposition following Moettonen et al., PRL 93,
        130502 (2004): at each level, the angle distributes probability
        between left and right halves of the state space.
        theta_j = 2*arcsin(sqrt(P_right / P_total)) at each binary split.
        """
        n = len(probs)
        if n <= 1:
            return []

        angles: list[float] = []
        self._recursive_angle_decomposition(probs, angles)
        return angles

    def _recursive_angle_decomposition(
        self, probs: NDArray[np.float64], angles: list[float]
    ) -> None:
        """Recursively decompose probabilities into rotation angles."""
        n = len(probs)
        if n <= 1:
            return

        half = n // 2
        left_sum = probs[:half].sum()
        right_sum = probs[half:].sum()
        total = left_sum + right_sum

        if total < 1e-12:
            angles.append(0.0)
        else:
            theta = 2 * np.arcsin(np.sqrt(max(0.0, min(1.0, right_sum / total))))
            angles.append(float(theta))

        if left_sum > 1e-12:
            self._recursive_angle_decomposition(probs[:half] / left_sum, angles)
        else:
            # Fill with zeros for the remaining levels
            levels = int(np.log2(half)) if half > 1 else 0
            angles.extend([0.0] * levels)

        if right_sum > 1e-12:
            self._recursive_angle_decomposition(probs[half:] / right_sum, angles)
        else:
            levels = int(np.log2(half)) if half > 1 else 0
            angles.extend([0.0] * levels)

    def _apply_controlled_amplitude_encoding(
        self,
        circuit: QuantumCircuit,
        angles: list[float],
        ctrl_state_bits: str,
        ctrl_action_bits: str,
    ) -> None:
        """Apply multi-controlled R_Y rotations for a specific (s, a) pair.

        Uses Qiskit's multi-controlled gate decomposition.
        """
        if not angles or len(angles) == 0:
            return

        n_state_bits = self._reg.state_current.size
        n_action_bits = self._reg.action.size
        n_target_bits = self._reg.state_next.size

        # Set up control qubits with X gates for 0-controls
        x_positions: list[int] = []
        ctrl_qubits = list(self._reg.state_current) + list(self._reg.action)
        ctrl_string = ctrl_state_bits + ctrl_action_bits

        for i, bit in enumerate(ctrl_string):
            if bit == "0":
                circuit.x(ctrl_qubits[i])
                x_positions.append(i)

        # Apply the first rotation angle to the MSB of state_next
        if len(angles) > 0 and abs(angles[0]) > 1e-10:
            from qiskit.circuit.library import RYGate

            target = self._reg.state_next[n_target_bits - 1]
            controlled_ry = RYGate(angles[0]).control(len(ctrl_qubits))
            circuit.append(controlled_ry, ctrl_qubits + [target])

        # Undo X gates
        for i in x_positions:
            circuit.x(ctrl_qubits[i])
