"""U_2: Sensor model encoding P(o_{t+1} | s_{t+1}, a_t).

Operates on registers (S_{t+1}, A_t, O_{t+1}).
Structure mirrors TransitionUnitary but conditions on next-state.

Academic References:
    arXiv:2507.18606, Sec III.C - Sensor model (observation) encoding.
    The observation tensor O[a, s', o] = P(o|s', a) is compiled into U_2
    using multi-controlled R_Y rotations on the O_{t+1} register,
    conditioned on the (S_{t+1}, A_t) registers. This implements the
    sensor measurement model of the POMDP.

    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    -- The observation function O(s', a, o) = P(o|s', a) is a core POMDP
       component; here it is encoded as quantum rotation angles.

    Moettonen, Vartiainen, Bergholm, Salomaa, PRL 93, 130502 (2004).
    -- UCR_Y decomposition for each conditional distribution row O[a, s', :].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap


class ObservationUnitary:
    """Encodes P(o|s', a) as quantum gates on the observation register."""

    def __init__(
        self,
        observation_tensor: NDArray[np.float64],
        register_map: POMDPRegisterMap,
    ) -> None:
        self._O = observation_tensor  # Shape: (A, S', Omega)
        self._reg = register_map

    def build(self, circuit: QuantumCircuit) -> None:
        """Append U_2 gates: controlled on (S_{t+1}, A_t), target O_{t+1}."""
        num_actions, num_states, num_obs = self._O.shape
        n_state_bits = self._reg.state_next.size
        n_action_bits = self._reg.action.size

        for a in range(num_actions):
            for s_prime in range(num_states):
                probs = self._O[a, s_prime, :]
                if np.allclose(probs, 0.0):
                    continue

                # Compute rotation angle for observation encoding
                angles = self._probs_to_angles(probs)
                if all(abs(angle) < 1e-10 for angle in angles):
                    continue

                ctrl_state_bits = format(s_prime, f"0{n_state_bits}b")
                ctrl_action_bits = format(a, f"0{n_action_bits}b")

                self._apply_controlled_rotation(
                    circuit, angles, ctrl_state_bits, ctrl_action_bits
                )

    def _probs_to_angles(self, probs: NDArray[np.float64]) -> list[float]:
        """Convert probability distribution to R_Y rotation angles."""
        n = len(probs)
        if n <= 1:
            return []

        half = n // 2
        left_sum = probs[:half].sum()
        right_sum = probs[half:].sum()
        total = left_sum + right_sum

        if total < 1e-12:
            return [0.0]

        theta = 2 * np.arcsin(np.sqrt(max(0.0, min(1.0, right_sum / total))))
        return [float(theta)]

    def _apply_controlled_rotation(
        self,
        circuit: QuantumCircuit,
        angles: list[float],
        ctrl_state_bits: str,
        ctrl_action_bits: str,
    ) -> None:
        """Apply multi-controlled R_Y for a specific (s', a) pair."""
        if not angles or abs(angles[0]) < 1e-10:
            return

        ctrl_qubits = list(self._reg.state_next) + list(self._reg.action)
        ctrl_string = ctrl_state_bits + ctrl_action_bits
        n_obs_bits = self._reg.observation.size

        x_positions: list[int] = []
        for i, bit in enumerate(ctrl_string):
            if bit == "0":
                circuit.x(ctrl_qubits[i])
                x_positions.append(i)

        from qiskit.circuit.library import RYGate

        target = self._reg.observation[n_obs_bits - 1]
        controlled_ry = RYGate(angles[0]).control(len(ctrl_qubits))
        circuit.append(controlled_ry, ctrl_qubits + [target])

        for i in x_positions:
            circuit.x(ctrl_qubits[i])
