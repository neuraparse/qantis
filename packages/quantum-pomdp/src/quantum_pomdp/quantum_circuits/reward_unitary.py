"""U_3: Reward function encoding E[r_{t+1} | s_t, a_t].

Operates on registers (S_t, A_t, R_{t+1}).
Reward values are encoded as rotation angles proportional to
the expected reward, allowing extraction via amplitude estimation.

Academic References:
    arXiv:2507.18606, Sec III.D - Reward function encoding. The reward
    matrix R[s, a] = E[r|s, a] is normalized to [0, 1] and encoded as
    R_Y rotation angles theta = 2*arcsin(sqrt(r_norm)) on the R_{t+1}
    register. This enables reward estimation via amplitude estimation
    (BIQAE) without separate reward measurement circuits.

    Li, Vidwans, Wang, Soley, "Harnessing Bayesian Statistics to
    Accelerate IQAE", Quantum 10, 1962 (Jan 14, 2026).
    DOI: 10.22331/q-2026-01-14-1962, arXiv:2507.23074.
    -- BIQAE can be used to estimate the encoded reward amplitude,
       providing near-optimal sample complexity O(1/epsilon).

    Brassard, Hoyer, Mosca, Tapp, "Quantum Amplitude Amplification
    and Estimation", Contemporary Math 305:53-74 (2002).
    -- Foundational amplitude estimation framework for extracting
       the encoded reward value from the quantum state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit

from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap


class RewardUnitary:
    """Encodes E[r|s, a] as quantum rotations on the reward register."""

    def __init__(
        self,
        reward_matrix: NDArray[np.float64],
        register_map: POMDPRegisterMap,
    ) -> None:
        self._R = reward_matrix  # Shape: (S, A)
        self._reg = register_map
        self._normalized = self._normalize_rewards()

    def _normalize_rewards(self) -> NDArray[np.float64]:
        """Normalize R[s,a] to [0, 1] range for angle encoding."""
        r_min = self._R.min()
        r_max = self._R.max()
        if np.isclose(r_min, r_max):
            return np.full_like(self._R, 0.5)
        return (self._R - r_min) / (r_max - r_min)

    def build(self, circuit: QuantumCircuit) -> None:
        """Append U_3 gates: controlled on (S_t, A_t), target R_{t+1}.

        Encodes normalized reward as R_Y rotation angle on the first
        qubit of the reward register (arXiv:2507.18606, Sec III.D).
        The angle theta = 2*arcsin(sqrt(r_norm)) maps each reward value
        to an amplitude that can be extracted via BIQAE (Quantum 10:1962).
        """
        num_states, num_actions = self._R.shape
        n_state_bits = self._reg.state_current.size
        n_action_bits = self._reg.action.size

        for s in range(num_states):
            for a in range(num_actions):
                r_norm = self._normalized[s, a]
                if r_norm < 1e-10:
                    continue

                theta = 2 * np.arcsin(np.sqrt(max(0.0, min(1.0, r_norm))))
                if abs(theta) < 1e-10:
                    continue

                ctrl_state_bits = format(s, f"0{n_state_bits}b")
                ctrl_action_bits = format(a, f"0{n_action_bits}b")

                ctrl_qubits = list(self._reg.state_current) + list(self._reg.action)
                ctrl_string = ctrl_state_bits + ctrl_action_bits

                x_positions: list[int] = []
                for i, bit in enumerate(ctrl_string):
                    if bit == "0":
                        circuit.x(ctrl_qubits[i])
                        x_positions.append(i)

                from qiskit.circuit.library import RYGate

                target = self._reg.reward[0]
                controlled_ry = RYGate(float(theta)).control(len(ctrl_qubits))
                circuit.append(controlled_ry, ctrl_qubits + [target])

                for i in x_positions:
                    circuit.x(ctrl_qubits[i])
