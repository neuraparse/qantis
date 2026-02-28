"""Complete quantum belief update circuit from arXiv:2507.18606.

The circuit implements the following sequence (Figure 3 of QBRL paper):

|0>_{s_t}   ---[ U(b) ]---[ U_1 ]---[     ]---[     ]---[ G^k(o) ]--- Measure
|0>_{a_t}   ---[ U(a) ]---[     ]---[     ]---[     ]---[        ]---
|0>_{s_t+1} ---[      ]---[     ]---[ U_2 ]---[     ]---[        ]--- Measure
|0>_{o_t+1} ---[      ]---[     ]---[     ]---[     ]---[        ]--- Measure
|0>_{r_t+1} ---[      ]---[     ]---[     ]---[ U_3 ]---[        ]--- Measure

Where:
- U(b): Belief state preparation (amplitude encoding)
- U(a): Action encoding (computational basis)
- U_1:  Transition dynamics P(s'|s,a)  -- arXiv:2507.18606 Sec III.B
- U_2:  Sensor model P(o|s',a)         -- arXiv:2507.18606 Sec III.C
- U_3:  Reward function E[r|s,a]       -- arXiv:2507.18606 Sec III.D
- G^k(o): Amplitude amplification for observation o (Brassard et al. 2002)

Academic References:
    arXiv:2507.18606 - "Hybrid quantum-classical algorithm for near-optimal
    planning in POMDPs" (Jul 2025), Fig. 3.
    -- Complete circuit composition: U(b), U(a), U_1, U_2, U_3, G^k(o).

    Brassard, Hoyer, Mosca, Tapp, "Quantum Amplitude Amplification
    and Estimation", Contemporary Math 305:53-74 (2002).
    -- G^k(o) amplitude amplification for the belief update step.

    Moettonen, Vartiainen, Bergholm, Salomaa, PRL 93, 130502 (2004).
    -- UCR_Y decomposition for U(b), U_1, U_2, U_3 sub-circuits.

    Qiskit v2.3 (Jan 2026): SamplerV2 used for circuit execution.
    Circuit depth on NISQ hardware is dominated by the multi-controlled
    rotations in U_1 and G^k(o). For Heron R3 (156 qubits), POMDPs
    up to ~|S|=16, |A|=4 are feasible without circuit cutting.

    Cai et al., "Quantum Error Mitigation", Rev. Mod. Phys. 95,
    045005 (2023).
    -- Composable error mitigation applied to measurement results
       post-execution to improve belief state reconstruction fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit

from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.models.pomdp import POMDPModel
from quantum_pomdp.quantum_circuits.amplitude_amplifier import BeliefAmplitudeAmplifier
from quantum_pomdp.quantum_circuits.observation_unitary import ObservationUnitary
from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap
from quantum_pomdp.quantum_circuits.rejection_sampling import QuantumRejectionSampler
from quantum_pomdp.quantum_circuits.reward_unitary import RewardUnitary
from quantum_pomdp.quantum_circuits.state_encoder import BeliefStateEncoder
from quantum_pomdp.quantum_circuits.transition_unitary import TransitionUnitary


@dataclass
class BeliefUpdateCircuitConfig:
    """Configuration for quantum belief update circuit construction."""

    reward_precision_bits: int = 4
    use_amplitude_amplification: bool = True
    aa_iterations: int | None = None
    include_reward_register: bool = True
    use_hardware_compatible_encoding: bool = False


class QuantumBeliefUpdateCircuit:
    """Constructs the complete quantum belief update circuit.

    Implements the full circuit from arXiv:2507.18606 Fig. 3, composing
    belief encoding, transition/observation/reward unitaries, and
    Grover-style amplitude amplification (Brassard et al., 2002).
    """

    def __init__(
        self,
        pomdp: POMDPModel,
        config: BeliefUpdateCircuitConfig | None = None,
    ) -> None:
        self._pomdp = pomdp
        self._config = config or BeliefUpdateCircuitConfig()

        self._reg = POMDPRegisterMap.from_dimensions(
            state_qubits=pomdp.state_qubits,
            action_qubits=pomdp.action_qubits,
            observation_qubits=pomdp.observation_qubits,
            reward_bits=self._config.reward_precision_bits,
        )

        self._state_encoder = BeliefStateEncoder(self._reg)
        self._transition = TransitionUnitary(pomdp.transition_tensor, self._reg)
        self._observation_u = ObservationUnitary(pomdp.observation_tensor, self._reg)
        self._reward = RewardUnitary(pomdp.reward_matrix, self._reg)
        self._qrs = QuantumRejectionSampler(self._reg)
        self._amplifier = BeliefAmplitudeAmplifier(self._qrs, self._reg)

    @property
    def register_map(self) -> POMDPRegisterMap:
        return self._reg

    @property
    def total_qubits(self) -> int:
        return self._reg.total_qubits

    def build(
        self,
        belief: BeliefState,
        action: int,
        observation: int | None = None,
        acceptance_probability: float | None = None,
    ) -> QuantumCircuit:
        """Construct the full quantum belief update circuit.

        Args:
            belief: Current belief distribution b_t.
            action: Action taken a_t.
            observation: If provided, applies amplitude amplification G^k(o).
                If None, measures observation register to sample o.
            acceptance_probability: P(o|b,a) for calibrating AA iterations.

        Returns:
            Complete circuit ready for transpilation and execution.
        """
        qc = QuantumCircuit(
            *self._reg.all_quantum_registers,
            self._reg.classical,
        )

        # Step 1: Encode belief state |b_t>
        amplitudes = belief.to_amplitudes()
        if self._config.use_hardware_compatible_encoding:
            self._state_encoder.encode_belief_with_rotations(
                belief.probabilities, qc
            )
        else:
            self._state_encoder.encode_belief(amplitudes, qc)

        # Step 2: Encode action |a_t>
        self._state_encoder.encode_action(action, qc)

        # Step 3: Apply transition dynamics U_1
        self._transition.build(qc)

        # Step 4: Apply sensor model U_2
        self._observation_u.build(qc)

        # Step 5: Apply reward encoding U_3 (optional)
        if self._config.include_reward_register:
            self._reward.build(qc)

        # Step 6: Amplitude amplification for observation (if specified)
        if (
            observation is not None
            and self._config.use_amplitude_amplification
        ):
            # Build state preparation circuit (everything up to this point)
            state_prep = qc.copy()
            self._amplifier.build(
                state_prep_circuit=state_prep,
                observation=observation,
                circuit=qc,
                num_iterations=self._config.aa_iterations,
                acceptance_probability=acceptance_probability,
            )

        # Step 7: Measurement of state_next and observation registers
        meas_qubits = list(self._reg.state_next) + list(self._reg.observation)
        n_meas = len(meas_qubits)
        for i, q in enumerate(meas_qubits):
            if i < self._reg.classical.size:
                qc.measure(q, self._reg.classical[i])

        return qc

    def build_without_aa(
        self,
        belief: BeliefState,
        action: int,
    ) -> QuantumCircuit:
        """Build circuit without amplitude amplification (for comparison)."""
        return self.build(belief, action, observation=None)

    def get_circuit_info(self, belief: BeliefState, action: int) -> dict[str, int]:
        """Return circuit resource estimates."""
        circuit = self.build_without_aa(belief, action)
        return {
            "total_qubits": self._reg.total_qubits,
            "circuit_depth": circuit.depth(),
            "gate_count": sum(circuit.count_ops().values()),
            "cx_count": circuit.count_ops().get("cx", 0),
        }
