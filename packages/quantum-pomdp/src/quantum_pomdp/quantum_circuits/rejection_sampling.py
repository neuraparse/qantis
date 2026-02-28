"""Quantum rejection sampling for Bayesian network inference.

Implements the core algorithm from arXiv:2507.18606 (Section 3):

1. Prepare joint state: |Psi> = sqrt(P(e))|Q,e> + sqrt(1-P(e))|Q,e_bar>
2. Apply evidence phase-flip operator S_e
3. Apply diffusion operator S_0
4. Compose amplitude amplification operator G = B * S_0 * B^dag * S_e
5. Iterate G^k times where k = O(P(e)^{-1/2})
6. Measure to obtain sample from P(Q|E=e)

The quadratic speedup over classical rejection sampling (which requires
O(P(e)^{-1}) iterations) is the key quantum advantage for POMDPs.

Academic References:
    Ozols, Roetteler, Roland, "Quantum Rejection Sampling",
    ACM Trans. Computation Theory 5(3):11 (2013).
    -- Core QRS algorithm. Converts Bayesian network inference into
       amplitude amplification: sampling from P(Q|E=e) requires only
       O(P(e)^{-1/2}) quantum circuit evaluations vs O(P(e)^{-1}) classical.

    Brassard, Hoyer, Mosca, Tapp, "Quantum Amplitude Amplification
    and Estimation", Contemporary Math 305:53-74 (2002).
    -- Grover-style amplitude amplification (AA) operator G = S_0 * S_e
       used in steps 2-5 above. Provides the quadratic speedup.

    Li, Vidwans, Wang, Soley, "Harnessing Bayesian Statistics to
    Accelerate IQAE", Quantum 10, 1962 (Jan 14, 2026).
    DOI: 10.22331/q-2026-01-14-1962, arXiv:2507.23074.
    -- BIQAE provides adaptive iteration count selection for QRS,
       replacing fixed k with Bayesian posterior-guided scheduling.
       Integration point: estimate_optimal_iterations() can be replaced
       with BIQAE's adaptive K-schedule for improved sample efficiency.

    Ramoa & Santos, "Bayesian Quantum Amplitude Estimation",
    Quantum 9, 1856 (Sep 2025). arXiv:2412.04394.
    -- Noise-aware BAE variant that accounts for hardware noise in the
       amplitude estimation, providing more robust iteration estimates
       on NISQ devices.

    arXiv:2507.18606, Sec III - Full integration of QRS into the POMDP
    belief update circuit (Fig. 3). The evidence is the observation
    register matching value o, yielding P(e) = P(o|b,a).
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap


class QuantumRejectionSampler:
    """Quantum rejection sampling with Grover-style amplitude amplification.

    Implements QRS (Ozols et al., ACM TOCT 2013) combined with Grover AA
    (Brassard et al., Contemp. Math 2002) for POMDP belief update.
    """

    def __init__(self, register_map: POMDPRegisterMap) -> None:
        self._reg = register_map

    def build_evidence_oracle(
        self,
        observation: int,
        circuit: QuantumCircuit,
    ) -> None:
        """S_e: Phase-flip oracle that marks states consistent with evidence.

        Flips the phase of states where the observation register matches
        the given observation value. Implemented as multi-controlled Z gate.
        """
        n_obs_bits = self._reg.observation.size
        obs_bits = format(observation, f"0{n_obs_bits}b")

        # Apply X gates for 0-controls
        x_positions: list[int] = []
        for i, bit in enumerate(obs_bits):
            if bit == "0":
                circuit.x(self._reg.observation[i])
                x_positions.append(i)

        # Multi-controlled Z: flip phase when all observation qubits are |1>
        if n_obs_bits == 1:
            circuit.z(self._reg.observation[0])
        else:
            # MCZ = H on target, then MCX, then H on target
            target = self._reg.observation[n_obs_bits - 1]
            controls = list(self._reg.observation[: n_obs_bits - 1])
            circuit.h(target)
            if len(controls) == 1:
                circuit.cx(controls[0], target)
            else:
                from qiskit.circuit.library import XGate
                circuit.append(XGate().control(len(controls)), controls + [target])
            circuit.h(target)

        # Undo X gates
        for i in x_positions:
            circuit.x(self._reg.observation[i])

    def build_diffusion_operator(
        self,
        state_prep_circuit: QuantumCircuit,
        circuit: QuantumCircuit,
    ) -> None:
        """S_0: Grover diffusion operator reflecting about initial state.

        S_0 = 2|Psi><Psi| - I = B * (2|0><0| - I) * B^dag

        where B is the state preparation circuit.
        """
        # Apply B^dag (inverse of state preparation).
        # Plain inverse() works when state prep uses only real-param gates
        # (use_hardware_compatible_encoding=True → RY rotations → IBM-serializable).
        # Fall back to annotated=True for circuits with Initialize (complex params),
        # which appears in the non-hardware-compatible path (unit tests only).
        from qiskit.circuit.exceptions import CircuitError

        try:
            inv_prep = state_prep_circuit.inverse()
        except CircuitError:
            inv_prep = state_prep_circuit.inverse(annotated=True)
        circuit.compose(inv_prep, inplace=True)

        # Apply 2|0><0| - I on all qubits (except ancilla)
        work_qubits = (
            list(self._reg.state_current)
            + list(self._reg.action)
            + list(self._reg.state_next)
            + list(self._reg.observation)
        )

        # X on all, then MCZ, then X on all
        for q in work_qubits:
            circuit.x(q)

        if len(work_qubits) > 1:
            target = work_qubits[-1]
            controls = work_qubits[:-1]
            circuit.h(target)
            if len(controls) == 1:
                circuit.cx(controls[0], target)
            elif len(controls) > 1:
                from qiskit.circuit.library import XGate
                circuit.append(XGate().control(len(controls)), controls + [target])
            circuit.h(target)

        for q in work_qubits:
            circuit.x(q)

        # Apply B (state preparation)
        circuit.compose(state_prep_circuit, inplace=True)

    def estimate_optimal_iterations(self, acceptance_probability: float) -> int:
        """Compute optimal number of amplitude amplification iterations.

        k_opt = floor(pi / (4 * arcsin(sqrt(P(e)))) - 1/2)

        Per Brassard, Hoyer, Mosca, Tapp, Contemp. Math 305:53-74 (2002):
        for low acceptance probabilities, k ~ O(P(e)^{-1/2}), providing
        quadratic speedup over classical O(P(e)^{-1}) rejection sampling.

        Note: BIQAE (Li et al., Quantum 10:1962, 2026) can adaptively
        select k via Bayesian posterior, potentially replacing this
        fixed-formula approach for improved sample efficiency.
        """
        if acceptance_probability <= 0 or acceptance_probability >= 1:
            return 1

        theta = np.arcsin(np.sqrt(acceptance_probability))
        if theta < 1e-10:
            return 1

        k = int(np.floor(np.pi / (4.0 * theta) - 0.5))
        return max(1, min(k, 50))  # Cap at 50 to limit circuit depth
