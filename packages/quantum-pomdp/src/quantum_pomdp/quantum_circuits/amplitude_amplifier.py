"""G^k(o): Amplitude amplification for a specific observation.

Wraps QuantumRejectionSampler to construct the complete amplification
circuit for a given observation value. Applied to the full belief update
circuit, amplifying the amplitude of states consistent with the observed
evidence, thereby performing the Bayesian update in quantum superposition.

Academic References:
    Brassard, Hoyer, Mosca, Tapp, "Quantum Amplitude Amplification
    and Estimation", Contemporary Math 305:53-74 (2002).
    -- Grover-style amplitude amplification: each iteration of G applies
       the evidence oracle S_e followed by the diffusion operator S_0,
       rotating the state vector by angle 2*arcsin(sqrt(P(e))) towards
       the target subspace. After k iterations, success probability is
       sin^2((2k+1)*arcsin(sqrt(P(e)))).

    Li, Vidwans, Wang, Soley, "Harnessing Bayesian Statistics to
    Accelerate IQAE", Quantum 10, 1962 (Jan 14, 2026).
    DOI: 10.22331/q-2026-01-14-1962, arXiv:2507.23074.
    -- BIQAE adaptive iteration selection: instead of using the fixed
       k_opt formula, BIQAE uses a Bayesian posterior over the unknown
       amplitude to adaptively choose iteration counts, achieving ~14%
       fewer oracle queries than standard IQAE.

    Ramoa & Santos, "Bayesian Quantum Amplitude Estimation",
    Quantum 9, 1856 (Sep 2025). arXiv:2412.04394.
    -- Noise-aware BAE variant: provides calibrated iteration counts
       that account for NISQ hardware depolarizing noise.

    arXiv:2507.18606, Sec III, Fig. 3 - G^k(o) is the final stage of
    the quantum belief update circuit, applied after U_1, U_2, U_3.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit

from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap
from quantum_pomdp.quantum_circuits.rejection_sampling import QuantumRejectionSampler


class BeliefAmplitudeAmplifier:
    """Amplitude amplification for POMDP belief update.

    Implements G^k(o) from Brassard et al. (2002) with optional
    BIQAE-guided adaptive iteration selection (Quantum 10:1962, 2026).
    """

    def __init__(
        self,
        rejection_sampler: QuantumRejectionSampler,
        register_map: POMDPRegisterMap,
    ) -> None:
        self._qrs = rejection_sampler
        self._reg = register_map

    def build(
        self,
        state_prep_circuit: QuantumCircuit,
        observation: int,
        circuit: QuantumCircuit,
        num_iterations: int | None = None,
        acceptance_probability: float | None = None,
    ) -> None:
        """Append G^k(o) amplitude amplification iterations to circuit.

        Each iteration consists of:
        1. Evidence oracle S_e (phase flip on observation match)
        2. Diffusion operator S_0 = B * (2|0><0|-I) * B^dag

        Args:
            state_prep_circuit: The state preparation circuit B.
            observation: The observed evidence value.
            circuit: Target circuit to append gates to.
            num_iterations: Number of Grover iterations. Auto-estimated if None.
            acceptance_probability: P(o|b,a) for auto-estimating iterations.
        """
        if num_iterations is None:
            if acceptance_probability is not None:
                num_iterations = self._qrs.estimate_optimal_iterations(
                    acceptance_probability
                )
            else:
                num_iterations = 1  # Conservative default

        # Each Grover iteration applies G = S_0 * S_e per Brassard et al. (2002).
        # Total oracle queries: 2*num_iterations + 1, providing O(P(e)^{-1/2})
        # complexity vs classical O(P(e)^{-1}).
        for _ in range(num_iterations):
            # Step 1: Evidence oracle S_e (phase flip on observation match)
            self._qrs.build_evidence_oracle(observation, circuit)

            # Step 2: Diffusion operator S_0 = B * (2|0><0|-I) * B^dag
            self._qrs.build_diffusion_operator(state_prep_circuit, circuit)
