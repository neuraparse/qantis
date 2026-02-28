"""Quantum belief updater - circuit execution coordinator.

Coordinates the execution of quantum belief update circuits
on quantum backends, handling transpilation, error mitigation,
and result post-processing.

Academic References:
    arXiv:2507.18606 - "Hybrid quantum-classical algorithm for near-optimal
    planning in POMDPs" (Jul 2025).
    -- The quantum belief update circuit (Fig. 3) is constructed,
       transpiled, executed, and post-processed in this module.

    Cai et al., "Quantum Error Mitigation", Rev. Mod. Phys. 95,
    045005 (2023).
    -- Composable error mitigation framework integrated via the
       mitigation_pipeline parameter. Supported techniques include:
       * Zero-noise extrapolation (ZNE)
       * Probabilistic error cancellation (PEC)
       * Twirled readout error extinction (T-REx)
       * M3 (Matrix-free Measurement Mitigation)
       These are applied to raw measurement counts before belief
       state reconstruction.

    Qiskit v2.3 (Jan 2026): SamplerV2 primitives used for circuit
    execution. The transpile step maps the abstract circuit to the
    target backend's native gate set (e.g., IBM Heron R3 with
    CX + single-qubit gates). EstimatorV2 can be used for direct
    expectation value estimation as an alternative to sampling.

    Li, Vidwans, Wang, Soley, "Harnessing Bayesian Statistics to
    Accelerate IQAE", Quantum 10, 1962 (Jan 14, 2026).
    DOI: 10.22331/q-2026-01-14-1962, arXiv:2507.23074.
    -- BIQAE can be integrated here for adaptive amplitude estimation
       of the belief state, replacing fixed-shot sampling with
       Bayesian-optimal measurement scheduling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

import numpy as np
from numpy.typing import NDArray

from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.models.pomdp import POMDPModel

logger = logging.getLogger(__name__)


@dataclass
class BeliefUpdateResult:
    """Result from a quantum belief update."""
    posterior_belief: BeliefState
    raw_counts: dict[str, int] | None = None
    mitigated_counts: dict[str, int] | None = None
    circuit_depth: int = 0
    execution_time_s: float = 0.0
    used_quantum: bool = False


@dataclass
class QuantumBeliefUpdater:
    """Coordinates quantum belief update circuit execution.

    Handles the full pipeline (arXiv:2507.18606, Fig. 3):
    1. Build belief update circuit (U(b), U(a), U_1, U_2, U_3, G^k(o))
    2. Transpile for target backend (Qiskit v2.3, Heron R3)
    3. Apply error mitigation (Cai et al., Rev. Mod. Phys. 2023)
    4. Execute on backend (SamplerV2)
    5. Post-process measurement results to posterior belief
    """
    model: POMDPModel
    circuit_config: Any = None
    backend: Any = None
    mitigation_pipeline: Any = None
    shots: int = 4096

    def __post_init__(self) -> None:
        if self.backend is not None:
            from quantum_pomdp.quantum_circuits.belief_update import (
                BeliefUpdateCircuitConfig,
                QuantumBeliefUpdateCircuit,
            )
            if self.circuit_config is None:
                self.circuit_config = BeliefUpdateCircuitConfig(
                    use_amplitude_amplification=True,
                    aa_iterations=1,
                    include_reward_register=False,
                )
            self._circuit_builder = QuantumBeliefUpdateCircuit(self.model, self.circuit_config)
        else:
            self._circuit_builder = None

    def update(
        self,
        belief: BeliefState,
        action: int,
        observation: int,
    ) -> BeliefUpdateResult:
        """Perform a quantum belief update.

        Falls back to classical update if backend is not available.
        """
        if self.backend is None:
            return self._classical_update(belief, action, observation)

        try:
            return self._quantum_update(belief, action, observation)
        except Exception as e:
            logger.warning("Quantum update failed (%s), using classical fallback", e)
            return self._classical_update(belief, action, observation)

    def _quantum_update(
        self,
        belief: BeliefState,
        action: int,
        observation: int,
    ) -> BeliefUpdateResult:
        """Execute quantum belief update circuit."""
        import time

        # Build circuit
        circuit = self._circuit_builder.build(
            belief, action=action, observation=observation,
        )

        # Transpile
        transpiled_circuits = self.backend.transpile([circuit])

        # Execute
        from quantum_common.backends.base import ExecutionRequest

        t0 = time.perf_counter()
        request = ExecutionRequest(circuits=transpiled_circuits, shots=self.shots)
        result = self.backend.execute(request)
        exec_time = time.perf_counter() - t0

        raw_counts = result.counts[0]

        # Apply error mitigation (Cai et al., Rev. Mod. Phys. 95, 045005, 2023)
        if self.mitigation_pipeline is not None:
            mitigated_counts = self.mitigation_pipeline.apply(
                circuit, self.backend, raw_counts, self.shots,
            )
        else:
            mitigated_counts = raw_counts

        # Extract posterior belief from measurement results
        posterior = BeliefState.from_quantum_measurement(
            mitigated_counts,
            num_states=self.model.num_states,
            num_state_qubits=self.model.state_qubits,
        )

        return BeliefUpdateResult(
            posterior_belief=posterior,
            raw_counts=raw_counts,
            mitigated_counts=mitigated_counts,
            circuit_depth=circuit.depth(),
            execution_time_s=exec_time,
            used_quantum=True,
        )

    def _classical_update(
        self,
        belief: BeliefState,
        action: int,
        observation: int,
    ) -> BeliefUpdateResult:
        """Classical Bayesian belief update (fallback).

        Per Kaelbling et al. (1998), Eq. 4: b'(s') = P(o|s',a) *
        sum_s T(s'|s,a) * b(s) / P(o|b,a). Used when quantum backend
        is unavailable or circuit execution fails.
        """
        posterior = belief.classical_update(
            action=action,
            observation=observation,
            transition_tensor=self.model.transition_tensor,
            observation_tensor=self.model.observation_tensor,
        )
        return BeliefUpdateResult(
            posterior_belief=posterior,
            used_quantum=False,
        )
