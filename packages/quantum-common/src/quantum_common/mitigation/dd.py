"""Dynamical Decoupling (DD) mitigation stage.

Adds Dynamical Decoupling pulse sequences to quantum circuits to suppress
dephasing and crosstalk on idle qubits during long gate sequences. DD is
particularly important for:

    - QAOA with many alternation layers (long gate sequences).
    - Belief-update circuits in QBRL with observation-unitary waits.
    - Amplitude-amplification circuits with repeated Grover operator calls.

Unlike ZNE / PEC, DD does not consume extra shots — it rewrites the circuit
before execution. This stage therefore implements :class:`apply` as a
pass-through on counts and exposes a separate :meth:`rewrite_circuit` hook
called before submission by backends that respect it.

Supported sequences (arXiv:quant-ph/9803057, Viola–Lloyd 1998; Quiroz &
Lidar, PRA 88, 052329, 2013):

    - "X2"   — two X pulses symmetrically placed (first-order decoupling).
    - "XY4"  — four alternating X, Y, X, Y pulses (second-order).
    - "CPMG" — Carr-Purcell-Meiboom-Gill, Y-pulse sequence tuned for T2.
    - "EDD"  — Eulerian DD, combines XY4 with its reverse for robustness
      against pulse errors (Viola, Knill, Lloyd, PRA 60, 1999).

Academic References:
    Viola, Knill, Lloyd, "Dynamical Decoupling of Open Quantum Systems,"
        PRL 82, 2417 (1999).
    Pokharel, Anand, Fortman, Lidar, "Demonstration of fidelity improvement
        using dynamical decoupling with superconducting qubits," PRL 121,
        220502 (2018) — first experimental gains on IBM hardware.
    Ezzell et al., "Dynamical Decoupling for Superconducting Qubits: a
        performance survey," PRX Quantum 4, 010320 (2023) — empirical
        sequence selection on IBM fleet.
    Ji & Polian, "Dynamical Decoupling for QAOA on Superconducting
        Hardware," arXiv:2508.10456, 2025 — the QAOA-specific
        application motivating this module.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import logging

from quantum_common.mitigation.pipeline import MitigationStrategy

logger = logging.getLogger(__name__)


@dataclass
class DDStrategy(MitigationStrategy):
    """Dynamical Decoupling rewrite stage.

    DD operates by inserting pulse sequences on idle qubits during the
    execution of other qubits' gates. The rewrite is performed via Qiskit's
    ``PadDynamicalDecoupling`` transpiler pass when Qiskit Runtime is the
    execution target, or via a generic pass-through otherwise.

    Attributes
    ----------
    sequence : {"X2", "XY4", "CPMG", "EDD"}
        DD pulse sequence family.
    skip_reset : bool
        When True, DD pulses are not inserted on qubits that were just
        reset — avoids double-initialization noise on dynamic-circuit
        backends (IBM Heron R3 supports this).
    pulse_alignment : int
        Hardware pulse-grid alignment in dt units. 16 is the current
        IBM Heron R3 value; set to 1 for other backends.
    """

    sequence: Literal["X2", "XY4", "CPMG", "EDD"] = "XY4"
    skip_reset: bool = True
    pulse_alignment: int = 16

    @property
    def name(self) -> str:
        return f"DD({self.sequence})"

    def apply(
        self,
        circuit: Any,
        backend: Any,
        counts: dict[str, int],
        shots: int,
    ) -> dict[str, int]:
        """DD is a circuit-rewrite, not a post-processing step.

        The actual rewrite happens in :meth:`rewrite_circuit`, invoked by
        the backend abstraction *before* the shots are submitted. This
        :meth:`apply` therefore returns counts unchanged; it exists so the
        DD stage can participate in the same :class:`MitigationPipeline`
        chain as ZNE / PEC / readout correction and so it can be
        introspected, logged, and toggled uniformly.
        """
        logger.debug(
            "DD(%s) — pulse-level rewrite happens at backend transpile time",
            self.sequence,
        )
        return counts

    def rewrite_circuit(self, circuit: Any, backend: Any) -> Any:
        """Insert DD pulses using Qiskit's PadDynamicalDecoupling if possible.

        Falls back to returning the circuit unchanged when Qiskit is not
        available or when the backend does not expose pulse-grid info. Ken
        Robbins-style advice applies here: do not silently add DD to
        simulators where it has no effect; log and skip.
        """
        try:
            from qiskit.circuit.library import XGate, YGate
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
        except ImportError:
            logger.warning("Qiskit unavailable; skipping DD rewrite")
            return circuit

        dd_sequence = self._resolve_sequence(XGate(), YGate())

        durations = getattr(backend, "instruction_durations", None)
        if durations is None:
            logger.debug("Backend has no instruction_durations; DD rewrite skipped")
            return circuit

        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(
                    durations=durations,
                    dd_sequence=dd_sequence,
                    pulse_alignment=self.pulse_alignment,
                    skip_reset_qubits=self.skip_reset,
                ),
            ]
        )
        return pm.run(circuit)

    def _resolve_sequence(self, x_gate: Any, y_gate: Any) -> list[Any]:
        if self.sequence == "X2":
            return [x_gate, x_gate]
        if self.sequence == "XY4":
            return [x_gate, y_gate, x_gate, y_gate]
        if self.sequence == "CPMG":
            return [y_gate, y_gate]
        # EDD: XY4 + its reverse for pulse-error robustness
        return [x_gate, y_gate, x_gate, y_gate, y_gate, x_gate, y_gate, x_gate]
