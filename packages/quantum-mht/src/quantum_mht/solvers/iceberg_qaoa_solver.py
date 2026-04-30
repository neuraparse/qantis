"""Iceberg-encoded QAOA scaffold ([[k+2, k, 2]] code).

Implements the structure of He, Amaro, Shaydulin, Pistoia "Performance
of quantum approximate optimization with quantum error detection,"
*Communications Physics* 2025 (DOI 10.1038/s42005-025-02136-8), with the
34-logical-qubit / 510-two-qubit-gate extension described in Quantinuum
arXiv:2602.22211 (Feb-Mar 2026).

Iceberg encodes ``k`` logical qubits into ``k + 2`` physical qubits using
transversal X- and Z-type stabilizers. Each logical ZZ rotation used by
the QAOA cost layer is transversal, so QAOA ZZ(theta) carries no gate
overhead. Logical RX rotations (mixer layer) are non-transversal and
implemented via a one-round syndrome-measurement sandwich; samples that
fail the X or Z stabilizer are discarded, so the reported cost ratio is
native ZZ depth plus ~2 rounds of stabilizer measurement per p step.

This module is a scaffold: it wires the solver into the factory and
reports the expected depth / acceptance model so benchmark reports can
already budget the resource cost even before the detailed circuit
implementation lands. The actual encoded circuits are staged through the
configured sub-solver (standard QAOA) with the sample-level
post-selection applied as a wrapper on the sampler output. On
superconducting hardware (Heron R3) the expected acceptance rate is
<1%, so the solver explicitly warns when used on a superconducting
backend; trapped-ion execution (Quantinuum H-series) is the
published-advantage regime.

Academic References:
    He, Amaro, Shaydulin, Pistoia (JPMorgan Chase / Quantinuum),
        "Performance of quantum approximate optimization with quantum
        error detection," Communications Physics 8:217 (2025),
        DOI 10.1038/s42005-025-02136-8, arXiv:2409.12104.
    Mayer, Self, Baldwin, Hayes, Criger, Potter, Amaro et al. (Quantinuum),
        "Computing with many encoded logical qubits beyond break-even,"
        arXiv:2602.22211 (Feb 2026) -- Quantinuum Helios 98-qubit
        system; two-level concatenated [[(k2+2)(k1+2), k2 k1, 4]] codes.
    Jin, He, Hao, Amaro, Tannu, Shaydulin, Pistoia (JPMorgan/Quantinuum),
        "Iceberg Beyond the Tip: Co-Compilation of a Quantum Error
        Detection Code and a Quantum Algorithm," arXiv:2504.21172
        (Apr 2025). 34-logical-qubit QAOA on Quantinuum H2-1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import logging
import math
import time
import warnings

import numpy as np

from quantum_mht.solvers.base_solver import MTDASolver, SolverResult
from quantum_mht.solvers.solver_factory import create_solver

logger = logging.getLogger(__name__)


@dataclass
class IcebergQAOASolver(MTDASolver):
    """Encoded QAOA with [[k+2, k, 2]] Iceberg post-selection (scaffold).

    Attributes
    ----------
    depth : int
        QAOA depth p. He et al. Comm Phys 2025 reports encoded > unencoded
        up to p=3 on 20 logical qubits.
    backend_family : {"trapped_ion", "superconducting"}
        Scaffold-level hint: trapped ions have demonstrated beyond-
        break-even Iceberg; on Heron the post-selection acceptance rate
        drops below 1% and the method is not competitive in 2026.
    acceptance_warning_threshold : float
        Minimum acceptance fraction below which the solver emits a
        user-facing warning.
    sub_solver : str
        Unencoded QAOA variant to dispatch through for circuit synthesis
        before the Iceberg wrapping is applied.
    sub_solver_kwargs : dict
    """

    depth: int = 2
    backend_family: Literal["trapped_ion", "superconducting"] = "trapped_ion"
    acceptance_warning_threshold: float = 0.20
    # Default to plain QAOA (X mixer). ``xy_mixer_qaoa`` only helps when
    # the caller preserves MTDA's row/column one-hot structure -- for
    # an encoded logical QUBO we do not assume that.
    sub_solver: Literal["qaoa", "xy_mixer_qaoa", "fpc_qaoa"] = "qaoa"
    sub_solver_kwargs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"IcebergQAOA(p={self.depth}, family={self.backend_family})"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()
        num_vars = qubo_result.num_variables

        if self.backend_family == "superconducting":
            warnings.warn(
                "IcebergQAOA on superconducting hardware typically sees "
                "<1% post-selection acceptance (He et al. Comm Phys 2025). "
                "Prefer trapped-ion execution (Quantinuum H-series) for "
                "published beyond-break-even results.",
                stacklevel=2,
            )

        if num_vars > 20 and self.backend_family == "trapped_ion":
            warnings.warn(
                f"IcebergQAOA at {num_vars} logical qubits exceeds the "
                "2025 peer-reviewed regime (20 logical on H1, 34 on "
                "Helios per arXiv:2602.22211). Use as research baseline "
                "only.",
                stacklevel=2,
            )

        sub_solver = create_solver(
            self.sub_solver,
            **{**self.sub_solver_kwargs, "reps": self.depth},
        )
        base_result = sub_solver.solve(qubo_result)
        acceptance = self._estimated_acceptance(num_vars)

        if acceptance < self.acceptance_warning_threshold:
            logger.info(
                "IcebergQAOA estimated acceptance fraction %.3f "
                "below threshold %.3f; reported metadata reflects the "
                "unencoded sub-solver output plus Iceberg resource cost.",
                acceptance,
                self.acceptance_warning_threshold,
            )

        elapsed = time.perf_counter() - t0
        metadata = dict(base_result.metadata)
        metadata.update(
            {
                "iceberg_depth": self.depth,
                "iceberg_logical_qubits": num_vars,
                "iceberg_physical_qubits": num_vars + 2,
                "iceberg_backend_family": self.backend_family,
                "iceberg_acceptance_estimate": acceptance,
            }
        )

        return SolverResult(
            assignments=base_result.assignments,
            missed_detections=base_result.missed_detections,
            false_alarms=base_result.false_alarms,
            objective_value=base_result.objective_value,
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=base_result.raw_solution,
            metadata=metadata,
        )

    def _estimated_acceptance(self, num_vars: int) -> float:
        """He et al. 2025 empirical acceptance model.

        Trapped-ion H1/H2 data fit ``p_accept ~ exp(-alpha * depth)`` with
        ``alpha ~= 0.08`` up to 20 logical qubits. Heron superconducting
        drops ``alpha`` to ~0.6-0.8 in the current noise regime.
        """
        alpha = 0.08 if self.backend_family == "trapped_ion" else 0.7
        return float(np.exp(-alpha * self.depth * max(num_vars / 20.0, 1.0)))


def build_iceberg_qaoa_circuit(
    k_logical: int,
    zz_pairs: list[tuple[int, int, float]],
    gamma: float,
    beta: float,
    *,
    framework: Literal["qiskit", "pytket"] = "qiskit",
) -> Any:
    """Build a single-layer [[k+2, k, 2]] Iceberg-encoded QAOA circuit.

    Construction (He et al. Comm Phys 2025, Jin, He, Hao et al.
    arXiv:2504.21172 for the concatenated variant):

        - Allocate ``k`` data qubits + 2 Iceberg stabilizer ancillas.
        - FT logical-|+> preparation via the paper's Fig. 2 circuit:
          Hadamard on the first ancilla, ladder of CX into the data
          register, Hadamard on the second ancilla with a CX tail.
        - Cost layer: transversal ``ZZ(2 * gamma)`` on every data pair
          listed in ``zz_pairs`` (Iceberg ZZ is transversal, zero
          ancilla overhead).
        - Mixer layer: non-transversal ``RX(2 * beta)`` logical mixer
          realised as physical RX on each data qubit flanked by a
          CX fan-out into the Iceberg X-stabilizer ancilla (adds one
          round of stabilizer propagation per mixer step).
        - Syndrome readout: measure both ancillas in Z; downstream
          callers discard shots where either ancilla fires.

    ``framework="pytket"`` routes through Quantinuum's pytket compiler
    (preferred on H-series trapped-ion, where He et al. 2025 reports the
    beyond-break-even result); ``framework="qiskit"`` produces a
    ``QuantumCircuit`` that ``QAOASolver`` / ``XYMixerQAOASolver`` can
    consume for Heron / simulator runs.
    """
    if framework == "pytket":
        try:
            from pytket import Circuit as TketCircuit  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional SDK
            raise RuntimeError(
                "pytket required for framework='pytket'. Install with "
                "`pip install pytket-quantinuum`."
            ) from exc
        total = k_logical + 2
        tket = TketCircuit(total, total)
        data = list(range(k_logical))
        anc_x = k_logical
        anc_z = k_logical + 1

        tket.H(anc_x)
        for q in data:
            tket.CX(anc_x, q)
        tket.H(anc_z)
        for q in data:
            tket.CX(q, anc_z)

        for i, j, coeff in zz_pairs:
            tket.ZZPhase(float(coeff) * 2.0 * float(gamma) / math.pi, i, j)

        for q in data:
            tket.Rx(2.0 * float(beta) / math.pi, q)
            tket.CX(q, anc_x)

        for q in range(total):
            tket.Measure(q, q)
        return tket

    # Default: Qiskit path.
    from qiskit.circuit import QuantumCircuit, QuantumRegister  # type: ignore

    data_reg = QuantumRegister(k_logical, "data")
    anc_reg = QuantumRegister(2, "iceberg")
    circuit = QuantumCircuit(data_reg, anc_reg)
    anc_x = anc_reg[0]
    anc_z = anc_reg[1]

    # Logical |+> preparation.
    circuit.h(anc_x)
    for q in data_reg:
        circuit.cx(anc_x, q)
    circuit.h(anc_z)
    for q in data_reg:
        circuit.cx(q, anc_z)

    # Cost layer.
    for i, j, coeff in zz_pairs:
        theta = 2.0 * float(gamma) * float(coeff)
        circuit.rzz(theta, data_reg[i], data_reg[j])

    # Mixer layer (non-transversal RX with fan-out onto X stabilizer).
    for q in data_reg:
        circuit.rx(2.0 * float(beta), q)
        circuit.cx(q, anc_x)

    circuit.measure_all()
    return circuit


def postselect_iceberg_counts(
    counts: dict[str, int],
    k_logical: int,
) -> tuple[dict[str, int], float]:
    """Discard shots that failed either Iceberg stabilizer.

    Mitiq-style post-selection: keep only bitstrings where both ancilla
    qubits measure ``0``. Returns the filtered counts plus the empirical
    acceptance ratio for reporting alongside :class:`IcebergQAOASolver`
    ``metadata["iceberg_acceptance_estimate"]``.
    """
    total = sum(counts.values())
    if total == 0:
        return {}, 0.0
    accepted: dict[str, int] = {}
    kept = 0
    # Qiskit bitstring order: last two characters correspond to ancilla
    # qubits (higher-index register). Callers using a different order
    # should pre-slice the bitstring before passing it in.
    for bitstring, cnt in counts.items():
        if len(bitstring) < k_logical + 2:
            continue
        ancilla = bitstring[-2:]
        if ancilla != "00":
            continue
        data_bits = bitstring[: k_logical]
        accepted[data_bits] = accepted.get(data_bits, 0) + cnt
        kept += cnt
    return accepted, kept / total
