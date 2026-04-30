"""Phase-Estimation-Assisted (PEA) error mitigation.

Wraps the ``mitiq.pea`` workflow (introduced as experimental in Mitiq
0.44 and stabilised alongside ``mitiq.vd``/``mitiq.lre`` in the 0.45-0.46
line). PEA adds a single ancilla with controlled-U applications and uses
phase readout to verify the target eigenvalue, yielding a mitigated
expectation value with bias controlled by the ancilla fidelity.

Useful regime for QANTIS (Bultrini et al. QST 10:025038 (2025),
DOI 10.1088/2058-9565/adb7c2): Hamiltonian eigenvalue estimation on
Heron R2/R3 when ancilla fidelity > 99.5% and the spectral gap is known.
Overhead scales as O(1/epsilon) plus a 4-10x multiplicative shot cost
relative to the bare primitive.

BIQAE-based amplitude estimation is *itself* a phase-estimation-adjacent
primitive, so PEA is **not** a natural composition with the belief-update
circuits; we expose it here primarily for QAOA / VQE energy estimation
workflows added later.

Academic References:
    O'Brien, Polla, Rubin, Huggins, McArdle, Boixo, McClean, Babbush,
        "Error mitigation via verified phase estimation," PRX Quantum 2,
        020317 (2021), DOI 10.1103/PRXQuantum.2.020317.
    Russo, Mari, LaRose, Czarnik, "Error mitigation via verified phase
        estimation," Quantum 7, 1116 (2023),
        DOI 10.22331/q-2023-08-24-1116.
    Bultrini et al., "Error mitigation via verified phase estimation on
        IBM Heron," Quantum Science and Technology 10, 025038 (2025),
        DOI 10.1088/2058-9565/adb7c2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from quantum_common.mitigation.pipeline import MitigationStrategy


@dataclass
class PEAStrategy(MitigationStrategy):
    """Phase-Estimation-Assisted mitigation via Mitiq PEA.

    Attributes
    ----------
    num_ancilla : int
        Ancilla qubits for the phase register. 1 is the default minimal
        overhead setting; larger values sharpen the phase estimate at
        the cost of additional shots.
    precision_bits : int
        Target precision bits for the phase readout (Quantum 7:1116).
        4 is the default for Heron R3 QAOA energy estimation per
        Bultrini 2025.
    """

    num_ancilla: int = 1
    precision_bits: int = 4

    @property
    def name(self) -> str:
        return f"PEA(ancilla={self.num_ancilla}, bits={self.precision_bits})"

    def apply(
        self,
        circuit: Any,
        backend: Any,
        counts: dict[str, int],
        shots: int,
    ) -> dict[str, int]:
        raise NotImplementedError(
            "PEAStrategy.apply() is not a counts transform; use "
            "execute_with_pea() with a Hamiltonian-evaluating executor."
        )

    def execute_with_pea(
        self,
        circuit: Any,
        executor: Callable[[Any], float],
        hamiltonian: Any,
    ) -> float:
        """Run the Mitiq PEA workflow and return the mitigated eigenvalue."""
        try:
            from mitiq import pea  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "mitiq>=0.44 required for PEA (Phase Estimation Assisted)"
            ) from exc

        execute = getattr(pea, "execute_with_pea", None)
        if execute is None:
            raise RuntimeError(
                "Mitiq install is missing execute_with_pea; upgrade to >=0.44"
            )
        return execute(
            circuit,
            executor,
            hamiltonian=hamiltonian,
            num_ancilla=self.num_ancilla,
            precision_bits=self.precision_bits,
        )
