"""Virtual Distillation (VD) error mitigation via Mitiq >= 0.44.

Virtual Distillation suppresses incoherent errors by measuring
observables on ``rho^M / Tr(rho^M)`` rather than ``rho`` directly. The
effective density matrix converges to the dominant-eigenvector
projector exponentially in the copy count ``M`` (Huggins PRX 11, 041036
(2021); Koczor PRX 11, 031057 (2021)).

Recent peer-reviewed refinements integrated here:

    - Huo & Li, Quantum 9, 1618 (2025), DOI 10.22331/q-2025-03-18-1618
      -- "Dual-State Purification," single-copy randomized-measurement
      variant that removes VD's usual 2n-qubit ancilla penalty. This is
      the method the Mitiq 0.44+ workflow exposes as
      ``technique="dual_state"``.
    - Cao et al., npj Quantum Info 11:22 (2025),
      DOI 10.1038/s41534-025-00961-x -- 27-qubit Eagle hardware demo:
      2-copy VD cut bias ~10x at 3x shot overhead for local Hamiltonian
      expectation values; shallow depth only.

Sweet spot for QANTIS: QAOA energy estimation with local-Z observables
and depth <= 30 layers. Not appropriate for BIQAE (global-phase
observable) or highly coherent-noise-dominated circuits -- prefer ZNE/LRE
in those regimes (Resende et al. Rep. Prog. Phys. 88:086501 (2025)
provides the post-Cai 2023 survey comparing bias/variance/overhead
trade-offs).

Integration notes:
    - :meth:`apply` is a counts-level no-op; VD is an observable-level
      procedure invoked by callers that know how to evaluate expectation
      values.
    - :meth:`execute_with_vd` calls ``mitiq.vd.execute_with_vd`` under
      the hood with the configured dual-state / swap-test technique.

Academic References:
    Huggins, McArdle, O'Brien et al., "Virtual Distillation for Quantum
        Error Mitigation," PRX 11, 041036 (2021),
        DOI 10.1103/PhysRevX.11.041036.
    Koczor, "Exponential Error Suppression for Near-Term Quantum
        Devices," PRX 11, 031057 (2021),
        DOI 10.1103/PhysRevX.11.031057.
    Huo & Li, "Dual-State Purification for Practical Quantum Error
        Mitigation," Quantum 9, 1618 (2025),
        DOI 10.22331/q-2025-03-18-1618.
    Cao et al., "Experimental demonstration of virtual distillation
        for Hamiltonian expectation values on a 27-qubit superconducting
        processor," npj Quantum Info 11, 22 (2025),
        DOI 10.1038/s41534-025-00961-x.
    Resende, Endo, Cai, Benjamin, "Quantum error mitigation in the
        NISQ-to-early-FTQC era," Rep. Prog. Phys. 88, 086501 (2025),
        DOI 10.1088/1361-6633/ade4f1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from quantum_common.mitigation.pipeline import MitigationStrategy


@dataclass
class VDStrategy(MitigationStrategy):
    """Virtual Distillation wrapper for local-observable QAOA paths.

    Attributes
    ----------
    num_copies : int
        Number of virtual copies ``M``. 2 is the production default
        validated in Cao 2025; larger M increases bias suppression at
        multiplicative shot overhead.
    technique : {"dual_state", "swap_test"}
        Mitiq 0.44+ backend selector. "dual_state" follows
        Huo-Li Quantum 9, 1618 (2025) and avoids doubling qubit count.
    """

    num_copies: int = 2
    technique: Literal["dual_state", "swap_test"] = "dual_state"

    @property
    def name(self) -> str:
        return f"VD(M={self.num_copies}, {self.technique})"

    def apply(
        self,
        circuit: Any,
        backend: Any,
        counts: dict[str, int],
        shots: int,
    ) -> dict[str, int]:
        raise NotImplementedError(
            "VDStrategy.apply() is not a counts transform; use "
            "execute_with_vd() with an observable-evaluating executor."
        )

    def execute_with_vd(
        self,
        circuit: Any,
        executor: Callable[[Any], float],
        observable: Any,
    ) -> float:
        """Run VD and return the mitigated expectation value."""
        try:
            from mitiq import vd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "mitiq>=0.44 required for Virtual Distillation"
            ) from exc

        execute = getattr(vd, "execute_with_vd", None)
        if execute is None:
            raise RuntimeError(
                "Mitiq install is missing execute_with_vd; upgrade to >=0.44"
            )
        return execute(
            circuit,
            executor,
            observable=observable,
            num_copies=self.num_copies,
            technique=self.technique,
        )
