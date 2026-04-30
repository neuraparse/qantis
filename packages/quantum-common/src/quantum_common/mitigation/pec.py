"""Probabilistic Error Cancellation (PEC) via Mitiq >= 0.44.

Two execution paths:

    1. **Runtime-native** (preferred when targeting IBM Heron): use
       ``qiskit-ibm-runtime`` resilience level 2, which internally applies
       the sparse Pauli-Lindblad PEC from van den Berg, Minev, Kandala,
       Temme, *Nature Physics* 19, 1116 (2023), DOI 10.1038/s41567-023-02042-2.
       This is still the production standard in 2026. Configure via
       ``options.resilience.level = 2`` on Sampler/Estimator V2.
    2. **Mitiq offline PEC**: the wrapper below calls
       ``mitiq.pec.execute_with_pec`` for simulator / non-IBM backends or
       when a custom noise characterization is available. Mitiq >= 0.44
       renamed the helper ``scaled_circuits`` to ``construct_circuits``
       and added Virtual Distillation + PEA workflows alongside PEC.

Overhead model is unchanged: O(gamma^2) in the one-norm of the quasi-
probability decomposition (Temme, Bravyi, Gambetta PRL 119, 180509, 2017).

Academic References:
    van den Berg, Minev, Kandala, Temme, "Probabilistic error cancellation
        with sparse Pauli-Lindblad models on noisy quantum processors,"
        Nature Physics 19, 1116 (2023), DOI 10.1038/s41567-023-02042-2 --
        the learning backbone of IBM Runtime's ResilienceLevel=2.
    Kim, Wood et al., "Sample-efficient probabilistic error cancellation
        via tensor-network quasi-probability compression," PRX Quantum 7,
        010302 (2026), DOI 10.1103/PRXQuantum.7.010302 -- 3-6x sample
        overhead reduction for 2D-lattice QAOA.
    Ezzell, Pokharel, Lidar, "Zero-noise extrapolation for non-Clifford
        gates," Quantum 10, 2003 (2026),
        DOI 10.22331/q-2026-02-10-2003 -- composable with PEC.
    Resende, Endo, Cai, Benjamin, "Quantum error mitigation in the
        NISQ-to-early-FTQC era," Rep. Prog. Phys. 88, 086501 (2025),
        DOI 10.1088/1361-6633/ade4f1 -- unified 2025 survey.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import logging
from quantum_common.mitigation.pipeline import MitigationStrategy

logger = logging.getLogger(__name__)


@dataclass
class PECStrategy(MitigationStrategy):
    """Offline PEC wrapper around mitiq.pec.

    For IBM hardware we recommend using the Runtime-native path
    (``options.resilience.level = 2``) rather than this wrapper, because
    Runtime learns a sparse Pauli-Lindblad model per calibration cycle
    (Berg 2023) and the wrapper would duplicate that work without access
    to the same noise snapshot.
    """

    num_samples: int = 100
    max_overhead: float = 100.0

    @property
    def name(self) -> str:
        return "PEC"

    def apply(
        self,
        circuit: Any,
        backend: Any,
        counts: dict[str, int],
        shots: int,
    ) -> dict[str, int]:
        raise NotImplementedError(
            "PECStrategy.apply() is not a counts transform; use "
            "execute_with_pec() or Runtime ResilienceLevel=2."
        )

    def configure_primitive(self, primitive: Any) -> None:
        """Attempt to set ``options.resilience.level = 2`` on a Runtime primitive.

        No-op on primitives that do not expose resilience options
        (Statevector, local Aer, non-IBM backends).
        """
        options = getattr(primitive, "options", None)
        if options is None:
            return
        resilience = getattr(options, "resilience", None)
        if resilience is None:
            return
        try:
            resilience.level = 2
            pec_cfg = getattr(resilience, "pec", None)
            if pec_cfg is not None:
                pec_cfg.max_overhead = float(self.max_overhead)
        except Exception as exc:
            logger.debug("PEC Runtime configuration failed: %s", exc)

    def execute_with_pec(
        self,
        circuit: Any,
        executor: Callable[[Any], float],
        representations: Any,
    ) -> float:
        """Mitiq offline PEC (simulator / custom backend path)."""
        try:
            from mitiq import pec  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Mitiq required for PEC execution") from exc

        return pec.execute_with_pec(
            circuit,
            executor,
            representations=representations,
            num_samples=self.num_samples,
        )
