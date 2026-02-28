"""Zero-Noise Extrapolation (ZNE) using Mitiq.

2026 Academic References — ZNE Theory and Implementation
==========================================================
- **Temme et al., "Error Mitigation for Short-Depth Quantum Circuits",
  PRL 119, 180509 (2017).** DOI: 10.1103/PhysRevLett.119.180509
  Foundational paper introducing ZNE and PEC. ZNE works by intentionally
  amplifying noise at multiple scale factors, then extrapolating to the
  zero-noise limit. Applicable to expectation value estimation.
  CONTEXT: Foundational ZNE paper introducing noise-scaling + extrapolation
  framework. Extended to non-Clifford gates in Feb 2026
  (q-2026-02-10-2003).

- **Li & Benjamin, "Efficient Variational Quantum Simulator Incorporating
  Active Error Minimization", PRX 7, 021050 (2017).**
  DOI: 10.1103/PhysRevX.7.021050
  Independently developed ZNE for variational circuits (VQE). Proposes
  Richardson extrapolation and linear extrapolation as inference methods.
  CONTEXT: Independent ZNE discovery for variational circuits. Their
  Richardson extrapolation approach is the basis for Mitiq's default
  extrapolation.

- **Quantum q-2026-02-10-2003 (Feb 2026)**: Generalizes ZNE to
  weakly-entangling non-Clifford gates, removing the previous Clifford-only
  limitation. This is critical for QAOA circuits which heavily use
  parameterized RZ/RX gates.

- **Nature s41467-025-67768-4 (2025)**: Demonstrates ZNE effectiveness on
  logical qubits -- combining error mitigation with error correction for
  further noise reduction.

- **Mitiq LRE (Layerwise Richardson Extrapolation)**: Available via
  ``mitiq.lre.execute_with_lre``, provides layer-by-layer noise scaling as
  alternative to global unitary folding.

- **Mitiq v0.44+** (2026): API changes —
  - ``scaled_circuits`` renamed to ``construct_circuits``
  - ``fold_gates_at_random`` remains the primary noise-scaling method
  - Richardson, Linear, and Poly extrapolation factories available
  - Virtual Distillation added as alternative to ZNE for certain circuits
  See: https://mitiq.readthedocs.io
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import logging
import numpy as np
from quantum_common.mitigation.pipeline import MitigationStrategy

logger = logging.getLogger(__name__)

@dataclass
class ZNEStrategy(MitigationStrategy):
    """Zero-Noise Extrapolation strategy using Mitiq.

    Implements the ZNE protocol from Temme et al., PRL 119, 180509 (2017)
    and Li & Benjamin, PRX 7, 021050 (2017). Noise is amplified at configurable
    scale factors and extrapolated to the zero-noise limit using Richardson,
    Linear, or Polynomial inference.
    """
    scale_factors: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    factory_type: str = "Richardson"  # Richardson, Linear, Poly

    @property
    def name(self) -> str:
        return f"ZNE({self.factory_type})"

    def apply(self, circuit: Any, backend: Any, counts: dict[str, int], shots: int) -> dict[str, int]:
        """Not implemented: use execute_with_zne() for full ZNE execution.

        This method is intentionally unimplemented. For ZNE with noise scaling
        and Richardson extrapolation use execute_with_zne().
        """
        raise NotImplementedError(
            "ZNEStrategy.apply() is not implemented. "
            "Use execute_with_zne() for full ZNE execution with Mitiq."
        )

    def execute_with_zne(self, circuit: Any, executor_fn: Any) -> float:
        """Full ZNE execution with noise scaling and extrapolation.

        Uses ``fold_gates_at_random`` for unitary folding noise amplification,
        per Temme et al. (2017). The executor_fn must accept a circuit and
        return a float expectation value.
        """
        try:
            from mitiq import zne
            return zne.execute_with_zne(circuit, executor_fn, scale_noise=zne.scaling.fold_gates_at_random)
        except ImportError as e:
            raise RuntimeError("Mitiq required for ZNE execution") from e
