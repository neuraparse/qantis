"""qantis_engine.infer: belief-update / posterior-conditioning facade.

Thin re-export of the hardware-validated belief-update primitives from
``quantum_pomdp``. The API is intentionally narrow: callers state what they
need (a posterior given prior + observation + model) and the engine selects
the right primitive (FPAA, BIQAE, two-phase calibrated BIQAE).

Academic reference: arXiv:2603.00785 (QANTIS, Feb 2026) for the platform
context; arXiv:2507.18606 for the underlying hybrid POMDP architecture;
Quantum 10:1962 (Jan 14, 2026) for BIQAE.
"""
from __future__ import annotations

from qantis_engine.infer.api import PosteriorReport, posterior

__all__ = ["posterior", "PosteriorReport"]
