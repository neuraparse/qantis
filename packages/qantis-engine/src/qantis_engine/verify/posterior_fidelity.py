"""Posterior-fidelity diagnostics: Hellinger and total-variation distance.

Used to measure how close an estimated posterior (BIQAE / FPAA / sampler-
derived) is to the analytic Bayes update. Hellinger is the default because
it is bounded in [0, 1] and the QANTIS paper reports Hellinger throughout
(maximum 0.015 across the T=8 closed-loop Tiger run, per arXiv:2603.00785).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class FidelityReport:
    """Result of a posterior-fidelity comparison."""

    hellinger: float
    total_variation: float
    kl_divergence: float
    is_tight: bool
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)


def hellinger(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0))


def total_variation(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))


def posterior_fidelity(
    estimated: NDArray[np.float64],
    reference: NDArray[np.float64],
    threshold: float = 0.015,
) -> FidelityReport:
    """Compare ``estimated`` to ``reference`` (typically analytic Bayes)."""
    h = hellinger(estimated, reference)
    tv = total_variation(estimated, reference)
    return FidelityReport(
        hellinger=h,
        total_variation=tv,
        kl_divergence=kl(reference, estimated),
        is_tight=h <= threshold,
        threshold=threshold,
        metadata={
            "n_states": int(np.asarray(estimated).size),
            "qantis_t8_threshold": 0.015,
        },
    )
