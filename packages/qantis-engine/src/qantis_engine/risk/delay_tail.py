"""Delay-tail probability estimation under truncated regenerative simulation.

Implements the Srikant (Feb 2026, arXiv:2602.09059) protocol: classical
regenerative-cycle simulation has unbounded state spaces, but quantum circuits
have fixed depth and finite registers. The fix is *truncated* regenerative
simulation: cap each regenerative cycle at horizon H, accept the bias, and
report it explicitly. BIQAE then estimates the truncated tail with quadratic
1/sqrt(p) sample-complexity.

Use cases for QANTIS:
    - assignment latency tail in MTDA
    - queue length / waiting time in distributed sensor networks
    - missed-deadline probability in real-time POMDP planning

The only knob the user picks is ``horizon`` — a longer horizon reduces the
truncation bias at the cost of deeper effective oracle. We compute and report
the truncation gap as a diagnostic so downstream code can flag biased runs.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from qantis_engine.risk.tail_probability import tail_probability


@dataclass
class DelayTailReport:
    """Result of a delay tail probability estimation."""

    tail_probability: float
    credible_interval: tuple[float, float]
    threshold: float
    horizon: int
    truncation_bias_estimate: float
    sample_cost: int
    mode: str
    diagnostics: dict[str, Any]


def estimate_delay_tail(
    cycle_sampler: Callable[[np.random.Generator, int], float],
    threshold: float,
    horizon: int,
    mode: str = "biqae",
    n_samples: int = 1024,
    confidence: float = 0.95,
    seed: int = 0,
) -> DelayTailReport:
    """Estimate ``P(delay >= threshold)`` from truncated regenerative cycles.

    Parameters
    ----------
    cycle_sampler : ``(rng, horizon) -> delay`` returning the maximum delay
        observed in one regenerative cycle of length up to ``horizon``.
    threshold : tail threshold (delay value).
    horizon : per-cycle truncation horizon. Larger horizon -> smaller
        truncation bias -> deeper effective oracle.
    mode : "classical" or "biqae".
    n_samples : sample budget.
    confidence : credibility level.
    seed : deterministic seed.
    """
    rng_seed = np.random.default_rng(seed)

    def sample_with_horizon(rng: np.random.Generator) -> float:
        return cycle_sampler(rng, horizon)

    def event(delay: float) -> bool:
        return float(delay) >= threshold

    n_bias = max(64, n_samples // 16)
    long_h = horizon * 2
    short_count = sum(
        1 for _ in range(n_bias) if cycle_sampler(rng_seed, horizon) >= threshold
    )
    long_count = sum(
        1 for _ in range(n_bias) if cycle_sampler(rng_seed, long_h) >= threshold
    )
    bias_est = float(abs(short_count - long_count) / max(n_bias, 1))

    rep = tail_probability(
        sampler=sample_with_horizon,
        event=event,
        mode=mode,
        n_samples=n_samples,
        confidence=confidence,
        seed=seed,
    )

    return DelayTailReport(
        tail_probability=rep.estimate,
        credible_interval=rep.credible_interval,
        threshold=threshold,
        horizon=horizon,
        truncation_bias_estimate=bias_est,
        sample_cost=rep.sample_cost + 2 * n_bias,
        mode=rep.mode,
        diagnostics={
            "n_bias_samples_per_horizon": n_bias,
            "compared_horizons": (horizon, long_h),
            **rep.diagnostics,
        },
    )
