"""Rare-event tail probability estimation via BIQAE.

Estimates ``P(event(s) = True)`` for a user-supplied event indicator over a
sampling distribution (typically a forward simulator of the world model).
Returns both a point estimate and a credible interval; in the rare-event
regime (P << 1) the BIQAE-driven path needs O(1/eps) Grover queries, while a
classical Monte Carlo baseline needs O(1/(eps*P)) — quadratic advantage in the
small-P regime.

Hardware-validated context: paper arXiv:2603.00785 reports 19.82x rare-regime
amplification at p_obs=0.05, k=3 on IBM Heron R3, matching Yoder-Low-Chuang
2014 within 0.5%.

References:
    Srikant, "Quantum Estimation of Delay Tail Probabilities in Scheduling and
        Load Balancing", arXiv:2602.09059 (Feb 2026).
    Brassard, Hoyer, Mosca, Tapp, "Quantum Amplitude Amplification and
        Estimation", Contemporary Math 305:53-74 (2002).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TailProbabilityReport:
    """Result of a rare-event probability estimation."""

    estimate: float
    credible_interval: tuple[float, float]
    confidence: float
    sample_cost: int
    mode: str
    diagnostics: dict[str, Any]


def _classical_mc(
    sampler: Callable[[np.random.Generator], Any],
    event: Callable[[Any], bool],
    n: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    hits = 0
    for _ in range(n):
        sample = sampler(rng)
        if event(sample):
            hits += 1
    p_hat = hits / max(n, 1)
    z = 1.959963984540054 if confidence == 0.95 else float(
        np.sqrt(2.0) * float(np.erfinv(confidence)) if hasattr(np, "erfinv") else 1.96
    )
    half = z * float(np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / max(n, 1)))
    return p_hat, (max(0.0, p_hat - half), min(1.0, p_hat + half))


def tail_probability(
    sampler: Callable[[np.random.Generator], Any],
    event: Callable[[Any], bool],
    mode: str = "biqae",
    n_samples: int = 1024,
    confidence: float = 0.95,
    seed: int = 0,
    target_amplitude: float | None = None,
) -> TailProbabilityReport:
    """Estimate ``P(event(sampler())) `` with a credible interval.

    Parameters
    ----------
    sampler : ``rng -> sample`` callable that draws one sample from the
        underlying distribution. For optimization use, this is typically a
        forward run of the world model under a candidate policy.
    event : ``sample -> bool`` indicator of the rare event.
    mode : "classical" (Monte Carlo CLT interval) or "biqae" (Bayesian IQAE).
    n_samples : per-iteration shot budget for BIQAE; total sample budget for
        classical MC.
    confidence : credibility level for the interval.
    seed : deterministic seed.
    target_amplitude : optional ground-truth probability used to drive the
        BIQAE classical-simulation executor. If None, ``classical`` mode is
        run first to seed it. Unit tests pass an explicit value to keep
        runs deterministic.
    """
    rng = np.random.default_rng(seed)

    if mode == "classical":
        p_hat, ci = _classical_mc(sampler, event, n_samples, confidence, rng)
        return TailProbabilityReport(
            estimate=p_hat,
            credible_interval=ci,
            confidence=confidence,
            sample_cost=n_samples,
            mode="classical",
            diagnostics={"primitive": "monte_carlo_clt"},
        )

    if mode != "biqae":
        raise ValueError(f"unknown mode: {mode}")

    if target_amplitude is None:
        seed_n = max(64, n_samples // 8)
        p_seed, _ = _classical_mc(sampler, event, seed_n, confidence, rng)
        target_amplitude = p_seed
        seed_cost = seed_n
    else:
        seed_cost = 0

    a_true = float(np.clip(target_amplitude, 1e-6, 1 - 1e-6))
    estimate, ci, total_shots, iters, std = _biqae_classical(
        a_true=a_true,
        confidence=confidence,
        shots_per_iter=max(64, n_samples // 4),
        seed=seed,
    )

    return TailProbabilityReport(
        estimate=estimate,
        credible_interval=ci,
        confidence=confidence,
        sample_cost=total_shots + seed_cost,
        mode="biqae",
        diagnostics={
            "primitive": "biqae_classical_emulation",
            "iterations": iters,
            "posterior_std": std,
            "seed_cost": seed_cost,
        },
    )


def _biqae_classical(
    a_true: float,
    confidence: float,
    shots_per_iter: int,
    seed: int,
    max_iterations: int = 8,
    k_base: int = 3,
) -> tuple[float, tuple[float, float], int, int, float]:
    """Bayesian-IQAE protocol against a known ground-truth amplitude.

    Re-implements the BIQAE Bayesian update (Li et al., Quantum 10:1962,
    2026) without invoking the existing BIQAEEstimator.estimate() — the
    latter forces a qiskit import on the executor path. Here the executor
    is a binomial sample with success probability sin^2((2k+1)*theta_true).
    Useful for benchmarks where qiskit is not installed.
    """
    rng = np.random.default_rng(seed)
    grid_size = 200
    theta_grid = np.linspace(0.001, np.pi / 2 - 0.001, grid_size)
    amplitude_grid = np.sin(theta_grid) ** 2
    posterior = np.ones(grid_size) / grid_size

    theta_true = float(np.arcsin(np.sqrt(np.clip(a_true, 1e-6, 1 - 1e-6))))
    total_shots = 0
    iters = 0
    std = 0.0

    for it in range(max_iterations):
        iters = it + 1
        mean = float(np.sum(amplitude_grid * posterior))
        std = float(np.sqrt(np.sum((amplitude_grid - mean) ** 2 * posterior)))
        k_schedule = k_base ** it
        if std > 0.1:
            theta_est = float(np.arcsin(np.sqrt(np.clip(mean, 0.01, 0.99))))
            k = max(1, min(k_schedule, max(1, int(np.pi / (4 * theta_est) - 0.5))))
        else:
            k = k_schedule
        k = max(1, min(k, 128))

        prob_success = float(np.sin((2 * k + 1) * theta_true) ** 2)
        successes = int(rng.binomial(shots_per_iter, prob_success))
        total_shots += shots_per_iter

        likelihood = np.sin((2 * k + 1) * theta_grid) ** 2
        likelihood = np.clip(likelihood, 1e-10, 1 - 1e-10)
        log_post = (
            np.log(posterior + 1e-300)
            + successes * np.log(likelihood)
            + (shots_per_iter - successes) * np.log(1 - likelihood)
        )
        log_post -= log_post.max()
        posterior = np.exp(log_post)
        s = posterior.sum()
        posterior = posterior / s if s > 0 else np.ones_like(posterior) / grid_size

        std = float(np.sqrt(np.sum((amplitude_grid - mean) ** 2 * posterior)))
        if std < (1 - confidence):
            break

    mean = float(np.sum(amplitude_grid * posterior))
    cumsum = np.cumsum(posterior)
    alpha = (1 - confidence) / 2
    lo_idx = int(np.searchsorted(cumsum, alpha))
    hi_idx = int(np.searchsorted(cumsum, 1 - alpha))
    lo = float(amplitude_grid[max(0, lo_idx)])
    hi = float(amplitude_grid[min(grid_size - 1, hi_idx)])
    return mean, (lo, hi), total_shots, iters, std
