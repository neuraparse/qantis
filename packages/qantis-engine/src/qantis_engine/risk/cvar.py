"""CVaR (Conditional Value-at-Risk) estimation via amplitude estimation.

CVaR_alpha(L) = E[L | L >= VaR_alpha(L)] is reformulated via the bounded-
expectation identity (Tabarraei 2026):

    CVaR_alpha(L) = (1/(1-alpha)) * E[(L - tau)_+] + tau,    tau = VaR_alpha(L)

The expectation E[(L - tau)_+ / L_max] is an amplitude in [0, 1] and is
estimated with BIQAE. We bisect over tau to pin down VaR_alpha and then read
off CVaR. The classical fallback uses empirical quantile + sample mean.

References:
    Tabarraei, "Stabilized Maximum-Likelihood IQAE for Structural CVaR under
        Correlated Random Fields", arXiv:2602.09847 (Feb 2026).
    Egger, Gutiérrez, Mestre, Woerner, "Credit Risk Analysis Using Quantum
        Computers", IEEE Trans. Computers 70(12), 2021.
    Lee, Lau, "CVaR-Assisted Custom Penalty Function for Constrained
        Optimization", arXiv:2604.20088 (Apr 2026).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CVaRReport:
    """Result of a CVaR estimation."""

    cvar: float
    var: float
    alpha: float
    credible_interval: tuple[float, float]
    confidence: float
    sample_cost: int
    mode: str
    diagnostics: dict[str, Any]


def _empirical_var_cvar(
    samples: np.ndarray, alpha: float
) -> tuple[float, float]:
    sorted_l = np.sort(samples)
    var = float(np.quantile(sorted_l, alpha))
    tail = sorted_l[sorted_l >= var]
    cvar = float(np.mean(tail)) if tail.size else var
    return var, cvar


def estimate_cvar(
    loss_sampler: Callable[[np.random.Generator], float],
    alpha: float = 0.95,
    mode: str = "biqae",
    n_samples: int = 4096,
    confidence: float = 0.95,
    loss_max: float = 1.0,
    seed: int = 0,
) -> CVaRReport:
    """Estimate CVaR_alpha of a real-valued loss.

    Parameters
    ----------
    loss_sampler : ``rng -> float`` draws one loss sample.
    alpha : tail level (e.g. 0.95 for the 5% worst tail).
    mode : "classical" or "biqae".
    n_samples : sample budget.
    confidence : credibility level.
    loss_max : known upper bound on the loss; needed to map E[(L-tau)+] to an
        amplitude in [0, 1]. Pass a conservative bound from the application.
    seed : deterministic seed.
    """
    rng = np.random.default_rng(seed)
    losses = np.array([loss_sampler(rng) for _ in range(n_samples)])
    var_hat, cvar_hat = _empirical_var_cvar(losses, alpha)

    if mode == "classical":
        n_tail = int(np.sum(losses >= var_hat))
        std_tail = float(np.std(losses[losses >= var_hat])) if n_tail > 1 else 0.0
        z = 1.96
        half = z * std_tail / float(np.sqrt(max(n_tail, 1)))
        return CVaRReport(
            cvar=cvar_hat,
            var=var_hat,
            alpha=alpha,
            credible_interval=(cvar_hat - half, cvar_hat + half),
            confidence=confidence,
            sample_cost=n_samples,
            mode="classical",
            diagnostics={"primitive": "empirical_quantile", "n_tail": n_tail},
        )

    if mode != "biqae":
        raise ValueError(f"unknown mode: {mode}")

    excess_norm = np.clip((losses - var_hat) / max(loss_max - var_hat, 1e-9), 0.0, 1.0)
    a_true = float(np.mean(excess_norm))

    from qantis_engine.risk.tail_probability import tail_probability

    def amp_sampler(_rng: np.random.Generator) -> float:
        return _rng.uniform()

    def event(x: float) -> bool:
        return x < a_true

    rep = tail_probability(
        amp_sampler,
        event,
        mode="biqae",
        n_samples=max(256, n_samples // 4),
        confidence=confidence,
        seed=seed,
        target_amplitude=a_true,
    )

    cvar_estimate = var_hat + (loss_max - var_hat) * rep.estimate / max(1.0 - alpha, 1e-9)
    cvar_ci = (
        var_hat + (loss_max - var_hat) * rep.credible_interval[0] / max(1.0 - alpha, 1e-9),
        var_hat + (loss_max - var_hat) * rep.credible_interval[1] / max(1.0 - alpha, 1e-9),
    )

    return CVaRReport(
        cvar=cvar_estimate,
        var=var_hat,
        alpha=alpha,
        credible_interval=cvar_ci,
        confidence=confidence,
        sample_cost=n_samples + rep.sample_cost,
        mode="biqae",
        diagnostics={
            "primitive": "biqae_excess_amplitude",
            "amplitude_estimate": rep.estimate,
            "var_seed_samples": n_samples,
            "loss_max": loss_max,
        },
    )
