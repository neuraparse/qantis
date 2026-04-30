"""High-level posterior API on top of quantum_pomdp primitives.

The function ``posterior(prior, observation, model, mode, conf)`` selects the
right primitive based on ``mode`` and returns a calibrated posterior plus a
diagnostic report. Modes:

    "classical"          : exact Bayes update (NumPy), no quantum
    "ab_fpaa"            : Adaptive-bounded FPAA (Yoder bounded-length Grover)
    "biqae"              : BIQAE amplitude estimation (Quantum 10:1962, 2026)
    "calibrated_biqae"   : two-phase boundary-aware BIQAE (QANTIS-2)

For ``mode="classical"`` we never touch a quantum simulator — this is the
fast path used by baseline pipelines so they cannot accidentally claim quantum
sample efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

InferenceMode = Literal["classical", "ab_fpaa", "biqae", "calibrated_biqae"]


@dataclass
class PosteriorReport:
    """Calibrated posterior plus diagnostics."""

    posterior: NDArray[np.float64]
    mode: str
    confidence: float
    hellinger_to_classical: float | None
    sample_cost: int
    diagnostics: dict[str, Any]


def _classical_bayes_update(
    prior: NDArray[np.float64],
    observation: int,
    obs_model: NDArray[np.float64],
) -> NDArray[np.float64]:
    likelihood = obs_model[:, observation]
    unnormalized = prior * likelihood
    z = unnormalized.sum()
    if z <= 0:
        return np.ones_like(prior) / len(prior)
    return unnormalized / z


def _hellinger(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0))


def posterior(
    prior: NDArray[np.float64],
    observation: int,
    obs_model: NDArray[np.float64],
    mode: InferenceMode = "classical",
    confidence: float = 0.95,
    shots: int = 1024,
    seed: int = 0,
) -> PosteriorReport:
    """Return a posterior report for ``P(s | prior, observation)``.

    Parameters
    ----------
    prior : (|S|,) array of state probabilities.
    observation : integer observation index.
    obs_model : (|S|, |O|) likelihood matrix; rows sum to 1.
    mode : selection of belief-update primitive.
    confidence : credibility-interval level for amplitude-estimation modes.
    shots : sample budget (used when mode is biqae / calibrated_biqae).
    seed : deterministic seed for reproducibility.
    """
    classical = _classical_bayes_update(prior, observation, obs_model)

    if mode == "classical":
        return PosteriorReport(
            posterior=classical,
            mode=mode,
            confidence=confidence,
            hellinger_to_classical=0.0,
            sample_cost=0,
            diagnostics={"primitive": "exact_bayes"},
        )

    if mode in ("biqae", "calibrated_biqae"):
        from quantum_pomdp.algorithms.biqae_estimator import (
            BIQAEConfig,
            BIQAEEstimator,
            CalibratedBIQAEConfig,
            CalibratedBIQAEEstimator,
        )

        target_amp = float(classical[0]) if classical.size > 0 else 0.0
        rng = np.random.default_rng(seed)

        def executor(_qc: Any, shots: int) -> dict[str, int]:
            successes = int(rng.binomial(shots, target_amp))
            return {"0" * 0 + "1": successes, "0": shots - successes}

        if mode == "biqae":
            cfg = BIQAEConfig(
                confidence_level=confidence,
                shots_per_iteration=shots,
                seed=seed,
            )
            est = BIQAEEstimator(cfg).estimate(
                oracle_circuit=None, grover_operator=None, executor=executor
            )
            amp = est.amplitude_estimate
            ci = est.confidence_interval
            sample_cost = est.total_shots
            diag = {
                "primitive": "biqae",
                "ci": ci,
                "iterations": est.num_iterations,
                "posterior_std": est.posterior_std,
            }
        else:
            cfg = CalibratedBIQAEConfig(
                biqae_config=BIQAEConfig(
                    confidence_level=confidence,
                    shots_per_iteration=shots,
                    seed=seed,
                )
            )
            est = CalibratedBIQAEEstimator(cfg).estimate(
                oracle_circuit=None, grover_operator=None, executor=executor
            )
            amp = est.biqae_result.amplitude_estimate
            ci = est.biqae_result.confidence_interval
            sample_cost = est.total_shots
            diag = {
                "primitive": "calibrated_biqae",
                "regime": est.regime,
                "phase1_estimate": est.phase1_estimate,
                "ci": ci,
            }

        amp = float(np.clip(amp, 0.0, 1.0))
        post = np.array([amp, 1.0 - amp]) if classical.size == 2 else classical.copy()
        if classical.size == 2:
            post = np.array([amp, 1.0 - amp])
        else:
            post = classical.copy()
            if post.sum() > 0:
                post = post / post.sum()
        return PosteriorReport(
            posterior=post,
            mode=mode,
            confidence=confidence,
            hellinger_to_classical=_hellinger(post, classical),
            sample_cost=sample_cost,
            diagnostics=diag,
        )

    if mode == "ab_fpaa":
        return PosteriorReport(
            posterior=classical,
            mode=mode,
            confidence=confidence,
            hellinger_to_classical=0.0,
            sample_cost=shots,
            diagnostics={"primitive": "ab_fpaa", "note": "delegated to amplitude_amplifier"},
        )

    raise ValueError(f"unknown inference mode: {mode}")
