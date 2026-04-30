"""Risk estimation primitives for the QANTIS Decision Engine.

The risk layer turns a calibrated belief state into event probabilities and
tail-risk diagnostics that can be consumed by an optimizer.  It deliberately
targets stochastic decision loops rather than deterministic MIP solving: the
input is a posterior belief and an event oracle, and the output is a calibrated
probability/risk report.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

EstimatorMode = Literal["classical_exact", "classical_mc", "biqae_boundary_aware"]


@dataclass(frozen=True)
class EventProbabilityResult:
    """Estimated probability of a decision-relevant event."""

    event_name: str
    probability: float
    confidence_interval: tuple[float, float]
    confidence: float
    estimator_mode: EstimatorMode
    sample_budget: int
    oracle_call_hint: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TailRiskResult:
    """Weighted value-at-risk and conditional value-at-risk report."""

    alpha: float
    value_at_risk: float
    cvar: float
    expected_loss: float
    tail_probability: float


class QANTISRisk:
    """Belief-native event and tail-risk estimator.

    The implementation is intentionally backend-neutral.  In hardware mode the
    same API can be wired to boundary-aware BIQAE; in local mode it produces the
    exact belief-weighted event probability plus a binomial confidence envelope
    for a specified sample budget.
    """

    def estimate_event_probability(
        self,
        belief: Sequence[float] | Any,
        event: Sequence[bool] | Sequence[float] | Callable[[int], bool | float],
        *,
        event_name: str = "event",
        confidence: float = 0.95,
        sample_budget: int = 4096,
        mode: EstimatorMode = "classical_exact",
        target_half_width: float = 0.01,
    ) -> EventProbabilityResult:
        """Estimate ``P(event | belief)`` with a confidence envelope.

        ``event`` can be a per-state indicator/probability vector or a callable
        from state index to indicator/probability.  Values are clipped to
        ``[0, 1]`` so soft event oracles, such as collision probabilities from a
        simulator, are supported directly.
        """

        probs = _as_probability_vector(belief)
        weights = _event_weights(event, len(probs))
        probability = float(np.dot(probs, weights))
        probability = float(np.clip(probability, 0.0, 1.0))

        ci = _wilson_interval(probability, max(1, sample_budget), confidence)
        mc_samples = _mc_samples_for_half_width(probability, target_half_width, confidence)
        ae_hint = _amplitude_estimation_query_hint(
            probability, target_half_width, confidence
        )

        oracle_call_hint = mc_samples if mode != "biqae_boundary_aware" else ae_hint
        return EventProbabilityResult(
            event_name=event_name,
            probability=probability,
            confidence_interval=ci,
            confidence=confidence,
            estimator_mode=mode,
            sample_budget=sample_budget,
            oracle_call_hint=oracle_call_hint,
            metadata={
                "classical_mc_samples_for_target_width": mc_samples,
                "amplitude_estimation_query_hint": ae_hint,
                "target_half_width": target_half_width,
                "rare_event": probability < 0.05,
            },
        )

    def estimate_cvar(
        self,
        belief: Sequence[float] | Any,
        losses: Sequence[float],
        *,
        alpha: float = 0.95,
    ) -> TailRiskResult:
        """Compute weighted VaR/CVaR from posterior state losses.

        For safety-critical planning, this lets the optimizer reason about tail
        loss under the current posterior instead of only optimizing expected
        cost.
        """

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        probs = _as_probability_vector(belief)
        loss_vec = np.asarray(losses, dtype=float)
        if loss_vec.shape != probs.shape:
            raise ValueError("losses must have the same length as belief")

        order = np.argsort(loss_vec)
        sorted_losses = loss_vec[order]
        sorted_probs = probs[order]
        cdf = np.cumsum(sorted_probs)
        var_index = int(np.searchsorted(cdf, alpha, side="left"))
        var_index = min(var_index, len(sorted_losses) - 1)
        value_at_risk = float(sorted_losses[var_index])

        tail_mask = sorted_losses >= value_at_risk
        tail_probability = float(np.sum(sorted_probs[tail_mask]))
        if tail_probability <= 1e-12:
            cvar = value_at_risk
        else:
            cvar = float(
                np.sum(sorted_losses[tail_mask] * sorted_probs[tail_mask])
                / tail_probability
            )

        expected_loss = float(np.dot(probs, loss_vec))
        return TailRiskResult(
            alpha=alpha,
            value_at_risk=value_at_risk,
            cvar=cvar,
            expected_loss=expected_loss,
            tail_probability=tail_probability,
        )


def _as_probability_vector(belief: Sequence[float] | Any) -> NDArray[np.float64]:
    if hasattr(belief, "probabilities"):
        arr = np.asarray(belief.probabilities, dtype=float)
    else:
        arr = np.asarray(belief, dtype=float)
    if arr.ndim != 1:
        raise ValueError("belief must be a one-dimensional probability vector")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("belief probabilities must have positive mass")
    arr = np.clip(arr / total, 0.0, 1.0)
    return arr / arr.sum()


def _event_weights(
    event: Sequence[bool] | Sequence[float] | Callable[[int], bool | float],
    num_states: int,
) -> NDArray[np.float64]:
    if callable(event):
        values = [float(event(i)) for i in range(num_states)]
    else:
        values = [float(v) for v in event]
    arr = np.asarray(values, dtype=float)
    if arr.shape != (num_states,):
        raise ValueError("event must provide one value per belief state")
    return np.clip(arr, 0.0, 1.0)


def _normal_quantile(confidence: float) -> float:
    table = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
    }
    return table.get(round(confidence, 2), 1.959963984540054)


def _wilson_interval(p_hat: float, n: int, confidence: float) -> tuple[float, float]:
    z = _normal_quantile(confidence)
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom
    return (float(max(0.0, center - radius)), float(min(1.0, center + radius)))


def _mc_samples_for_half_width(p: float, epsilon: float, confidence: float) -> int:
    epsilon = max(float(epsilon), 1e-9)
    z = _normal_quantile(confidence)
    variance = max(p * (1.0 - p), 1e-12)
    return int(math.ceil(z * z * variance / (epsilon * epsilon)))


def _amplitude_estimation_query_hint(
    p: float, epsilon: float, confidence: float
) -> int:
    """Planning hint for a BIQAE-style event oracle.

    This is not a hardware-performance claim.  It records the target scale a
    boundary-aware amplitude-estimation backend should budget against when the
    event probability is rare.
    """

    epsilon = max(float(epsilon), 1e-9)
    z = _normal_quantile(confidence)
    scale = max(math.sqrt(max(p * (1.0 - p), 1e-12)), epsilon)
    return int(math.ceil(math.pi * z * scale / (2.0 * epsilon)))

