"""Chance constraint primitive: ``P(event(x)) <= prob_bound`` with confidence.

A ``ChanceConstraint`` is a first-class object the optimizer accepts. The
optimizer uses ``constraint.check(candidate)`` to filter or repair candidate
solutions; ``check`` returns a report with the BIQAE-estimated probability and
the confidence interval, so the constraint decision is *certified* rather than
a noisy point estimate.

Why a class rather than a function: the optimizer needs to know the
constraint's confidence and sample-budget shape to decide whether to spend
more queries when the credible interval straddles the bound.

References:
    Lee, Lau, "CVaR-Assisted Custom Penalty Function", arXiv:2604.20088, 2026.
    arXiv:2512.03925 — chance-constrained Unit Commitment with quantum
        annealing (D-Wave + Gurobi precedent).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qantis_engine.risk.tail_probability import tail_probability


@dataclass
class ChanceConstraintReport:
    """Result of evaluating a chance constraint."""

    satisfied: bool
    estimated_probability: float
    credible_interval: tuple[float, float]
    sample_cost: int
    margin: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChanceConstraint:
    """A probabilistic constraint of the form ``P(event(x; world)) <= bound``.

    The constraint is evaluated by sampling ``world`` under the candidate
    decision ``x`` and counting event occurrences; the BIQAE pathway gives a
    rare-event sample-efficiency advantage when ``bound`` is small.

    Attributes
    ----------
    name : human-readable label, e.g. "collision".
    event : ``(decision, sample) -> bool`` indicator (decision is the
        candidate solution, sample is one draw from ``world``).
    world : ``rng -> sample`` simulator of the uncertain environment.
    prob_bound : the upper bound on event probability.
    confidence : confidence level for the credible interval.
    mode : "classical" or "biqae".
    n_samples : sample budget per evaluation.
    """

    name: str
    event: Callable[[Any, Any], bool]
    world: Callable[[np.random.Generator], Any]
    prob_bound: float
    confidence: float = 0.95
    mode: str = "biqae"
    n_samples: int = 1024

    def check(
        self, decision: Any, seed: int = 0, target_amplitude: float | None = None
    ) -> ChanceConstraintReport:
        """Evaluate the constraint at ``decision``."""

        def sampler(rng: np.random.Generator) -> Any:
            return self.world(rng)

        def event_fn(sample: Any) -> bool:
            return bool(self.event(decision, sample))

        rep = tail_probability(
            sampler=sampler,
            event=event_fn,
            mode=self.mode,
            n_samples=self.n_samples,
            confidence=self.confidence,
            seed=seed,
            target_amplitude=target_amplitude,
        )
        upper = rep.credible_interval[1]
        margin = self.prob_bound - rep.estimate
        return ChanceConstraintReport(
            satisfied=upper <= self.prob_bound,
            estimated_probability=rep.estimate,
            credible_interval=rep.credible_interval,
            sample_cost=rep.sample_cost,
            margin=margin,
            diagnostics={
                "name": self.name,
                "bound": self.prob_bound,
                "mode": self.mode,
                "upper_ci": upper,
            },
        )
