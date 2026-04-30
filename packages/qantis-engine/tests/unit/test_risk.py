"""Tests for qantis_engine.risk module."""
from __future__ import annotations

from typing import Any

import numpy as np

from qantis_engine.risk import (
    ChanceConstraint,
    estimate_cvar,
    estimate_delay_tail,
    tail_probability,
)


def test_tail_probability_classical_in_credible_interval() -> None:
    rng_state = {"p": 0.05}

    def world(rng: np.random.Generator) -> bool:
        return bool(rng.uniform() < rng_state["p"])

    def event(s: bool) -> bool:
        return s

    rep = tail_probability(
        sampler=world, event=event, mode="classical",
        n_samples=2048, confidence=0.95, seed=1,
    )
    assert 0.0 <= rep.estimate <= 0.2
    lo, hi = rep.credible_interval
    assert lo <= rep.estimate <= hi
    assert rep.sample_cost == 2048
    assert rep.mode == "classical"


def test_tail_probability_biqae_with_target_amplitude() -> None:
    target = 0.07

    def world(rng: np.random.Generator) -> bool:
        return bool(rng.uniform() < target)

    def event(s: bool) -> bool:
        return s

    rep = tail_probability(
        sampler=world, event=event, mode="biqae",
        n_samples=512, confidence=0.95, seed=42,
        target_amplitude=target,
    )
    assert 0.0 <= rep.estimate <= 1.0
    assert rep.credible_interval[0] <= rep.estimate <= rep.credible_interval[1]
    assert rep.sample_cost > 0
    assert rep.mode == "biqae"


def test_estimate_cvar_classical_recovers_empirical() -> None:
    def loss(rng: np.random.Generator) -> float:
        return float(rng.exponential(scale=1.0))

    rep = estimate_cvar(
        loss_sampler=loss, alpha=0.95, mode="classical",
        n_samples=4096, confidence=0.95, loss_max=8.0, seed=0,
    )
    assert rep.cvar > rep.var
    assert rep.alpha == 0.95
    assert rep.sample_cost == 4096


def test_estimate_cvar_biqae_runs() -> None:
    def loss(rng: np.random.Generator) -> float:
        return float(rng.exponential(scale=1.0))

    rep = estimate_cvar(
        loss_sampler=loss, alpha=0.9, mode="biqae",
        n_samples=512, confidence=0.9, loss_max=5.0, seed=2,
    )
    assert rep.cvar >= 0.0
    assert rep.alpha == 0.9
    assert rep.mode == "biqae"


def test_chance_constraint_satisfied_when_probability_low() -> None:
    target = 0.02

    def world(rng: np.random.Generator) -> Any:
        return bool(rng.uniform() < target)

    def event(decision: Any, sample: bool) -> bool:
        return sample

    cc = ChanceConstraint(
        name="collision",
        event=event,
        world=world,
        prob_bound=0.10,
        confidence=0.9,
        mode="classical",
        n_samples=4096,
    )
    rep = cc.check(decision={"foo": 1}, seed=3)
    assert rep.estimated_probability < 0.10
    assert rep.satisfied
    assert rep.margin > 0


def test_chance_constraint_violated_when_probability_high() -> None:
    target = 0.5

    def world(rng: np.random.Generator) -> Any:
        return bool(rng.uniform() < target)

    def event(decision: Any, sample: bool) -> bool:
        return sample

    cc = ChanceConstraint(
        name="collision",
        event=event,
        world=world,
        prob_bound=0.05,
        confidence=0.95,
        mode="classical",
        n_samples=2048,
    )
    rep = cc.check(decision={"foo": 1}, seed=4)
    assert not rep.satisfied
    assert rep.margin < 0


def test_delay_tail_truncation_bias_reported() -> None:
    rng = np.random.default_rng(0)

    def cycle_sampler(_rng: np.random.Generator, horizon: int) -> float:
        return float(min(rng.exponential(scale=2.0), horizon))

    rep = estimate_delay_tail(
        cycle_sampler=cycle_sampler,
        threshold=4.0,
        horizon=10,
        mode="classical",
        n_samples=1024,
        confidence=0.9,
        seed=5,
    )
    assert 0.0 <= rep.tail_probability <= 1.0
    assert rep.truncation_bias_estimate >= 0.0
    assert rep.horizon == 10
