"""Tests for the verify module."""
from __future__ import annotations

import numpy as np

from qantis_engine.verify import (
    check_assignment,
    closed_loop_regret,
    lp_assignment_lower_bound,
    posterior_fidelity,
)
from qantis_engine.verify.dual_bound import lagrangian_dual_lower_bound


def test_check_assignment_accepts_feasible() -> None:
    rep = check_assignment(
        assignments=[(0, 1), (1, 0), (2, 2)],
        n_tracks=3,
        n_meas=3,
        missed_detections=[],
        false_alarms=[],
    )
    assert rep.feasible
    assert rep.violations == []


def test_check_assignment_rejects_double_assignment() -> None:
    rep = check_assignment(
        assignments=[(0, 0), (1, 0)],
        n_tracks=2,
        n_meas=2,
        missed_detections=[],
        false_alarms=[1],
    )
    assert not rep.feasible
    assert any("measurement 0 assigned 2 times" in v for v in rep.violations)


def test_check_assignment_flags_inconsistent_missed_list() -> None:
    rep = check_assignment(
        assignments=[(0, 0)],
        n_tracks=2,
        n_meas=2,
        missed_detections=[],
        false_alarms=[1],
    )
    assert not rep.feasible
    assert any("missed_detections" in v for v in rep.violations)


def test_lp_lower_bound_is_optimal_for_assignment() -> None:
    cost = np.array(
        [[1.0, 4.0, 5.0],
         [4.0, 1.0, 5.0],
         [5.0, 5.0, 1.0]],
        dtype=np.float64,
    )
    rep = lp_assignment_lower_bound(cost)
    assert rep.lower_bound == 3.0
    assert rep.is_tight


def test_lagrangian_dual_lower_bound_runs() -> None:
    cost = np.array([2.0, 1.0, 3.0, 1.0])
    coupling = np.array([[1.0, 1.0, 1.0, 1.0]])
    rhs = np.array([2.0])
    rep = lagrangian_dual_lower_bound(cost, coupling, rhs, n_iters=32, step=0.1)
    assert rep.method == "lagrangian_subgradient"
    assert isinstance(rep.lower_bound, float)


def test_closed_loop_regret_zero_when_pipeline_matches_oracle() -> None:
    rep = closed_loop_regret(
        pipeline_rewards=[1.0, 2.0, 0.5],
        oracle_rewards=[1.0, 2.0, 0.5],
    )
    assert rep.cumulative_regret == 0.0
    assert rep.n_steps == 3


def test_posterior_fidelity_hellinger_zero_when_equal() -> None:
    p = np.array([0.3, 0.7])
    rep = posterior_fidelity(p, p, threshold=0.01)
    assert rep.hellinger < 1e-12
    assert rep.is_tight
