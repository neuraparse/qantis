from __future__ import annotations

import numpy as np
from qantis import (
    ConstraintNativeAssignmentOptimizer,
    QANTISDecisionEngine,
    QANTISRisk,
    QANTISVerify,
)


def test_risk_estimates_soft_event_probability_with_ci() -> None:
    risk = QANTISRisk()

    result = risk.estimate_event_probability(
        [0.7, 0.2, 0.1],
        [0.0, 0.5, 1.0],
        event_name="collision",
        sample_budget=2048,
        mode="biqae_boundary_aware",
    )

    assert result.event_name == "collision"
    assert np.isclose(result.probability, 0.2)
    assert result.confidence_interval[0] <= result.probability
    assert result.confidence_interval[1] >= result.probability
    assert result.metadata["amplitude_estimation_query_hint"] <= result.metadata[
        "classical_mc_samples_for_target_width"
    ]


def test_risk_estimates_weighted_cvar() -> None:
    risk = QANTISRisk()

    result = risk.estimate_cvar([0.6, 0.3, 0.1], [1.0, 3.0, 20.0], alpha=0.8)

    assert result.value_at_risk == 3.0
    assert result.cvar > result.expected_loss
    assert np.isclose(result.tail_probability, 0.4)


def test_constraint_native_assignment_returns_feasible_state_without_qubo_penalty() -> None:
    optimizer = ConstraintNativeAssignmentOptimizer(
        missed_detection_cost=4.0,
        false_alarm_cost=1.5,
    )
    costs = np.array(
        [
            [0.2, 5.0, 2.0],
            [4.0, 0.1, 3.0],
        ]
    )

    result = optimizer.solve(costs)

    assert result.is_feasible
    assert result.assignments == [(0, 0), (1, 1)]
    assert result.false_alarms == [2]
    assert np.isclose(result.objective_value, 1.8)
    assert result.metadata["uses_qubo_penalty"] is False


def test_constraint_native_assignment_respects_gates_and_missed_detection() -> None:
    optimizer = ConstraintNativeAssignmentOptimizer(missed_detection_cost=0.7)
    costs = np.array([[0.1], [0.2]])
    gates = np.array([[True], [False]])

    result = optimizer.solve(costs, gates)

    assert result.is_feasible
    assert result.assignments == [(0, 0)]
    assert result.missed_detections == [1]


def test_verify_repairs_duplicate_assignments_by_cost() -> None:
    verifier = QANTISVerify()
    costs = np.array(
        [
            [1.0, 10.0],
            [2.0, 0.5],
        ]
    )

    report = verifier.check_assignment(
        [(0, 0), (1, 0), (1, 1)],
        n_tracks=2,
        n_measurements=2,
        cost_matrix=costs,
    )

    assert report.feasible is False
    assert report.duplicate_tracks == [1]
    assert report.duplicate_measurements == [0]
    assert report.repaired_assignments == [(0, 0), (1, 1)]


def test_decision_engine_combines_optimize_verify_and_risk() -> None:
    engine = QANTISDecisionEngine()

    report = engine.optimize_assignment(
        [[0.2, 4.0], [4.0, 0.3]],
        belief=[0.99, 0.01],
        event=[0.0, 1.0],
        event_name="missed_detection",
    )

    assert report.optimization.is_feasible
    assert report.feasibility.feasible
    assert report.risk is not None
    assert np.isclose(report.risk.probability, 0.01)
    assert report.metadata["claim_scope"] == "belief-to-action loop, not general-purpose MIP"

