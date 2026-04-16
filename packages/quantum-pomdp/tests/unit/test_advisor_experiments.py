"""Tests for the publication-oriented advisor experiment helpers."""

from quantum_pomdp.analysis.advisor_experiments import (
    corridor_tiger_4state_simulator_report,
    scenario_resource_pathway_report,
    tiger_classical_baseline_report,
)


def test_tiger_baselines_listen_under_uniform_uncertainty() -> None:
    report = tiger_classical_baseline_report()
    uniform = next(item for item in report if item["belief_label"] == "uniform")
    assert uniform["recommended_action_names"]["POMCP"] == "listen"
    assert uniform["recommended_action_names"]["DESPOT"] == "listen"
    assert uniform["recommended_action_names"]["PBVI"] == "listen"


def test_tiger_baselines_open_safe_door_when_belief_is_concentrated() -> None:
    report = tiger_classical_baseline_report()
    strongly_left = next(item for item in report if item["belief_label"] == "strongly_left")
    strongly_right = next(item for item in report if item["belief_label"] == "strongly_right")

    assert strongly_left["recommended_action_names"]["POMCP"] == "open-right"
    assert strongly_left["recommended_action_names"]["DESPOT"] == "open-right"
    assert strongly_left["recommended_action_names"]["PBVI"] == "open-right"

    assert strongly_right["recommended_action_names"]["POMCP"] == "open-left"
    assert strongly_right["recommended_action_names"]["DESPOT"] == "open-left"
    assert strongly_right["recommended_action_names"]["PBVI"] == "open-left"


def test_corridor_tiger_4state_simulator_matches_classical_posterior() -> None:
    report = corridor_tiger_4state_simulator_report(shots=4096)
    assert report["num_qubits"] == 3
    assert report["hellinger_distance"] < 0.03


def test_resource_pathway_scales_with_problem_size() -> None:
    report = scenario_resource_pathway_report()
    by_label = {item["label"]: item for item in report}

    assert by_label["Tiger-2"]["total_circuit_qubits"] < by_label["CorridorTiger-4"]["total_circuit_qubits"]
    assert by_label["CorridorTiger-4"]["total_circuit_qubits"] < by_label["GridNavigation-4x4"]["total_circuit_qubits"]
    assert by_label["GridNavigation-4x4"]["total_circuit_qubits"] < by_label["GPSDenied-6x6"]["total_circuit_qubits"]

    assert by_label["GridNavigation-4x4"]["classical_queries_per_accept"] > by_label["Tiger-2"]["classical_queries_per_accept"]
    assert by_label["GPSDenied-6x6"]["asymptotic_sample_efficiency_gain"] > by_label["GPSDenied-4x4"]["asymptotic_sample_efficiency_gain"]
