"""Analysis and metrics for POMDP experiments."""

from quantum_pomdp.analysis.metrics import BeliefMetrics, ExperimentComparison, PlanningMetrics
from quantum_pomdp.analysis.scalability import ScalingAnalysis

try:
    from quantum_pomdp.analysis.advisor_experiments import (
        corridor_tiger_4state_simulator_report,
        create_corridor_tiger_pomdp,
        scenario_resource_pathway_report,
        tiger_classical_baseline_report,
    )
except ModuleNotFoundError as exc:
    if exc.name != "qiskit":
        raise
    missing_qiskit = exc

    def _missing_qiskit(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError(
            "quantum_pomdp.analysis advisor helpers require the optional qiskit dependency"
        ) from missing_qiskit

    corridor_tiger_4state_simulator_report = _missing_qiskit
    create_corridor_tiger_pomdp = _missing_qiskit
    scenario_resource_pathway_report = _missing_qiskit
    tiger_classical_baseline_report = _missing_qiskit

__all__ = [
    "corridor_tiger_4state_simulator_report",
    "create_corridor_tiger_pomdp",
    "BeliefMetrics",
    "PlanningMetrics",
    "ExperimentComparison",
    "ScalingAnalysis",
    "scenario_resource_pathway_report",
    "tiger_classical_baseline_report",
]
