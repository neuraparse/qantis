"""Tests for quantum_mht.experiments module.

Validates the experiment runners for QUBO scaling analysis,
annealing vs QAOA comparison, and classical vs quantum tracking.
"""

import numpy as np
import pytest

from quantum_mht.experiments.exp_qubo_scaling import run_scaling_experiment, ScalingResult
from quantum_mht.experiments.exp_annealing_vs_qaoa import run_comparison
from quantum_mht.experiments.exp_classical_vs_quantum import run_tracking_experiment


class TestScalingResult:
    """Test ScalingResult data class."""

    def test_creation(self) -> None:
        """ScalingResult should store all fields."""
        result = ScalingResult(
            n_targets=5,
            n_measurements=8,
            n_qubo_variables=53,
            penalty=10.0,
            build_time_s=0.01,
            solve_time_s=0.02,
            solver_name="hungarian",
            objective_value=-5.0,
            is_feasible=True,
        )
        assert result.n_targets == 5
        assert result.n_measurements == 8
        assert result.n_qubo_variables == 53
        assert result.penalty == 10.0
        assert result.solver_name == "hungarian"
        assert result.is_feasible is True


class TestRunScalingExperiment:
    """Test run_scaling_experiment function."""

    def test_small_target_counts(self) -> None:
        """Scaling experiment with small target counts should produce results."""
        results = run_scaling_experiment(
            target_counts=[2, 3],
            solvers=["hungarian"],
            seed=42,
        )
        assert len(results) > 0
        assert all(isinstance(r, ScalingResult) for r in results)

    def test_result_fields_populated(self) -> None:
        """All ScalingResult fields should have meaningful values."""
        results = run_scaling_experiment(
            target_counts=[2],
            solvers=["hungarian"],
            seed=42,
        )
        assert len(results) >= 1
        r = results[0]
        assert r.n_targets == 2
        assert r.n_measurements >= 2
        assert r.n_qubo_variables > 0
        assert r.penalty >= 0.0
        assert r.build_time_s >= 0.0
        assert r.solve_time_s >= 0.0
        assert r.solver_name == "hungarian"

    def test_increasing_problem_size(self) -> None:
        """Larger target counts should produce larger QUBO variable counts."""
        results = run_scaling_experiment(
            target_counts=[2, 3],
            solvers=["hungarian"],
            seed=42,
        )
        vars_by_size = {r.n_targets: r.n_qubo_variables for r in results}
        if 2 in vars_by_size and 3 in vars_by_size:
            assert vars_by_size[3] > vars_by_size[2]

    def test_multiple_solvers(self) -> None:
        """Should produce results for each solver specified."""
        results = run_scaling_experiment(
            target_counts=[2],
            solvers=["hungarian", "gnn"],
            seed=42,
        )
        solver_names = {r.solver_name for r in results}
        assert "hungarian" in solver_names
        assert "gnn" in solver_names


class TestRunComparison:
    """Test run_comparison (annealing vs QAOA comparison)."""

    def test_small_comparison(self) -> None:
        """Comparison with small target count should produce results."""
        results = run_comparison(
            target_counts=[2],
            seed=42,
        )
        # Should at least get results for hungarian and gnn baselines
        assert len(results) > 0

    def test_results_have_solver_names(self) -> None:
        """Each result should have a solver_name."""
        results = run_comparison(
            target_counts=[2],
            seed=42,
        )
        for r in results:
            assert r.solver_name is not None
            assert isinstance(r.solver_name, str)

    def test_results_have_feasibility(self) -> None:
        """Each result should have an is_feasible field."""
        results = run_comparison(
            target_counts=[2],
            seed=42,
        )
        for r in results:
            assert isinstance(r.is_feasible, bool)


class TestRunTrackingExperiment:
    """Test run_tracking_experiment (classical vs quantum end-to-end)."""

    def test_small_tracking_experiment(self) -> None:
        """Tracking experiment with small parameters should complete."""
        from quantum_mht.simulation.scenario_generator import crossing_targets

        scenarios = {"crossing_2": crossing_targets(n_targets=2)}
        results = run_tracking_experiment(
            scenarios=scenarios,
            solver_names=["hungarian"],
            num_steps=5,
            seed=42,
        )
        assert len(results) > 0

    def test_tracking_result_fields(self) -> None:
        """TrackingResult should have all expected fields populated."""
        from quantum_mht.simulation.scenario_generator import crossing_targets

        scenarios = {"crossing_2": crossing_targets(n_targets=2)}
        results = run_tracking_experiment(
            scenarios=scenarios,
            solver_names=["hungarian"],
            num_steps=3,
            seed=42,
        )
        assert len(results) >= 1
        r = results[0]
        assert r.scenario_name == "crossing_2"
        assert r.solver_name == "hungarian"
        assert r.num_steps == 3
        assert r.total_time_s >= 0.0
        assert r.avg_step_time_s >= 0.0

    def test_multiple_solvers_tracking(self) -> None:
        """Should produce results for each solver."""
        from quantum_mht.simulation.scenario_generator import crossing_targets

        scenarios = {"crossing_2": crossing_targets(n_targets=2)}
        results = run_tracking_experiment(
            scenarios=scenarios,
            solver_names=["hungarian", "gnn"],
            num_steps=3,
            seed=42,
        )
        solver_names = {r.solver_name for r in results}
        assert "hungarian" in solver_names
        assert "gnn" in solver_names
