"""Tests for MTDA solvers."""

import numpy as np
import pytest

from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder
from quantum_mht.solvers.solver_factory import create_solver


@pytest.fixture
def small_qubo():
    """Create a small QUBO for testing."""
    builder = MTDAQuboBuilder()
    cost_matrix = np.array([[1.0, 5.0], [5.0, 1.0]])
    return builder.build_from_cost_matrix(cost_matrix)


class TestAnnealingSolver:
    def test_simulated_annealing(self, small_qubo) -> None:
        try:
            solver = create_solver("annealing", use_simulator=True, num_reads=100)
            result = solver.solve(small_qubo)
            assert result.solve_time_s >= 0
            assert result.solver_name == "DWaveAnnealing"
        except ImportError:
            pytest.skip("neal not installed")


class TestHungarianSolver:
    def test_optimal_assignment(self, small_qubo) -> None:
        solver = create_solver("hungarian")
        result = solver.solve(small_qubo)
        assert result.is_feasible
        assert len(result.assignments) > 0
        assert result.solver_name == "Hungarian"

    def test_assignments_are_optimal(self, small_qubo) -> None:
        solver = create_solver("hungarian")
        result = solver.solve(small_qubo)
        # With cost [[1,5],[5,1]], optimal is (0,0) and (1,1)
        track_indices = {a[0] for a in result.assignments}
        meas_indices = {a[1] for a in result.assignments}
        # Should have 1:1 assignments
        assert len(track_indices) == len(result.assignments)
        assert len(meas_indices) == len(result.assignments)


class TestGNNSolver:
    def test_greedy_assignment(self, small_qubo) -> None:
        solver = create_solver("gnn")
        result = solver.solve(small_qubo)
        assert result.solver_name == "GNN"
        assert len(result.assignments) > 0


class TestSolverFactory:
    def test_create_hungarian(self) -> None:
        solver = create_solver("hungarian")
        assert solver.name == "Hungarian"

    def test_create_gnn(self) -> None:
        solver = create_solver("gnn")
        assert solver.name == "GNN"

    def test_unknown_solver_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown solver"):
            create_solver("nonexistent_solver")
