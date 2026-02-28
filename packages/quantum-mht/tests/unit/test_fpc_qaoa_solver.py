"""Comprehensive parametrized tests for FPC-QAOA solver and schedule functions.

Tests polynomial_schedule, trigonometric_schedule, digitize_schedule,
and FPCQAOASolver configuration and solve behavior.
"""

import numpy as np
import pytest

from quantum_mht.solvers.fpc_qaoa_solver import (
    FPCQAOASolver,
    _polynomial_schedule,
    _trigonometric_schedule,
    digitize_schedule,
)
from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder


@pytest.fixture
def tiny_qubo():
    """Smallest valid QUBO (2x2 cost matrix) for fast solver testing."""
    builder = MTDAQuboBuilder()
    cost_matrix = np.array([[1.0, 5.0], [5.0, 1.0]])
    return builder.build_from_cost_matrix(cost_matrix)


# ---------------------------------------------------------------------------
# Schedule function tests
# ---------------------------------------------------------------------------

class TestPolynomialSchedule:
    """Test _polynomial_schedule(t, coeffs) = sum(c_k * t^k)."""

    def test_constant_schedule(self) -> None:
        """Constant schedule f(t) = c0 for any t."""
        assert np.isclose(_polynomial_schedule(0.0, [3.0]), 3.0)
        assert np.isclose(_polynomial_schedule(0.5, [3.0]), 3.0)
        assert np.isclose(_polynomial_schedule(1.0, [3.0]), 3.0)

    def test_linear_schedule_at_endpoints(self) -> None:
        """Linear ramp f(t) = 0 + pi*t."""
        coeffs = [0.0, np.pi]
        assert np.isclose(_polynomial_schedule(0.0, coeffs), 0.0)
        assert np.isclose(_polynomial_schedule(1.0, coeffs), np.pi)
        assert np.isclose(_polynomial_schedule(0.5, coeffs), np.pi / 2)

    def test_quadratic_schedule(self) -> None:
        """f(t) = 1 + 2t + 3t^2."""
        coeffs = [1.0, 2.0, 3.0]
        assert np.isclose(_polynomial_schedule(0.0, coeffs), 1.0)
        assert np.isclose(_polynomial_schedule(1.0, coeffs), 6.0)
        assert np.isclose(_polynomial_schedule(0.5, coeffs), 1 + 1 + 0.75)  # 2.75

    def test_zero_coefficients(self) -> None:
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert np.isclose(_polynomial_schedule(t, [0.0, 0.0, 0.0]), 0.0)

    @pytest.mark.parametrize("t", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    def test_constant_schedule_parametrized(self, t: float) -> None:
        assert np.isclose(_polynomial_schedule(t, [np.pi / 4]), np.pi / 4)

    @pytest.mark.parametrize("coeffs,t,expected", [
        ([1.0], 0.5, 1.0),
        ([0.0, 1.0], 0.5, 0.5),
        ([0.0, 0.0, 1.0], 0.5, 0.25),  # t^2
        ([1.0, 1.0, 1.0], 1.0, 3.0),   # 1+1+1
    ])
    def test_parametrized_values(self, coeffs, t, expected) -> None:
        assert np.isclose(_polynomial_schedule(t, coeffs), expected)

    def test_returns_float(self) -> None:
        result = _polynomial_schedule(0.5, [1.0, 2.0])
        assert isinstance(result, float)


class TestTrigonometricSchedule:
    """Test _trigonometric_schedule(t, coeffs) = sum(c_k * sin((k+1)*pi*t))."""

    def test_zero_at_endpoints(self) -> None:
        """sin(k*pi*0)=0 and sin(k*pi*1)=0 for integer k."""
        coeffs = [1.0, 0.5, 0.25]
        assert np.isclose(_trigonometric_schedule(0.0, coeffs), 0.0, atol=1e-10)
        assert np.isclose(_trigonometric_schedule(1.0, coeffs), 0.0, atol=1e-10)

    def test_peak_at_half(self) -> None:
        """sin(pi/2) = 1, single coefficient: f(0.5) = c_0."""
        c = 2.5
        assert np.isclose(_trigonometric_schedule(0.5, [c]), c)

    def test_zero_coefficients(self) -> None:
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert np.isclose(_trigonometric_schedule(t, [0.0, 0.0]), 0.0)

    def test_single_coefficient(self) -> None:
        """f(t) = c * sin(pi * t)."""
        c = np.pi
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            expected = c * np.sin(np.pi * t)
            assert np.isclose(_trigonometric_schedule(t, [c]), expected)

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_returns_float_parametrized(self, t: float) -> None:
        result = _trigonometric_schedule(t, [1.0, 0.5])
        assert isinstance(result, float)

    def test_multiple_coefficients(self) -> None:
        """Verify sum formula for two-coefficient schedule."""
        t = 0.3
        c0, c1 = 1.0, 0.5
        expected = c0 * np.sin(np.pi * t) + c1 * np.sin(2 * np.pi * t)
        assert np.isclose(_trigonometric_schedule(t, [c0, c1]), expected)


class TestDigitizeSchedule:
    """Test digitize_schedule(schedule_fn, coeffs, depth)."""

    def test_output_length_equals_depth(self) -> None:
        for depth in [1, 2, 4, 8, 16]:
            result = digitize_schedule(_polynomial_schedule, [0.0, 1.0], depth)
            assert len(result) == depth

    def test_linear_schedule_is_monotone_increasing(self) -> None:
        """Linear ramp [0, pi] digitized should be monotonically increasing."""
        result = digitize_schedule(_polynomial_schedule, [0.0, np.pi], 8)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_constant_schedule_all_equal(self) -> None:
        """Constant schedule should give identical values at all depths."""
        c = np.pi / 4
        for depth in [2, 4, 8]:
            result = digitize_schedule(_polynomial_schedule, [c], depth)
            assert all(np.isclose(v, c) for v in result)

    def test_midpoint_evaluation(self) -> None:
        """Schedule evaluated at (i + 0.5) / depth for i = 0..p-1."""
        depth = 4
        coeffs = [0.0, 1.0]  # f(t) = t
        result = digitize_schedule(_polynomial_schedule, coeffs, depth)
        expected = [(i + 0.5) / depth for i in range(depth)]
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
    def test_trig_schedule_output_length(self, depth: int) -> None:
        result = digitize_schedule(_trigonometric_schedule, [1.0, 0.5], depth)
        assert len(result) == depth

    def test_returns_list(self) -> None:
        result = digitize_schedule(_polynomial_schedule, [1.0], 4)
        assert isinstance(result, list)

    def test_values_within_expected_range_for_linear(self) -> None:
        """Linear ramp [0, pi]: all values should be in (0, pi)."""
        result = digitize_schedule(_polynomial_schedule, [0.0, np.pi], 8)
        for v in result:
            assert 0.0 < v < np.pi


# ---------------------------------------------------------------------------
# FPCQAOASolver configuration tests
# ---------------------------------------------------------------------------

class TestFPCQAOASolverConfig:
    """Test FPCQAOASolver instantiation and configuration."""

    def test_default_values(self) -> None:
        solver = FPCQAOASolver()
        assert solver.depth == 8
        assert solver.num_schedule_params == 3
        assert solver.schedule_type == "polynomial"
        assert solver.optimizer_type == "cobyla"
        assert solver.optimizer_maxiter == 200

    def test_custom_depth(self) -> None:
        solver = FPCQAOASolver(depth=3)
        assert solver.depth == 3

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 8, 16])
    def test_parametrized_depth(self, depth: int) -> None:
        solver = FPCQAOASolver(depth=depth)
        assert solver.depth == depth

    @pytest.mark.parametrize("schedule_type", ["polynomial", "trigonometric"])
    def test_schedule_types(self, schedule_type: str) -> None:
        solver = FPCQAOASolver(schedule_type=schedule_type)
        assert solver.schedule_type == schedule_type

    @pytest.mark.parametrize("optimizer_type", ["cobyla", "spsa", "COBYLA", "SPSA"])
    def test_optimizer_types(self, optimizer_type: str) -> None:
        solver = FPCQAOASolver(optimizer_type=optimizer_type)
        assert solver.optimizer_type == optimizer_type

    def test_name_includes_depth_and_k(self) -> None:
        solver = FPCQAOASolver(depth=3, num_schedule_params=2)
        assert "3" in solver.name
        assert "2" in solver.name

    def test_name_includes_optimizer_type(self) -> None:
        cobyla_solver = FPCQAOASolver(optimizer_type="cobyla")
        spsa_solver = FPCQAOASolver(optimizer_type="spsa")
        assert "COBYLA" in cobyla_solver.name
        assert "SPSA" in spsa_solver.name

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_num_schedule_params(self, k: int) -> None:
        solver = FPCQAOASolver(num_schedule_params=k)
        assert solver.num_schedule_params == k

    def test_warm_start_disabled_by_default_custom_coeffs(self) -> None:
        solver = FPCQAOASolver()
        assert solver.warm_start_coeffs is None
        assert solver.use_warm_start is True

    def test_get_schedule_fn_polynomial(self) -> None:
        solver = FPCQAOASolver(schedule_type="polynomial")
        fn = solver._get_schedule_fn()
        assert fn is _polynomial_schedule

    def test_get_schedule_fn_trigonometric(self) -> None:
        solver = FPCQAOASolver(schedule_type="trigonometric")
        fn = solver._get_schedule_fn()
        assert fn is _trigonometric_schedule

    def test_get_schedule_fn_default_is_polynomial(self) -> None:
        solver = FPCQAOASolver(schedule_type="unknown")
        fn = solver._get_schedule_fn()
        assert fn is _polynomial_schedule


class TestBuildQAOAInitialPoint:
    """Test _build_qaoa_initial_point() digitization output."""

    def test_output_length_is_2p(self) -> None:
        """Output should be [gamma_1, beta_1, ..., gamma_p, beta_p] = 2*depth params."""
        for depth in [1, 2, 4, 8]:
            solver = FPCQAOASolver(depth=depth)
            gamma_coeffs = [0.0, np.pi]
            beta_coeffs = [np.pi / 4]
            params = solver._build_qaoa_initial_point(gamma_coeffs, beta_coeffs)
            assert len(params) == 2 * depth

    def test_interleaved_format(self) -> None:
        """Output should be [gamma_1, beta_1, gamma_2, beta_2, ...]."""
        solver = FPCQAOASolver(depth=2, schedule_type="polynomial")
        # gamma: linear [0, pi] → digitized: [pi/4, 3*pi/4]
        # beta: constant [pi/4] → digitized: [pi/4, pi/4]
        gamma_coeffs = [0.0, np.pi]
        beta_coeffs = [np.pi / 4]
        params = solver._build_qaoa_initial_point(gamma_coeffs, beta_coeffs)
        # params[0] = gamma_1 ≈ pi/4, params[1] = beta_1 = pi/4
        # params[2] = gamma_2 ≈ 3*pi/4, params[3] = beta_2 = pi/4
        assert len(params) == 4
        assert np.isclose(params[0], np.pi / 4)      # gamma_1
        assert np.isclose(params[1], np.pi / 4)      # beta_1
        assert np.isclose(params[2], 3 * np.pi / 4)  # gamma_2
        assert np.isclose(params[3], np.pi / 4)      # beta_2

    @pytest.mark.parametrize("depth,k", [
        (1, 1), (2, 2), (4, 3), (8, 3),
    ])
    def test_output_length_parametrized(self, depth: int, k: int) -> None:
        solver = FPCQAOASolver(depth=depth, num_schedule_params=k)
        gamma = [0.0, np.pi] + [0.0] * (k - 2)
        beta = [np.pi / 4] + [0.0] * (k - 1)
        params = solver._build_qaoa_initial_point(gamma, beta)
        assert len(params) == 2 * depth


class TestFPCQAOASolverSolve:
    """Test FPCQAOASolver.solve() with small QUBO instances."""

    def test_solve_returns_solver_result(self, tiny_qubo) -> None:
        """FPC-QAOA should return a SolverResult for a small instance."""
        try:
            from qiskit_algorithms.optimizers import COBYLA  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-algorithms not installed")

        solver = FPCQAOASolver(depth=1, num_schedule_params=2, optimizer_maxiter=10)
        result = solver.solve(tiny_qubo)

        from quantum_mht.solvers.base_solver import SolverResult
        assert isinstance(result, SolverResult)

    def test_solve_result_has_correct_fields(self, tiny_qubo) -> None:
        try:
            from qiskit_algorithms.optimizers import COBYLA  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-algorithms not installed")

        solver = FPCQAOASolver(depth=1, num_schedule_params=2, optimizer_maxiter=10)
        result = solver.solve(tiny_qubo)

        assert result.solver_name is not None
        assert result.solve_time_s >= 0
        assert result.raw_solution is not None
        assert len(result.raw_solution) == tiny_qubo.num_variables

    def test_solve_cobyla_optimizer(self, tiny_qubo) -> None:
        try:
            from qiskit_algorithms.optimizers import COBYLA  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-algorithms not installed")

        solver = FPCQAOASolver(
            depth=1, optimizer_type="cobyla", optimizer_maxiter=5
        )
        result = solver.solve(tiny_qubo)
        assert result.metadata["optimizer_type"] == "cobyla"

    def test_solve_spsa_optimizer(self, tiny_qubo) -> None:
        try:
            from qiskit_algorithms.optimizers import SPSA  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-algorithms not installed")

        solver = FPCQAOASolver(
            depth=1, optimizer_type="spsa", optimizer_maxiter=5
        )
        result = solver.solve(tiny_qubo)
        assert result.metadata["optimizer_type"] == "spsa"

    def test_solve_metadata_contains_schedule_info(self, tiny_qubo) -> None:
        try:
            from qiskit_algorithms.optimizers import COBYLA  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-algorithms not installed")

        solver = FPCQAOASolver(depth=1, optimizer_maxiter=5)
        result = solver.solve(tiny_qubo)
        md = result.metadata
        assert "depth" in md
        assert "schedule_type" in md
        assert "optimizer_type" in md
        assert "gamma_coeffs" in md
        assert "beta_coeffs" in md

    def test_solve_solution_is_binary(self, tiny_qubo) -> None:
        try:
            from qiskit_algorithms.optimizers import COBYLA  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-algorithms not installed")

        solver = FPCQAOASolver(depth=1, optimizer_maxiter=5)
        result = solver.solve(tiny_qubo)
        for bit in result.raw_solution:
            assert bit in (0, 1)

    @pytest.mark.parametrize("schedule_type", ["polynomial", "trigonometric"])
    def test_solve_both_schedule_types(self, tiny_qubo, schedule_type: str) -> None:
        try:
            from qiskit_algorithms.optimizers import COBYLA  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-algorithms not installed")

        solver = FPCQAOASolver(
            depth=1, schedule_type=schedule_type, optimizer_maxiter=5
        )
        result = solver.solve(tiny_qubo)
        assert result.metadata["schedule_type"] == schedule_type

    def test_solver_name_in_result(self, tiny_qubo) -> None:
        try:
            from qiskit_algorithms.optimizers import COBYLA  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-algorithms not installed")

        solver = FPCQAOASolver(depth=1, optimizer_maxiter=5)
        result = solver.solve(tiny_qubo)
        assert "FPC-QAOA" in result.solver_name
