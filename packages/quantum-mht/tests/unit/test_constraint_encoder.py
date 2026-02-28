"""Tests for quantum_mht.formulation.constraint_encoder module.

Validates the ConstraintEncoder class which converts MTDA assignment
constraints (row and column) into QUBO penalty terms following the
formulation in Stollenwerk et al., arXiv:2110.08346, Sec IV.
"""

import numpy as np
import pytest

from quantum_mht.formulation.constraint_encoder import ConstraintEncoder
from quantum_mht.formulation.association_variables import AssociationVariables


class TestConstraintEncoderCreation:
    """Test ConstraintEncoder instantiation and defaults."""

    def test_default_penalty_multiplier(self) -> None:
        """Default penalty_multiplier should be 1.5 (arXiv:2110.08346, Sec IV)."""
        encoder = ConstraintEncoder()
        assert encoder.penalty_multiplier == 1.5

    def test_custom_penalty_multiplier(self) -> None:
        """Should accept a custom penalty_multiplier."""
        encoder = ConstraintEncoder(penalty_multiplier=2.0)
        assert encoder.penalty_multiplier == 2.0


class TestAutoCalibratePenalty:
    """Test auto-calibration of penalty strength."""

    def test_positive_cost_matrix(self) -> None:
        """Auto-calibrated penalty should be a positive float proportional to cost magnitude."""
        encoder = ConstraintEncoder(penalty_multiplier=1.5)
        cost_matrix = np.array([[2.0, 4.0], [1.0, 3.0]])
        penalty = encoder.auto_calibrate_penalty(cost_matrix)
        assert isinstance(penalty, float)
        assert penalty > 0.0
        # Should be 1.5 * max(|c_{i,j}|) = 1.5 * 4.0 = 6.0
        assert np.isclose(penalty, 6.0)

    def test_proportional_to_magnitude(self) -> None:
        """Penalty should scale with cost matrix magnitude."""
        encoder = ConstraintEncoder(penalty_multiplier=1.5)
        small_costs = np.array([[1.0, 2.0], [0.5, 1.5]])
        large_costs = np.array([[10.0, 20.0], [5.0, 15.0]])

        small_penalty = encoder.auto_calibrate_penalty(small_costs)
        large_penalty = encoder.auto_calibrate_penalty(large_costs)
        assert large_penalty > small_penalty
        # Ratio should match the ratio of max abs values: 20/2 = 10
        assert np.isclose(large_penalty / small_penalty, 10.0)

    def test_zero_cost_matrix(self) -> None:
        """Zero cost matrix should return 0 penalty (multiplier * 0 = 0)."""
        encoder = ConstraintEncoder()
        cost_matrix = np.zeros((3, 4))
        penalty = encoder.auto_calibrate_penalty(cost_matrix)
        assert isinstance(penalty, float)
        assert penalty == 0.0

    def test_negative_costs(self) -> None:
        """Auto-calibrate should use absolute values of costs."""
        encoder = ConstraintEncoder(penalty_multiplier=1.5)
        cost_matrix = np.array([[-5.0, 2.0], [3.0, -1.0]])
        penalty = encoder.auto_calibrate_penalty(cost_matrix)
        # max(|c|) = 5.0, penalty = 1.5 * 5.0 = 7.5
        assert np.isclose(penalty, 7.5)

    def test_inf_values_ignored(self) -> None:
        """Infinite costs (from gating) should be excluded from calibration."""
        encoder = ConstraintEncoder(penalty_multiplier=1.5)
        cost_matrix = np.array([[2.0, np.inf], [np.inf, 3.0]])
        penalty = encoder.auto_calibrate_penalty(cost_matrix)
        # Only finite values: max(|2.0|, |3.0|) = 3.0
        assert np.isclose(penalty, 1.5 * 3.0)

    def test_all_inf_returns_fallback(self) -> None:
        """All-infinite cost matrix should return the fallback penalty of 10.0."""
        encoder = ConstraintEncoder()
        cost_matrix = np.full((2, 2), np.inf)
        penalty = encoder.auto_calibrate_penalty(cost_matrix)
        assert penalty == 10.0


class TestEncodeRowConstraints:
    """Test row constraint encoding: each track -> one measurement or missed."""

    def test_returns_dict_of_tuples(self) -> None:
        """Row constraints should return dict with (int, int) keys and float values."""
        encoder = ConstraintEncoder()
        variables = AssociationVariables(n_tracks=2, n_measurements=3)
        Q = encoder.encode_row_constraints(variables, penalty=5.0)
        assert isinstance(Q, dict)
        for key, val in Q.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], int)
            assert isinstance(key[1], int)
            assert isinstance(val, float)

    def test_nonempty_for_nontrivial_problem(self) -> None:
        """Should produce non-empty Q for problems with tracks and measurements."""
        encoder = ConstraintEncoder()
        variables = AssociationVariables(n_tracks=2, n_measurements=2)
        Q = encoder.encode_row_constraints(variables, penalty=3.0)
        assert len(Q) > 0

    def test_diagonal_terms_are_negative(self) -> None:
        """Diagonal (linear) penalty terms should be negative (from -x in (sum x - 1)^2)."""
        encoder = ConstraintEncoder()
        variables = AssociationVariables(
            n_tracks=1, n_measurements=2,
            include_missed=True, include_false_alarm=False,
        )
        Q = encoder.encode_row_constraints(variables, penalty=1.0)
        # For 1 track, row vars are: x_{0,0}, x_{0,1}, x_{0,miss}
        # Diagonal terms should have coefficient -1 * penalty
        for key, val in Q.items():
            if key[0] == key[1]:
                assert val < 0, f"Diagonal term {key} should be negative, got {val}"

    def test_off_diagonal_terms_are_positive(self) -> None:
        """Off-diagonal (quadratic) penalty terms should be positive (from 2*x_k*x_l)."""
        encoder = ConstraintEncoder()
        variables = AssociationVariables(
            n_tracks=1, n_measurements=2,
            include_missed=True, include_false_alarm=False,
        )
        Q = encoder.encode_row_constraints(variables, penalty=1.0)
        for key, val in Q.items():
            if key[0] != key[1]:
                assert val > 0, f"Off-diagonal term {key} should be positive, got {val}"

    def test_row_constraint_enforcement(self) -> None:
        """Verify that feasible assignments (exactly one per row) give zero penalty.

        For 1 track, 2 measurements, with missed detection:
        Row vars: x_{0,0}, x_{0,1}, x_{0,miss}
        Setting exactly one to 1 should yield zero net penalty contribution.
        """
        encoder = ConstraintEncoder()
        variables = AssociationVariables(
            n_tracks=1, n_measurements=2,
            include_missed=True, include_false_alarm=False,
        )
        penalty = 5.0
        Q = encoder.encode_row_constraints(variables, penalty=penalty)

        # Feasible solution: x_{0,0} = 1, rest = 0
        solution = np.zeros(variables.num_variables, dtype=int)
        idx_00 = variables.var_index(0, 0)
        solution[idx_00] = 1

        # Compute quadratic form: sum Q[i,j] * x_i * x_j
        energy = 0.0
        for (i, j), val in Q.items():
            energy += val * solution[i] * solution[j]

        # Penalty for (sum x - 1)^2 with sum x = 1 should be 0
        # But QUBO expansion gives: -1 + 0 + constant = -penalty + penalty*constant
        # The constant term (+1 from expansion) is NOT in Q, so energy = -penalty
        assert np.isclose(energy, -penalty), (
            f"Feasible single-assignment should give energy=-penalty={-penalty}, got {energy}"
        )


class TestEncodeColumnConstraints:
    """Test column constraint encoding: each measurement -> one track or false alarm."""

    def test_returns_dict_of_tuples(self) -> None:
        """Column constraints should return dict with (int, int) keys and float values."""
        encoder = ConstraintEncoder()
        variables = AssociationVariables(n_tracks=2, n_measurements=3)
        Q = encoder.encode_column_constraints(variables, penalty=5.0)
        assert isinstance(Q, dict)
        for key, val in Q.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(val, float)

    def test_nonempty_for_nontrivial_problem(self) -> None:
        """Should produce non-empty Q for problems with tracks and measurements."""
        encoder = ConstraintEncoder()
        variables = AssociationVariables(n_tracks=2, n_measurements=2)
        Q = encoder.encode_column_constraints(variables, penalty=3.0)
        assert len(Q) > 0

    def test_column_constraint_enforcement(self) -> None:
        """Feasible column assignment (exactly one per column) should give zero penalty.

        For 2 tracks, 1 measurement, with false alarm:
        Column vars for meas 0: x_{0,0}, x_{1,0}, x_{fa,0}
        Setting exactly one to 1 should yield zero net penalty contribution
        (excluding the constant offset).
        """
        encoder = ConstraintEncoder()
        variables = AssociationVariables(
            n_tracks=2, n_measurements=1,
            include_missed=False, include_false_alarm=True,
        )
        penalty = 5.0
        Q = encoder.encode_column_constraints(variables, penalty=penalty)

        # Feasible: x_{0,0} = 1, x_{1,0} = 0, x_{fa,0} = 0
        solution = np.zeros(variables.num_variables, dtype=int)
        idx_00 = variables.var_index(0, 0)
        solution[idx_00] = 1

        energy = 0.0
        for (i, j), val in Q.items():
            energy += val * solution[i] * solution[j]

        # Exactly one variable set => (1 - 1)^2 = 0 penalty, energy = -penalty (no constant in Q)
        assert np.isclose(energy, -penalty)

    def test_infeasible_double_assignment_penalized(self) -> None:
        """Double-assigning a measurement should incur extra penalty."""
        encoder = ConstraintEncoder()
        variables = AssociationVariables(
            n_tracks=2, n_measurements=1,
            include_missed=False, include_false_alarm=True,
        )
        penalty = 5.0
        Q = encoder.encode_column_constraints(variables, penalty=penalty)

        # Infeasible: x_{0,0} = 1, x_{1,0} = 1 (double assignment)
        solution_infeasible = np.zeros(variables.num_variables, dtype=int)
        solution_infeasible[variables.var_index(0, 0)] = 1
        solution_infeasible[variables.var_index(1, 0)] = 1

        energy_infeasible = 0.0
        for (i, j), val in Q.items():
            energy_infeasible += val * solution_infeasible[i] * solution_infeasible[j]

        # Feasible: x_{0,0} = 1 only
        solution_feasible = np.zeros(variables.num_variables, dtype=int)
        solution_feasible[variables.var_index(0, 0)] = 1

        energy_feasible = 0.0
        for (i, j), val in Q.items():
            energy_feasible += val * solution_feasible[i] * solution_feasible[j]

        # Infeasible should have higher energy
        assert energy_infeasible > energy_feasible
