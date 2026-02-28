"""Tests for MTDA QUBO builder."""

import numpy as np
import pytest

from quantum_mht.formulation.association_variables import AssociationVariables
from quantum_mht.formulation.constraint_encoder import ConstraintEncoder
from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder, QUBOResult


class TestAssociationVariables:
    def test_variable_count(self) -> None:
        vars = AssociationVariables(n_tracks=3, n_measurements=4)
        # 3*4 + 3 (missed) + 4 (false alarm) = 19
        assert vars.num_variables == 19

    def test_no_missed_or_false_alarm(self) -> None:
        vars = AssociationVariables(
            n_tracks=2, n_measurements=3,
            include_missed=False, include_false_alarm=False,
        )
        assert vars.num_variables == 6

    def test_decode_solution(self) -> None:
        vars = AssociationVariables(n_tracks=2, n_measurements=2)
        # x_{0,0}=1, x_{1,1}=1, rest=0
        solution = np.zeros(vars.num_variables, dtype=int)
        solution[vars.var_index(0, 0)] = 1
        solution[vars.var_index(1, 1)] = 1
        decoded = vars.decode_solution(solution)
        assert (0, 0) in decoded["assignments"]
        assert (1, 1) in decoded["assignments"]


class TestConstraintEncoder:
    def test_auto_calibrate(self) -> None:
        encoder = ConstraintEncoder(penalty_multiplier=1.5)
        cost_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        penalty = encoder.auto_calibrate_penalty(cost_matrix)
        assert penalty == pytest.approx(6.0)

    def test_row_constraints(self) -> None:
        encoder = ConstraintEncoder()
        vars = AssociationVariables(n_tracks=2, n_measurements=2)
        Q = encoder.encode_row_constraints(vars, penalty=10.0)
        assert len(Q) > 0


class TestMTDAQuboBuilder:
    def test_build_from_cost_matrix(self) -> None:
        builder = MTDAQuboBuilder()
        cost_matrix = np.array([[1.0, 5.0], [5.0, 1.0]])
        result = builder.build_from_cost_matrix(cost_matrix)
        assert isinstance(result, QUBOResult)
        assert result.num_variables > 0
        assert result.penalty > 0

    def test_to_bqm(self) -> None:
        builder = MTDAQuboBuilder()
        cost_matrix = np.array([[1.0, 3.0], [3.0, 1.0]])
        result = builder.build_from_cost_matrix(cost_matrix)
        try:
            bqm = result.to_bqm()
            assert bqm.num_variables == result.num_variables
        except ImportError:
            pytest.skip("dimod not installed")
