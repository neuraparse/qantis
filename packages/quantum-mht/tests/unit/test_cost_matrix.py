"""Tests for QUBO cost matrix builder."""

import numpy as np
import pytest

from quantum_mht.formulation.cost_matrix import CostMatrixBuilder


class TestCostMatrixBuilder:
    def test_basic_cost_matrix(self) -> None:
        builder = CostMatrixBuilder(gate_threshold=100.0)
        predicted = np.array([[10.0, 10.0], [20.0, 20.0]])
        measurements = np.array([[10.1, 10.1], [20.1, 20.1]])
        covariances = np.array([np.eye(2), np.eye(2)])

        cost_matrix = builder.build(predicted, measurements, covariances)
        assert cost_matrix.shape == (2, 2)
        # Diagonal should have lower cost (closer matches)
        assert cost_matrix[0, 0] < cost_matrix[0, 1]
        assert cost_matrix[1, 1] < cost_matrix[1, 0]

    def test_gating(self) -> None:
        builder = CostMatrixBuilder(gate_threshold=1.0)
        predicted = np.array([[0.0, 0.0]])
        measurements = np.array([[0.1, 0.1], [100.0, 100.0]])
        covariances = np.array([np.eye(2)])

        cost_matrix = builder.build(predicted, measurements, covariances)
        assert np.isfinite(cost_matrix[0, 0])
        assert np.isinf(cost_matrix[0, 1])  # Gated out

    def test_with_gating_mask(self) -> None:
        builder = CostMatrixBuilder(gate_threshold=1.0)
        predicted = np.array([[0.0, 0.0]])
        measurements = np.array([[0.1, 0.1], [100.0, 100.0]])
        covariances = np.array([np.eye(2)])

        cost_matrix, mask = builder.build_with_gating_mask(predicted, measurements, covariances)
        assert mask[0, 0] is True or mask[0, 0] == True
        assert mask[0, 1] is False or mask[0, 1] == False
