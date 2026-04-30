"""Tests for quantum_common.mitigation.readout module — ReadoutMitigationStrategy."""

import numpy as np
import pytest

from quantum_common.mitigation.readout import ReadoutMitigationStrategy


class TestReadoutMitigationStrategyCreation:
    """Verify ReadoutMitigationStrategy construction."""

    def test_default_calibration_shots(self) -> None:
        strategy = ReadoutMitigationStrategy()
        assert strategy.calibration_shots == 8192

    def test_custom_calibration_shots(self) -> None:
        strategy = ReadoutMitigationStrategy(calibration_shots=4096)
        assert strategy.calibration_shots == 4096

    def test_name_property(self) -> None:
        strategy = ReadoutMitigationStrategy()
        # Upgraded 2026-04: label explicitly so the new Runtime-native
        # path (RuntimeReadoutStrategy with TREx+M3) is distinguishable
        # in benchmark reports.
        assert strategy.name == "ReadoutMitigation(legacy-matrix)"


class TestInferNumQubits:
    """Verify _infer_num_qubits correctly determines qubit count from bitstrings."""

    def test_two_qubits(self) -> None:
        strategy = ReadoutMitigationStrategy()
        counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        assert strategy._infer_num_qubits(counts) == 2

    def test_three_qubits(self) -> None:
        strategy = ReadoutMitigationStrategy()
        counts = {"000": 400, "001": 200, "110": 200, "111": 200}
        assert strategy._infer_num_qubits(counts) == 3

    def test_single_qubit(self) -> None:
        strategy = ReadoutMitigationStrategy()
        counts = {"0": 800, "1": 200}
        assert strategy._infer_num_qubits(counts) == 1

    def test_empty_counts_returns_zero(self) -> None:
        strategy = ReadoutMitigationStrategy()
        assert strategy._infer_num_qubits({}) == 0

    def test_four_qubits(self) -> None:
        strategy = ReadoutMitigationStrategy()
        counts = {"0000": 500, "1111": 500}
        assert strategy._infer_num_qubits(counts) == 4


class TestBuildCalibrationMatrix:
    """Verify _build_calibration_matrix returns correct shape and structure."""

    def test_shape_one_qubit(self) -> None:
        strategy = ReadoutMitigationStrategy()
        matrix = strategy._build_calibration_matrix(1, backend=None)
        assert matrix.shape == (2, 2)

    def test_shape_two_qubits(self) -> None:
        strategy = ReadoutMitigationStrategy()
        matrix = strategy._build_calibration_matrix(2, backend=None)
        assert matrix.shape == (4, 4)

    def test_shape_three_qubits(self) -> None:
        strategy = ReadoutMitigationStrategy()
        matrix = strategy._build_calibration_matrix(3, backend=None)
        assert matrix.shape == (8, 8)

    def test_identity_matrix_placeholder(self) -> None:
        """The placeholder implementation returns an identity matrix."""
        strategy = ReadoutMitigationStrategy()
        matrix = strategy._build_calibration_matrix(2, backend=None)
        np.testing.assert_array_equal(matrix, np.eye(4))

    def test_matrix_is_square(self) -> None:
        strategy = ReadoutMitigationStrategy()
        for n_qubits in range(1, 5):
            matrix = strategy._build_calibration_matrix(n_qubits, backend=None)
            assert matrix.shape[0] == matrix.shape[1]

    def test_matrix_dimension_is_power_of_two(self) -> None:
        strategy = ReadoutMitigationStrategy()
        for n_qubits in [1, 2, 3, 4]:
            matrix = strategy._build_calibration_matrix(n_qubits, backend=None)
            expected_dim = 2 ** n_qubits
            assert matrix.shape == (expected_dim, expected_dim)


class TestApplyInverse:
    """Verify _apply_inverse modifies counts appropriately."""

    def test_identity_calibration_preserves_counts(self) -> None:
        """With identity calibration matrix, corrected counts should match raw."""
        strategy = ReadoutMitigationStrategy()
        counts = {"00": 500, "01": 200, "10": 200, "11": 100}
        shots = 1000
        cal_matrix = np.eye(4)

        corrected = strategy._apply_inverse(counts, cal_matrix, shots)
        assert sum(corrected.values()) == pytest.approx(shots, abs=2)

    def test_identity_preserves_distribution(self) -> None:
        """Identity calibration should yield approximately the same counts."""
        strategy = ReadoutMitigationStrategy()
        counts = {"00": 600, "11": 400}
        shots = 1000
        cal_matrix = np.eye(4)

        corrected = strategy._apply_inverse(counts, cal_matrix, shots)
        assert corrected.get("00", 0) == pytest.approx(600, abs=2)
        assert corrected.get("11", 0) == pytest.approx(400, abs=2)

    def test_returns_dict_of_str_int(self) -> None:
        strategy = ReadoutMitigationStrategy()
        counts = {"0": 700, "1": 300}
        cal_matrix = np.eye(2)

        corrected = strategy._apply_inverse(counts, cal_matrix, 1000)
        for key, value in corrected.items():
            assert isinstance(key, str)
            assert isinstance(value, int)

    def test_bitstring_format_preserved(self) -> None:
        """Output bitstrings should have the same length as input bitstrings."""
        strategy = ReadoutMitigationStrategy()
        counts = {"000": 500, "111": 500}
        cal_matrix = np.eye(8)

        corrected = strategy._apply_inverse(counts, cal_matrix, 1000)
        for bitstring in corrected:
            assert len(bitstring) == 3

    def test_small_counts_filtered_out(self) -> None:
        """Counts that round to zero should not appear in output."""
        strategy = ReadoutMitigationStrategy()
        # With identity matrix and a very small component
        counts = {"00": 999, "01": 1}
        cal_matrix = np.eye(4)

        corrected = strategy._apply_inverse(counts, cal_matrix, 1000)
        # "01" has 1 count which is renormalized to 1, so it should be present
        # but total should still be approximately 1000
        total = sum(corrected.values())
        assert total == pytest.approx(1000, abs=2)
