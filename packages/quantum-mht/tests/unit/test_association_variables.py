"""Comprehensive parametrized tests for AssociationVariables.

Tests the binary variable encoding for Multi-Target Data Association (MTDA)
as implemented in quantum_mht.formulation.association_variables.
"""

import numpy as np
import pytest

from quantum_mht.formulation.association_variables import AssociationVariables


class TestVariableCount:
    """Verify num_variables for different N, M configurations."""

    def test_basic_2_3(self) -> None:
        v = AssociationVariables(n_tracks=2, n_measurements=3)
        # 2*3 + 2 + 3 = 11
        assert v.num_variables == 11

    def test_basic_3_4(self) -> None:
        v = AssociationVariables(n_tracks=3, n_measurements=4)
        # 3*4 + 3 + 4 = 19
        assert v.num_variables == 19

    def test_no_slack_variables(self) -> None:
        v = AssociationVariables(
            n_tracks=2, n_measurements=3,
            include_missed=False, include_false_alarm=False,
        )
        assert v.num_variables == 6  # 2*3 only

    def test_only_missed(self) -> None:
        v = AssociationVariables(
            n_tracks=2, n_measurements=3,
            include_missed=True, include_false_alarm=False,
        )
        # 2*3 + 2 = 8
        assert v.num_variables == 8

    def test_only_false_alarm(self) -> None:
        v = AssociationVariables(
            n_tracks=2, n_measurements=3,
            include_missed=False, include_false_alarm=True,
        )
        # 2*3 + 3 = 9
        assert v.num_variables == 9

    @pytest.mark.parametrize("n_tracks,n_meas,expected", [
        (1, 1, 3),    # 1 + 1 + 1 = 3
        (2, 2, 8),    # 4 + 2 + 2 = 8
        (3, 4, 19),   # 12 + 3 + 4 = 19
        (5, 5, 35),   # 25 + 5 + 5 = 35
        (10, 15, 175),  # 150 + 10 + 15 = 175
    ])
    def test_parametrized_full(self, n_tracks, n_meas, expected) -> None:
        v = AssociationVariables(n_tracks=n_tracks, n_measurements=n_meas)
        assert v.num_variables == expected

    @pytest.mark.parametrize("n_tracks,n_meas", [
        (1, 1), (2, 2), (3, 3), (5, 5), (4, 6),
    ])
    def test_parametrized_no_slack(self, n_tracks, n_meas) -> None:
        v = AssociationVariables(
            n_tracks=n_tracks, n_measurements=n_meas,
            include_missed=False, include_false_alarm=False,
        )
        assert v.num_variables == n_tracks * n_meas

    @pytest.mark.parametrize("n_tracks,n_meas", [
        (1, 1), (2, 3), (3, 4),
    ])
    def test_parametrized_with_missed_only(self, n_tracks, n_meas) -> None:
        v = AssociationVariables(
            n_tracks=n_tracks, n_measurements=n_meas,
            include_missed=True, include_false_alarm=False,
        )
        assert v.num_variables == n_tracks * n_meas + n_tracks

    @pytest.mark.parametrize("n_tracks,n_meas", [
        (1, 1), (2, 3), (3, 4),
    ])
    def test_parametrized_with_fa_only(self, n_tracks, n_meas) -> None:
        v = AssociationVariables(
            n_tracks=n_tracks, n_measurements=n_meas,
            include_missed=False, include_false_alarm=True,
        )
        assert v.num_variables == n_tracks * n_meas + n_meas


class TestVarIndex:
    """Verify var_index() mapping for assignment and slack variables."""

    def test_assignment_var_index_ordering(self) -> None:
        """Assignment variables are laid out row-major (track-major)."""
        v = AssociationVariables(n_tracks=2, n_measurements=3,
                                  include_missed=False, include_false_alarm=False)
        # Row-major: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        assert v.var_index(0, 0) == 0
        assert v.var_index(0, 1) == 1
        assert v.var_index(0, 2) == 2
        assert v.var_index(1, 0) == 3
        assert v.var_index(1, 1) == 4
        assert v.var_index(1, 2) == 5

    def test_missed_detection_indices(self) -> None:
        """Missed detection variables (i, -1) follow assignment block."""
        v = AssociationVariables(n_tracks=2, n_measurements=2,
                                  include_missed=True, include_false_alarm=False)
        # Assignment: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # Missed: (0,-1)=4, (1,-1)=5
        assert v.var_index(0, -1) == 4
        assert v.var_index(1, -1) == 5

    def test_false_alarm_indices(self) -> None:
        """False alarm variables (-1, j) follow missed detection block."""
        v = AssociationVariables(n_tracks=2, n_measurements=2,
                                  include_missed=True, include_false_alarm=True)
        # Assignment: 0..3, Missed: 4..5, FA: 6..7
        assert v.var_index(-1, 0) == 6
        assert v.var_index(-1, 1) == 7

    @pytest.mark.parametrize("track,meas", [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 3),
    ])
    def test_var_index_non_negative(self, track, meas) -> None:
        v = AssociationVariables(n_tracks=3, n_measurements=4)
        idx = v.var_index(track, meas)
        assert idx >= 0

    @pytest.mark.parametrize("track,meas", [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
    ])
    def test_var_index_unique(self, track, meas) -> None:
        """Each (track, meas) pair maps to a unique index."""
        v = AssociationVariables(n_tracks=2, n_measurements=3)
        indices = [v.var_index(i, j) for i in range(2) for j in range(3)]
        assert len(set(indices)) == len(indices), "All var indices must be unique"

    def test_all_indices_unique_with_slack(self) -> None:
        v = AssociationVariables(n_tracks=3, n_measurements=4)
        indices = list(v._var_map.values())
        assert len(set(indices)) == len(indices)

    def test_invalid_pair_raises_key_error(self) -> None:
        v = AssociationVariables(n_tracks=2, n_measurements=2)
        with pytest.raises(KeyError):
            v.var_index(99, 99)

    def test_missing_slack_raises_when_disabled(self) -> None:
        v = AssociationVariables(n_tracks=2, n_measurements=2,
                                  include_missed=False, include_false_alarm=False)
        with pytest.raises(KeyError):
            v.var_index(0, -1)
        with pytest.raises(KeyError):
            v.var_index(-1, 0)


class TestVarLabel:
    """Verify human-readable variable labels."""

    @pytest.mark.parametrize("track,meas,expected", [
        (0, 0, "x_0_0"),
        (0, 1, "x_0_1"),
        (1, 0, "x_1_0"),
        (2, 3, "x_2_3"),
        (10, 15, "x_10_15"),
    ])
    def test_assignment_label(self, track, meas, expected) -> None:
        v = AssociationVariables(n_tracks=11, n_measurements=16)
        assert v.var_label(track, meas) == expected

    @pytest.mark.parametrize("track,expected", [
        (0, "miss_0"),
        (1, "miss_1"),
        (5, "miss_5"),
    ])
    def test_missed_label(self, track, expected) -> None:
        assert AssociationVariables(n_tracks=6, n_measurements=3).var_label(track, -1) == expected

    @pytest.mark.parametrize("meas,expected", [
        (0, "fa_0"),
        (1, "fa_1"),
        (3, "fa_3"),
    ])
    def test_false_alarm_label(self, meas, expected) -> None:
        assert AssociationVariables(n_tracks=2, n_measurements=4).var_label(-1, meas) == expected

    def test_label_starts_with_x_for_assignment(self) -> None:
        v = AssociationVariables(n_tracks=3, n_measurements=3)
        for i in range(3):
            for j in range(3):
                assert v.var_label(i, j).startswith("x_")

    def test_label_starts_with_miss_for_missed(self) -> None:
        v = AssociationVariables(n_tracks=3, n_measurements=3)
        for i in range(3):
            assert v.var_label(i, -1).startswith("miss_")

    def test_label_starts_with_fa_for_false_alarm(self) -> None:
        v = AssociationVariables(n_tracks=3, n_measurements=3)
        for j in range(3):
            assert v.var_label(-1, j).startswith("fa_")


class TestDecodeSolution:
    """Verify decode_solution returns correct assignments, missed, false alarms."""

    def test_decode_two_assignments(self) -> None:
        v = AssociationVariables(n_tracks=2, n_measurements=2)
        solution = np.zeros(v.num_variables, dtype=int)
        solution[v.var_index(0, 0)] = 1
        solution[v.var_index(1, 1)] = 1
        decoded = v.decode_solution(solution)
        assert (0, 0) in decoded["assignments"]
        assert (1, 1) in decoded["assignments"]
        assert len(decoded["missed_detections"]) == 0
        assert len(decoded["false_alarms"]) == 0

    def test_decode_missed_detection(self) -> None:
        v = AssociationVariables(n_tracks=2, n_measurements=2)
        solution = np.zeros(v.num_variables, dtype=int)
        solution[v.var_index(0, -1)] = 1  # track 0 is missed
        solution[v.var_index(1, 1)] = 1   # track 1 → meas 1
        decoded = v.decode_solution(solution)
        assert 0 in decoded["missed_detections"]
        assert (1, 1) in decoded["assignments"]

    def test_decode_false_alarm(self) -> None:
        v = AssociationVariables(n_tracks=2, n_measurements=2)
        solution = np.zeros(v.num_variables, dtype=int)
        solution[v.var_index(-1, 0)] = 1  # meas 0 is false alarm
        solution[v.var_index(0, 1)] = 1   # track 0 → meas 1
        decoded = v.decode_solution(solution)
        assert 0 in decoded["false_alarms"]
        assert (0, 1) in decoded["assignments"]

    def test_decode_all_zeros(self) -> None:
        v = AssociationVariables(n_tracks=3, n_measurements=4)
        solution = np.zeros(v.num_variables, dtype=int)
        decoded = v.decode_solution(solution)
        assert decoded["assignments"] == []
        assert decoded["missed_detections"] == []
        assert decoded["false_alarms"] == []

    def test_decode_result_keys_present(self) -> None:
        v = AssociationVariables(n_tracks=2, n_measurements=2)
        solution = np.zeros(v.num_variables, dtype=int)
        decoded = v.decode_solution(solution)
        assert "assignments" in decoded
        assert "missed_detections" in decoded
        assert "false_alarms" in decoded

    @pytest.mark.parametrize("n_tracks,n_meas", [
        (2, 2), (3, 4), (1, 3),
    ])
    def test_decode_full_assignment(self, n_tracks, n_meas) -> None:
        v = AssociationVariables(n_tracks=n_tracks, n_measurements=n_meas)
        solution = np.zeros(v.num_variables, dtype=int)
        # Assign track i → measurement i (for min(n_tracks, n_meas) pairs)
        n_pairs = min(n_tracks, n_meas)
        for i in range(n_pairs):
            solution[v.var_index(i, i)] = 1
        decoded = v.decode_solution(solution)
        for i in range(n_pairs):
            assert (i, i) in decoded["assignments"]

    def test_decode_solution_returns_lists(self) -> None:
        v = AssociationVariables(n_tracks=2, n_measurements=2)
        solution = np.zeros(v.num_variables, dtype=int)
        decoded = v.decode_solution(solution)
        assert isinstance(decoded["assignments"], list)
        assert isinstance(decoded["missed_detections"], list)
        assert isinstance(decoded["false_alarms"], list)

    def test_decode_partial_solution_shorter_than_num_variables(self) -> None:
        """decode_solution should handle shorter solution arrays gracefully."""
        v = AssociationVariables(n_tracks=3, n_measurements=3)
        # Provide only assignment block (no slack bits)
        solution = np.zeros(v.n_tracks * v.n_measurements, dtype=int)
        solution[v.var_index(0, 0)] = 1
        decoded = v.decode_solution(solution)
        assert (0, 0) in decoded["assignments"]


class TestAssociationVariablesProperties:
    """Test internal consistency and properties."""

    def test_var_map_size_matches_num_variables(self) -> None:
        v = AssociationVariables(n_tracks=3, n_measurements=4)
        assert len(v._var_map) == v.num_variables

    def test_indices_are_contiguous(self) -> None:
        """Variable indices should cover 0..num_variables-1 with no gaps."""
        v = AssociationVariables(n_tracks=3, n_measurements=4)
        indices = sorted(v._var_map.values())
        expected = list(range(v.num_variables))
        assert indices == expected

    @pytest.mark.parametrize("n_tracks,n_meas", [
        (1, 1), (2, 3), (3, 4), (5, 5),
    ])
    def test_contiguous_indices_parametrized(self, n_tracks, n_meas) -> None:
        v = AssociationVariables(n_tracks=n_tracks, n_measurements=n_meas)
        indices = sorted(v._var_map.values())
        assert indices == list(range(v.num_variables))

    def test_assignment_block_precedes_slack(self) -> None:
        """All assignment variable indices < missed detection indices < FA indices."""
        v = AssociationVariables(n_tracks=2, n_measurements=3)
        assignment_indices = [v.var_index(i, j) for i in range(2) for j in range(3)]
        missed_indices = [v.var_index(i, -1) for i in range(2)]
        fa_indices = [v.var_index(-1, j) for j in range(3)]
        assert max(assignment_indices) < min(missed_indices)
        assert max(missed_indices) < min(fa_indices)

    def test_different_configs_are_independent(self) -> None:
        v1 = AssociationVariables(n_tracks=2, n_measurements=2)
        v2 = AssociationVariables(n_tracks=3, n_measurements=4)
        assert v1.num_variables != v2.num_variables
        assert v1.n_tracks != v2.n_tracks

    def test_n_tracks_and_n_measurements_stored(self) -> None:
        v = AssociationVariables(n_tracks=4, n_measurements=7)
        assert v.n_tracks == 4
        assert v.n_measurements == 7

    def test_include_flags_stored(self) -> None:
        v1 = AssociationVariables(n_tracks=2, n_measurements=2,
                                   include_missed=True, include_false_alarm=False)
        assert v1.include_missed is True
        assert v1.include_false_alarm is False

        v2 = AssociationVariables(n_tracks=2, n_measurements=2,
                                   include_missed=False, include_false_alarm=True)
        assert v2.include_missed is False
        assert v2.include_false_alarm is True
