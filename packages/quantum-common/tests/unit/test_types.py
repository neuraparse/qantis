"""Tests for quantum_common.types module — enums and type aliases."""

import numpy as np
import pytest

from quantum_common.types import (
    BackendType,
    BitstringCounts,
    CostMatrix,
    JobStatus,
    ProbabilityDistribution,
    QuantumParadigm,
    QubitIndex,
    ShotCount,
    StateVector,
)


class TestQuantumParadigm:
    """Verify QuantumParadigm enum members and semantics."""

    def test_gate_based_exists(self) -> None:
        assert QuantumParadigm.GATE_BASED is not None

    def test_annealing_exists(self) -> None:
        assert QuantumParadigm.ANNEALING is not None

    def test_hybrid_exists(self) -> None:
        assert QuantumParadigm.HYBRID is not None

    def test_members_count(self) -> None:
        assert len(QuantumParadigm) == 3

    def test_members_are_distinct(self) -> None:
        members = [QuantumParadigm.GATE_BASED, QuantumParadigm.ANNEALING, QuantumParadigm.HYBRID]
        assert len(set(members)) == 3

    @pytest.mark.parametrize(
        "member_name",
        ["GATE_BASED", "ANNEALING", "HYBRID"],
    )
    def test_access_by_name(self, member_name: str) -> None:
        member = QuantumParadigm[member_name]
        assert member.name == member_name


class TestBackendType:
    """Verify BackendType enum members and their string values."""

    EXPECTED_MEMBERS = {
        "IBM_QUANTUM": "ibm_quantum",
        "AZURE_QUANTUM": "azure_quantum",
        "DWAVE": "dwave",
        "PENNYLANE": "pennylane",
        "LOCAL_AER": "local_aer",
        "LOCAL_PENNYLANE": "local_pennylane",
        "LOCAL_DWAVE_SIM": "local_dwave_sim",
    }

    def test_members_count(self) -> None:
        assert len(BackendType) == 7

    @pytest.mark.parametrize(
        "name,value",
        EXPECTED_MEMBERS.items(),
    )
    def test_member_value(self, name: str, value: str) -> None:
        member = BackendType[name]
        assert member.value == value

    def test_ibm_quantum_value(self) -> None:
        assert BackendType.IBM_QUANTUM.value == "ibm_quantum"

    def test_local_dwave_sim_value(self) -> None:
        assert BackendType.LOCAL_DWAVE_SIM.value == "local_dwave_sim"

    def test_construct_from_value(self) -> None:
        assert BackendType("pennylane") is BackendType.PENNYLANE


class TestJobStatus:
    """Verify JobStatus enum members."""

    def test_queued_exists(self) -> None:
        assert JobStatus.QUEUED is not None

    def test_running_exists(self) -> None:
        assert JobStatus.RUNNING is not None

    def test_completed_exists(self) -> None:
        assert JobStatus.COMPLETED is not None

    def test_failed_exists(self) -> None:
        assert JobStatus.FAILED is not None

    def test_cancelled_exists(self) -> None:
        assert JobStatus.CANCELLED is not None

    def test_members_count(self) -> None:
        assert len(JobStatus) == 5

    @pytest.mark.parametrize(
        "member_name",
        ["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"],
    )
    def test_access_by_name(self, member_name: str) -> None:
        assert JobStatus[member_name].name == member_name


class TestTypeAliases:
    """Verify that type aliases are defined and usable."""

    def test_state_vector_alias(self) -> None:
        vec: StateVector = np.array([1 + 0j, 0 + 0j], dtype=np.complex128)
        assert vec.dtype == np.complex128

    def test_probability_distribution_alias(self) -> None:
        probs: ProbabilityDistribution = np.array([0.5, 0.5], dtype=np.float64)
        assert probs.dtype == np.float64
        assert np.isclose(probs.sum(), 1.0)

    def test_cost_matrix_alias(self) -> None:
        cost: CostMatrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        assert cost.shape == (2, 2)

    def test_qubit_index_alias(self) -> None:
        idx: QubitIndex = 5
        assert isinstance(idx, int)

    def test_shot_count_alias(self) -> None:
        shots: ShotCount = 4096
        assert isinstance(shots, int)

    def test_bitstring_counts_alias(self) -> None:
        counts: BitstringCounts = {"00": 500, "01": 300, "10": 150, "11": 50}
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 1000
