"""Tests for quantum_common.backends.base module — ExecutionRequest/Result."""

import pytest

from quantum_common.backends.base import ExecutionRequest, ExecutionResult
from quantum_common.types import JobStatus


class TestExecutionRequest:
    """Verify ExecutionRequest dataclass creation and defaults."""

    def test_creation_with_defaults(self) -> None:
        req = ExecutionRequest(circuits=["circ1"])
        assert req.circuits == ["circ1"]
        assert req.shots == 4096
        assert req.options == {}
        assert req.tags == []

    def test_custom_shots(self) -> None:
        req = ExecutionRequest(circuits=["circ1"], shots=10_000)
        assert req.shots == 10_000

    def test_custom_options(self) -> None:
        opts = {"optimization_level": 3, "resilience_level": 1}
        req = ExecutionRequest(circuits=["circ1"], options=opts)
        assert req.options["optimization_level"] == 3
        assert req.options["resilience_level"] == 1

    def test_custom_tags(self) -> None:
        tags = ["production", "ibm_heron"]
        req = ExecutionRequest(circuits=["circ1"], tags=tags)
        assert req.tags == ["production", "ibm_heron"]

    def test_multiple_circuits(self) -> None:
        req = ExecutionRequest(circuits=["circ1", "circ2", "circ3"])
        assert len(req.circuits) == 3

    def test_default_options_are_independent(self) -> None:
        """Verify that mutable defaults are not shared across instances."""
        req1 = ExecutionRequest(circuits=["a"])
        req2 = ExecutionRequest(circuits=["b"])
        req1.options["key"] = "value"
        assert "key" not in req2.options

    def test_default_tags_are_independent(self) -> None:
        """Verify that mutable default tags are not shared across instances."""
        req1 = ExecutionRequest(circuits=["a"])
        req2 = ExecutionRequest(circuits=["b"])
        req1.tags.append("tag1")
        assert "tag1" not in req2.tags


class TestExecutionResult:
    """Verify ExecutionResult dataclass and its properties."""

    def test_creation_with_counts(self) -> None:
        result = ExecutionResult(counts=[{"00": 500, "11": 500}])
        assert len(result.counts) == 1
        assert result.counts[0]["00"] == 500

    def test_default_metadata(self) -> None:
        result = ExecutionResult(counts=[{"0": 1000}])
        assert result.metadata == {}

    def test_default_execution_time(self) -> None:
        result = ExecutionResult(counts=[{"0": 1000}])
        assert result.execution_time_s == 0.0

    def test_default_status(self) -> None:
        result = ExecutionResult(counts=[{"0": 1000}])
        assert result.status == JobStatus.COMPLETED

    def test_custom_status(self) -> None:
        result = ExecutionResult(counts=[{}], status=JobStatus.FAILED)
        assert result.status == JobStatus.FAILED

    def test_custom_metadata(self) -> None:
        meta = {"backend": "ibm_brisbane", "queue_position": 3}
        result = ExecutionResult(counts=[{"01": 100}], metadata=meta)
        assert result.metadata["backend"] == "ibm_brisbane"

    def test_custom_execution_time(self) -> None:
        result = ExecutionResult(counts=[{"01": 100}], execution_time_s=1.234)
        assert result.execution_time_s == pytest.approx(1.234)

    def test_raw_results_default(self) -> None:
        result = ExecutionResult(counts=[{"0": 100}])
        assert result.raw_results == []


class TestExecutionResultQuasiDists:
    """Verify quasi_dists property converts counts to probability distributions."""

    def test_single_circuit_uniform(self) -> None:
        result = ExecutionResult(counts=[{"00": 500, "11": 500}])
        dists = result.quasi_dists
        assert len(dists) == 1
        assert dists[0]["00"] == pytest.approx(0.5)
        assert dists[0]["11"] == pytest.approx(0.5)

    def test_single_circuit_skewed(self) -> None:
        result = ExecutionResult(counts=[{"00": 750, "01": 250}])
        dists = result.quasi_dists
        assert dists[0]["00"] == pytest.approx(0.75)
        assert dists[0]["01"] == pytest.approx(0.25)

    def test_multiple_circuits(self) -> None:
        result = ExecutionResult(
            counts=[
                {"0": 800, "1": 200},
                {"0": 300, "1": 700},
            ]
        )
        dists = result.quasi_dists
        assert len(dists) == 2
        assert dists[0]["0"] == pytest.approx(0.8)
        assert dists[1]["1"] == pytest.approx(0.7)

    def test_empty_counts_returns_empty_dict(self) -> None:
        result = ExecutionResult(counts=[{}])
        dists = result.quasi_dists
        assert dists == [{}]

    def test_distributions_sum_to_one(self) -> None:
        result = ExecutionResult(
            counts=[{"00": 100, "01": 200, "10": 300, "11": 400}]
        )
        dists = result.quasi_dists
        total = sum(dists[0].values())
        assert total == pytest.approx(1.0)

    def test_all_counts_in_one_bitstring(self) -> None:
        result = ExecutionResult(counts=[{"111": 4096}])
        dists = result.quasi_dists
        assert dists[0]["111"] == pytest.approx(1.0)
