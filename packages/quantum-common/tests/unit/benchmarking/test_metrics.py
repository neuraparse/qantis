"""Tests for quantum_common.benchmarking.metrics module."""

import pytest

from quantum_common.benchmarking.metrics import (
    BenchmarkMetric,
    ComparisonResult,
    MetricType,
)


class TestMetricType:
    """Verify MetricType enum members."""

    @pytest.mark.parametrize(
        "member_name",
        ["TIME", "ACCURACY", "FIDELITY", "COST", "QUBITS", "DEPTH", "CUSTOM"],
    )
    def test_member_exists(self, member_name: str) -> None:
        assert hasattr(MetricType, member_name)
        assert MetricType[member_name].name == member_name

    def test_members_count(self) -> None:
        assert len(MetricType) == 7

    def test_members_are_distinct(self) -> None:
        values = [m.value for m in MetricType]
        assert len(values) == len(set(values))


class TestBenchmarkMetric:
    """Verify BenchmarkMetric creation and repr."""

    def test_creation_with_all_fields(self) -> None:
        metric = BenchmarkMetric(
            name="execution_time",
            value=1.234,
            metric_type=MetricType.TIME,
            unit="seconds",
            lower_is_better=True,
            metadata={"backend": "ibm_heron"},
        )
        assert metric.name == "execution_time"
        assert metric.value == pytest.approx(1.234)
        assert metric.metric_type == MetricType.TIME
        assert metric.unit == "seconds"
        assert metric.lower_is_better is True
        assert metric.metadata["backend"] == "ibm_heron"

    def test_default_unit_is_empty(self) -> None:
        metric = BenchmarkMetric(name="acc", value=0.95, metric_type=MetricType.ACCURACY)
        assert metric.unit == ""

    def test_default_lower_is_better(self) -> None:
        metric = BenchmarkMetric(name="time", value=2.0, metric_type=MetricType.TIME)
        assert metric.lower_is_better is True

    def test_default_metadata_is_empty(self) -> None:
        metric = BenchmarkMetric(name="fid", value=0.99, metric_type=MetricType.FIDELITY)
        assert metric.metadata == {}

    def test_repr_contains_name_and_value(self) -> None:
        metric = BenchmarkMetric(
            name="test_metric",
            value=0.9876,
            metric_type=MetricType.ACCURACY,
            unit="ratio",
        )
        r = repr(metric)
        assert "test_metric" in r
        assert "0.9876" in r
        assert "ratio" in r

    def test_repr_format(self) -> None:
        metric = BenchmarkMetric(name="time", value=1.5, metric_type=MetricType.TIME, unit="s")
        expected = "BenchmarkMetric(time=1.5000 s)"
        assert repr(metric) == expected

    def test_accuracy_metric(self) -> None:
        metric = BenchmarkMetric(
            name="solution_quality",
            value=0.95,
            metric_type=MetricType.ACCURACY,
            lower_is_better=False,
        )
        assert metric.lower_is_better is False

    def test_custom_metric_type(self) -> None:
        metric = BenchmarkMetric(
            name="clops",
            value=25000.0,
            metric_type=MetricType.CUSTOM,
            unit="CLOPS",
        )
        assert metric.metric_type == MetricType.CUSTOM


class TestComparisonResult:
    """Verify ComparisonResult creation and computed properties."""

    @pytest.fixture()
    def quantum_time_metric(self) -> BenchmarkMetric:
        return BenchmarkMetric(name="q_time", value=2.0, metric_type=MetricType.TIME, unit="s")

    @pytest.fixture()
    def classical_time_metric(self) -> BenchmarkMetric:
        return BenchmarkMetric(name="c_time", value=10.0, metric_type=MetricType.TIME, unit="s")

    @pytest.fixture()
    def quantum_accuracy_metric(self) -> BenchmarkMetric:
        return BenchmarkMetric(name="q_acc", value=0.85, metric_type=MetricType.ACCURACY)

    @pytest.fixture()
    def classical_accuracy_metric(self) -> BenchmarkMetric:
        return BenchmarkMetric(name="c_acc", value=0.90, metric_type=MetricType.ACCURACY)

    def test_creation(
        self, quantum_time_metric: BenchmarkMetric, classical_time_metric: BenchmarkMetric
    ) -> None:
        result = ComparisonResult(
            problem_size=10,
            quantum_metrics=[quantum_time_metric],
            classical_metrics=[classical_time_metric],
        )
        assert result.problem_size == 10
        assert len(result.quantum_metrics) == 1
        assert len(result.classical_metrics) == 1

    def test_speedup_quantum_faster(
        self, quantum_time_metric: BenchmarkMetric, classical_time_metric: BenchmarkMetric
    ) -> None:
        """When quantum is faster, speedup > 1."""
        result = ComparisonResult(
            problem_size=10,
            quantum_metrics=[quantum_time_metric],  # 2.0s
            classical_metrics=[classical_time_metric],  # 10.0s
        )
        assert result.speedup is not None
        assert result.speedup == pytest.approx(5.0)
        assert result.speedup > 1.0

    def test_speedup_quantum_slower(self) -> None:
        """When quantum is slower, speedup < 1."""
        q = BenchmarkMetric(name="q", value=20.0, metric_type=MetricType.TIME)
        c = BenchmarkMetric(name="c", value=5.0, metric_type=MetricType.TIME)
        result = ComparisonResult(
            problem_size=5,
            quantum_metrics=[q],
            classical_metrics=[c],
        )
        assert result.speedup is not None
        assert result.speedup == pytest.approx(0.25)
        assert result.speedup < 1.0

    def test_speedup_no_time_metrics(self) -> None:
        """When no TIME metrics exist, speedup is None."""
        q = BenchmarkMetric(name="q", value=0.9, metric_type=MetricType.ACCURACY)
        c = BenchmarkMetric(name="c", value=0.8, metric_type=MetricType.ACCURACY)
        result = ComparisonResult(
            problem_size=10,
            quantum_metrics=[q],
            classical_metrics=[c],
        )
        assert result.speedup is None

    def test_accuracy_improvement_positive(
        self,
        quantum_accuracy_metric: BenchmarkMetric,
        classical_accuracy_metric: BenchmarkMetric,
    ) -> None:
        """Quantum worse accuracy: (0.85 - 0.90) / 0.90 < 0."""
        result = ComparisonResult(
            problem_size=10,
            quantum_metrics=[quantum_accuracy_metric],  # 0.85
            classical_metrics=[classical_accuracy_metric],  # 0.90
        )
        improvement = result.accuracy_improvement
        assert improvement is not None
        assert improvement == pytest.approx((0.85 - 0.90) / 0.90)

    def test_accuracy_improvement_quantum_better(self) -> None:
        q = BenchmarkMetric(name="q", value=0.95, metric_type=MetricType.ACCURACY)
        c = BenchmarkMetric(name="c", value=0.80, metric_type=MetricType.ACCURACY)
        result = ComparisonResult(
            problem_size=10,
            quantum_metrics=[q],
            classical_metrics=[c],
        )
        improvement = result.accuracy_improvement
        assert improvement is not None
        assert improvement > 0  # quantum is better

    def test_accuracy_improvement_no_accuracy_metrics(self) -> None:
        q = BenchmarkMetric(name="q", value=1.0, metric_type=MetricType.TIME)
        c = BenchmarkMetric(name="c", value=2.0, metric_type=MetricType.TIME)
        result = ComparisonResult(
            problem_size=5,
            quantum_metrics=[q],
            classical_metrics=[c],
        )
        assert result.accuracy_improvement is None

    def test_metadata_default_empty(
        self, quantum_time_metric: BenchmarkMetric, classical_time_metric: BenchmarkMetric
    ) -> None:
        result = ComparisonResult(
            problem_size=10,
            quantum_metrics=[quantum_time_metric],
            classical_metrics=[classical_time_metric],
        )
        assert result.metadata == {}

    def test_metadata_custom(
        self, quantum_time_metric: BenchmarkMetric, classical_time_metric: BenchmarkMetric
    ) -> None:
        result = ComparisonResult(
            problem_size=10,
            quantum_metrics=[quantum_time_metric],
            classical_metrics=[classical_time_metric],
            metadata={"problem": "TSP", "backend": "ibm_brisbane"},
        )
        assert result.metadata["problem"] == "TSP"
