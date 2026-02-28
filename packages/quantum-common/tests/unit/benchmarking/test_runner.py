"""Tests for quantum_common.benchmarking.runner module."""

import time

import pytest

from quantum_common.benchmarking.metrics import MetricType
from quantum_common.benchmarking.runner import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
)


def _dummy_function() -> int:
    """Simple function for benchmarking tests."""
    return sum(range(100))


def _slow_function() -> None:
    """Function with a measurable delay for timing tests."""
    time.sleep(0.005)


def _adder(a: int, b: int) -> int:
    return a + b


class TestBenchmarkCase:
    """Verify BenchmarkCase creation."""

    def test_creation_minimal(self) -> None:
        case = BenchmarkCase(name="test_case", fn=_dummy_function)
        assert case.name == "test_case"
        assert case.fn is _dummy_function
        assert case.args == ()
        assert case.kwargs == {}
        assert case.num_trials == 5
        assert case.warmup == 1

    def test_creation_with_args(self) -> None:
        case = BenchmarkCase(name="adder", fn=_adder, args=(3, 5))
        assert case.args == (3, 5)

    def test_creation_with_kwargs(self) -> None:
        case = BenchmarkCase(name="adder", fn=_adder, kwargs={"a": 3, "b": 5})
        assert case.kwargs == {"a": 3, "b": 5}

    def test_custom_trials_and_warmup(self) -> None:
        case = BenchmarkCase(name="test", fn=_dummy_function, num_trials=10, warmup=3)
        assert case.num_trials == 10
        assert case.warmup == 3


class TestBenchmarkResult:
    """Verify BenchmarkResult properties."""

    def test_mean_time_with_values(self) -> None:
        result = BenchmarkResult(
            case_name="test",
            metrics=[],
            trial_times=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        assert result.mean_time == pytest.approx(3.0)

    def test_std_time_with_values(self) -> None:
        result = BenchmarkResult(
            case_name="test",
            metrics=[],
            trial_times=[2.0, 2.0, 2.0, 2.0],
        )
        assert result.std_time == pytest.approx(0.0)

    def test_std_time_with_variance(self) -> None:
        result = BenchmarkResult(
            case_name="test",
            metrics=[],
            trial_times=[1.0, 3.0],
        )
        # std of [1, 3] = 1.0
        assert result.std_time == pytest.approx(1.0)

    def test_mean_time_empty(self) -> None:
        result = BenchmarkResult(case_name="test", metrics=[])
        assert result.mean_time == 0.0

    def test_std_time_empty(self) -> None:
        result = BenchmarkResult(case_name="test", metrics=[])
        assert result.std_time == 0.0

    def test_extra_default(self) -> None:
        result = BenchmarkResult(case_name="test", metrics=[])
        assert result.extra == {}


class TestBenchmarkSuite:
    """Verify BenchmarkSuite creation and add_case."""

    def test_creation(self) -> None:
        suite = BenchmarkSuite(name="test_suite")
        assert suite.name == "test_suite"
        assert suite.cases == []

    def test_add_case(self) -> None:
        suite = BenchmarkSuite(name="test_suite")
        suite.add_case("sum_test", _dummy_function)
        assert len(suite.cases) == 1
        assert suite.cases[0].name == "sum_test"

    def test_add_multiple_cases(self) -> None:
        suite = BenchmarkSuite(name="test_suite")
        suite.add_case("case1", _dummy_function)
        suite.add_case("case2", _slow_function)
        assert len(suite.cases) == 2

    def test_add_case_with_custom_trials(self) -> None:
        suite = BenchmarkSuite(name="test_suite")
        suite.add_case("custom", _dummy_function, num_trials=10)
        assert suite.cases[0].num_trials == 10

    def test_add_case_with_args(self) -> None:
        suite = BenchmarkSuite(name="test_suite")
        suite.add_case("adder", _adder, 3, 5)
        assert suite.cases[0].args == (3, 5)


class TestBenchmarkRunner:
    """Verify BenchmarkRunner executes suites and produces results."""

    def test_run_suite_with_simple_function(self) -> None:
        suite = BenchmarkSuite(name="simple_suite")
        suite.add_case("dummy", _dummy_function, num_trials=3)

        runner = BenchmarkRunner()
        results = runner.run_suite(suite)

        assert len(results) == 1
        assert results[0].case_name == "dummy"

    def test_results_have_trial_times(self) -> None:
        suite = BenchmarkSuite(name="timing_suite")
        suite.add_case("timed", _dummy_function, num_trials=5)

        runner = BenchmarkRunner()
        results = runner.run_suite(suite)

        assert len(results[0].trial_times) == 5
        assert all(t >= 0 for t in results[0].trial_times)

    def test_results_have_time_metric(self) -> None:
        suite = BenchmarkSuite(name="metric_suite")
        suite.add_case("metric_case", _dummy_function, num_trials=3)

        runner = BenchmarkRunner()
        results = runner.run_suite(suite)

        time_metrics = [m for m in results[0].metrics if m.metric_type == MetricType.TIME]
        assert len(time_metrics) == 1
        assert time_metrics[0].unit == "seconds"
        assert time_metrics[0].value > 0

    def test_slow_function_has_measurable_time(self) -> None:
        suite = BenchmarkSuite(name="slow_suite")
        suite.add_case("slow", _slow_function, num_trials=3)

        runner = BenchmarkRunner()
        results = runner.run_suite(suite)

        # Each trial should take at least ~5ms
        assert results[0].mean_time >= 0.003

    def test_runner_accumulates_results(self) -> None:
        runner = BenchmarkRunner()

        suite1 = BenchmarkSuite(name="suite1")
        suite1.add_case("case1", _dummy_function, num_trials=2)
        runner.run_suite(suite1)

        suite2 = BenchmarkSuite(name="suite2")
        suite2.add_case("case2", _dummy_function, num_trials=2)
        runner.run_suite(suite2)

        assert len(runner.results) == 2

    def test_multiple_cases_in_suite(self) -> None:
        suite = BenchmarkSuite(name="multi")
        suite.add_case("fast", _dummy_function, num_trials=2)
        suite.add_case("slow", _slow_function, num_trials=2)

        runner = BenchmarkRunner()
        results = runner.run_suite(suite)

        assert len(results) == 2
        assert results[0].case_name == "fast"
        assert results[1].case_name == "slow"

    def test_last_result_captured(self) -> None:
        suite = BenchmarkSuite(name="result_check")
        suite.add_case("adder", _adder, 10, 20, num_trials=3)

        runner = BenchmarkRunner()
        results = runner.run_suite(suite)

        assert results[0].extra.get("last_result") == 30
