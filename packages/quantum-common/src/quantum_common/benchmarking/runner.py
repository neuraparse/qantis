"""Benchmark runner framework.

2026 Academic References — Quantum Benchmarking Best Practices
================================================================
- **Reproducibility**: All benchmark cases use configurable ``num_trials``
  and ``warmup`` parameters to ensure statistical significance. Results
  report mean +/- std deviation following best practices from:
  Lubinski et al., "Application-Oriented Performance Benchmarks for Quantum
  Computing", IEEE Trans. Quantum Eng. 4, 3100332 (2023).
  DOI: 10.1109/TQE.2023.3253761

- **Quantum Volume**: Cross-platform benchmark defined by IBM. See
  ``benchmarking/metrics.py`` for QV and CLOPS metric definitions.

- **Warm-up runs**: Essential for JIT compilation effects in simulators
  (Qiskit Aer v0.17, PennyLane v0.44 lightning.qubit) and for establishing
  stable Runtime Sessions on IBM Quantum hardware.

- **arXiv:2512.08245** (Dec 2025): Discusses shot count impact on
  benchmark reliability — 10K default shots may be insufficient for
  problems with >13 qubits (state space > 8192).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable
import logging
import numpy as np
from quantum_common.benchmarking.metrics import BenchmarkMetric, MetricType

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkCase:
    """Single benchmark case to execute."""
    name: str
    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    num_trials: int = 5
    warmup: int = 1

@dataclass
class BenchmarkResult:
    """Result from running a benchmark case."""
    case_name: str
    metrics: list[BenchmarkMetric]
    trial_times: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.trial_times)) if self.trial_times else 0.0

    @property
    def std_time(self) -> float:
        return float(np.std(self.trial_times)) if self.trial_times else 0.0

@dataclass
class BenchmarkSuite:
    """Collection of benchmark cases."""
    name: str
    cases: list[BenchmarkCase] = field(default_factory=list)

    def add_case(self, name: str, fn: Callable, *args: Any, num_trials: int = 5, **kwargs: Any) -> None:
        self.cases.append(BenchmarkCase(name=name, fn=fn, args=args, kwargs=kwargs, num_trials=num_trials))

@dataclass
class BenchmarkRunner:
    """Execute benchmark suites and collect results.

    Follows the benchmarking methodology from Lubinski et al., IEEE Trans.
    Quantum Eng. 4, 3100332 (2023) — warm-up, multiple trials, and
    statistical aggregation of timing and quality metrics.
    """
    results: list[BenchmarkResult] = field(default_factory=list)

    def run_suite(self, suite: BenchmarkSuite) -> list[BenchmarkResult]:
        logger.info("Running benchmark suite: %s (%d cases)", suite.name, len(suite.cases))
        suite_results = []
        for case in suite.cases:
            result = self._run_case(case)
            suite_results.append(result)
            self.results.append(result)
        return suite_results

    def _run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        logger.info("Running benchmark: %s", case.name)
        # Warmup
        for _ in range(case.warmup):
            case.fn(*case.args, **case.kwargs)

        trial_times = []
        last_result = None
        for trial in range(case.num_trials):
            t0 = time.perf_counter()
            last_result = case.fn(*case.args, **case.kwargs)
            elapsed = time.perf_counter() - t0
            trial_times.append(elapsed)

        metrics = [
            BenchmarkMetric(
                name=f"{case.name}_time",
                value=float(np.mean(trial_times)),
                metric_type=MetricType.TIME,
                unit="seconds",
                metadata={"std": float(np.std(trial_times))},
            )
        ]

        return BenchmarkResult(
            case_name=case.name,
            metrics=metrics,
            trial_times=trial_times,
            extra={"last_result": last_result},
        )
