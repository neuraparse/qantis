"""Benchmark runner: orchestrates seed * deadline * pipeline grids.

A single ``run_benchmark`` call executes one (scenario, pipeline, deadline,
seed) tuple and returns a ``BenchResult``. Higher-level grids are composed by
the caller. The runner is intentionally synchronous and stateless so that
experiments are reproducible bit-for-bit given a seed.

The result is written to a JSON-serializable dictionary so the existing
``quantum_common.benchmarking`` storage layer can ingest it without
additional adapters.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from qantis_engine.bench.metrics import (
    ClosedLoopMetrics,
    compute_closed_loop_metrics,
)


@dataclass
class BenchConfig:
    """Configuration of a single benchmark run."""

    scenario_name: str
    pipeline_name: str
    deadline_ms: float
    seed: int
    n_steps: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchResult:
    """Output of a single benchmark run."""

    config: BenchConfig
    metrics: ClosedLoopMetrics
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "config": asdict(self.config),
            "metrics": {
                "bench": asdict(self.metrics.bench_metrics),
                "cumulative_regret": self.metrics.cumulative_regret,
                "pipeline_reward": self.metrics.pipeline_reward,
                "oracle_reward": self.metrics.oracle_reward,
            },
            "pipeline_metadata": self.pipeline_metadata,
        }
        return out


def run_benchmark(
    scenario_factory: Callable[[int, int], Any],
    pipeline_factory: Callable[[BenchConfig, Any], Any],
    config: BenchConfig,
) -> BenchResult:
    """Run one (scenario, pipeline, deadline, seed) tuple and report metrics.

    Parameters
    ----------
    scenario_factory : ``(n_steps, seed) -> scenario`` constructor.
    pipeline_factory : ``(config, scenario) -> pipeline_outputs`` runner. The
        pipeline must return a dict with keys:
            - ``assignments`` : list of per-step assignment lists
            - ``missed``      : list of per-step missed-detection lists
            - ``false_alarms``: list of per-step false-alarm lists
            - ``latency_ms``  : list of per-step solve times
            - ``sample_cost`` : list of per-step oracle counts
            - ``rewards``     : list of per-step pipeline rewards
            - ``oracle_rewards`` : list of per-step oracle rewards
            - ``ground_truth_ids`` : list of per-step ground-truth ID arrays
            - ``hellinger``   : optional list of per-step posterior Hellinger
            - ``metadata``    : pipeline diagnostics
    config : ``BenchConfig``.
    """
    scenario = scenario_factory(config.n_steps, config.seed)
    pipeline_outputs = pipeline_factory(config, scenario)

    metrics = compute_closed_loop_metrics(
        pipeline_assignments_per_step=pipeline_outputs["assignments"],
        pipeline_missed_per_step=pipeline_outputs["missed"],
        pipeline_false_per_step=pipeline_outputs["false_alarms"],
        pipeline_latency_ms_per_step=pipeline_outputs["latency_ms"],
        sample_cost_per_step=pipeline_outputs["sample_cost"],
        deadline_ms=config.deadline_ms,
        ground_truth_ids_per_step=pipeline_outputs["ground_truth_ids"],
        pipeline_rewards=pipeline_outputs["rewards"],
        oracle_rewards=pipeline_outputs["oracle_rewards"],
        posterior_hellinger_per_step=pipeline_outputs.get("hellinger", []),
    )
    return BenchResult(
        config=config,
        metrics=metrics,
        pipeline_metadata=pipeline_outputs.get("metadata", {}),
    )
