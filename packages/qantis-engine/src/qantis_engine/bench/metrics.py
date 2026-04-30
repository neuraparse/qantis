"""Closed-loop tracking metrics for the QANTIS-vs-Gurobi benchmark.

The metric set follows the MOT16 / CLEAR-MOT family but with deadline-aware
extensions: every miss, switch, and false track is tagged with the time-step
deadline that produced it, so deadline-vs-quality plots come for free.

Definitions:
    - ID switches  : one per timestep where ``track i`` flips from being
                     associated with ``measurement_a`` to ``measurement_b``
                     while the underlying ground-truth target identity is
                     unchanged.
    - Missed tracks: ground-truth targets that were *visible* this timestep
                     (in field of view, above SNR floor) but the pipeline
                     produced no assignment for them.
    - False tracks : assignments produced that have no ground-truth target
                     within gating threshold.
    - Deadline miss: pipeline's solve-time exceeded the configured deadline.
                     Counted but the partial output is still scored.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class BenchMetrics:
    """Per-step metrics aggregated over a full episode."""

    id_switches: int
    missed_tracks: int
    false_tracks: int
    deadline_misses: int
    decision_latency_ms_p50: float
    decision_latency_ms_p95: float
    sample_cost_total: int
    posterior_hellinger_mean: float
    posterior_hellinger_max: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClosedLoopMetrics:
    """Composite closed-loop metrics including regret."""

    bench_metrics: BenchMetrics
    cumulative_regret: float
    pipeline_reward: float
    oracle_reward: float


def _compute_id_switches(
    pipeline_assignments_per_step: Sequence[Sequence[tuple[int, int]]],
    ground_truth_ids_per_step: Sequence[NDArray[np.int_]],
) -> int:
    switches = 0
    last_assignment: dict[int, int] = {}
    for step_idx, assignments in enumerate(pipeline_assignments_per_step):
        gt_ids = ground_truth_ids_per_step[step_idx]
        current: dict[int, int] = {}
        for track_idx, meas_idx in assignments:
            if 0 <= meas_idx < len(gt_ids):
                gt_id = int(gt_ids[meas_idx])
                current[gt_id] = track_idx
        for gt_id, track_idx in current.items():
            previous = last_assignment.get(gt_id)
            if previous is not None and previous != track_idx:
                switches += 1
        last_assignment = current
    return switches


def compute_closed_loop_metrics(
    pipeline_assignments_per_step: Sequence[Sequence[tuple[int, int]]],
    pipeline_missed_per_step: Sequence[Sequence[int]],
    pipeline_false_per_step: Sequence[Sequence[int]],
    pipeline_latency_ms_per_step: Sequence[float],
    sample_cost_per_step: Sequence[int],
    deadline_ms: float,
    ground_truth_ids_per_step: Sequence[NDArray[np.int_]],
    pipeline_rewards: Sequence[float],
    oracle_rewards: Sequence[float],
    posterior_hellinger_per_step: Sequence[float] = (),
) -> ClosedLoopMetrics:
    """Aggregate per-step pipeline outputs into closed-loop metrics."""
    n_steps = len(pipeline_assignments_per_step)
    if n_steps == 0:
        raise ValueError("at least one step required")

    id_switches = _compute_id_switches(
        pipeline_assignments_per_step, ground_truth_ids_per_step
    )
    missed = sum(len(m) for m in pipeline_missed_per_step)
    false_t = sum(len(f) for f in pipeline_false_per_step)
    latencies = np.asarray(pipeline_latency_ms_per_step, dtype=np.float64)
    deadline_misses = int(np.sum(latencies > deadline_ms))
    p50 = float(np.median(latencies)) if latencies.size else 0.0
    p95 = float(np.quantile(latencies, 0.95)) if latencies.size else 0.0
    sample_total = int(np.sum(sample_cost_per_step))
    if posterior_hellinger_per_step:
        h_arr = np.asarray(posterior_hellinger_per_step, dtype=np.float64)
        h_mean = float(np.mean(h_arr))
        h_max = float(np.max(h_arr))
    else:
        h_mean = h_max = 0.0

    bench = BenchMetrics(
        id_switches=id_switches,
        missed_tracks=missed,
        false_tracks=false_t,
        deadline_misses=deadline_misses,
        decision_latency_ms_p50=p50,
        decision_latency_ms_p95=p95,
        sample_cost_total=sample_total,
        posterior_hellinger_mean=h_mean,
        posterior_hellinger_max=h_max,
        metadata={
            "n_steps": n_steps,
            "deadline_ms": deadline_ms,
        },
    )

    pipeline_reward = float(np.sum(pipeline_rewards))
    oracle_reward = float(np.sum(oracle_rewards))
    return ClosedLoopMetrics(
        bench_metrics=bench,
        cumulative_regret=oracle_reward - pipeline_reward,
        pipeline_reward=pipeline_reward,
        oracle_reward=oracle_reward,
    )
