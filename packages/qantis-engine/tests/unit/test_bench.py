"""Tests for the benchmark harness."""
from __future__ import annotations

import numpy as np

from qantis_engine.bench import (
    BenchConfig,
    compute_closed_loop_metrics,
    generate_rare_event_mtda,
    run_benchmark,
)
from qantis_engine.bench.pipelines import (
    run_classical_pf_gurobi,
    run_qantis_full,
    run_qantis_infer_gurobi,
    run_smc_gurobi,
)


def test_rare_event_mtda_scenario_shape() -> None:
    sc = generate_rare_event_mtda(n_steps=5, n_targets=3, p_miss=0.2, seed=0)
    assert len(sc.frames) == 5
    for frame in sc.frames:
        assert frame.cost_matrix.shape[0] == 3


def test_compute_closed_loop_metrics_basic() -> None:
    n_steps = 4
    assignments = [[(0, 0), (1, 1)] for _ in range(n_steps)]
    missed: list[list[int]] = [[] for _ in range(n_steps)]
    false_alarms: list[list[int]] = [[] for _ in range(n_steps)]
    latency = [10.0, 20.0, 30.0, 40.0]
    sample_cost = [0, 0, 0, 0]
    gt_ids = [np.array([0, 1], dtype=np.int_) for _ in range(n_steps)]
    rewards = [-1.0] * n_steps
    oracle = [-0.5] * n_steps

    m = compute_closed_loop_metrics(
        pipeline_assignments_per_step=assignments,
        pipeline_missed_per_step=missed,
        pipeline_false_per_step=false_alarms,
        pipeline_latency_ms_per_step=latency,
        sample_cost_per_step=sample_cost,
        deadline_ms=25.0,
        ground_truth_ids_per_step=gt_ids,
        pipeline_rewards=rewards,
        oracle_rewards=oracle,
    )
    assert m.bench_metrics.id_switches == 0
    assert m.bench_metrics.deadline_misses == 2
    assert m.cumulative_regret == n_steps * (-0.5 - (-1.0))


def test_pipeline_a_classical_pf_runs() -> None:
    sc = generate_rare_event_mtda(n_steps=4, n_targets=3, p_miss=0.15, seed=1)
    cfg = BenchConfig(
        scenario_name="rare_event_mtda",
        pipeline_name="classical_pf_gurobi",
        deadline_ms=100.0,
        seed=1,
        n_steps=4,
    )
    out = run_classical_pf_gurobi(cfg, sc)
    assert len(out["assignments"]) == 4
    assert len(out["latency_ms"]) == 4


def test_pipeline_b_smc_runs() -> None:
    sc = generate_rare_event_mtda(n_steps=3, n_targets=2, seed=2)
    cfg = BenchConfig(
        scenario_name="rare_event_mtda",
        pipeline_name="smc_gurobi",
        deadline_ms=100.0,
        seed=2,
        n_steps=3,
    )
    out = run_smc_gurobi(cfg, sc)
    assert len(out["assignments"]) == 3


def test_pipeline_c_qantis_infer_runs() -> None:
    sc = generate_rare_event_mtda(n_steps=2, n_targets=2, seed=3)
    cfg = BenchConfig(
        scenario_name="rare_event_mtda",
        pipeline_name="qantis_infer_gurobi",
        deadline_ms=200.0,
        seed=3,
        n_steps=2,
    )
    out = run_qantis_infer_gurobi(cfg, sc)
    assert len(out["assignments"]) == 2
    assert sum(out["sample_cost"]) > 0


def test_pipeline_d_qantis_full_runs() -> None:
    sc = generate_rare_event_mtda(n_steps=2, n_targets=2, seed=4)
    cfg = BenchConfig(
        scenario_name="rare_event_mtda",
        pipeline_name="qantis_full",
        deadline_ms=200.0,
        seed=4,
        n_steps=2,
    )
    out = run_qantis_full(cfg, sc)
    assert len(out["assignments"]) == 2


def test_run_benchmark_end_to_end() -> None:
    cfg = BenchConfig(
        scenario_name="rare_event_mtda",
        pipeline_name="classical_pf_gurobi",
        deadline_ms=100.0,
        seed=5,
        n_steps=3,
    )

    def factory(n_steps: int, seed: int):
        return generate_rare_event_mtda(n_steps=n_steps, n_targets=2, seed=seed)

    res = run_benchmark(
        scenario_factory=factory,
        pipeline_factory=run_classical_pf_gurobi,
        config=cfg,
    )
    assert res.metrics.bench_metrics.metadata["n_steps"] == 3
    record = res.to_dict()
    assert "config" in record
    assert "metrics" in record
