"""qantis_engine.bench: QANTIS-vs-Gurobi closed-loop benchmark harness.

The harness implements the v0 specification documented in the architecture
proposal:

    Scenarios :
        - rare_event_mtda                : multi-target tracking with low SNR
                                           and rare-observation regime
        - chance_constrained_planning    : robot/drone path planning under
                                           rare hazard events
        - pomdp_mpc                      : belief-state MPC for short-horizon
                                           replanning under partial obs

    Pipelines :
        A : classical_pf_gurobi          (particle filter + Hungarian)
        B : smc_gurobi                   (SMC + Hungarian)
        C : qantis_infer_gurobi          (QANTIS-Infer + Hungarian)
        D : qantis_full                  (QANTIS-Infer + QANTIS-Optimize)

    Metrics :
        - id_switches, missed_tracks, false_tracks
        - decision_latency_ms, deadline_miss_rate
        - sample_cost (oracle calls)
        - posterior_hellinger
        - closed_loop_regret

The runner orchestrates the seed * deadline * pipeline grid and writes a
single JSON record per run, suitable for downstream analysis with the
existing ``quantum_common.benchmarking`` infrastructure.
"""
from __future__ import annotations

from qantis_engine.bench.metrics import (
    BenchMetrics,
    ClosedLoopMetrics,
    compute_closed_loop_metrics,
)
from qantis_engine.bench.runner import BenchConfig, BenchResult, run_benchmark
from qantis_engine.bench.scenarios.rare_event_mtda import (
    MTDAScenario,
    generate_rare_event_mtda,
)

__all__ = [
    "BenchMetrics",
    "ClosedLoopMetrics",
    "compute_closed_loop_metrics",
    "BenchConfig",
    "BenchResult",
    "run_benchmark",
    "MTDAScenario",
    "generate_rare_event_mtda",
]
