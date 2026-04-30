"""End-to-end smoke run of the QANTIS Decision Engine.

Spins up a 20-step rare-event MTDA scenario, runs all four benchmark
pipelines (A/B/C/D), and prints a side-by-side comparison of:
    - cumulative regret
    - id switches
    - decision latency p95
    - sample cost
    - posterior Hellinger (only for C, D)

Run with:

    .venv/bin/python scripts/qantis_engine_smoke.py

This is the user-facing demo that the architecture is wired end to end and
the four-pillar API integrates cleanly. It is NOT a research benchmark —
the seed grid and step count are tiny so it completes in a few seconds.
"""
from __future__ import annotations

import json
from typing import Any

from qantis_engine.bench import (
    BenchConfig,
    generate_rare_event_mtda,
    run_benchmark,
)
from qantis_engine.bench.pipelines import (
    run_classical_pf_gurobi,
    run_qantis_full,
    run_qantis_infer_gurobi,
    run_smc_gurobi,
)

PIPELINES = [
    ("A_classical_pf_gurobi", run_classical_pf_gurobi),
    ("B_smc_gurobi", run_smc_gurobi),
    ("C_qantis_infer_gurobi", run_qantis_infer_gurobi),
    ("D_qantis_full", run_qantis_full),
]


def main() -> None:
    n_steps = 20
    n_targets = 4
    p_miss = 0.15
    deadline_ms = 100.0

    rows: list[dict[str, Any]] = []
    for name, fn in PIPELINES:
        cfg = BenchConfig(
            scenario_name="rare_event_mtda",
            pipeline_name=name,
            deadline_ms=deadline_ms,
            seed=42,
            n_steps=n_steps,
        )
        result = run_benchmark(
            scenario_factory=lambda n, s: generate_rare_event_mtda(
                n_steps=n, n_targets=n_targets, p_miss=p_miss, seed=s
            ),
            pipeline_factory=fn,
            config=cfg,
        )
        m = result.metrics
        rows.append(
            {
                "pipeline": name,
                "regret": round(m.cumulative_regret, 3),
                "id_switches": m.bench_metrics.id_switches,
                "missed": m.bench_metrics.missed_tracks,
                "false_tracks": m.bench_metrics.false_tracks,
                "p95_latency_ms": round(m.bench_metrics.decision_latency_ms_p95, 3),
                "deadline_misses": m.bench_metrics.deadline_misses,
                "sample_cost": m.bench_metrics.sample_cost_total,
                "post_h_max": round(m.bench_metrics.posterior_hellinger_max, 4),
            }
        )

    width = 24
    cols = list(rows[0].keys())
    header = " | ".join(c.rjust(width if c == "pipeline" else 12) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(
            str(r[c]).rjust(width if c == "pipeline" else 12) for c in cols
        )
        print(line)
    print()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
