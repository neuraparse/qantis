"""Benchmark pipelines for QANTIS-vs-Gurobi MTDA closed-loop comparison.

Four pipelines, all sharing the same scenario stream and seed:

    A : classical_pf_gurobi   — particle filter + Hungarian assignment
    B : smc_gurobi            — SMC + Hungarian assignment (heavier filter)
    C : qantis_infer_gurobi   — QANTIS-Infer (BIQAE) + Hungarian
    D : qantis_full           — QANTIS-Infer + QANTIS-Optimize (hybrid B&B)

The Hungarian solver is used in place of Gurobi as the *deterministic
optimisation* baseline in the assignment regime: Hungarian is
provably optimal (Kuhn 1955) and the LP relaxation has the integrality
property, so for pure assignment Hungarian == Gurobi. For chance-
constrained extensions we add a Gurobi adapter behind a feature flag.
"""
from __future__ import annotations

from qantis_engine.bench.pipelines.classical_pf_gurobi import (
    PipelineOutputs,
    run_classical_pf_gurobi,
)
from qantis_engine.bench.pipelines.qantis_full import run_qantis_full
from qantis_engine.bench.pipelines.qantis_infer_gurobi import run_qantis_infer_gurobi
from qantis_engine.bench.pipelines.smc_gurobi import run_smc_gurobi

__all__ = [
    "PipelineOutputs",
    "run_classical_pf_gurobi",
    "run_smc_gurobi",
    "run_qantis_infer_gurobi",
    "run_qantis_full",
]
