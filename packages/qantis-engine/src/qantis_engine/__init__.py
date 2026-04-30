"""QANTIS Decision Engine.

Four-pillar API for stochastic decision problems under partial observability,
rare observations, and chance constraints:

    qantis_engine.infer    : belief update / posterior conditioning
    qantis_engine.risk     : rare-event probability / CVaR / chance constraints
    qantis_engine.optimize : constraint-native quantum/hybrid optimizer
    qantis_engine.verify   : feasibility, dual bound, closed-loop regret

Built on top of the quantum_pomdp and quantum_mht packages and their
hardware-validated primitives (BIQAE, AB-FPAA, XY-mixer QAOA, Iceberg [[4,2,2]],
FPC-QAOA). The 2026 literature anchors are documented in module docstrings.

Top-level slogan: "Gurobi solves deterministic models. QANTIS solves the
belief-to-action loop under fixed decision and sample budgets."
"""
from __future__ import annotations

from qantis_engine import bench, infer, optimize, risk, verify

__all__ = ["infer", "risk", "optimize", "verify", "bench"]
__version__ = "0.1.0"
