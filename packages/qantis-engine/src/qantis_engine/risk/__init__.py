"""qantis_engine.risk: rare-event / tail-risk / chance-constraint estimator.

Wraps the BIQAE primitives from ``quantum_pomdp.algorithms.biqae_estimator``
into a risk-product API: the optimizer can pose questions like
"what is P(collision) under this policy with 95% credibility?" and receive
both a point estimate and a credible interval.

Academic anchors (all 2026, verified on arXiv):
    - arXiv:2602.09059 (Srikant, Feb 2026) — Quantum Estimation of Delay Tail
      Probabilities in Scheduling and Load Balancing. Quadratic 1/sqrt(p) vs
      1/p sample-complexity in the rare-event regime via truncated regenerative
      simulation.
    - arXiv:2602.09847 (Tabarraei, Feb 2026) — Stabilized MLE-IQAE for
      Structural CVaR under Correlated Random Fields. CVaR-via-bounded-
      expectation reformulation that plugs into amplitude estimation.
    - arXiv:2603.15664 (Mar 2026) — QAE for Catastrophe Insurance Tail-Risk
      Pricing. NISQ-noise-aware empirical convergence study.
    - arXiv:2604.20088 (Lee-Lau, Apr 2026) — CVaR-Assisted Custom Penalty for
      Constrained Optimization. Slack-free nonlinear penalty templates.
    - arXiv:2512.03925 (Dec 2025) — Joint Chance Constraints with Quantum
      Annealing. Direct D-Wave-vs-Gurobi precedent on chance-constrained UC.
"""
from __future__ import annotations

from qantis_engine.risk.chance_constraint import ChanceConstraint, ChanceConstraintReport
from qantis_engine.risk.cvar import CVaRReport, estimate_cvar
from qantis_engine.risk.delay_tail import DelayTailReport, estimate_delay_tail
from qantis_engine.risk.tail_probability import TailProbabilityReport, tail_probability

__all__ = [
    "tail_probability",
    "TailProbabilityReport",
    "estimate_cvar",
    "CVaRReport",
    "ChanceConstraint",
    "ChanceConstraintReport",
    "estimate_delay_tail",
    "DelayTailReport",
]
