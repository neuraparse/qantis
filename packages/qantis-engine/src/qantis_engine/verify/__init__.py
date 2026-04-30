"""qantis_engine.verify: feasibility, dual bounds, regret, posterior fidelity.

The verification layer enforces the project's honest-framing rule (see
``feedback_honest_framing`` in QANTIS memory): every optimization output
must be accompanied by a verification report that includes:

    - ``feasibility``      : strict checker — does ``x`` satisfy all hard
                             constraints? Returns the *list* of violated
                             constraints, never just a boolean.
    - ``dual_bound``       : LP-relaxation lower bound (or Lagrangian dual)
                             for the optimal value, so we can compute an
                             optimality-gap proxy without invoking Gurobi.
    - ``regret``           : closed-loop regret evaluator vs an oracle policy.
                             This is *the* metric that lets us claim we beat
                             a Gurobi-centered pipeline under fixed deadlines.
    - ``posterior_fidelity``: Hellinger / TV distance between an estimated
                              posterior and the analytic Bayes update.
"""
from __future__ import annotations

from qantis_engine.verify.dual_bound import (
    DualBoundReport,
    lp_assignment_lower_bound,
)
from qantis_engine.verify.feasibility import FeasibilityReport, check_assignment
from qantis_engine.verify.posterior_fidelity import (
    FidelityReport,
    posterior_fidelity,
)
from qantis_engine.verify.regret import RegretReport, closed_loop_regret

__all__ = [
    "FeasibilityReport",
    "check_assignment",
    "DualBoundReport",
    "lp_assignment_lower_bound",
    "RegretReport",
    "closed_loop_regret",
    "FidelityReport",
    "posterior_fidelity",
]
