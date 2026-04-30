"""QANTIS Decision Engine.

The engine is organized around four cores:

* QANTIS-Infer: posterior belief updates from noisy observations.
* QANTIS-Risk: rare-event probability and tail-risk estimation.
* QANTIS-Optimize: constraint-native action/assignment optimization.
* QANTIS-Verify: feasibility, posterior-fidelity, and repair diagnostics.

This framing keeps QANTIS out of the weak "quantum replacement for Gurobi"
position.  Deterministic solvers optimize deterministic models; QANTIS targets
the online belief-to-action loop where uncertainty, rare observations, and
closed-loop verification are first-class objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qantis.optimize import ConstraintNativeAssignmentOptimizer, FeasibleAssignmentResult
from qantis.risk import EventProbabilityResult, QANTISRisk
from qantis.verify import FeasibilityReport, PosteriorFidelityReport, QANTISVerify


@dataclass(frozen=True)
class AssignmentDecisionReport:
    """End-to-end assignment decision with verification metadata."""

    optimization: FeasibleAssignmentResult
    feasibility: FeasibilityReport
    risk: EventProbabilityResult | None = None
    posterior_fidelity: PosteriorFidelityReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QANTISDecisionEngine:
    """Composable inference/risk/optimization/verification facade."""

    risk: QANTISRisk = field(default_factory=QANTISRisk)
    verifier: QANTISVerify = field(default_factory=QANTISVerify)
    assignment_optimizer: ConstraintNativeAssignmentOptimizer = field(
        default_factory=ConstraintNativeAssignmentOptimizer
    )

    def infer(
        self,
        belief_updater: Any,
        belief: Any,
        *,
        action: int,
        observation: int,
    ) -> Any:
        """Run a pluggable QANTIS-Infer updater.

        ``belief_updater`` can be the existing ``QuantumBeliefUpdater`` or a
        classical test double.  Keeping this dependency inverted lets the
        product API stay stable while hardware backends evolve.
        """

        if not hasattr(belief_updater, "update"):
            raise TypeError("belief_updater must expose update(belief, action, observation)")
        return belief_updater.update(belief, action, observation)

    def estimate_event_probability(self, *args: Any, **kwargs: Any) -> EventProbabilityResult:
        """Delegate to QANTIS-Risk."""

        return self.risk.estimate_event_probability(*args, **kwargs)

    def optimize_assignment(
        self,
        cost_matrix: NDArray[np.float64] | list[list[float]],
        *,
        gate_mask: NDArray[np.bool_] | list[list[bool]] | None = None,
        belief: Any | None = None,
        event: Any | None = None,
        event_name: str = "decision_risk",
    ) -> AssignmentDecisionReport:
        """Optimize an assignment decision and verify it.

        Optional ``belief`` + ``event`` arguments attach a risk report to the
        same decision, enabling fixed-latency benchmark comparisons against
        deterministic optimization pipelines.
        """

        optimization = self.assignment_optimizer.solve(cost_matrix, gate_mask)
        costs = np.asarray(cost_matrix, dtype=float)
        feasibility = self.verifier.check_assignment(
            optimization.assignments,
            n_tracks=costs.shape[0],
            n_measurements=costs.shape[1],
            cost_matrix=costs,
        )
        risk_report = None
        if belief is not None and event is not None:
            risk_report = self.risk.estimate_event_probability(
                belief,
                event,
                event_name=event_name,
                mode="biqae_boundary_aware",
            )
        return AssignmentDecisionReport(
            optimization=optimization,
            feasibility=feasibility,
            risk=risk_report,
            metadata={
                "engine": "QANTIS Decision Engine",
                "claim_scope": "belief-to-action loop, not general-purpose MIP",
            },
        )

