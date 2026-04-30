"""QANTIS - Quantum Autonomous Navigation, Tracking & Intelligence System.

QANTIS is organized as a decision engine for stochastic, partially observable,
rare-event autonomy loops: infer calibrated beliefs, estimate risk, optimize in
native feasible spaces, and verify the resulting action.
"""

from qantis.decision_engine import AssignmentDecisionReport, QANTISDecisionEngine
from qantis.optimize import ConstraintNativeAssignmentOptimizer, FeasibleAssignmentResult
from qantis.risk import EventProbabilityResult, QANTISRisk, TailRiskResult
from qantis.verify import FeasibilityReport, PosteriorFidelityReport, QANTISVerify

__version__ = "0.1.0"

__all__ = [
    "AssignmentDecisionReport",
    "ConstraintNativeAssignmentOptimizer",
    "EventProbabilityResult",
    "FeasibilityReport",
    "FeasibleAssignmentResult",
    "PosteriorFidelityReport",
    "QANTISDecisionEngine",
    "QANTISRisk",
    "QANTISVerify",
    "TailRiskResult",
    "__version__",
]
