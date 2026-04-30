"""Classical repair + local search after quantum sampling.

Every quantum-emitted candidate is repaired to feasibility before scoring.
This is intentional: we never claim the sampler alone produces feasibility,
because in any noisy sampling regime occasional infeasible bitstrings will
appear and silent acceptance corrupts benchmarks.
"""
from __future__ import annotations

from qantis_engine.optimize.repair.feasibility_repair import (
    AssignmentRepair,
    PermutationRepair,
    repair_assignment,
)
from qantis_engine.optimize.repair.local_search import swap_search, two_opt

__all__ = [
    "AssignmentRepair",
    "PermutationRepair",
    "repair_assignment",
    "two_opt",
    "swap_search",
]
