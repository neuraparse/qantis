"""Solvers for the constraint-native optimizer.

    - ``feasible_sampler``    : unified sampler interface backed by XY-mixer
      QAOA, FPC-QAOA, or a classical Hamming-weight Gibbs sampler fallback.
    - ``hybrid_bnb``          : classical branch-and-bound where the quantum
      sampler is the primal heuristic feeding incumbent solutions.
    - ``benders_qc``          : Benders decomposition with the quantum
      sub-problem (arXiv:2601.14024).
    - ``lagrangian_decomp``   : Lagrangian decomposition with quantum-handled
      sub-problems (arXiv:2604.22194).
"""
from __future__ import annotations

from qantis_engine.optimize.solvers.benders_qc import BendersQCSolver
from qantis_engine.optimize.solvers.feasible_sampler import (
    FeasibleSampler,
    FeasibleSamplerResult,
    HammingWeightGibbsSampler,
)
from qantis_engine.optimize.solvers.hybrid_bnb import HybridBnBSolver
from qantis_engine.optimize.solvers.lagrangian_decomp import LagrangianDecomposition

__all__ = [
    "FeasibleSampler",
    "FeasibleSamplerResult",
    "HammingWeightGibbsSampler",
    "HybridBnBSolver",
    "BendersQCSolver",
    "LagrangianDecomposition",
]
