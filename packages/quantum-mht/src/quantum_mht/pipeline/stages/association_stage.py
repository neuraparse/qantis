"""Association stage: QUBO build + quantum/classical solve.

Implements the core data association step of the MHT pipeline by:
    1. Building the MTDA QUBO matrix from the cost matrix and constraints
       (Stollenwerk et al., arXiv:2110.08346, Sec III-IV).
    2. Solving the QUBO using the configured solver (quantum annealing,
       QAOA, or classical baseline).

This is Stage 3 of the predict -> gate -> QUBO -> solve -> update pipeline.

Academic References:
    Stollenwerk et al., "Adiabatic Quantum Computing for Multi Object Tracking",
        Fraunhofer FKIE, arXiv:2110.08346, 2021 -- QUBO-based MTDA association.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995 -- data
        association problem formulation.
"""
from __future__ import annotations
from dataclasses import dataclass
from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder, QUBOResult
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult
import numpy as np
from numpy.typing import NDArray

@dataclass
class AssociationStage:
    qubo_builder: MTDAQuboBuilder
    solver: MTDASolver

    def process(self, cost_matrix: NDArray[np.float64], gate_mask: NDArray[np.bool_] | None = None) -> SolverResult:
        """Build QUBO from cost matrix and solve (arXiv:2110.08346)."""
        qubo = self.qubo_builder.build_from_cost_matrix(cost_matrix, gate_mask)
        return self.solver.solve(qubo)
