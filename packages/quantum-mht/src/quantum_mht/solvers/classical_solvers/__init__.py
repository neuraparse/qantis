"""Classical baseline solvers for MTDA.

Provides GNN, Hungarian, JPDA, and Reid MHT solvers as classical baselines
for benchmarking quantum annealing and QAOA solvers.
"""
from quantum_mht.solvers.classical_solvers.gnn_solver import GNNSolver
from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver
from quantum_mht.solvers.classical_solvers.jpda_solver import JPDASolver
from quantum_mht.solvers.classical_solvers.reid_mht_solver import ReidMHTSolver

__all__ = [
    "GNNSolver",
    "HungarianSolver",
    "JPDASolver",
    "ReidMHTSolver",
]
