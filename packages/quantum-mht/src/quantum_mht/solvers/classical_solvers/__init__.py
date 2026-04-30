"""Classical baseline solvers for MTDA.

Includes the full 2024-2026 peer-reviewed baseline matrix:

    - Hungarian (Kuhn 1955) -- optimal assignment, scipy-backed.
    - JPDA (Bar-Shalom & Li 1995) -- soft-association baseline.
    - Reid MHT + Murty k-best -- multi-hypothesis tracker.
    - GNN greedy -- latency-minimal baseline.
    - NVIDIA cuOpt 26.02 -- GPU MIP baseline (optional GPU install).
    - Gurobi 13.0 -- commercial MIP SOTA (optional license).
    - PT-ICM (Aadit et al. Nat. Commun. 16:2025) -- Numba-accelerated
      Katzgraber-group parallel tempering + Houdayer isoenergetic
      cluster moves; the most critical reviewer-ammunition classical
      baseline of 2025-2026.
    - Fixstars Amplify v1.5+ adapter -- unified gateway to Amplify AE,
      Toshiba SQBM+, Fujitsu DA, Hitachi CMOS behind one API key.
    - Tensor-network classical baseline (quimb loopy-BP) -- pre-empts
      the Tindall/Patra/Begusic rebuttal pattern.
"""
from quantum_mht.solvers.classical_solvers.gnn_solver import GNNSolver
from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver
from quantum_mht.solvers.classical_solvers.jpda_solver import JPDASolver
from quantum_mht.solvers.classical_solvers.reid_mht_solver import ReidMHTSolver
from quantum_mht.solvers.classical_solvers.cuopt_solver import CuOptSolver
from quantum_mht.solvers.classical_solvers.gurobi_solver import GurobiSolver
from quantum_mht.solvers.classical_solvers.pticm_solver import PTICMSolver
from quantum_mht.solvers.classical_solvers.fixstars_solver import (
    FixstarsAmplifySolver,
)
from quantum_mht.solvers.classical_solvers.tn_baseline_solver import (
    TensorNetworkBaselineSolver,
)

__all__ = [
    "GNNSolver",
    "HungarianSolver",
    "JPDASolver",
    "ReidMHTSolver",
    "CuOptSolver",
    "GurobiSolver",
    "PTICMSolver",
    "FixstarsAmplifySolver",
    "TensorNetworkBaselineSolver",
]
