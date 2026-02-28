"""Solver factory with strategy pattern.

Implements the Strategy design pattern for MTDA solver selection. The factory
enables runtime switching between quantum and classical solvers without
modifying pipeline code.

Solver Selection Criteria:
    - Problem size (qubits): Small (<50) -> direct QPU or QAOA;
      Medium (50-200) -> FPC-QAOA or annealing; Large (>200) -> LeapHybrid.
    - Hardware availability: D-Wave QPU, IBM Quantum, or classical-only.
    - Accuracy requirements: Hungarian (optimal baseline) vs quantum (heuristic).
    - Latency constraints: GNN (fastest) vs annealing (moderate) vs QAOA (slowest).

Available Solvers:
    - "annealing": D-Wave quantum annealing (AnnealingSolver)
    - "qaoa": IBM Quantum QAOA (QAOASolver, Farhi et al. 2014)
    - "fpc_qaoa": FPC-QAOA (arXiv:2512.21181, Saavedra-Pino et al. Dec 2025)
    - "hybrid": D-Wave LeapHybrid (HybridSolver)
    - "hungarian": Optimal 2D assignment (Kuhn 1955, O(n^3))
    - "gnn": Global Nearest Neighbor greedy (Blackman 1986)

Academic References:
    Stollenwerk et al., arXiv:2110.08346, 2021 -- QUBO-based MTDA.
    Gamma et al., "Design Patterns", 1994 -- Strategy pattern.
"""
from __future__ import annotations
from quantum_mht.solvers.base_solver import MTDASolver

def create_solver(solver_type: str, **kwargs) -> MTDASolver:
    """Create a solver by type name (Strategy pattern).

    See module docstring for solver selection criteria and available types.
    """
    solvers = {
        "annealing": _create_annealing,
        "qaoa": _create_qaoa,
        "fpc_qaoa": _create_fpc_qaoa,
        "hybrid": _create_hybrid,
        "hungarian": _create_hungarian,
        "gnn": _create_gnn,
    }
    if solver_type not in solvers:
        raise ValueError(f"Unknown solver: {solver_type}. Available: {list(solvers.keys())}")
    return solvers[solver_type](**kwargs)

def _create_annealing(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.annealing_solver import AnnealingSolver
    return AnnealingSolver(**kwargs)

def _create_qaoa(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.qaoa_solver import QAOASolver
    return QAOASolver(**kwargs)

def _create_fpc_qaoa(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.fpc_qaoa_solver import FPCQAOASolver
    return FPCQAOASolver(**kwargs)

def _create_hybrid(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.hybrid_solver import HybridSolver
    return HybridSolver(**kwargs)

def _create_hungarian(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver
    return HungarianSolver(**kwargs)

def _create_gnn(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.gnn_solver import GNNSolver
    return GNNSolver(**kwargs)
