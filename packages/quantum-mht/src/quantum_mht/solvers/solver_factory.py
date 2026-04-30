"""Solver factory with strategy pattern.

Implements the Strategy design pattern for MTDA solver selection. The factory
enables runtime switching between quantum and classical solvers without
modifying pipeline code.

Solver Selection Criteria:
    - Problem size (qubits): Small (<50) -> direct QPU or QAOA;
      Medium (50-200) -> FPC-QAOA, XY-mixer QAOA, RQAOA, or annealing;
      Large (>200) -> LeapHybrid (BQM) or LeapHybridNL (nonlinear).
    - Hardware availability: D-Wave QPU, IBM Quantum, or classical-only.
    - Accuracy requirements: Hungarian (optimal baseline) vs quantum (heuristic).
    - Latency constraints: GNN (fastest) vs annealing (moderate) vs QAOA (slowest).
    - Classical baseline tier: Gurobi 13.0 (commercial), cuOpt 26.02 (GPU),
      Hungarian (pure Python / scipy).

Available Solvers:
    Quantum:
        - "annealing"   : D-Wave quantum annealing (AnnealingSolver)
        - "qaoa"        : IBM Quantum QAOA (QAOASolver, Farhi et al. 2014)
        - "fpc_qaoa"    : FPC-QAOA (arXiv:2512.21181, Saavedra-Pino et al. Dec 2025)
        - "xy_mixer_qaoa": XY-mixer IWS-QAOA for one-hot constraints
                          (arXiv:2604.02083, Apr 2026)
        - "recursive_qaoa": Recursive QAOA (arXiv:2602.07483, 2026);
                          variable-elimination via correlation contraction
        - "hybrid"      : D-Wave LeapHybridBQM (HybridSolver)
        - "hybrid_nl"   : D-Wave LeapHybridNL (HybridSolver, nl_mode=True)
    Classical:
        - "hungarian"   : Optimal 2D assignment (Kuhn 1955, O(n^3))
        - "gnn"         : Global Nearest Neighbor greedy (Blackman 1986)
        - "cuopt"       : NVIDIA cuOpt 26.02 MIP baseline (GPU)
        - "gurobi"      : Gurobi 13.0 MIP baseline (commercial MIP SOTA)

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
        "xy_mixer_qaoa": _create_xy_mixer_qaoa,
        "recursive_qaoa": _create_recursive_qaoa,
        "para_qaoa": _create_para_qaoa,
        "iceberg_qaoa": _create_iceberg_qaoa,
        "hybrid": _create_hybrid,
        "hybrid_nl": _create_hybrid_nl,
        "hybrid_cqm": _create_hybrid_cqm,
        "hungarian": _create_hungarian,
        "gnn": _create_gnn,
        "cuopt": _create_cuopt,
        "gurobi": _create_gurobi,
        "pticm": _create_pticm,
        "fixstars": _create_fixstars,
        "tn_baseline": _create_tn_baseline,
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

def _create_xy_mixer_qaoa(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.xy_mixer_qaoa_solver import XYMixerQAOASolver
    return XYMixerQAOASolver(**kwargs)

def _create_recursive_qaoa(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.recursive_qaoa_solver import RecursiveQAOASolver
    return RecursiveQAOASolver(**kwargs)

def _create_para_qaoa(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.para_qaoa_solver import ParaQAOASolver
    return ParaQAOASolver(**kwargs)

def _create_iceberg_qaoa(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.iceberg_qaoa_solver import IcebergQAOASolver
    return IcebergQAOASolver(**kwargs)

def _create_hybrid(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.hybrid_solver import HybridSolver
    kwargs.setdefault("nl_mode", False)
    return HybridSolver(**kwargs)

def _create_hybrid_nl(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.hybrid_solver import HybridSolver
    kwargs["nl_mode"] = True
    return HybridSolver(**kwargs)

def _create_hybrid_cqm(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.hybrid_solver import HybridSolver
    kwargs["cqm_mode"] = True
    return HybridSolver(**kwargs)

def _create_hungarian(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver
    return HungarianSolver(**kwargs)

def _create_gnn(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.gnn_solver import GNNSolver
    return GNNSolver(**kwargs)

def _create_cuopt(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.cuopt_solver import CuOptSolver
    return CuOptSolver(**kwargs)

def _create_gurobi(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.gurobi_solver import GurobiSolver
    return GurobiSolver(**kwargs)

def _create_pticm(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.pticm_solver import PTICMSolver
    return PTICMSolver(**kwargs)

def _create_fixstars(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.fixstars_solver import (
        FixstarsAmplifySolver,
    )
    return FixstarsAmplifySolver(**kwargs)

def _create_tn_baseline(**kwargs) -> MTDASolver:
    from quantum_mht.solvers.classical_solvers.tn_baseline_solver import (
        TensorNetworkBaselineSolver,
    )
    return TensorNetworkBaselineSolver(**kwargs)
