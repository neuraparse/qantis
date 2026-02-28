"""Quantum Multi-Hypothesis Tracking (MHT) package.

Hybrid quantum-classical pipeline for multi-target data association:
    Classical: Kalman prediction, chi-squared gating, track management
    Quantum:   QUBO optimization via D-Wave annealing or QAOA

Key modules:
    formulation:  MTDA-to-QUBO conversion (arXiv:2110.08346)
    solvers:      Quantum (annealing, QAOA) and classical (Hungarian, MHT) solvers
    tracking:     Kalman filtering, gating, track lifecycle management
    fusion:       Multi-sensor fusion (centralized + Covariance Intersection)
    simulation:   Benchmark scenario generation (CV/CT/CA dynamics)
    pipeline:     End-to-end predict -> gate -> solve -> update pipeline stages
"""
__version__ = "0.1.0"

__all__ = [
    "__version__",
]
