"""qantis_engine.optimize: constraint-native quantum/hybrid optimizer.

The four central choices that distinguish QANTIS-Optimize from a Gurobi-style
QUBO+penalty pipeline:

    1. **Feasible-subspace encoding** (encodings/) instead of penalty-QUBO.
       Hamming-weight operators (arXiv:2601.01516), valid-permutation oracles
       (arXiv:2603.21283), colored-permutation flow encodings (arXiv:2604.04570),
       and generalized-Dicke states (arXiv:2601.16396) hard-confine the
       evolution to feasible bitstrings. Penalty distortion never enters the
       cost landscape.

    2. **Constraint-preserving sampler** (solvers/feasible_sampler) — wraps
       the existing XYMixerQAOA / FPCQAOA / Iceberg solvers as primal
       heuristics that produce a stream of *feasible* candidates with
       associated cost values.

    3. **Decomposition over monolithic QUBO** (solvers/hybrid_bnb,
       solvers/benders_qc, solvers/lagrangian_decomp) — the quantum sampler
       owns a bounded-width subproblem, while a classical master tracks the
       global solution. arXiv:2509.11040, arXiv:2511.19501, arXiv:2603.21374,
       arXiv:2604.22194.

    4. **Classical repair + verify after sampling** (repair/) — every
       quantum-emitted candidate is repaired to feasibility (Hungarian for
       matching, 2-opt for routing) before scoring. We never claim feasibility
       from the sampler alone.

The high-level entry point is :func:`solve` in ``optimize.api``.
"""
from __future__ import annotations

from qantis_engine.optimize import encodings, repair, solvers
from qantis_engine.optimize.api import OptimizeReport, solve

__all__ = ["solve", "OptimizeReport", "encodings", "solvers", "repair"]
