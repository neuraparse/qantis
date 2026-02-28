"""Experiment: QUBO scaling analysis.

Analyzes how QUBO problem size scales with number of targets and measurements.
Measures variable count, constraint count, and solve time.

QUBO Scaling (arXiv:2110.08346, Table 1):
    For N tracks and M measurements, total QUBO variables = N*M + N + M.
    The number of quadratic terms in the QUBO matrix grows as O((N*M)^2) in
    the worst case (dense constraints). Gating reduces this significantly by
    zeroing out infeasible assignments.

    Problem Size Scaling:
        N=3,  M=5  ->  23 qubits,   ~500 QUBO terms
        N=5,  M=8  ->  53 qubits,   ~2800 QUBO terms
        N=10, M=15 -> 175 qubits,   ~30K QUBO terms
        N=20, M=30 -> 650 qubits,   ~420K QUBO terms (Advantage2 feasible)
        N=50, M=75 -> 3875 qubits,  ~15M QUBO terms (LeapHybrid required)

    This experiment empirically validates these scaling predictions and measures
    actual solve times for different solver backends.

Academic References:
    Stollenwerk et al., "Adiabatic Quantum Computing for Multi Object Tracking",
        Fraunhofer FKIE, arXiv:2110.08346, 2021, Table 1 -- QUBO scaling
        analysis for MTDA problems.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995 -- problem
        size characterization for multi-target association.

Usage:
    uv run python -m quantum_mht.experiments.exp_qubo_scaling
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from quantum_mht.formulation.cost_matrix import CostMatrixBuilder
from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder
from quantum_mht.solvers.solver_factory import create_solver

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Result from a single scaling experiment (arXiv:2110.08346, Table 1)."""
    n_targets: int
    n_measurements: int
    n_qubo_variables: int
    penalty: float
    build_time_s: float
    solve_time_s: float
    solver_name: str
    objective_value: float
    is_feasible: bool


def run_scaling_experiment(
    target_counts: list[int] | None = None,
    meas_ratio: float = 1.5,
    solvers: list[str] | None = None,
    seed: int = 42,
) -> list[ScalingResult]:
    """Run QUBO scaling experiment across problem sizes.

    Args:
        target_counts: List of target counts to test.
        meas_ratio: Measurements per target ratio.
        solvers: List of solver names to benchmark.
        seed: Random seed for reproducibility.
    """
    if target_counts is None:
        target_counts = [3, 5, 8, 10, 15]
    if solvers is None:
        solvers = ["hungarian", "gnn"]

    rng = np.random.default_rng(seed)
    results: list[ScalingResult] = []

    for n_targets in target_counts:
        n_meas = int(n_targets * meas_ratio)
        logger.info("Testing %d targets x %d measurements", n_targets, n_meas)

        # Generate synthetic data
        predicted_states = rng.uniform(0, 100, size=(n_targets, 2))
        measurements = predicted_states + rng.normal(0, 2, size=(n_targets, 2))
        # Add extra measurements (clutter)
        extra = rng.uniform(0, 100, size=(n_meas - n_targets, 2))
        measurements = np.vstack([measurements, extra]) if n_meas > n_targets else measurements[:n_meas]
        covariances = np.array([np.eye(2) * 4.0 for _ in range(n_targets)])

        # Build QUBO (arXiv:2110.08346: N*M + N + M variables)
        builder = MTDAQuboBuilder()
        t0 = time.perf_counter()
        cost_matrix = builder.cost_builder.build(predicted_states, measurements, covariances)
        qubo_result = builder.build_from_cost_matrix(cost_matrix)
        build_time = time.perf_counter() - t0

        logger.info(
            "  QUBO: %d variables, penalty=%.2f, build_time=%.4fs",
            qubo_result.num_variables, qubo_result.penalty, build_time,
        )

        # Solve with each solver
        for solver_name in solvers:
            try:
                solver = create_solver(solver_name)
                solver_result = solver.solve(qubo_result)
                results.append(ScalingResult(
                    n_targets=n_targets,
                    n_measurements=n_meas,
                    n_qubo_variables=qubo_result.num_variables,
                    penalty=qubo_result.penalty,
                    build_time_s=build_time,
                    solve_time_s=solver_result.solve_time_s,
                    solver_name=solver_name,
                    objective_value=solver_result.objective_value,
                    is_feasible=solver_result.is_feasible,
                ))
                logger.info(
                    "  %s: solve_time=%.4fs, objective=%.2f, feasible=%s",
                    solver_name, solver_result.solve_time_s,
                    solver_result.objective_value, solver_result.is_feasible,
                )
            except Exception as e:
                logger.warning("  %s failed: %s", solver_name, e)

    return results


def print_results(results: list[ScalingResult]) -> None:
    """Print scaling results as a table."""
    print(f"\n{'Targets':>8} {'Meas':>6} {'Vars':>6} {'Solver':>12} {'Build(s)':>10} {'Solve(s)':>10} {'Obj':>10} {'Feasible':>8}")
    print("-" * 82)
    for r in results:
        print(
            f"{r.n_targets:>8} {r.n_measurements:>6} {r.n_qubo_variables:>6} "
            f"{r.solver_name:>12} {r.build_time_s:>10.4f} {r.solve_time_s:>10.4f} "
            f"{r.objective_value:>10.2f} {str(r.is_feasible):>8}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_scaling_experiment()
    print_results(results)
