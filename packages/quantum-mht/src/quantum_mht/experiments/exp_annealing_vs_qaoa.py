"""Experiment: D-Wave Annealing vs IBM QAOA comparison.

Compares quantum annealing (D-Wave) and gate-model QAOA (IBM) solvers
for multi-target data association across problem sizes.

Comparison Methodology:
    Both solvers receive identical QUBO instances derived from the same
    synthetic tracking data. The comparison measures:
    - Solution quality (objective value, feasibility)
    - Solve time (wall-clock, including compilation overhead for QAOA)
    - Scalability across problem sizes (3, 5, 8 targets)

    Classical baselines (Hungarian, GNN) provide reference optimal and
    greedy solutions for calibration.

Solver Configurations:
    - D-Wave Annealing: SimulatedAnnealingSampler (local), 500 reads.
      On QPU: Advantage2 (4400+ qubits, Zephyr topology), Ocean SDK 9.x.
    - IBM QAOA: StatevectorSampler (exact), p=1 layer (Farhi et al. 2014).
      On hardware: SamplerV2 with 100K shots (arXiv:2512.08245 mitigation).
    - FPC-QAOA: Fixed-parameter-count schedule (arXiv:2512.21181).

Academic References:
    Stollenwerk et al., arXiv:2110.08346, 2021 -- QUBO formulation for MTDA.
    Farhi, Goldstone & Gutmann, arXiv:1411.4028, 2014 -- QAOA algorithm.
    Saavedra-Pino et al., arXiv:2512.21181, Dec 2025 -- FPC-QAOA.
    Ihara, Scientific Reports 15:24294, 2025 -- reverse annealing warm-start.
    McCormick et al., arXiv:2209.00615, 2022 -- diabatic quantum annealing.
    arXiv:2512.08245, Dec 2025 -- SamplerV2 shot count analysis.

Usage:
    uv run python -m quantum_mht.experiments.exp_annealing_vs_qaoa
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder
from quantum_mht.solvers.solver_factory import create_solver

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result from annealing vs QAOA solver comparison."""
    n_targets: int
    solver_name: str
    solve_time_s: float
    objective_value: float
    is_feasible: bool
    num_assignments: int
    metadata: dict = field(default_factory=dict)


def run_comparison(
    target_counts: list[int] | None = None,
    meas_ratio: float = 1.5,
    seed: int = 42,
) -> list[ComparisonResult]:
    """Compare annealing vs QAOA solvers.

    Args:
        target_counts: Problem sizes to test.
        meas_ratio: Measurements per target.
        seed: Random seed.
    """
    if target_counts is None:
        target_counts = [3, 5, 8]

    rng = np.random.default_rng(seed)
    results: list[ComparisonResult] = []

    # Define solver configurations for comparison
    # Annealing: D-Wave SimulatedAnnealingSampler (local dev mode)
    # Hungarian: optimal classical baseline (Kuhn 1955, O(n^3))
    # GNN: greedy classical baseline (Blackman 1986)
    solver_configs = {
        "annealing_sim": {"type": "annealing", "kwargs": {"use_simulator": True, "num_reads": 500}},
        "hungarian": {"type": "hungarian", "kwargs": {}},
        "gnn": {"type": "gnn", "kwargs": {}},
    }

    # Try to add QAOA (Farhi et al. 2014) if qiskit-optimization 0.7+ available
    try:
        import qiskit_optimization  # noqa: F401
        solver_configs["qaoa"] = {"type": "qaoa", "kwargs": {"reps": 1, "use_simulator": True}}
    except ImportError:
        logger.info("qiskit-optimization not available, skipping QAOA")

    for n_targets in target_counts:
        n_meas = int(n_targets * meas_ratio)
        logger.info("=== %d targets x %d measurements ===", n_targets, n_meas)

        # Generate synthetic tracking data
        true_positions = rng.uniform(10, 90, size=(n_targets, 2))
        measurements = true_positions + rng.normal(0, 3, size=(n_targets, 2))
        extra_clutter = rng.uniform(0, 100, size=(max(0, n_meas - n_targets), 2))
        if len(extra_clutter) > 0:
            all_meas = np.vstack([measurements, extra_clutter])
        else:
            all_meas = measurements
        covariances = np.array([np.eye(2) * 9.0 for _ in range(n_targets)])

        # Build QUBO
        builder = MTDAQuboBuilder()
        cost_matrix = builder.cost_builder.build(true_positions, all_meas, covariances)
        qubo_result = builder.build_from_cost_matrix(cost_matrix)
        logger.info("QUBO: %d variables", qubo_result.num_variables)

        # Solve with each solver
        for name, config in solver_configs.items():
            try:
                solver = create_solver(config["type"], **config["kwargs"])
                t0 = time.perf_counter()
                sol = solver.solve(qubo_result)
                elapsed = time.perf_counter() - t0

                results.append(ComparisonResult(
                    n_targets=n_targets,
                    solver_name=name,
                    solve_time_s=elapsed,
                    objective_value=sol.objective_value,
                    is_feasible=sol.is_feasible,
                    num_assignments=len(sol.assignments),
                ))
                logger.info(
                    "  %s: %.4fs, obj=%.2f, %d assignments, feasible=%s",
                    name, elapsed, sol.objective_value, len(sol.assignments), sol.is_feasible,
                )
            except Exception as e:
                logger.warning("  %s failed: %s", name, e)

    return results


def print_comparison(results: list[ComparisonResult]) -> None:
    """Print comparison table."""
    print(f"\n{'Targets':>8} {'Solver':>16} {'Time(s)':>10} {'Objective':>12} {'Assigns':>8} {'Feasible':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r.n_targets:>8} {r.solver_name:>16} {r.solve_time_s:>10.4f} "
            f"{r.objective_value:>12.2f} {r.num_assignments:>8} {str(r.is_feasible):>8}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_comparison()
    print_comparison(results)
