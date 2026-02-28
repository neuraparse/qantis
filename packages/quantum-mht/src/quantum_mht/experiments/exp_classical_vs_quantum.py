"""Experiment: Classical vs Quantum end-to-end tracking comparison.

End-to-end comparison of classical and quantum MTDA solvers on simulated
multi-target tracking scenarios, evaluating full pipeline performance
(predict -> gate -> QUBO -> solve -> update).

Comparison Methodology:
    Each solver is evaluated on identical scenarios with the same random seed,
    measuring:
    - Total tracking time and per-step latency
    - Number of confirmed tracks (track quality)
    - Assignment counts, missed detections, false alarms
    - Scalability across scenario complexity (5, 10 targets; variable clutter)

    Classical baselines:
    - Hungarian: optimal 2D assignment (Kuhn 1955), O(n^3)
    - GNN: greedy assignment (Blackman 1986), O(NM log(NM))

    Quantum solvers:
    - D-Wave annealing with SimulatedAnnealingSampler (local dev mode)
    - On QPU: Advantage2 (4400+ qubits, Zephyr topology), Ocean SDK 9.x

Academic References:
    Stollenwerk et al., arXiv:2110.08346, 2021 -- QUBO-based MTDA pipeline.
    Kuhn, "The Hungarian Method", Naval Res. Logistics 2(1-2):83-97, 1955.
    Blackman, "Multiple Target Tracking with Radar Applications", 1986.
    Bernardin & Stiefelhagen 2008 -- MOTA/MOTP evaluation methodology.
    Ristani et al. 2016 -- IDF1 identity preservation metric.

Usage:
    uv run python -m quantum_mht.experiments.exp_classical_vs_quantum
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from quantum_mht.simulation.scenario_generator import crossing_targets, dense_clutter
from quantum_mht.pipeline.tracking_pipeline import TrackingPipeline
from quantum_mht.solvers.solver_factory import create_solver
from quantum_mht.tracking.track_manager import TrackManager
from quantum_mht.tracking.kalman_filter import KalmanFilter
from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result from a full end-to-end tracking experiment.

    Captures pipeline-level metrics for quantum vs classical comparison.
    """
    scenario_name: str
    solver_name: str
    num_steps: int
    total_time_s: float
    avg_step_time_s: float
    final_confirmed_tracks: int
    total_assignments: int
    total_missed: int
    total_false_alarms: int


def run_tracking_experiment(
    scenarios: dict | None = None,
    solver_names: list[str] | None = None,
    num_steps: int = 30,
    seed: int = 42,
) -> list[TrackingResult]:
    """Run end-to-end tracking comparison.

    Args:
        scenarios: Dict of scenario_name -> SimulationWorld.
        solver_names: Solvers to compare.
        num_steps: Simulation steps per scenario.
        seed: Random seed.
    """
    if scenarios is None:
        scenarios = {
            "crossing_5": crossing_targets(n_targets=5),
            "crossing_10": crossing_targets(n_targets=10),
            "dense_clutter_3": dense_clutter(n_targets=3, clutter_rate=3.0),
        }
    if solver_names is None:
        solver_names = ["hungarian", "gnn"]
        # Add annealing if available
        try:
            import neal  # noqa: F401
            solver_names.append("annealing")
        except ImportError:
            pass

    results: list[TrackingResult] = []

    for scenario_name, world_template in scenarios.items():
        for solver_name in solver_names:
            logger.info("=== %s with %s ===", scenario_name, solver_name)

            # Fresh world and pipeline for each run
            rng = np.random.default_rng(seed)

            # Recreate the scenario for a fresh run
            if "crossing_5" in scenario_name:
                world = crossing_targets(n_targets=5)
            elif "crossing_10" in scenario_name:
                world = crossing_targets(n_targets=10)
            elif "dense_clutter" in scenario_name:
                world = dense_clutter(n_targets=3, clutter_rate=3.0)
            else:
                world = crossing_targets(n_targets=5)

            solver_kwargs = {}
            if solver_name == "annealing":
                solver_kwargs = {"use_simulator": True, "num_reads": 100}

            try:
                solver = create_solver(solver_name, **solver_kwargs)
            except (ValueError, ImportError) as e:
                logger.warning("Solver %s not available: %s", solver_name, e)
                continue

            # Full pipeline: predict -> gate -> QUBO -> solve -> update (arXiv:2110.08346)
            pipeline = TrackingPipeline(solver=solver)

            total_assignments = 0
            total_missed = 0
            total_false_alarms = 0

            t0 = time.perf_counter()
            for step in range(num_steps):
                scan = world.step(rng)
                result = pipeline.process_scan(scan)
                total_assignments += len(result.solver_result.assignments)
                total_missed += len(result.solver_result.missed_detections)
                total_false_alarms += len(result.solver_result.false_alarms)
            total_time = time.perf_counter() - t0

            final_confirmed = len(pipeline.track_manager.confirmed_tracks)
            avg_step = total_time / num_steps

            results.append(TrackingResult(
                scenario_name=scenario_name,
                solver_name=solver_name,
                num_steps=num_steps,
                total_time_s=total_time,
                avg_step_time_s=avg_step,
                final_confirmed_tracks=final_confirmed,
                total_assignments=total_assignments,
                total_missed=total_missed,
                total_false_alarms=total_false_alarms,
            ))

            logger.info(
                "  %s: %.2fs total, %.4fs/step, %d confirmed tracks, %d assigns",
                solver_name, total_time, avg_step, final_confirmed, total_assignments,
            )

    return results


def print_tracking_results(results: list[TrackingResult]) -> None:
    """Print tracking comparison table."""
    print(f"\n{'Scenario':>18} {'Solver':>12} {'Time(s)':>8} {'ms/step':>8} {'Tracks':>7} {'Assigns':>8} {'Missed':>7} {'FA':>5}")
    print("-" * 95)
    for r in results:
        print(
            f"{r.scenario_name:>18} {r.solver_name:>12} {r.total_time_s:>8.2f} "
            f"{r.avg_step_time_s * 1000:>8.1f} {r.final_confirmed_tracks:>7} "
            f"{r.total_assignments:>8} {r.total_missed:>7} {r.total_false_alarms:>5}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_tracking_experiment()
    print_tracking_results(results)
