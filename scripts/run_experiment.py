#!/usr/bin/env python3
"""Run quantum autonomous systems experiments.

Usage:
    uv run python scripts/run_experiment.py --config configs/experiments/pomdp_default.yaml
    uv run python scripts/run_experiment.py --config configs/experiments/mht_default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantum autonomous experiment")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and show plan without executing",
    )
    parser.add_argument(
        "--backend-override", type=str, default=None,
        help="Override backend type (e.g., local_aer, ibm_quantum)",
    )
    args = parser.parse_args()

    from quantum_common.config.loader import load_experiment_config
    from quantum_common.logging.setup import setup_logging

    config = load_experiment_config(args.config)
    setup_logging(
        level=config.logging.level,
        log_format=config.logging.format,
        log_file=config.logging.log_file,
    )
    logger = logging.getLogger(__name__)

    logger.info("Loaded experiment config: %s", config.name)
    logger.info("Backend: %s", config.backend.backend_type)
    logger.info("Case: %s", config.case_parameters.get("scenario", "unknown"))

    if args.dry_run:
        logger.info("Dry run mode - config validated successfully")
        print(f"Experiment: {config.name}")
        print(f"Backend: {config.backend.backend_type}")
        print(f"Mitigation: {config.mitigation.strategies}")
        print(f"Problem sizes: {config.benchmark.problem_sizes}")
        return

    case = config.case_parameters.get("scenario", "")
    experiment_name = config.name

    if "pomdp" in experiment_name.lower():
        _run_pomdp_experiment(config, logger)
    elif "mht" in experiment_name.lower():
        _run_mht_experiment(config, logger)
    else:
        logger.error("Unknown experiment case: %s", experiment_name)
        sys.exit(1)


def _run_pomdp_experiment(config: object, logger: logging.Logger) -> None:
    """Run POMDP belief estimation experiment."""
    logger.info("Starting POMDP experiment...")

    from quantum_pomdp.scenarios.tiger_problem import create_tiger_pomdp
    from quantum_pomdp.models.belief_state import BeliefState
    from quantum_pomdp.algorithms.qbrl import QBRLPlanner, QBRLConfig

    scenario = config.case_parameters.get("scenario", "tiger")

    if scenario == "tiger":
        params = config.case_parameters.get("tiger", {})
        model = create_tiger_pomdp(
            listen_accuracy=params.get("listen_accuracy", 0.85),
        )
        logger.info("Created Tiger POMDP: %d states, %d actions", model.num_states, model.num_actions)

        qbrl_params = config.case_parameters.get("qbrl", {})
        qbrl_config = QBRLConfig(
            horizon=qbrl_params.get("horizon", 2),
            num_samples=qbrl_params.get("num_samples", 100),
            use_quantum=qbrl_params.get("use_quantum", False),
        )
        planner = QBRLPlanner(model, qbrl_config)

        belief = BeliefState.uniform(model.num_states)
        num_episodes = params.get("num_episodes", 10)
        max_steps = params.get("max_steps_per_episode", 20)

        total_reward = 0.0
        for ep in range(num_episodes):
            episode_reward = 0.0
            b = belief
            for step in range(max_steps):
                action = planner.select_action(b)
                reward = model.expected_reward(b.probabilities, action)
                episode_reward += reward
            total_reward += episode_reward
            if (ep + 1) % 10 == 0:
                logger.info("Episode %d/%d, avg reward: %.2f", ep + 1, num_episodes, total_reward / (ep + 1))

        logger.info("POMDP experiment complete. Total reward: %.2f", total_reward)

    elif scenario == "grid_navigation":
        from quantum_pomdp.scenarios.grid_navigation import create_grid_navigation_pomdp
        params = config.case_parameters.get("grid_navigation", {})
        for grid_size in params.get("grid_sizes", [4]):
            model = create_grid_navigation_pomdp(grid_size=grid_size)
            logger.info("Grid %dx%d: %d states, %d qubits", grid_size, grid_size, model.num_states, model.total_circuit_qubits)

    logger.info("POMDP experiment finished.")


def _run_mht_experiment(config: object, logger: logging.Logger) -> None:
    """Run MHT tracking experiment."""
    logger.info("Starting MHT experiment...")

    from quantum_mht.simulation.scenario_generator import crossing_targets, dense_clutter
    from quantum_mht.pipeline.tracking_pipeline import TrackingPipeline
    from quantum_mht.solvers.solver_factory import create_solver

    import numpy as np

    scenario = config.case_parameters.get("scenario", "crossing")
    sim_params = config.case_parameters.get("simulation", {})
    num_steps = sim_params.get("num_steps", 50)

    if scenario == "crossing":
        world = crossing_targets(n_targets=5)
    elif scenario == "dense_clutter":
        world = dense_clutter(n_targets=3, clutter_rate=5.0)
    else:
        world = crossing_targets(n_targets=5)

    solver = create_solver("annealing", use_simulator=True, num_reads=100)
    pipeline = TrackingPipeline(solver=solver)

    rng = np.random.default_rng(42)
    for step in range(num_steps):
        scan = world.step(rng)
        result = pipeline.process_scan(scan)
        if (step + 1) % 10 == 0:
            logger.info(
                "Step %d: %d tracks (%d confirmed), %d measurements",
                step + 1, result.num_tracks, result.num_confirmed, result.num_measurements,
            )

    logger.info("MHT experiment finished. Final tracks: %d", len(pipeline.track_manager.confirmed_tracks))


if __name__ == "__main__":
    main()
