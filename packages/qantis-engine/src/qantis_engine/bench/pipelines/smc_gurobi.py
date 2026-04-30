"""Pipeline B: SMC particle filter + Hungarian.

Heavier sequential-Monte-Carlo filter; same downstream Hungarian step.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from qantis_engine.bench.pipelines.classical_pf_gurobi import (
    PipelineOutputs,
    _assign_with_hungarian,
    _oracle_step_reward,
)
from qantis_engine.bench.runner import BenchConfig
from qantis_engine.bench.scenarios.rare_event_mtda import MTDAScenario


def run_smc_gurobi(config: BenchConfig, scenario: MTDAScenario) -> dict[str, Any]:
    out = PipelineOutputs()
    rng = np.random.default_rng(config.seed)
    n_tracks = scenario.n_targets
    n_particles = 64

    particle_perturb_std = scenario.measurement_noise * 0.5

    for frame in scenario.frames:
        n_meas = frame.cost_matrix.shape[1]
        particle_costs = np.stack(
            [
                frame.cost_matrix
                + rng.normal(0, particle_perturb_std, size=frame.cost_matrix.shape)
                for _ in range(n_particles)
            ]
        )
        cost = np.mean(np.clip(particle_costs, 0.0, 25.0), axis=0)

        t0 = time.perf_counter()
        assigns, missed, false_a = _assign_with_hungarian(cost, n_tracks, n_meas)
        t = (time.perf_counter() - t0) * 1000.0

        out.assignments.append(assigns)
        out.missed.append(missed)
        out.false_alarms.append(false_a)
        out.latency_ms.append(t)
        out.sample_cost.append(n_particles)
        reward = float(-sum(frame.cost_matrix[r, c] for r, c in assigns))
        out.rewards.append(reward)
        out.oracle_rewards.append(_oracle_step_reward(frame, n_tracks))
        out.ground_truth_ids.append(frame.measurement_target_id)
        out.hellinger.append(0.0)

    out.metadata = {"pipeline": "smc_gurobi", "n_particles": n_particles}
    return out.as_dict()
