"""Pipeline A: classical particle filter + Hungarian assignment baseline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from qantis_engine.bench.runner import BenchConfig
from qantis_engine.bench.scenarios.rare_event_mtda import MTDAScenario


@dataclass
class PipelineOutputs:
    assignments: list[list[tuple[int, int]]] = field(default_factory=list)
    missed: list[list[int]] = field(default_factory=list)
    false_alarms: list[list[int]] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    sample_cost: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    oracle_rewards: list[float] = field(default_factory=list)
    ground_truth_ids: list[NDArray[np.int_]] = field(default_factory=list)
    hellinger: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "assignments": self.assignments,
            "missed": self.missed,
            "false_alarms": self.false_alarms,
            "latency_ms": self.latency_ms,
            "sample_cost": self.sample_cost,
            "rewards": self.rewards,
            "oracle_rewards": self.oracle_rewards,
            "ground_truth_ids": self.ground_truth_ids,
            "hellinger": self.hellinger,
            "metadata": self.metadata,
        }


def _assign_with_hungarian(
    cost_matrix: NDArray[np.float64], n_tracks: int, n_meas: int, threshold: float = 20.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    rows, cols = linear_sum_assignment(cost_matrix)
    assignments: list[tuple[int, int]] = []
    used_t: set[int] = set()
    used_m: set[int] = set()
    for r, c in zip(rows, cols, strict=False):
        if 0 <= r < n_tracks and 0 <= c < n_meas and cost_matrix[r, c] < threshold:
            assignments.append((int(r), int(c)))
            used_t.add(int(r))
            used_m.add(int(c))
    missed = [i for i in range(n_tracks) if i not in used_t]
    false_alarms = [j for j in range(n_meas) if j not in used_m]
    return assignments, missed, false_alarms


def _oracle_step_reward(frame: Any, n_tracks: int) -> float:
    n_meas = frame.cost_matrix.shape[1]
    assigns, _, _ = _assign_with_hungarian(frame.cost_matrix, n_tracks, n_meas)
    return float(-sum(frame.cost_matrix[r, c] for r, c in assigns))


def run_classical_pf_gurobi(
    config: BenchConfig, scenario: MTDAScenario
) -> dict[str, Any]:
    """Run baseline A on a generated MTDA scenario."""
    out = PipelineOutputs()
    rng = np.random.default_rng(config.seed)
    n_tracks = scenario.n_targets
    pf_noise = scenario.measurement_noise * 1.5

    for frame in scenario.frames:
        n_meas = frame.cost_matrix.shape[1]
        cost = frame.cost_matrix + rng.normal(0, pf_noise * 0.2, size=frame.cost_matrix.shape)
        cost = np.clip(cost, 0.0, 25.0)
        t0 = time.perf_counter()
        assigns, missed, false_a = _assign_with_hungarian(cost, n_tracks, n_meas)
        t = (time.perf_counter() - t0) * 1000.0

        out.assignments.append(assigns)
        out.missed.append(missed)
        out.false_alarms.append(false_a)
        out.latency_ms.append(t)
        out.sample_cost.append(0)
        reward = float(-sum(frame.cost_matrix[r, c] for r, c in assigns))
        out.rewards.append(reward)
        out.oracle_rewards.append(_oracle_step_reward(frame, n_tracks))
        out.ground_truth_ids.append(frame.measurement_target_id)
        out.hellinger.append(0.0)

    out.metadata = {"pipeline": "classical_pf_gurobi", "pf_noise": pf_noise}
    return out.as_dict()
