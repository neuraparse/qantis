"""Closed-loop regret evaluator.

Regret = oracle_reward - pipeline_reward over a fixed-length episode under
a fixed seed. The oracle is the policy that knows the ground truth
observation sequence and assigns optimally; the pipeline only sees noisy
observations and must replan online.

This is the metric the QANTIS-vs-Gurobi paper claim rests on. Optimality
gap on a single static QUBO is irrelevant: what matters is the *closed-loop*
reward under the same deadline budget on the same scenario. ``RegretReport``
records both per-step and cumulative regret so we can plot regret-vs-time
curves rather than only one number.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class RegretReport:
    """Result of a closed-loop regret evaluation."""

    cumulative_regret: float
    per_step_regret: NDArray[np.float64]
    pipeline_reward: float
    oracle_reward: float
    n_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


def closed_loop_regret(
    pipeline_rewards: Sequence[float],
    oracle_rewards: Sequence[float],
    metadata: dict[str, Any] | None = None,
) -> RegretReport:
    """Compute closed-loop regret from per-step reward sequences.

    Both sequences must be aligned in time (same number of steps) and use the
    same scenario seed. We do not normalize by oracle reward because that
    obscures cases where both pipelines are near-zero.
    """
    pipeline_arr = np.asarray(pipeline_rewards, dtype=np.float64)
    oracle_arr = np.asarray(oracle_rewards, dtype=np.float64)
    if pipeline_arr.shape != oracle_arr.shape:
        raise ValueError(
            f"shape mismatch: pipeline {pipeline_arr.shape} vs oracle {oracle_arr.shape}"
        )
    per_step = oracle_arr - pipeline_arr
    return RegretReport(
        cumulative_regret=float(per_step.sum()),
        per_step_regret=per_step,
        pipeline_reward=float(pipeline_arr.sum()),
        oracle_reward=float(oracle_arr.sum()),
        n_steps=int(pipeline_arr.size),
        metadata=metadata or {},
    )


def regret_from_callbacks(
    pipeline_step: Callable[[int], float],
    oracle_step: Callable[[int], float],
    n_steps: int,
    metadata: dict[str, Any] | None = None,
) -> RegretReport:
    """Drive both policies for ``n_steps`` and compute closed-loop regret."""
    pipeline_rewards = [pipeline_step(t) for t in range(n_steps)]
    oracle_rewards = [oracle_step(t) for t in range(n_steps)]
    return closed_loop_regret(pipeline_rewards, oracle_rewards, metadata=metadata)
