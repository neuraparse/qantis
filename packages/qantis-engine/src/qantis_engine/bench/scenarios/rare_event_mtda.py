"""Rare-event MTDA scenario generator.

Generates a sequence of frames containing:
    - ground-truth targets with persistent IDs
    - noisy measurements (clutter + missed detection)
    - cost matrices (negative log-likelihood ratios)

The scenario is parameterised by:
    - n_targets       : number of ground-truth targets
    - p_miss          : per-target missed-detection probability
    - clutter_rate    : Poisson rate of false alarms per frame
    - measurement_noise: stddev of observation Gaussian
    - n_steps         : episode length

The cost matrix is built so that Hungarian assignment is the *oracle*
solution at each frame (assuming the cost matrix is reported faithfully).
This is intentional: it lets us run a Hungarian-baseline pipeline that we
know is the static-frame optimum, and any closed-loop regret comes purely
from belief tracking — not from the static-frame solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MTDAScenarioFrame:
    """One frame of the MTDA scenario."""

    cost_matrix: NDArray[np.float64]
    measurement_target_id: NDArray[np.int_]
    visible_target_ids: NDArray[np.int_]
    target_states: NDArray[np.float64]


@dataclass
class MTDAScenario:
    """A rare-event MTDA scenario."""

    frames: list[MTDAScenarioFrame]
    n_targets: int
    n_steps: int
    p_miss: float
    clutter_rate: float
    measurement_noise: float
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_rare_event_mtda(
    n_steps: int = 50,
    n_targets: int = 5,
    p_miss: float = 0.15,
    clutter_rate: float = 2.0,
    measurement_noise: float = 0.5,
    seed: int = 0,
    field_size: float = 10.0,
) -> MTDAScenario:
    """Generate a rare-event MTDA scenario."""
    rng = np.random.default_rng(seed)

    target_pos = rng.uniform(0, field_size, size=(n_targets, 2))
    target_vel = rng.normal(0.0, 0.5, size=(n_targets, 2))

    frames: list[MTDAScenarioFrame] = []
    for _ in range(n_steps):
        target_pos = target_pos + target_vel * 0.1
        for d in range(2):
            mask_lo = target_pos[:, d] < 0
            mask_hi = target_pos[:, d] > field_size
            target_vel[mask_lo, d] *= -1
            target_vel[mask_hi, d] *= -1
            target_pos[mask_lo, d] = 0.0
            target_pos[mask_hi, d] = field_size

        visible_mask = rng.uniform(size=n_targets) > p_miss
        visible_ids = np.flatnonzero(visible_mask)
        n_visible = visible_ids.size
        n_clutter = int(rng.poisson(clutter_rate))

        true_meas = (
            target_pos[visible_ids]
            + rng.normal(0, measurement_noise, size=(n_visible, 2))
            if n_visible > 0
            else np.zeros((0, 2))
        )
        clutter_meas = (
            rng.uniform(0, field_size, size=(n_clutter, 2)) if n_clutter > 0 else np.zeros((0, 2))
        )
        meas = np.concatenate([true_meas, clutter_meas], axis=0)
        meas_id = np.concatenate(
            [visible_ids, -np.ones(n_clutter, dtype=np.int_)]
        ).astype(np.int_)
        order = rng.permutation(meas.shape[0])
        meas = meas[order]
        meas_id = meas_id[order]

        cost = np.zeros((n_targets, meas.shape[0]), dtype=np.float64)
        for i in range(n_targets):
            for j in range(meas.shape[0]):
                d = float(np.linalg.norm(target_pos[i] - meas[j]))
                cost[i, j] = (
                    0.5 * (d / measurement_noise) ** 2
                    if d < 4.0 * measurement_noise
                    else 25.0
                )

        frames.append(
            MTDAScenarioFrame(
                cost_matrix=cost,
                measurement_target_id=meas_id,
                visible_target_ids=visible_ids,
                target_states=target_pos.copy(),
            )
        )

    return MTDAScenario(
        frames=frames,
        n_targets=n_targets,
        n_steps=n_steps,
        p_miss=p_miss,
        clutter_rate=clutter_rate,
        measurement_noise=measurement_noise,
        metadata={"seed": seed, "field_size": field_size},
    )
