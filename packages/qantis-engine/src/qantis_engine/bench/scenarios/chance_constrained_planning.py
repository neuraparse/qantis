"""Chance-constrained planning scenario generator.

A simple grid-world planner where at each step the agent chooses a
next-cell action; the world reveals a (possibly empty) hazard observation;
the constraint ``P(collision) <= delta`` must be satisfied with confidence.

This scenario is the v0 stand-in for the full MPC benchmark — it tests
that QANTIS-Risk's chance-constraint API integrates correctly with
QANTIS-Optimize and QANTIS-Verify under a fixed deadline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class PlanningScenario:
    """A chance-constrained planning scenario."""

    grid_size: int
    hazard_locations: NDArray[np.int_]
    hazard_appear_prob: float
    start: tuple[int, int]
    goal: tuple[int, int]
    n_steps: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_chance_constrained_planning(
    grid_size: int = 10,
    n_hazards: int = 3,
    hazard_appear_prob: float = 0.05,
    n_steps: int = 30,
    seed: int = 0,
) -> PlanningScenario:
    """Generate a small chance-constrained grid-world planning scenario."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(grid_size * grid_size, size=n_hazards + 2, replace=False)
    hazard = np.array(
        [(int(c // grid_size), int(c % grid_size)) for c in cells[:n_hazards]],
        dtype=np.int_,
    )
    start = (int(cells[-2] // grid_size), int(cells[-2] % grid_size))
    goal = (int(cells[-1] // grid_size), int(cells[-1] % grid_size))
    return PlanningScenario(
        grid_size=grid_size,
        hazard_locations=hazard,
        hazard_appear_prob=hazard_appear_prob,
        start=start,
        goal=goal,
        n_steps=n_steps,
        seed=seed,
        metadata={"n_hazards": n_hazards},
    )
