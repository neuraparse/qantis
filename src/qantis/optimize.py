"""Constraint-native optimization primitives for QANTIS.

This module intentionally avoids converting every decision problem into a
penalty QUBO.  For assignment-style decisions it keeps the hard constraints in
the state representation: every candidate emitted by the search is feasible by
construction, and missed detections / false alarms are modeled as native slack
decisions rather than large quadratic penalties.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FeasibleAssignmentResult:
    """Result from a feasible-subspace assignment search."""

    assignments: list[tuple[int, int]]
    missed_detections: list[int]
    false_alarms: list[int]
    objective_value: float
    solve_time_s: float
    optimizer_name: str
    explored_states: int
    lower_bound: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_feasible(self) -> bool:
        tracks = [i for i, _ in self.assignments]
        measurements = [j for _, j in self.assignments]
        return len(tracks) == len(set(tracks)) and len(measurements) == len(set(measurements))


@dataclass
class ConstraintNativeAssignmentOptimizer:
    """Beam-search optimizer over the feasible assignment subspace.

    The search state stores only partial matchings.  This gives QANTIS a clean
    non-QUBO baseline for the later quantum/hybrid sampler: a mixer or sampler
    should propose states in this same feasible space, not rely on penalties to
    repair infeasible samples after the fact.
    """

    missed_detection_cost: float = 5.0
    false_alarm_cost: float = 3.0
    beam_width: int | None = None

    @property
    def name(self) -> str:
        if self.beam_width is None:
            return "QANTIS-Optimize(feasible-exact)"
        return f"QANTIS-Optimize(feasible-beam-{self.beam_width})"

    def solve(
        self,
        cost_matrix: NDArray[np.float64] | list[list[float]],
        gate_mask: NDArray[np.bool_] | list[list[bool]] | None = None,
    ) -> FeasibleAssignmentResult:
        """Solve a single-frame assignment problem without penalty terms."""

        t0 = time.perf_counter()
        costs = np.asarray(cost_matrix, dtype=float)
        if costs.ndim != 2:
            raise ValueError("cost_matrix must be two-dimensional")
        n_tracks, n_measurements = costs.shape
        if gate_mask is None:
            gates = np.ones_like(costs, dtype=bool)
        else:
            gates = np.asarray(gate_mask, dtype=bool)
            if gates.shape != costs.shape:
                raise ValueError("gate_mask must match cost_matrix shape")

        states: list[_PartialAssignment] = [
            _PartialAssignment(cost=0.0, assignments=(), used_measurements=frozenset(), missed=())
        ]
        explored_states = 0

        for track in range(n_tracks):
            next_states: list[_PartialAssignment] = []
            for state in states:
                explored_states += 1
                for measurement in range(n_measurements):
                    if measurement in state.used_measurements or not gates[track, measurement]:
                        continue
                    next_states.append(
                        state.extend_assignment(
                            track,
                            measurement,
                            float(costs[track, measurement]),
                        )
                    )
                next_states.append(state.extend_missed(track, self.missed_detection_cost))

            states = _keep_best(next_states, self.beam_width)

        final_states: list[tuple[float, _PartialAssignment, list[int]]] = []
        all_measurements = set(range(n_measurements))
        for state in states:
            false_alarms = sorted(all_measurements - set(state.used_measurements))
            total_cost = state.cost + self.false_alarm_cost * len(false_alarms)
            final_states.append((total_cost, state, false_alarms))

        if not final_states:
            raise RuntimeError("no feasible assignment state was generated")

        objective, best, false_alarms = min(final_states, key=lambda item: item[0])
        lower_bound = _relaxed_assignment_lower_bound(costs, gates, self.missed_detection_cost)
        elapsed = time.perf_counter() - t0

        return FeasibleAssignmentResult(
            assignments=list(best.assignments),
            missed_detections=list(best.missed),
            false_alarms=false_alarms,
            objective_value=float(objective),
            solve_time_s=elapsed,
            optimizer_name=self.name,
            explored_states=explored_states,
            lower_bound=float(lower_bound),
            metadata={
                "n_tracks": n_tracks,
                "n_measurements": n_measurements,
                "beam_width": self.beam_width,
                "encoding": "feasible-subspace",
                "uses_qubo_penalty": False,
            },
        )


@dataclass(frozen=True)
class _PartialAssignment:
    cost: float
    assignments: tuple[tuple[int, int], ...]
    used_measurements: frozenset[int]
    missed: tuple[int, ...]

    def extend_assignment(
        self, track: int, measurement: int, assignment_cost: float
    ) -> _PartialAssignment:
        return _PartialAssignment(
            cost=self.cost + assignment_cost,
            assignments=(*self.assignments, (track, measurement)),
            used_measurements=self.used_measurements | {measurement},
            missed=self.missed,
        )

    def extend_missed(self, track: int, missed_cost: float) -> _PartialAssignment:
        return _PartialAssignment(
            cost=self.cost + missed_cost,
            assignments=self.assignments,
            used_measurements=self.used_measurements,
            missed=(*self.missed, track),
        )


def _keep_best(
    states: list[_PartialAssignment], beam_width: int | None
) -> list[_PartialAssignment]:
    if beam_width is None or len(states) <= beam_width:
        return states
    return heapq.nsmallest(beam_width, states, key=lambda state: state.cost)


def _relaxed_assignment_lower_bound(
    costs: NDArray[np.float64], gates: NDArray[np.bool_], missed_cost: float
) -> float:
    """Cheap admissible lower bound that ignores measurement collisions."""

    row_bounds = []
    for i in range(costs.shape[0]):
        feasible_costs = costs[i, gates[i]]
        best_assignment = float(np.min(feasible_costs)) if feasible_costs.size else float("inf")
        row_bounds.append(min(best_assignment, missed_cost))
    return float(np.sum(row_bounds))
