"""Feasibility repair for assignment / permutation candidates.

Two repair operators:
    - ``AssignmentRepair`` : projects a one-hot bitstring onto an exact
      track-to-measurement matching using scipy's Hungarian algorithm. Cost
      matrix is reconstructed from the bitstring's per-cell scores.
    - ``PermutationRepair`` : projects a free integer vector onto the nearest
      permutation by greedy slot reassignment.

Both operators are *deterministic* given the same scores: rerunning a benchmark
with the same seed reproduces the same repaired candidate. This is critical
for honest comparison between quantum and classical pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


@dataclass
class AssignmentRepair:
    """Repair an MTDA-style one-hot candidate to a feasible matching.

    ``n_tracks`` rows must each be assigned to either one of ``n_meas`` columns
    or to the missed-detection slot. ``n_meas`` columns must each be assigned
    to either one of ``n_tracks`` rows or to the false-alarm slot. The repair
    runs Hungarian on the cost matrix derived from per-cell sampler scores.
    """

    n_tracks: int
    n_meas: int

    def repair(
        self,
        scores: NDArray[np.float64],
        cost_matrix: NDArray[np.float64] | None = None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Produce a feasible matching.

        Parameters
        ----------
        scores : (n_tracks, n_meas) per-cell sampler probability of assignment.
        cost_matrix : optional explicit cost matrix; if provided, repair
            minimises this objective instead of scoring.

        Returns
        -------
        assignments : list of (track, meas) pairs.
        missed : list of track indices left unassigned.
        false_alarms : list of measurement indices left unassigned.
        """
        n_t, n_m = self.n_tracks, self.n_meas
        if cost_matrix is None:
            cost_matrix = -np.asarray(scores, dtype=np.float64)
        else:
            cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
        if cost_matrix.shape != (n_t, n_m):
            raise ValueError(
                f"cost_matrix shape {cost_matrix.shape} != ({n_t}, {n_m})"
            )

        size = max(n_t, n_m)
        big = float(np.max(cost_matrix)) + 1.0 if cost_matrix.size else 1.0
        padded = np.full((size, size), big, dtype=np.float64)
        padded[:n_t, :n_m] = cost_matrix
        rows, cols = linear_sum_assignment(padded)

        assignments: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_meas: set[int] = set()
        for r, c in zip(rows, cols, strict=False):
            if r < n_t and c < n_m and cost_matrix[r, c] < big:
                assignments.append((int(r), int(c)))
                used_tracks.add(int(r))
                used_meas.add(int(c))
        missed = [i for i in range(n_t) if i not in used_tracks]
        false_alarms = [j for j in range(n_m) if j not in used_meas]
        return assignments, missed, false_alarms


@dataclass
class PermutationRepair:
    """Project a free integer vector onto a valid permutation of ``[0, n)``."""

    n: int

    def repair(self, decoded: NDArray[np.int_]) -> NDArray[np.int_]:
        if decoded.shape != (self.n,):
            raise ValueError(f"expected ({self.n},), got {decoded.shape}")
        used: set[int] = set()
        out = np.full(self.n, -1, dtype=np.int_)
        for t in range(self.n):
            v = int(decoded[t]) % self.n
            if v not in used:
                out[t] = v
                used.add(v)
        missing = [i for i in range(self.n) if i not in used]
        empties = [t for t in range(self.n) if out[t] == -1]
        for t, v in zip(empties, missing, strict=False):
            out[t] = v
        return out


def repair_assignment(
    scores: NDArray[np.float64],
    n_tracks: int,
    n_meas: int,
    cost_matrix: NDArray[np.float64] | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Functional facade over :class:`AssignmentRepair`."""
    return AssignmentRepair(n_tracks=n_tracks, n_meas=n_meas).repair(
        scores=scores, cost_matrix=cost_matrix
    )
