"""Verification and repair tools for QANTIS decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FeasibilityReport:
    """Hard-constraint verification report."""

    feasible: bool
    duplicate_tracks: list[int]
    duplicate_measurements: list[int]
    repaired_assignments: list[tuple[int, int]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PosteriorFidelityReport:
    """Distance diagnostics between two posterior beliefs."""

    hellinger: float
    total_variation: float
    fidelity: float


class QANTISVerify:
    """Trust layer for belief-to-action decisions."""

    def check_assignment(
        self,
        assignments: list[tuple[int, int]],
        *,
        n_tracks: int,
        n_measurements: int,
        cost_matrix: NDArray[np.float64] | list[list[float]] | None = None,
    ) -> FeasibilityReport:
        """Verify and, if needed, greedily repair duplicate assignments."""

        duplicate_tracks = _duplicates([i for i, _ in assignments])
        duplicate_measurements = _duplicates([j for _, j in assignments])
        feasible = not duplicate_tracks and not duplicate_measurements
        repaired = assignments
        if not feasible:
            repaired = self.repair_assignment(
                assignments,
                n_tracks=n_tracks,
                n_measurements=n_measurements,
                cost_matrix=cost_matrix,
            )
        return FeasibilityReport(
            feasible=feasible,
            duplicate_tracks=duplicate_tracks,
            duplicate_measurements=duplicate_measurements,
            repaired_assignments=repaired,
            metadata={
                "n_tracks": n_tracks,
                "n_measurements": n_measurements,
                "repair_applied": not feasible,
            },
        )

    def repair_assignment(
        self,
        assignments: list[tuple[int, int]],
        *,
        n_tracks: int,
        n_measurements: int,
        cost_matrix: NDArray[np.float64] | list[list[float]] | None = None,
    ) -> list[tuple[int, int]]:
        """Keep a minimum-cost conflict-free subset of proposed assignments."""

        if cost_matrix is None:
            costs = np.zeros((n_tracks, n_measurements), dtype=float)
        else:
            costs = np.asarray(cost_matrix, dtype=float)

        unique_candidates = sorted(
            set(assignments),
            key=lambda pair: float(costs[pair[0], pair[1]])
            if 0 <= pair[0] < n_tracks and 0 <= pair[1] < n_measurements
            else float("inf"),
        )
        used_tracks: set[int] = set()
        used_measurements: set[int] = set()
        repaired: list[tuple[int, int]] = []
        for track, measurement in unique_candidates:
            if not (0 <= track < n_tracks and 0 <= measurement < n_measurements):
                continue
            if track in used_tracks or measurement in used_measurements:
                continue
            repaired.append((track, measurement))
            used_tracks.add(track)
            used_measurements.add(measurement)
        return sorted(repaired)

    def posterior_fidelity(
        self,
        reference: NDArray[np.float64] | list[float] | Any,
        candidate: NDArray[np.float64] | list[float] | Any,
    ) -> PosteriorFidelityReport:
        """Compute Hellinger, total-variation, and fidelity diagnostics."""

        p = _as_probability_vector(reference)
        q = _as_probability_vector(candidate)
        if p.shape != q.shape:
            raise ValueError("posterior vectors must have the same length")
        hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
        total_variation = float(0.5 * np.sum(np.abs(p - q)))
        fidelity = float(np.sum(np.sqrt(p * q)) ** 2)
        return PosteriorFidelityReport(
            hellinger=hellinger,
            total_variation=total_variation,
            fidelity=fidelity,
        )


def _duplicates(values: list[int]) -> list[int]:
    seen: set[int] = set()
    dupes: set[int] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return sorted(dupes)


def _as_probability_vector(values: NDArray[np.float64] | list[float] | Any) -> NDArray[np.float64]:
    if hasattr(values, "probabilities"):
        arr = np.asarray(values.probabilities, dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if arr.ndim != 1 or total <= 0.0:
        raise ValueError("posterior must be a one-dimensional positive vector")
    arr = np.clip(arr / total, 0.0, 1.0)
    return arr / arr.sum()

