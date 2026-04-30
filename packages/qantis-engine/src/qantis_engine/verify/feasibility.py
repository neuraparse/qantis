"""Strict feasibility checker for assignment-style outputs.

Returns a ``FeasibilityReport`` that lists every violated constraint, not
just a boolean. The benchmark harness records this list verbatim in run
logs, so a "looks ok" run that silently violates a row constraint cannot
masquerade as a clean result.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class FeasibilityReport:
    """Result of a feasibility check."""

    feasible: bool
    violations: list[str] = field(default_factory=list)
    metadata: dict[str, int] = field(default_factory=dict)


def check_assignment(
    assignments: Sequence[tuple[int, int]],
    n_tracks: int,
    n_meas: int,
    missed_detections: Sequence[int] = (),
    false_alarms: Sequence[int] = (),
) -> FeasibilityReport:
    """Verify a track-to-measurement assignment.

    Hard constraints:
        - each track index appears in at most one (track, meas) pair
        - each measurement index appears in at most one (track, meas) pair
        - track indices in [0, n_tracks); measurement indices in [0, n_meas)
        - missed_detections list contains exactly tracks not assigned
        - false_alarms list contains exactly measurements not assigned
    """
    violations: list[str] = []
    track_use: dict[int, int] = {}
    meas_use: dict[int, int] = {}
    for r, c in assignments:
        if not 0 <= r < n_tracks:
            violations.append(f"track index out of range: {r}")
        if not 0 <= c < n_meas:
            violations.append(f"measurement index out of range: {c}")
        track_use[r] = track_use.get(r, 0) + 1
        meas_use[c] = meas_use.get(c, 0) + 1

    for r, count in track_use.items():
        if count > 1:
            violations.append(f"track {r} assigned {count} times")
    for c, count in meas_use.items():
        if count > 1:
            violations.append(f"measurement {c} assigned {count} times")

    assigned_tracks = set(track_use)
    assigned_meas = set(meas_use)
    expected_missed = set(range(n_tracks)) - assigned_tracks
    expected_false = set(range(n_meas)) - assigned_meas
    if set(missed_detections) != expected_missed:
        violations.append(
            f"missed_detections {sorted(missed_detections)} != expected {sorted(expected_missed)}"
        )
    if set(false_alarms) != expected_false:
        violations.append(
            f"false_alarms {sorted(false_alarms)} != expected {sorted(expected_false)}"
        )

    return FeasibilityReport(
        feasible=len(violations) == 0,
        violations=violations,
        metadata={
            "n_assignments": len(assignments),
            "n_tracks": n_tracks,
            "n_meas": n_meas,
            "n_violations": len(violations),
        },
    )
