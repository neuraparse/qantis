"""Track state and lifecycle management.

Implements the track data structure and lifecycle state machine for multi-target
tracking. Each track maintains a Kalman-filtered state estimate, covariance
matrix, and lifecycle status (tentative -> confirmed -> deleted).

Track State Machine (Blackman & Popoli 1999, Ch 4):
    TENTATIVE: Newly initiated track, not yet confirmed. Requires M hits
               within N scans to transition to CONFIRMED.
    CONFIRMED: Track has passed the M/N confirmation test and is considered
               a real target.
    DELETED:   Track has accumulated too many consecutive missed detections
               and is removed from active tracking.

The hit/miss counters support the M/N logic used by TrackManager for
lifecycle transitions.

Academic References:
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 4 -- track lifecycle management, state machine,
        and M/N confirmation logic.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995 -- track state estimation framework.
    Kalman, "A New Approach to Linear Filtering and Prediction Problems",
        J. Basic Engineering 82(1):35-45, 1960 -- state estimation used in
        track update.

Quantum Pipeline Role (2026):
    Track lifecycle management (tentative->confirmed->deleted) operates
    entirely in the classical domain. The quantum advantage in our
    hybrid pipeline applies specifically to the DATA ASSOCIATION step
    (QUBO optimization via quantum annealing or QAOA). Track state
    estimation (Kalman filter) and management (M/N logic) remain
    classical -- this is by design, as these operations are already
    efficient classically and would not benefit from quantum processing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray

class TrackStatus(Enum):
    """Track lifecycle states (Blackman & Popoli 1999, Ch 4)."""
    TENTATIVE = auto()   # Awaiting M/N confirmation
    CONFIRMED = auto()   # Active, confirmed target
    DELETED = auto()     # Removed due to consecutive misses

@dataclass
class Track:
    """Single target track with Kalman-filtered state estimation.

    Maintains the track state vector, covariance, and lifecycle counters.
    The state vector is typically [x, vx, y, vy] for a constant-velocity
    model (Li & Jilkov 2003).

    References:
        Blackman & Popoli 1999, Ch 4 -- track data structure and lifecycle.
        Kalman 1960 -- state estimation framework.
    """
    track_id: int
    state: NDArray[np.float64]  # [x, vx, y, vy] or similar
    covariance: NDArray[np.float64]
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: int = 1
    misses: int = 0
    age: int = 1
    _history: list[NDArray[np.float64]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._history.append(self.state.copy())

    @property
    def position(self) -> NDArray[np.float64]:
        return self.state[:len(self.state) // 2]

    @property
    def velocity(self) -> NDArray[np.float64]:
        return self.state[len(self.state) // 2:]

    def record_hit(self) -> None:
        self.hits += 1
        self.misses = 0
        self.age += 1

    def record_miss(self) -> None:
        self.misses += 1
        self.age += 1

    def update_state(self, new_state: NDArray[np.float64], new_covariance: NDArray[np.float64]) -> None:
        self.state = new_state
        self.covariance = new_covariance
        self._history.append(new_state.copy())
