"""Track lifecycle management with M/N confirmation logic.

Manages the creation, confirmation, and deletion of tracks using the M/N
logic described in Blackman & Popoli 1999, Ch 4.3.

M/N Confirmation Logic:
    A tentative track is promoted to CONFIRMED status when it accumulates
    M = confirm_hits detections within N = confirm_window consecutive scans.
    This prevents false tracks from being confirmed based on clutter.

Track Deletion:
    A track is deleted (DELETED status) after delete_misses consecutive
    scans without a measurement association. This removes tracks that have
    left the surveillance region or whose targets have stopped.

Academic References:
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 4.3 -- M/N confirmation logic, track
        initiation and deletion policies.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995 --
        track management framework.

Quantum Pipeline Role (2026):
    Track management (creation, M/N confirmation, deletion) is the
    CLASSICAL bookkeeping layer that wraps the quantum association step.
    The pipeline boundary is:
      Classical: predict -> gate -> [build QUBO] ... [decode solution] -> update -> manage
      Quantum:                      [solve QUBO via annealing/QAOA]
    TrackManager operates entirely in the classical domain. The number
    of active tracks (N) directly determines QUBO size (N*M variables),
    so aggressive deletion of stale tracks is important for keeping
    problems within D-Wave Advantage2 QPU capacity (4400+ qubits).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from quantum_mht.tracking.track import Track, TrackStatus

@dataclass
class TrackManager:
    """Manage track creation, confirmation, and deletion.

    Implements M/N confirmation logic (Blackman & Popoli 1999, Ch 4.3):
    - confirm_hits (M): required detections for confirmation
    - confirm_window (N): window of scans for M/N test
    - delete_misses: consecutive misses before track deletion

    References:
        Blackman & Popoli 1999, Ch 4.3 -- M/N logic, track management.
    """
    # M/N confirmation parameters (Blackman & Popoli 1999, Ch 4.3)
    confirm_hits: int = 3      # M: required hits
    confirm_window: int = 5    # N: window size
    delete_misses: int = 5     # consecutive misses for deletion
    _tracks: list[Track] = field(default_factory=list)
    _next_id: int = field(default=0, init=False)

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.status != TrackStatus.DELETED]

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.status == TrackStatus.CONFIRMED]

    def create_track(self, state: NDArray[np.float64], covariance: NDArray[np.float64]) -> Track:
        track = Track(track_id=self._next_id, state=state, covariance=covariance)
        self._next_id += 1
        self._tracks.append(track)
        return track

    def update_lifecycle(self) -> None:
        """Apply M/N confirmation and deletion rules (Blackman & Popoli 1999, Ch 4.3)."""
        for track in self._tracks:
            if track.status == TrackStatus.DELETED:
                continue
            if track.status == TrackStatus.TENTATIVE and track.hits >= self.confirm_hits:
                track.status = TrackStatus.CONFIRMED
            if track.misses >= self.delete_misses:
                track.status = TrackStatus.DELETED

    def get_predicted_states(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        tracks = self.active_tracks
        if not tracks:
            return np.empty((0, 0)), np.empty((0, 0, 0))
        states = np.array([t.position for t in tracks])
        covs = np.array([t.covariance[:len(t.position), :len(t.position)] for t in tracks])
        return states, covs
