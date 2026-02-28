"""Track management stage of the MHT tracking pipeline.

Handles track lifecycle transitions after association and update:
    - Records missed detections for unassigned tracks.
    - Applies M/N confirmation logic (tentative -> confirmed).
    - Deletes tracks with excessive consecutive misses.

This is Stage 5 (final) of the predict -> gate -> QUBO -> solve -> update
pipeline.

Academic References:
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 4.3 -- M/N confirmation logic, track
        initiation and deletion policies.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995 --
        track management in MHT context.
"""
from __future__ import annotations
from dataclasses import dataclass
from quantum_mht.tracking.track_manager import TrackManager

@dataclass
class ManagementStage:
    track_manager: TrackManager

    def process(self, missed_track_indices: list[int]) -> None:
        """Record misses and apply lifecycle rules (Blackman & Popoli 1999, Ch 4.3)."""
        active = self.track_manager.active_tracks
        for idx in missed_track_indices:
            if idx < len(active):
                active[idx].record_miss()
        self.track_manager.update_lifecycle()
