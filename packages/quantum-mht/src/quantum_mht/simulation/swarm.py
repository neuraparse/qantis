"""Drone swarm coordinator for multi-platform surveillance.

Coordinates a swarm of surveillance drones, managing formation geometry
and coverage area computation. The swarm operates in a decentralized
formation with each drone maintaining its own sensor and local processor.

Patrol Formation:
    Drones are arranged in a regular polygon (circular formation) around a
    center point, maximizing coverage overlap for robust multi-sensor tracking.

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995, Ch 8
        -- multi-platform sensor management.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 10 -- multi-platform coordination for
        surveillance and tracking.

Quantum Benchmark Role (2026):
    Multi-drone swarms increase the number of measurements per scan (M)
    due to overlapping FOV coverage, which directly increases QUBO size.
    Swarm scenarios test the scalability boundary of direct QPU embedding
    vs LeapHybrid decomposition. Formation geometry affects measurement
    redundancy, which CI fusion (covariance_intersection.py) leverages
    to improve track state estimates before QUBO construction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from quantum_mht.simulation.drone import Drone

@dataclass
class DroneSwarm:
    """Coordinate a swarm of surveillance drones.

    Manages formation geometry and aggregate coverage area for
    multi-platform tracking scenarios.
    """
    drones: list[Drone] = field(default_factory=list)

    def add_drone(self, drone: Drone) -> None:
        self.drones.append(drone)

    @property
    def coverage_area(self) -> float:
        return sum(np.pi * d.fov_radius ** 2 for d in self.drones)

    def patrol_formation(self, center: tuple[float, float], radius: float) -> None:
        """Arrange drones in circular formation for maximum coverage overlap."""
        n = len(self.drones)
        for i, drone in enumerate(self.drones):
            angle = 2 * np.pi * i / n
            drone.position = np.array([
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
            ])
