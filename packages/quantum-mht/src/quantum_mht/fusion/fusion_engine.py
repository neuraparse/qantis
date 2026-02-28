"""Centralized multi-sensor fusion engine.

Implements centralized (measurement-level) fusion for multi-sensor tracking.
In centralized fusion, all sensor measurements are collected at a central
processing node and merged into a single measurement scan before association.

Fusion Architectures:
    - Centralized (this module): All measurements merged at measurement level.
      Simple but requires high-bandwidth communication to the fusion center.
    - Distributed: Each sensor runs a local tracker; track-level estimates
      are fused using Covariance Intersection (see covariance_intersection.py).

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995, Ch 8 -- centralized vs distributed
        multi-sensor fusion architectures.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 10 -- multi-sensor tracking systems.

Quantum Pipeline Role (2026):
    Centralized fusion architecture chosen for quantum pipeline
    compatibility: all sensor measurements are collected at a central
    node before QUBO construction. This ensures the cost matrix
    (log-likelihood ratios) accounts for all available sensor data.
    Distributed quantum optimization (DQAOA) is emerging but requires
    quantum network connectivity not yet available for mobile platforms.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from quantum_mht.fusion.measurement import Measurement, MeasurementScan

@dataclass
class FusionEngine:
    """Centralized fusion of measurements from multiple sensors.

    Merges all sensor scans into a single MeasurementScan for unified
    association processing (Bar-Shalom & Li 1995, Ch 8).
    """
    sensors: list[int] = field(default_factory=list)

    def fuse_scans(self, scans: list[MeasurementScan]) -> MeasurementScan:
        """Fuse measurements from multiple sensor scans at same timestamp."""
        all_measurements: list[Measurement] = []
        timestamp = scans[0].timestamp if scans else 0.0
        for scan in scans:
            all_measurements.extend(scan.measurements)
        return MeasurementScan(measurements=all_measurements, timestamp=timestamp)
