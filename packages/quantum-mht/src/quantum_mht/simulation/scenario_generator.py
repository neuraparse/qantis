"""Predefined test scenarios for MHT algorithm benchmarking.

Provides standardized simulation scenarios that exercise different aspects
of multi-target tracking: crossing trajectories, dense clutter, and
multi-sensor (swarm) configurations.

Scenario Design Principles:
    - Crossing targets: tests track identity maintenance during proximity
      events (the hardest case for data association).
    - Dense clutter: tests false alarm rejection and track confirmation logic.
    - Swarm patrol: tests multi-sensor fusion and distributed coverage.

These scenarios enable reproducible evaluation and comparison of quantum
vs classical MTDA solvers under controlled conditions.

Academic References:
    Bernardin & Stiefelhagen, "Evaluating Multiple Object Tracking
        Performance: The CLEAR MOT Metrics", J. Image Video Process.,
        2008 -- standardized MOT evaluation methodology.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 12 -- benchmark scenario design for
        tracking system evaluation.

Quantum Benchmark Role (2026):
    Predefined scenarios provide standardized test cases for quantum vs
    classical solver comparison:
    - crossing_targets: Hardest association case, tests quantum advantage
      in resolving ambiguous assignments that confuse greedy/GNN solvers.
    - dense_clutter: Tests QUBO scalability as M (measurements) grows
      due to false alarms; stresses QPU capacity limits.
    - swarm_patrol: Tests multi-sensor fusion + quantum association
      end-to-end; evaluates centralized QUBO with fused measurement scans.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from quantum_mht.simulation.world import SimulationWorld
from quantum_mht.simulation.target import Target
from quantum_mht.simulation.drone import Drone
from quantum_mht.simulation.dynamics import ConstantVelocity, ConstantTurn
from quantum_mht.fusion.sensor_model import LinearSensor

def crossing_targets(n_targets: int = 5, speed: float = 2.0) -> SimulationWorld:
    """Scenario with crossing target trajectories.

    Targets start at center and move outward at equal angular spacing.
    Tests track identity maintenance during proximity/crossing events
    (the most challenging case for data association).
    """
    targets = []
    rng = np.random.default_rng(42)
    for i in range(n_targets):
        angle = 2 * np.pi * i / n_targets
        state = np.array([50.0, speed * np.cos(angle), 50.0, speed * np.sin(angle)])
        targets.append(Target(target_id=i, initial_state=state))
    drone = Drone(drone_id=0, position=np.array([50.0, 50.0]), fov_radius=100.0)
    return SimulationWorld(targets=targets, drones=[drone])

def dense_clutter(n_targets: int = 3, clutter_rate: float = 5.0) -> SimulationWorld:
    """Scenario with high clutter density.

    Tests false alarm rejection and track confirmation under high
    clutter rates. Clutter follows a Poisson process (Bar-Shalom & Li 1995).
    """
    targets = []
    for i in range(n_targets):
        state = np.array([20.0 + i * 30.0, 1.0, 50.0, 0.5])
        targets.append(Target(target_id=i, initial_state=state))
    drone = Drone(drone_id=0, position=np.array([50.0, 50.0]), fov_radius=100.0)
    return SimulationWorld(targets=targets, drones=[drone], clutter_rate=clutter_rate)

def swarm_patrol(n_drones: int = 3, n_targets: int = 5) -> SimulationWorld:
    """Multi-drone swarm patrol scenario.

    Tests multi-sensor fusion with distributed drone platforms providing
    overlapping coverage. Evaluates centralized fusion (FusionEngine) and
    CI-based distributed fusion (CovarianceIntersection).
    """
    targets = []
    rng = np.random.default_rng(42)
    for i in range(n_targets):
        state = rng.uniform([10, -1, 10, -1], [90, 1, 90, 1])
        targets.append(Target(target_id=i, initial_state=state))
    drones = []
    for i in range(n_drones):
        angle = 2 * np.pi * i / n_drones
        pos = np.array([50 + 30 * np.cos(angle), 50 + 30 * np.sin(angle)])
        drones.append(Drone(drone_id=i, position=pos, fov_radius=40.0))
    return SimulationWorld(targets=targets, drones=drones)
