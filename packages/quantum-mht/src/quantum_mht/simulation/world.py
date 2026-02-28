"""2D/3D simulation environment for MHT algorithm evaluation.

Provides a discrete-time simulation world that generates ground-truth target
trajectories and noisy sensor measurements for testing and benchmarking MHT
algorithms. The simulation includes:

    - Target motion with configurable dynamics models (CV, CT, CA)
    - Multi-drone sensor platforms with field-of-view constraints
    - Probabilistic detection with configurable P_D
    - Poisson-distributed clutter (false alarm) generation
    - Bounded 2D environment with configurable extent

The generated measurement scans serve as input to the tracking pipeline,
enabling reproducible evaluation of quantum vs classical solvers.

Academic References:
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995 -- multi-target simulation
        framework and measurement generation model.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 12 -- simulation design for tracking
        system evaluation.
    Li & Jilkov, "Survey of Maneuvering Target Tracking Part I: Dynamic
        Models", IEEE TAES 39(4):1333-1364, 2003 -- target motion models
        used in simulation.

Quantum Benchmark Role (2026):
    SimulationWorld provides the controlled environment for reproducible
    quantum vs classical solver benchmarking. Each time step generates a
    MeasurementScan that feeds into the QUBO pipeline. The configurable
    clutter rate and detection probability allow systematic evaluation
    of solver performance across difficulty levels (low clutter/high P_D
    = easy; high clutter/low P_D = hard). Ground-truth trajectories
    enable OSPA and CLEAR MOT metric computation for solver comparison.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from quantum_mht.simulation.target import Target
from quantum_mht.simulation.drone import Drone
from quantum_mht.fusion.measurement import Measurement, MeasurementScan

@dataclass
class SimulationWorld:
    """2D simulation environment for MHT testing.

    Generates ground-truth trajectories and noisy measurements at each
    discrete time step. Clutter follows a Poisson process with rate
    clutter_rate (Bar-Shalom & Li 1995, Ch 1).
    """
    targets: list[Target] = field(default_factory=list)
    drones: list[Drone] = field(default_factory=list)
    clutter_rate: float = 0.5
    bounds: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0)
    dt: float = 1.0
    time: float = 0.0

    def step(self, rng: np.random.Generator) -> MeasurementScan:
        """Advance simulation by one time step and generate measurements."""
        self.time += self.dt
        # Move targets
        for target in self.targets:
            target.step(self.dt, rng)
        # Move drones
        for drone in self.drones:
            drone.step(self.dt)
        # Generate measurements
        measurements: list[Measurement] = []
        for drone in self.drones:
            for target in self.targets:
                obs = drone.observe(target.state, rng)
                if obs is not None:
                    measurements.append(Measurement(
                        position=obs,
                        covariance=drone.sensor.measurement_covariance(),
                        sensor_id=drone.drone_id,
                        timestamp=self.time,
                    ))
        # Add Poisson-distributed clutter (Bar-Shalom & Li 1995, Ch 1)
        n_clutter = rng.poisson(self.clutter_rate)
        for _ in range(n_clutter):
            x = rng.uniform(self.bounds[0], self.bounds[1])
            y = rng.uniform(self.bounds[2], self.bounds[3])
            measurements.append(Measurement(
                position=np.array([x, y]),
                covariance=np.eye(2) * 10.0,
                sensor_id=-1,
                timestamp=self.time,
            ))
        return MeasurementScan(measurements=measurements, timestamp=self.time)
