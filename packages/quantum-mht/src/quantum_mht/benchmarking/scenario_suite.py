"""Predefined benchmark scenarios for MHT evaluation.

Provides a standardized suite of simulation scenarios for reproducible
benchmarking of quantum and classical MTDA solvers. Each scenario exercises
different tracking challenges (crossing, clutter, multi-sensor).

Scenario selection follows the methodology from the CLEAR MOT benchmark
(Bernardin & Stiefelhagen 2008), which emphasizes controlled evaluation
across progressively harder conditions.

Academic References:
    Bernardin & Stiefelhagen, "Evaluating Multiple Object Tracking
        Performance: The CLEAR MOT Metrics", J. Image and Video Processing,
        2008 -- benchmark scenario methodology.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 12 -- scenario design for tracking evaluation.
    Stollenwerk et al., arXiv:2110.08346, 2021 -- quantum vs classical
        solver comparison across problem sizes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from quantum_mht.simulation.scenario_generator import crossing_targets, dense_clutter, swarm_patrol

@dataclass
class ScenarioSuite:
    """Suite of benchmark scenarios for MHT evaluation.

    Provides standardized scenarios with increasing difficulty for
    reproducible quantum vs classical solver comparison.
    """
    name: str = "default"
    num_steps: int = 50

    def get_scenarios(self) -> dict:
        return {
            "crossing_5": crossing_targets(n_targets=5),
            "crossing_10": crossing_targets(n_targets=10),
            "dense_clutter": dense_clutter(n_targets=3, clutter_rate=5.0),
            "swarm_3x5": swarm_patrol(n_drones=3, n_targets=5),
        }
