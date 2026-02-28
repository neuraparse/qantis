"""D-Wave LeapHybrid solver for large MTDA problems.

Uses D-Wave's LeapHybrid classical-quantum decomposition service for MTDA
problems that exceed direct QPU embedding capacity. LeapHybrid automatically
decomposes the QUBO into subproblems, solves smaller subproblems on the QPU
(Advantage2, 4400+ qubits, Zephyr topology), and combines results using
classical heuristics.

Architecture:
    - Cloud-based hybrid solver via D-Wave Leap API
    - Automatic problem decomposition (classical-quantum split)
    - Supports BQM problems with up to ~1M variables
    - Time-limited execution (default 5s, configurable)

This is the recommended solver for large-scale MTDA problems (>100 qubits)
where direct QPU embedding is infeasible due to qubit connectivity constraints.

Academic References:
    Stollenwerk et al., arXiv:2110.08346, 2021 -- QUBO formulation for MTDA.
    D-Wave Ocean SDK 9.x documentation -- LeapHybridSampler architecture,
        Advantage2 system specifications (4400+ qubits, Zephyr topology).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
import logging
import numpy as np
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

logger = logging.getLogger(__name__)

@dataclass
class HybridSolver(MTDASolver):
    """D-Wave LeapHybrid solver for problems too large for direct embedding.

    LeapHybrid decomposes the QUBO into classical and quantum subproblems,
    routing quantum-amenable subproblems to the Advantage2 QPU while solving
    the remainder classically. This enables MTDA problems with hundreds of
    tracks/measurements that exceed direct QPU embedding limits.

    References:
        D-Wave Ocean SDK 9.x -- LeapHybridSampler, Advantage2 (Zephyr topology).
    """
    time_limit_s: int = 5

    @property
    def name(self) -> str:
        return "LeapHybrid"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()
        bqm = qubo_result.to_bqm()

        try:
            from dwave.system import LeapHybridSampler
        except ImportError as e:
            raise RuntimeError("dwave-ocean-sdk required for hybrid solver") from e

        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm, time_limit=self.time_limit_s)

        elapsed = time.perf_counter() - t0
        best = sampleset.first
        solution = np.array([int(best.sample[i]) for i in range(qubo_result.num_variables)])
        decoded = qubo_result.variables.decode_solution(solution)

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=float(best.energy),
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata={"time_limit": self.time_limit_s},
        )
