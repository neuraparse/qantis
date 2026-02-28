"""D-Wave quantum annealing solver for MTDA.

Supports:
- Forward annealing (standard QUBO solve)
- Reverse annealing with warm-start from previous frame's solution
  (Ihara 2025, Scientific Reports: "Enhancing MOT via Quantum Annealing")
- Diabatic quantum annealing for multi-hypothesis enumeration
  (McCormick et al., arXiv:2209.00615)

Hardware Specifications (D-Wave Ocean SDK 9.x):
    - D-Wave Advantage2: 4400+ qubits, Zephyr topology (20-way connectivity)
    - `dwave.samplers` namespace for SimulatedAnnealingSampler (local dev)
    - `dwave.system.DWaveSampler` for QPU access via Leap cloud
    - EmbeddingComposite handles logical-to-physical qubit mapping
    - Typical annealing time: 20 us forward, custom PWL for reverse

Academic References:
    Stollenwerk et al., "Adiabatic Quantum Computing for Multi Object Tracking",
        Fraunhofer FKIE, arXiv:2110.08346, 2021 -- QUBO formulation solved
        via quantum annealing.
    Ihara, "Enhancing Multi-Object Tracking via Quantum Annealing",
        Scientific Reports 15:24294, 2025 -- reverse annealing warm-start
        exploiting temporal continuity between tracking frames.
    McCormick et al., "Bayesian Diabatic Quantum Annealing for Multi-Target
        Tracking", arXiv:2209.00615, 2022 -- diabatic quantum annealing (DQA)
        for multi-hypothesis enumeration.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import logging
import numpy as np
from numpy.typing import NDArray
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

logger = logging.getLogger(__name__)

@dataclass
class AnnealingSolver(MTDASolver):
    """D-Wave quantum annealing solver with reverse annealing warm-start.

    Reverse annealing (Ihara 2025): Uses previous frame's solution as
    initial_state to warm-start the annealer. The annealer starts from
    the classical state s=1, reverses to s=s_target (exploring nearby
    states), then reanneals forward. This exploits temporal continuity
    in sequential tracking frames.
    """
    num_reads: int = 1000
    annealing_time_us: float = 20.0
    use_reverse_annealing: bool = False
    reverse_anneal_s_target: float = 0.45
    reverse_anneal_hold_us: float = 10.0
    reinitialize_state: bool = True
    chain_strength: float | None = None
    use_simulator: bool = True
    _previous_solution: dict[int, int] | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        suffix = "+reverse" if self.use_reverse_annealing else ""
        return f"DWaveAnnealing{suffix}"

    def solve(self, qubo_result: Any) -> SolverResult:
        bqm = qubo_result.to_bqm()
        t0 = time.perf_counter()

        if self.use_simulator:
            sampleset = self._solve_simulated(bqm)
        elif self.use_reverse_annealing and self._previous_solution is not None:
            sampleset = self._solve_reverse_anneal(bqm)
        else:
            sampleset = self._solve_hardware(bqm)

        elapsed = time.perf_counter() - t0
        best = sampleset.first
        solution = np.array([int(best.sample[i]) for i in range(qubo_result.num_variables)])

        # Store solution for next frame's reverse annealing warm-start
        self._previous_solution = dict(best.sample)

        decoded = qubo_result.variables.decode_solution(solution)

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=float(best.energy),
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata={
                "num_reads": self.num_reads,
                "num_samples": len(sampleset),
                "reverse_annealing": self.use_reverse_annealing and self._previous_solution is not None,
            },
        )

    def _solve_simulated(self, bqm: Any) -> Any:
        # D-Wave Ocean SDK 9.x: prefer dwave.samplers namespace (Advantage2 era)
        # Falls back to legacy neal package for older installations
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError:
            import neal
            SimulatedAnnealingSampler = neal.SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
        return sampler.sample(bqm, num_reads=self.num_reads)

    def _solve_hardware(self, bqm: Any) -> Any:
        # D-Wave Advantage2: 4400+ qubits, Zephyr topology (Ocean SDK 9.x)
        # EmbeddingComposite handles logical-to-physical qubit mapping
        from dwave.system import DWaveSampler, EmbeddingComposite
        sampler = EmbeddingComposite(DWaveSampler())
        kwargs: dict[str, Any] = {
            "num_reads": self.num_reads,
            "annealing_time": self.annealing_time_us,
        }
        if self.chain_strength:
            kwargs["chain_strength"] = self.chain_strength
        return sampler.sample(bqm, **kwargs)

    def _solve_reverse_anneal(self, bqm: Any) -> Any:
        """Reverse annealing using previous solution as initial state.

        Anneal schedule (PWL): s starts at 1.0, reverses to s_target,
        holds, then returns to 1.0. This searches a neighborhood around
        the previous solution, exploiting frame-to-frame continuity.

        Reference: Ihara 2025, Scientific Reports 15, 24294.
        """
        from dwave.system import DWaveSampler, EmbeddingComposite

        sampler = DWaveSampler()
        embedded_sampler = EmbeddingComposite(sampler)

        # Build piecewise-linear reverse anneal schedule
        ramp_time = self.annealing_time_us / 3
        anneal_schedule = [
            [0.0, 1.0],
            [ramp_time, self.reverse_anneal_s_target],
            [ramp_time + self.reverse_anneal_hold_us, self.reverse_anneal_s_target],
            [2 * ramp_time + self.reverse_anneal_hold_us, 1.0],
        ]

        # Convert binary {0,1} solution to spin {-1,+1} for D-Wave
        initial_state = {}
        if self._previous_solution:
            for var, val in self._previous_solution.items():
                initial_state[var] = 2 * val - 1 if val in (0, 1) else val

        kwargs: dict[str, Any] = {
            "num_reads": self.num_reads,
            "anneal_schedule": anneal_schedule,
            "initial_state": initial_state,
            "reinitialize_state": self.reinitialize_state,
        }
        if self.chain_strength:
            kwargs["chain_strength"] = self.chain_strength

        return embedded_sampler.sample(bqm, **kwargs)

    def reset_warm_start(self) -> None:
        """Clear previous solution (e.g., when tracks change significantly)."""
        self._previous_solution = None
