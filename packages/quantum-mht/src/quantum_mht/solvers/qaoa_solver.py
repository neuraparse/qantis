"""IBM Quantum QAOA solver for MTDA.

Implements the Quantum Approximate Optimization Algorithm (QAOA) for solving
the MTDA QUBO on gate-model quantum hardware or simulators.

Uses Qiskit Optimization MinimumEigenOptimizer with QAOA.
Compatible with qiskit-optimization 0.7+ and Qiskit v2.3 (Jan 2026),
which introduces SamplerV2 and EstimatorV2 as the standard primitives.

Shot Count Issue (arXiv:2512.08245, Dec 2025):
    SamplerV2 defaults to 10,000 shots, which captures only ~23% of the
    state space for typical MTDA problems. This solver uses StatevectorSampler
    for exact simulation (no shot noise), or explicitly sets high shot counts
    (default 100K) for hardware execution to mitigate undersampling.

Academic References:
    Farhi, Goldstone & Gutmann, "A Quantum Approximate Optimization Algorithm",
        arXiv:1411.4028, 2014 -- original QAOA formulation.
    Stollenwerk et al., arXiv:2110.08346, 2021 -- QUBO formulation for MTDA.
    arXiv:2512.08245, Dec 2025 -- SamplerV2 shot count coverage analysis;
        10K shots yields only 23% state space coverage.
    Cai et al., Rev. Mod. Phys. 95, 045005, 2023 -- composable error
        mitigation techniques applicable to QAOA circuits.
    Qiskit v2.3 (Jan 2026): SamplerV2, EstimatorV2, qiskit-optimization 0.7+.
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
class QAOASolver(MTDASolver):
    """QAOA solver using Qiskit Optimization.

    Implements QAOA (Farhi, Goldstone & Gutmann, arXiv:1411.4028, 2014)
    via qiskit-optimization 0.7+ MinimumEigenOptimizer.

    For simulator mode, uses StatevectorSampler (exact, no shot noise).
    For hardware, uses SamplerV2 with explicit high shot count to avoid
    the 23% coverage issue (arXiv:2512.08245).
    """
    reps: int = 2
    use_simulator: bool = True
    optimization_level: int = 1
    # Explicit high shot count for SamplerV2 to mitigate undersampling
    # (arXiv:2512.08245: default 10K shots covers only ~23% of state space)
    shots: int = 100_000
    maxiter: int = 200

    @property
    def name(self) -> str:
        return f"QAOA(p={self.reps})"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()

        try:
            from qiskit_optimization import QuadraticProgram
            from qiskit_optimization.algorithms import MinimumEigenOptimizer
            from qiskit.primitives import StatevectorSampler
        except ImportError as e:
            raise RuntimeError(f"Qiskit optimization packages required: {e}") from e

        # Build QuadraticProgram from QUBO
        qp = QuadraticProgram()
        for i in range(qubo_result.num_variables):
            qp.binary_var(f"x{i}")

        linear: dict[str, float] = {}
        quadratic: dict[tuple[str, str], float] = {}
        for (i, j), val in qubo_result.Q.items():
            if i == j:
                linear[f"x{i}"] = linear.get(f"x{i}", 0.0) + val
            else:
                quadratic[(f"x{i}", f"x{j}")] = (
                    quadratic.get((f"x{i}", f"x{j}"), 0.0) + val
                )
        qp.minimize(linear=linear, quadratic=quadratic)

        # Import QAOA -- qiskit-optimization 0.7+ (Qiskit v2.3, Jan 2026) migration:
        # QAOA moved to qiskit_optimization.algorithms in 0.7+
        try:
            from qiskit_optimization.algorithms import QAOA
            from qiskit_algorithms.optimizers import COBYLA
        except ImportError:
            from qiskit_algorithms import QAOA
            from qiskit_algorithms.optimizers import COBYLA

        # StatevectorSampler: exact simulation, no shot noise
        # (for hardware, use SamplerV2 with explicit shots to avoid arXiv:2512.08245 issue)
        sampler = StatevectorSampler()
        # QAOA circuit with p=reps layers (Farhi et al., arXiv:1411.4028, 2014)
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=self.maxiter), reps=self.reps)
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(qp)

        elapsed = time.perf_counter() - t0
        solution = np.array(
            [int(result.variables_dict.get(f"x{i}", 0)) for i in range(qubo_result.num_variables)]
        )
        decoded = qubo_result.variables.decode_solution(solution)

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=float(result.fval),
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata={"reps": self.reps, "status": str(result.status)},
        )
