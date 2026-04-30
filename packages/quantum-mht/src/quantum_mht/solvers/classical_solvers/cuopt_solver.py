"""NVIDIA cuOpt 26.02 classical baseline solver for MTDA.

NVIDIA cuOpt v26.02 (Feb 2026) added Papilo MIP presolve, PDLP Stable3 mode,
cuDSS barrier, and concurrent MIP solving. At release it closed four
previously-unsolved MIPLIB instances and reported ~67% better primal-gap than
open-source CPU solvers on the MIPLIB primal suite.

For QANTIS this is a non-negotiable classical baseline: any claim that
QANTIS's quantum-mht beats a "GPU-accelerated classical solver" must be
benchmarked against cuOpt 26.02 specifically, not cuOpt 25.08 or dwave-neal.

The implementation wraps the cuOpt ``mip_solver`` callable when the package
is importable. When cuOpt is not installed (CPU-only dev environments, CI
without GPU), the solver raises a descriptive RuntimeError rather than
silently falling back — benchmarks must honestly report whether cuOpt ran.

Installation (GPU host only):
    pip install cuopt

Academic References:
    NVIDIA cuOpt release notes v26.02, Feb 2026.
    NVIDIA Developer Blog, "Learn How NVIDIA cuOpt Accelerates Mixed-Integer
        Optimization Using Primal Heuristics," Feb 2026.
    Kuhn, "The Hungarian Method for the Assignment Problem," 1955 — optimal
        baseline we cross-check cuOpt against on single-frame instances.
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
class CuOptSolver(MTDASolver):
    """NVIDIA cuOpt 26.02 MIP solver baseline.

    Attributes
    ----------
    time_limit_s : float
        Wall-clock budget. We benchmark against 60s and 300s time limits
        to match reviewer expectations for "Gurobi 60s / 300s" style
        comparisons.
    optimality_tolerance : float
        MIP gap tolerance. cuOpt's default of 1e-4 is suitable for MTDA
        where costs are negative log-likelihoods and relative gap matters
        more than absolute.
    gpu_id : int
        Device index for multi-GPU hosts; cuOpt picks GPU 0 by default.
    """

    time_limit_s: float = 60.0
    optimality_tolerance: float = 1e-4
    gpu_id: int = 0

    @property
    def name(self) -> str:
        return f"cuOpt26.02({self.time_limit_s:.0f}s)"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()

        try:
            import cuopt  # type: ignore
            from cuopt import mip_solver  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "NVIDIA cuOpt 26.02 is not installed. This solver is intended "
                "as a classical GPU baseline for QANTIS benchmarks and must "
                "not be silently skipped. Install on a GPU host: `pip install "
                f"cuopt`. Underlying error: {e}"
            ) from e

        num_vars = qubo_result.num_variables
        linear, quadratic = self._split_qubo(qubo_result.Q)

        # cuOpt MIP model — express QUBO as a MILP with quadratic objective
        # via McCormick linearization on off-diagonal terms. This is the
        # standard QUBO->MILP transformation that cuOpt accepts natively.
        model = self._build_mip_model(
            mip_solver, num_vars, linear, quadratic
        )
        params = mip_solver.SolverParameters()
        params.time_limit = float(self.time_limit_s)
        params.mip_gap = float(self.optimality_tolerance)
        params.gpu_id = int(self.gpu_id)

        result = mip_solver.Solve(model, params)
        elapsed = time.perf_counter() - t0

        solution = np.array(
            [int(round(result.variable_values[i])) for i in range(num_vars)],
            dtype=int,
        )
        decoded = qubo_result.variables.decode_solution(solution)
        objective = self._evaluate(qubo_result.Q, solution)

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=float(objective),
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata={
                "time_limit_s": self.time_limit_s,
                "mip_gap_tolerance": self.optimality_tolerance,
                "cuopt_version": getattr(cuopt, "__version__", "unknown"),
                "status": str(getattr(result, "status", "unknown")),
            },
        )

    def _split_qubo(
        self, Q: dict[tuple[int, int], float]
    ) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
        linear: dict[int, float] = {}
        quadratic: dict[tuple[int, int], float] = {}
        for (i, j), val in Q.items():
            if i == j:
                linear[i] = linear.get(i, 0.0) + val
            else:
                key = (min(i, j), max(i, j))
                quadratic[key] = quadratic.get(key, 0.0) + val
        return linear, quadratic

    def _build_mip_model(
        self,
        mip_solver,
        num_vars: int,
        linear: dict[int, float],
        quadratic: dict[tuple[int, int], float],
    ):
        """Build a binary MILP whose objective matches the QUBO.

        Off-diagonal QUBO terms ``q_ij x_i x_j`` are lifted via McCormick
        linearization using auxiliary binaries ``y_ij`` with the classical
        three-inequality relaxation. cuOpt handles this efficiently on GPU.
        """
        model = mip_solver.Model()
        x_vars = [model.add_binary(f"x{i}") for i in range(num_vars)]
        obj_terms: list = []
        for i, coeff in linear.items():
            obj_terms.append((coeff, x_vars[i]))
        for (i, j), coeff in quadratic.items():
            y_ij = model.add_binary(f"y_{i}_{j}")
            model.add_constraint(y_ij <= x_vars[i])
            model.add_constraint(y_ij <= x_vars[j])
            model.add_constraint(y_ij >= x_vars[i] + x_vars[j] - 1)
            obj_terms.append((coeff, y_ij))
        model.set_objective(
            mip_solver.Sense.MINIMIZE,
            sum(c * v for c, v in obj_terms),
        )
        return model

    def _evaluate(
        self, Q: dict[tuple[int, int], float], x: np.ndarray
    ) -> float:
        energy = 0.0
        for (i, j), val in Q.items():
            energy += val * float(x[i]) * float(x[j])
        return energy
