"""Gurobi 13.0 classical baseline solver for MTDA.

Gurobi 13.0 (Nov 2025) remains the commercial MIP state-of-the-art as of
Apr 2026: ~16% faster MIP vs 12.x, 2× MINLP speedup, GPU-PDHG for LP, and
a nonlinear barrier method. Any "quantum beats classical MIP" claim from
QANTIS must be benchmarked against Gurobi 13.0 with reasonable time limits
(60s / 300s) rather than pre-12 numbers that reviewers will discard.

This solver uses Gurobi's native quadratic objective API (``setObjective``
with a ``QuadExpr``), so no McCormick lift is required. Gurobi handles
binary quadratic objectives directly and more efficiently than the
linearized path used by cuOpt.

License notes:
    Gurobi requires a commercial or academic license. On licensed hosts
    the ``gurobipy`` package imports and ``gurobipy.Model()`` succeeds.
    In CI / CPU-only dev environments without a license, this solver
    raises a descriptive RuntimeError — benchmarks must honestly record
    whether Gurobi ran.

Academic References:
    Gurobi Optimization, "Gurobi 13.0 Release Notes," Nov 2025.
    Gurobi Business Wire, "Gurobi Releases Version 13.0 with Improved
        Performance and New Solving Capabilities," Nov 18, 2025.
    Kuhn, "The Hungarian Method for the Assignment Problem," 1955 —
        optimal baseline we cross-check Gurobi against on single-frame
        instances.
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
class GurobiSolver(MTDASolver):
    """Gurobi 13.0 MIP solver baseline with native quadratic objective.

    Attributes
    ----------
    time_limit_s : float
        Wall-clock budget. Standard benchmark grid is {60, 300}.
    mip_gap : float
        Relative MIP gap termination tolerance. Gurobi's default of 1e-4
        is appropriate for MTDA.
    threads : int
        Thread count for parallel MIP. 0 = Gurobi default (all cores).
    output_flag : int
        Gurobi ``OutputFlag`` parameter; 0 silences the solver output.
    """

    time_limit_s: float = 60.0
    mip_gap: float = 1e-4
    threads: int = 0
    output_flag: int = 0

    @property
    def name(self) -> str:
        return f"Gurobi13({self.time_limit_s:.0f}s)"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()

        try:
            import gurobipy as gp  # type: ignore
            from gurobipy import GRB  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Gurobi 13.0 is not installed / licensed. This solver is the "
                "commercial classical baseline for QANTIS benchmarks and must "
                "not be silently skipped. Install and license gurobipy, then "
                f"re-run. Underlying error: {e}"
            ) from e

        num_vars = qubo_result.num_variables

        model = gp.Model("MTDA_QUBO")
        model.Params.OutputFlag = int(self.output_flag)
        model.Params.TimeLimit = float(self.time_limit_s)
        model.Params.MIPGap = float(self.mip_gap)
        if self.threads > 0:
            model.Params.Threads = int(self.threads)

        x = model.addVars(num_vars, vtype=GRB.BINARY, name="x")

        obj = gp.QuadExpr()
        for (i, j), val in qubo_result.Q.items():
            if i == j:
                obj.addTerms(val, x[i])
            else:
                obj.add(val * x[i] * x[j])
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        elapsed = time.perf_counter() - t0
        if model.SolCount == 0:
            raise RuntimeError(
                f"Gurobi produced no feasible solution within {self.time_limit_s}s"
            )

        solution = np.array(
            [int(round(x[i].X)) for i in range(num_vars)], dtype=int
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
                "mip_gap_tolerance": self.mip_gap,
                "gurobi_status": int(model.Status),
                "gurobi_gap": float(model.MIPGap),
                "gurobi_runtime_s": float(model.Runtime),
                "gurobi_version": ".".join(map(str, gp.gurobi.version())),
            },
        )

    def _evaluate(
        self, Q: dict[tuple[int, int], float], x: np.ndarray
    ) -> float:
        energy = 0.0
        for (i, j), val in Q.items():
            energy += val * float(x[i]) * float(x[j])
        return energy
