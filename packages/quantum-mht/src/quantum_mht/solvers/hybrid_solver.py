"""D-Wave LeapHybrid solvers (BQM / CQM / NL-Stride) for large MTDA problems.

Routes QUBO-sized Multi-Target Data Association instances to one of three
D-Wave Leap hybrid services, chosen according to the problem's structural
fingerprint rather than by name:

    - ``LeapHybridSampler``   — BQM, default. Up to ~1M binary variables,
      5s wall-clock floor. Correct default for pure-binary pairwise QUBOs
      like ours. Osaba & Miranda-Rodriguez, IEEE Access 13:4724 (2025),
      DOI 10.1109/ACCESS.2025.3525620 shows BQM/CQM beat NL on pure-binary
      Max-Cut-class problems at p<0.01 across 15-instance benchmarks —
      which is the regime our MTDA QUBO occupies.
    - ``LeapHybridCQMSampler`` — CQM. Use when explicit equality/inequality
      constraints exist natively (mixed binary/integer/real). Up to
      ~500k variables, 100k constraints.
    - ``LeapHybridNLSampler``  — NL (rebranded "Stride" 2026-02-02). Only
      when the problem carries higher-order (cubic+) or permutation
      structure. MTDA's one-hot row/column constraints are quadratic
      after Stollenwerk encoding, so NL is NOT the right default here.
      Kept behind a ``has_higher_order_terms`` guard for future MTDA
      extensions (cross-frame triplet continuity, logical OR gates).

The NL fallback remains available for experimentation, but the solver
emits a warning and falls back to BQM whenever NL is requested on a
pure-binary pairwise QUBO — honest engineering beats silent performance
regressions (Osaba 2025, Section V).

Academic References:
    Osaba & Miranda-Rodriguez, IEEE Access 13:4724 (2025),
        DOI 10.1109/ACCESS.2025.3525620 -- NL-vs-CQM-vs-BQM benchmark.
    Stollenwerk et al., arXiv:2110.08346 -- MTDA QUBO formulation.
    Robbins, "Exploring Hybrid Quantum Solvers" (D-Wave Medium, 2024)
        -- CQM vs NL formulation-driven selection heuristics.
    D-Wave Ocean SDK 9.3 -- LeapHybrid* samplers; Stride rebrand in
        Leap release notes, 2026-02-02.
"""
from __future__ import annotations

import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

logger = logging.getLogger(__name__)


# Capacity limits per Leap solver (Ocean SDK 9.3 release notes, 2026-04).
NL_MAX_VARS = 2_000_000
CQM_MAX_VARS = 500_000
CQM_MAX_CONSTRAINTS = 100_000
BQM_MAX_VARS = 1_000_000


@dataclass
class HybridSolver(MTDASolver):
    """D-Wave Leap hybrid routing layer.

    Attributes
    ----------
    time_limit_s : int
        Wall-clock budget passed to the selected Leap sampler.
    nl_mode : bool
        When True, request ``LeapHybridNLSampler`` ("Stride"). Falls back
        to BQM if the problem is pure-binary pairwise — per Osaba 2025
        benchmark, NL underperforms BQM on that regime.
    cqm_mode : bool
        When True, use ``LeapHybridCQMSampler``. Preferred when the
        caller constructs a constrained model with native equality or
        inequality constraints.
    """

    time_limit_s: int = 5
    nl_mode: bool = False
    cqm_mode: bool = False

    @property
    def name(self) -> str:
        if self.nl_mode:
            return "LeapHybridNL-Stride"
        if self.cqm_mode:
            return "LeapHybridCQM"
        return "LeapHybridBQM"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()
        self._check_capacity(qubo_result.num_variables)

        if self.cqm_mode:
            return self._solve_cqm(qubo_result, t0)

        if self.nl_mode:
            if not getattr(qubo_result, "has_higher_order_terms", False):
                logger.warning(
                    "NL solver requested but QUBO is pure-binary pairwise; "
                    "per Osaba IEEE Access 13:4724 (2025, "
                    "DOI 10.1109/ACCESS.2025.3525620) NL underperforms BQM "
                    "in this regime. Falling back to BQM."
                )
            else:
                bqm = self._to_bqm_or_runtime_error(qubo_result)
                result = self._solve_nl(bqm, qubo_result, t0)
                if result is not None:
                    return result
                logger.warning(
                    "LeapHybridNLSampler unavailable; falling back to BQM."
                )

        bqm = self._to_bqm_or_runtime_error(qubo_result)
        return self._solve_bqm(bqm, qubo_result, t0)

    def _to_bqm_or_runtime_error(self, qubo_result: Any) -> Any:
        try:
            return qubo_result.to_bqm()
        except ModuleNotFoundError as e:
            raise RuntimeError("dwave-ocean-sdk required for hybrid solver") from e

    def _check_capacity(self, num_vars: int) -> None:
        cap = self._selected_cap()
        if num_vars > cap:
            raise RuntimeError(
                f"{self.name} capacity exceeded: "
                f"{num_vars} variables > {cap} cap"
            )

    def _selected_cap(self) -> int:
        if self.nl_mode:
            return NL_MAX_VARS
        if self.cqm_mode:
            return CQM_MAX_VARS
        return BQM_MAX_VARS

    def _solve_bqm(self, bqm, qubo_result: Any, t0: float) -> SolverResult:
        try:
            from dwave.system import LeapHybridSampler
        except ImportError as e:
            raise RuntimeError("dwave-ocean-sdk required for hybrid solver") from e

        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm, time_limit=self.time_limit_s)

        elapsed = time.perf_counter() - t0
        best = sampleset.first
        solution = np.array(
            [int(best.sample[i]) for i in range(qubo_result.num_variables)]
        )
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
                "time_limit": self.time_limit_s,
                "mode": "bqm",
            },
        )

    def _solve_cqm(self, qubo_result: Any, t0: float) -> SolverResult:
        """Route through LeapHybridCQMSampler when native constraints matter."""
        try:
            import dimod
            from dwave.system import LeapHybridCQMSampler
        except ImportError as e:
            raise RuntimeError("dwave-ocean-sdk required for CQM hybrid") from e

        num_vars = qubo_result.num_variables
        cqm = dimod.ConstrainedQuadraticModel()
        x = [dimod.Binary(f"x{i}") for i in range(num_vars)]

        linear: dict[int, float] = {}
        quadratic: dict[tuple[int, int], float] = {}
        for (i, j), val in qubo_result.Q.items():
            if i == j:
                linear[i] = linear.get(i, 0.0) + val
            else:
                quadratic[(i, j)] = quadratic.get((i, j), 0.0) + val

        objective = sum(coef * x[i] for i, coef in linear.items())
        for (i, j), coef in quadratic.items():
            objective += coef * x[i] * x[j]
        cqm.set_objective(objective)

        variables = qubo_result.variables
        n_tracks = getattr(variables, "n_tracks", 0)
        n_meas = getattr(variables, "n_measurements", 0)
        for i in range(n_tracks):
            row = [x[variables.var_index(i, j)] for j in range(n_meas)]
            if getattr(variables, "include_missed", False):
                row.append(x[variables.var_index(i, -1)])
            if row:
                cqm.add_constraint(sum(row) == 1, label=f"row_{i}")
        for j in range(n_meas):
            col = [x[variables.var_index(i, j)] for i in range(n_tracks)]
            if getattr(variables, "include_false_alarm", False):
                col.append(x[variables.var_index(-1, j)])
            if col:
                cqm.add_constraint(sum(col) == 1, label=f"col_{j}")

        sampler = LeapHybridCQMSampler()
        sampleset = sampler.sample_cqm(cqm, time_limit=self.time_limit_s)
        feasible = sampleset.filter(lambda row: row.is_feasible)
        chosen = feasible.first if len(feasible) else sampleset.first

        elapsed = time.perf_counter() - t0
        solution = np.array(
            [int(chosen.sample[f"x{i}"]) for i in range(num_vars)]
        )
        decoded = qubo_result.variables.decode_solution(solution)
        energy = float(
            sum(val * solution[i] * solution[j] for (i, j), val in qubo_result.Q.items())
        )

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=energy,
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata={
                "time_limit": self.time_limit_s,
                "mode": "cqm",
                "feasible_sample_count": len(feasible),
            },
        )

    def _solve_nl(
        self, bqm, qubo_result: Any, t0: float
    ) -> SolverResult | None:
        """Route through LeapHybridNLSampler; return None if unavailable."""
        try:
            from dwave.optimization import Model  # type: ignore
            from dwave.system import LeapHybridNLSampler  # type: ignore
        except ImportError:
            return None

        num_vars = qubo_result.num_variables
        model = Model()
        x = model.binary(num_vars)

        linear = np.zeros(num_vars)
        quad_coeffs: list[tuple[int, int, float]] = []
        for (i, j), val in qubo_result.Q.items():
            if i == j:
                linear[i] += val
            elif i < j:
                quad_coeffs.append((i, j, val))
            else:
                quad_coeffs.append((j, i, val))

        objective = (linear * x).sum()
        for i, j, val in quad_coeffs:
            objective += val * x[i] * x[j]
        model.minimize(objective)

        sampler = LeapHybridNLSampler()
        result = sampler.sample(model, time_limit=self.time_limit_s)
        # LeapHybridNLSampler.sample returns a Future on Ocean >= 0.6;
        # resolve before reading the variable state so x.state(0) is
        # guaranteed to hold the final sample.
        if hasattr(result, "result"):
            with suppress(Exception):
                result.result()
        elif hasattr(result, "resolve"):
            with suppress(Exception):
                result.resolve()

        elapsed = time.perf_counter() - t0
        best = np.asarray(x.state(0), dtype=int)
        decoded = qubo_result.variables.decode_solution(best)
        energy = float(
            sum(val * best[i] * best[j] for (i, j), val in qubo_result.Q.items())
        )

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=energy,
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=best,
            metadata={
                "time_limit": self.time_limit_s,
                "mode": "nl_stride",
            },
        )
