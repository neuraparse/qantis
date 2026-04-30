"""Fixstars Amplify SDK adapter as classical Ising-machine baseline.

Fixstars Amplify (Engine / SDK >= v1.5.0, 2026-04) exposes a unified
Python interface for running QUBO / HUBO problems on multiple physics-
inspired backends:

    - Amplify AE (Fixstars' GPU simulated-annealing engine),
    - Toshiba SQBM+ (dSB / bSB simulated-bifurcation),
    - Fujitsu Digital Annealer,
    - Hitachi CMOS Annealer,
    - and D-Wave Leap samplers through a shared abstraction.

For QANTIS this is the single most cost-effective multi-vendor baseline
-- one integration + one API key gives us several published classical
annealing comparators behind the ``client_type`` selector. MTDA QUBO
sizes (175-3875 variables) sit comfortably in the Amplify free-tier
envelope.

Academic / product references:
    Fixstars Amplify Engine page -- https://amplify.fixstars.com/en/engine
    Fixstars Amplify SDK v1 docs -- https://amplify.fixstars.com/en/docs/amplify/v1/
    Kittichaikoonkij et al. (2025) -- Chulalongkorn benchmark comparing
        Amplify AE vs D-Wave Leap vs Gurobi on 3SAT / QAP / TSP, with
        Fixstars achieving best accuracy on most instances.
    Shaglel et al., arXiv:2507.22117 (Jul 2025) -- Max-Cut benchmark
        where SBM and hybrid SA tied or beat D-Wave on medium scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import time
import logging
import numpy as np

from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

logger = logging.getLogger(__name__)


@dataclass
class FixstarsAmplifySolver(MTDASolver):
    """Run MTDA QUBO via the Fixstars Amplify SDK.

    Attributes
    ----------
    client_type : {"ae", "sqbm", "fujitsu_da", "hitachi_cmos"}
        Backend to dispatch through Amplify's unified client layer.
    token : str | None
        Amplify API token. When ``None`` the adapter reads
        ``FIXSTARS_AMPLIFY_TOKEN`` from the environment.
    time_limit_ms : int
        Wall-clock budget per solve, expressed in milliseconds as
        required by the Amplify client API.
    num_reads : int
        Number of independent samples to request.
    """

    client_type: Literal["ae", "sqbm", "fujitsu_da", "hitachi_cmos"] = "ae"
    token: str | None = None
    time_limit_ms: int = 1_000
    num_reads: int = 20

    @property
    def name(self) -> str:
        return f"Fixstars({self.client_type})"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()

        try:
            from amplify import (  # type: ignore
                Model,
                VariableGenerator,
                solve,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Fixstars Amplify SDK not installed. Install with "
                "`pip install amplify`. Underlying error: " + str(exc)
            ) from exc

        client = self._build_client()
        num_vars = qubo_result.num_variables

        gen = VariableGenerator()
        x = gen.array("Binary", num_vars)

        objective = 0.0
        for (i, j), val in qubo_result.Q.items():
            if i == j:
                objective = objective + val * x[i]
            else:
                objective = objective + val * x[i] * x[j]

        model = Model(objective)
        result = solve(model, client, num_solves=self.num_reads)
        if not result.solutions:
            raise RuntimeError("Fixstars Amplify returned no feasible solutions")
        best = min(result.solutions, key=lambda sol: sol.objective)

        solution = np.array(
            [int(round(best.values[x[i]])) for i in range(num_vars)], dtype=int
        )
        decoded = qubo_result.variables.decode_solution(solution)
        elapsed = time.perf_counter() - t0

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=float(best.objective),
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata={
                "client_type": self.client_type,
                "time_limit_ms": self.time_limit_ms,
                "num_reads": self.num_reads,
            },
        )

    def _build_client(self):
        import os

        from amplify import (  # type: ignore
            FixstarsClient,
            ToshibaSQBM2Client,
            FujitsuDA4Client,
            HitachiCMOSAnnealerClient,
        )

        token = self.token or os.environ.get("FIXSTARS_AMPLIFY_TOKEN")
        if not token:
            raise RuntimeError(
                "Fixstars Amplify requires an API token via "
                "`FixstarsAmplifySolver(token=...)` or the "
                "FIXSTARS_AMPLIFY_TOKEN environment variable."
            )

        clients = {
            "ae": FixstarsClient,
            "sqbm": ToshibaSQBM2Client,
            "fujitsu_da": FujitsuDA4Client,
            "hitachi_cmos": HitachiCMOSAnnealerClient,
        }
        try:
            cls = clients[self.client_type]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Fixstars client_type: {self.client_type}"
            ) from exc

        client = cls()
        client.token = token
        params = getattr(client, "parameters", None)
        if params is not None:
            timeout = getattr(params, "timeout", None)
            if timeout is not None:
                try:
                    params.timeout = self.time_limit_ms
                except Exception:
                    logger.debug("Fixstars client timeout field unsupported")
        return client
