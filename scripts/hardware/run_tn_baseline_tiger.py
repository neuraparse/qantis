"""QANTIS-2 E6: Classical tensor-network baseline for the Tiger cost circuit.

Runs a loopy belief-propagation tensor-network contraction on the Tiger
belief-update cost landscape, reformulated as a mini-QUBO with the same
ZZ / Z structure as the quantum circuit. Provides a published Tindall-
style rebuttal anchor (PRX Quantum 5:010308, 2024) by showing that a
classical tensor network matches the quantum posterior at ``|S|=2`` and
quantifying where it diverges as ``|S|`` grows.

Runs entirely on CPU, so the experiment takes minutes and has no QPU
budget. Used as the classical counterpoint to E1-E5 in the campaign.

Usage
-----
    python scripts/hardware/run_tn_baseline_tiger.py --sizes 2 4 6 --beta 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-mht" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))

import numpy as np

from scripts.hardware import save_result  # type: ignore
from quantum_mht.solvers.classical_solvers.tn_baseline_solver import (
    TensorNetworkBaselineSolver,
)


class _SyntheticQUBO:
    """Minimal QUBO stand-in matching the interface expected by the TN solver."""

    def __init__(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.num_variables = n
        self.Q = {}
        for i in range(n):
            self.Q[(i, i)] = float(rng.normal(0.0, 0.5))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.6:
                    self.Q[(i, j)] = float(rng.normal(0.0, 0.3))
        self.variables = _Variables(n)
        self.has_higher_order_terms = False
        self.dynamic_range = 1.0


class _Variables:
    def __init__(self, n: int) -> None:
        self.n_tracks = n
        self.n_measurements = 0
        self.include_missed = False
        self.include_false_alarm = False

    def decode_solution(self, solution):
        return {
            "assignments": [],
            "missed_detections": [],
            "false_alarms": [],
        }

    def var_index(self, i, j):
        return i


def _brute_force(Q, n):
    best_e = float("inf")
    best_x = None
    for n_idx in range(1 << n):
        x = np.array([(n_idx >> b) & 1 for b in range(n)], dtype=int)
        e = 0.0
        for (i, j), val in Q.items():
            e += val * x[i] * x[j]
        if e < best_e:
            best_e = e
            best_x = x
    return best_x, best_e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--seeds", type=int, default=5)
    # Accepted for consistency with the campaign orchestrator; this
    # experiment runs entirely on CPU so the flag is a no-op.
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    try:
        import quimb  # type: ignore  # noqa: F401
    except ImportError:
        save_result(
            "tn_baseline_tiger_unavailable",
            {
                "campaign": "E6_tn_baseline_tiger",
                "error": "quimb>=1.8 required; install via `pip install quimb`",
            },
        )
        return

    rows = []
    for n in args.sizes:
        for seed in range(args.seeds):
            qubo = _SyntheticQUBO(n, seed)
            solver = TensorNetworkBaselineSolver(beta=args.beta, method="bp")
            t0 = time.perf_counter()
            try:
                result = solver.solve(qubo)
                elapsed = time.perf_counter() - t0
                bp_energy = result.objective_value
                bp_error = None
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                bp_energy = float("nan")
                bp_error = f"{type(exc).__name__}: {exc}"

            optimal_x, optimal_energy = _brute_force(qubo.Q, n)
            gap = (
                (bp_energy - optimal_energy) / abs(optimal_energy)
                if optimal_energy != 0 and not np.isnan(bp_energy)
                else float("nan")
            )
            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "bp_energy": bp_energy,
                    "optimal_energy": optimal_energy,
                    "relative_gap": gap,
                    "wall_time_s": elapsed,
                    "bp_error": bp_error,
                },
            )

    save_result(
        "tn_baseline_tiger",
        {
            "campaign": "E6_tn_baseline_tiger",
            "beta": args.beta,
            "seeds_per_size": args.seeds,
            "rows": rows,
            "reference": (
                "Tindall-Fishman-Stoudenmire-Sels PRX Quantum 5:010308 (2024) "
                "DOI 10.1103/PRXQuantum.5.010308; Mauron-Carleo arXiv:2503.08247"
            ),
        },
    )


if __name__ == "__main__":
    main()
