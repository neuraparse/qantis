"""QANTIS-2 E3: Yoder-Low-Chuang bounded-length FPAA at P(o) boundaries.

Sweeps the number of Grover iterations ``k in {1, 3, 5}`` at four
observation-probability regimes and records the measured amplification
factor + Hellinger distance. ``k`` is chosen both manually and via
:func:`quantum_pomdp.quantum_circuits.amplitude_amplifier.bounded_length_fpaa`
which implements the Yoder-Low-Chuang 2014 length formula. The sweep
lets reviewers see the depth-vs-gain tradeoff that the current paper
only reports at ``k=1``.

Boundary P(o) values probed:

    - 0.01, 0.05 (rare-observation regime, headline 3.6-4.2x claim)
    - 0.95, 0.99 (near-certain, boundary-BIQAE regime)

Usage
-----
Simulator:

    python scripts/hardware/run_bounded_fpaa_boundary.py --dry-run

Hardware on Pittsburgh:

    IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_bounded_fpaa_boundary.py \
        --backend ibm_pittsburgh --shots 4096
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

import numpy as np

from scripts.hardware import (  # type: ignore
    run_on_aer_batched,
    run_on_heron_batched,
    save_partial,
    save_result,
)
from quantum_pomdp.quantum_circuits.amplitude_amplifier import bounded_length_fpaa


def _build_1q_grover(theta: float, k: int):
    """1-qubit Grover amplification oracle for the amplitude a = sin(theta)**2.

    P(|1>) after k iterations is ``sin((2k+1) * theta)**2``.
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.ry(2 * theta, 0)
    for _ in range(k):
        qc.z(0)  # S_chi (mark |1>)
        qc.ry(-2 * theta, 0)
        qc.z(0)  # S_0 (reflect around |0>)
        qc.ry(2 * theta, 0)
    qc.measure(0, 0)
    return qc


def _row_from_counts(
    p_obs: float, k: int, shots: int, counts: dict[str, int]
) -> dict[str, float]:
    theta = math.asin(math.sqrt(max(min(p_obs, 1.0 - 1e-6), 1e-6)))
    expected_p1 = math.sin((2 * k + 1) * theta) ** 2
    total = sum(counts.values()) or 1
    p1_measured = counts.get("1", 0) / total
    amplification = p1_measured / p_obs if p_obs > 0 else float("nan")
    hellinger = math.sqrt(
        0.5 * ((math.sqrt(p1_measured) - math.sqrt(expected_p1)) ** 2
               + (math.sqrt(1 - p1_measured) - math.sqrt(1 - expected_p1)) ** 2)
    )
    return {
        "p_obs": p_obs,
        "k": k,
        "theta": theta,
        "expected_p1": expected_p1,
        "measured_p1": p1_measured,
        "amplification": amplification,
        "hellinger_vs_ideal": hellinger,
        "shots": shots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="ibm_pittsburgh")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument(
        "--p-obs", nargs="+", type=float, default=[0.01, 0.05, 0.95, 0.99],
    )
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[1, 3, 5],
    )
    parser.add_argument(
        "--failure-tolerance", type=float, default=0.01,
        help="Yoder-Low-Chuang delta used by bounded_length_fpaa.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Build every circuit up-front so the backend dispatch is one
    # Sampler.run([...]) call instead of N queue waits.
    plan: list[tuple[float, int, bool]] = []
    circuits = []
    auto_k_rows: list[dict] = []
    for p_obs in args.p_obs:
        omega = math.sqrt(p_obs)
        auto_k = bounded_length_fpaa(omega, args.failure_tolerance)
        auto_k_rows.append({"p_obs": p_obs, "yoder_bounded_length": auto_k})
        for k in args.ks:
            theta = math.asin(math.sqrt(max(min(p_obs, 1 - 1e-6), 1e-6)))
            plan.append((p_obs, k, k == auto_k))
            circuits.append(_build_1q_grover(theta, k))

    payload: dict = {
        "campaign": "E3_bounded_fpaa_boundary",
        "backend": args.backend,
        "dry_run": args.dry_run,
        "shots": args.shots,
        "failure_tolerance": args.failure_tolerance,
        "num_circuits": len(circuits),
        "yoder_auto_k": auto_k_rows,
        "rows": [],
        "reference": (
            "Yoder-Low-Chuang PRL 113:210501 (2014); "
            "Utsumi-Nakata Quantum 10:2024 (2026)."
        ),
    }
    save_partial("bounded_fpaa_boundary", payload)

    if args.dry_run:
        counts_list = run_on_aer_batched(circuits, args.shots)
        payload["job_id"] = None
    else:
        counts_list, job_id = run_on_heron_batched(
            circuits,
            backend_name=args.backend,
            shots=args.shots,
            checkpoint_name="bounded_fpaa_boundary",
        )
        payload["job_id"] = job_id

    for (p_obs, k, is_yoder), counts in zip(plan, counts_list):
        row = _row_from_counts(p_obs, k, args.shots, counts)
        row["is_yoder_bound"] = is_yoder
        payload["rows"].append(row)
        save_partial("bounded_fpaa_boundary", payload)

    save_result("bounded_fpaa_boundary", payload)


if __name__ == "__main__":
    main()
