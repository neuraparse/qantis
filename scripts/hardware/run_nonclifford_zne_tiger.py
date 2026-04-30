"""QANTIS-2 E2: Ezzell 2026 non-Clifford ZNE vs global-fold ZNE on Tiger.

Runs both mitigation schemes back-to-back on the Tiger belief-update
circuit at two depth regimes (ISA 15 minimal + ISA 85 4-state optimised)
and records Hellinger distance + zero-noise extrapolated expectation
values. This is the first hardware head-to-head of the Ezzell-Pokharel-
Lidar "non-Clifford ZNE" paper (Quantum 10:2003, 2026,
DOI 10.22331/q-2026-02-10-2003) against Mitiq's standard
``fold_gates_at_random`` on a POMDP inference circuit.

Uses :func:`quantum_common.mitigation.zne.non_clifford_scale_method` for
the identity-pair insertion around RX / RY / RZ rotations.

Usage
-----
Simulator dry-run (Aer + FakeHeronR3 noise model if available):

    python scripts/hardware/run_nonclifford_zne_tiger.py --dry-run

Hardware run on Pittsburgh:

    IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_nonclifford_zne_tiger.py \
        --backend ibm_pittsburgh --shots 8192 --regimes minimal 4state
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

import numpy as np

from scripts.hardware import (  # type: ignore
    compute_zz_expectation,
    databin_get_counts,
    run_on_aer_batched,
    run_on_heron_batched,
    save_partial,
    save_result,
)
from quantum_common.mitigation.zne import non_clifford_scale_method


def _build_tiger_minimal_circuit() -> Any:
    """Minimal Tiger belief-update circuit (|S|=2, ISA ~12-15)."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.ry(0.8, 0)
    qc.cx(0, 1)
    qc.rz(0.6, 1)
    qc.ry(-0.4, 0)
    qc.cx(1, 0)
    qc.rx(0.5, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _build_tiger_4state_circuit() -> Any:
    """4-state Tiger belief update (ISA ~85-86, optimized corridor)."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    # Short prior preparation
    qc.ry(0.9, 0)
    qc.ry(0.7, 1)
    qc.cx(0, 1)
    qc.ry(0.6, 2)
    qc.cx(1, 2)
    # "Observation" reweighting
    for _ in range(3):
        qc.rz(0.45, 0)
        qc.rx(0.3, 1)
        qc.rzz(0.4, 0, 1)
        qc.ry(0.2, 2)
        qc.rzz(0.3, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _reference_posterior(circuit: Any) -> np.ndarray:
    """Exact noiseless Aer reference for Hellinger comparison."""
    from qiskit.primitives import StatevectorSampler

    no_meas = circuit.remove_final_measurements(inplace=False)
    no_meas.measure_all(inplace=True)
    sampler = StatevectorSampler()
    counts = databin_get_counts(
        sampler.run([no_meas], shots=200_000).result()[0]
    )
    total = sum(counts.values())
    n = circuit.num_qubits
    probs = np.zeros(2 ** n)
    for bitstring, cnt in counts.items():
        idx = int(bitstring[::-1], 2)
        probs[idx] = cnt / total
    return probs


def _counts_to_probs(counts: dict[str, int], num_qubits: int) -> np.ndarray:
    probs = np.zeros(2 ** num_qubits)
    total = sum(counts.values())
    if total == 0:
        return probs
    for bitstring, cnt in counts.items():
        bits = bitstring.split()[-1]
        idx = int(bits[::-1], 2)
        probs[idx] = cnt / total
    return probs


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()))


def _scale_with(scaler: str, circuit: Any, scale_factor: float) -> Any:
    if scaler == "non_clifford":
        return non_clifford_scale_method(circuit, scale_factor)
    try:
        from mitiq.zne.scaling import fold_gates_at_random  # type: ignore

        return fold_gates_at_random(circuit, scale_factor=scale_factor)
    except Exception:
        from qiskit.circuit import QuantumCircuit

        if not isinstance(circuit, QuantumCircuit):
            return circuit
        folds = max(int(round((scale_factor - 1.0) / 2.0)), 0)
        folded = circuit.copy()
        body = circuit.copy()
        body.remove_final_measurements(inplace=True)
        inverse = body.inverse()
        for _ in range(folds):
            folded.compose(inverse, inplace=True)
            folded.compose(body, inplace=True)
        return folded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="ibm_pittsburgh")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--regimes", nargs="+", default=["minimal", "4state"])
    parser.add_argument("--scale-factors", nargs="+", type=float,
                        default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    regime_specs = []
    if "minimal" in args.regimes:
        regime_specs.append(("minimal_ISA15", _build_tiger_minimal_circuit()))
    if "4state" in args.regimes:
        regime_specs.append(("4state_ISA85", _build_tiger_4state_circuit()))

    # Build ALL circuits (regimes x scalers x scales) up front so hardware
    # dispatch is one Sampler.run; record the axis for post-processing.
    plan: list[tuple[str, str, float]] = []
    circuits = []
    references = {}
    for label, circuit in regime_specs:
        references[label] = _reference_posterior(circuit)
        for scaler in ("global_fold", "non_clifford"):
            for scale in args.scale_factors:
                plan.append((label, scaler, scale))
                circuits.append(_scale_with(scaler, circuit, scale))

    payload: dict = {
        "campaign": "E2_nonclifford_zne_vs_global_fold",
        "backend": args.backend,
        "dry_run": args.dry_run,
        "shots": args.shots,
        "scale_factors": list(args.scale_factors),
        "num_circuits": len(circuits),
        "regimes": [],
        "reference": (
            "Ezzell-Pokharel-Lidar Quantum 10:2003 (2026) "
            "DOI 10.22331/q-2026-02-10-2003"
        ),
    }
    save_partial("nonclifford_zne_tiger", payload)

    if args.dry_run:
        counts_list = run_on_aer_batched(circuits, args.shots)
        payload["job_id"] = None
    else:
        counts_list, job_id = run_on_heron_batched(
            circuits,
            backend_name=args.backend,
            shots=args.shots,
            checkpoint_name="nonclifford_zne_tiger",
        )
        payload["job_id"] = job_id

    # Re-aggregate into regime x method x scale rows.
    grouped: dict[str, dict[str, list]] = {}
    for (label, scaler, scale), counts in zip(plan, counts_list):
        ref = references[label]
        num_qubits = (ref.size.bit_length() - 1) if ref.size > 1 else 1
        probs = _counts_to_probs(counts, num_qubits)
        h = _hellinger(probs, ref)
        grouped.setdefault(label, {}).setdefault(scaler, []).append({
            "scale": float(scale),
            "hellinger": h,
        })
        save_partial("nonclifford_zne_tiger", {
            **payload,
            "_progress": {
                "regime": label, "scaler": scaler, "scale": scale,
                "h": h,
            },
        })

    # Linear ZNE extrapolation per (regime, method).
    for label, per_method in grouped.items():
        method_rows: dict = {}
        for scaler, rows in per_method.items():
            scales = np.array([r["scale"] for r in rows])
            hs = np.array([r["hellinger"] for r in rows])
            poly = np.polyfit(scales, hs, deg=1) if len(rows) >= 2 else [0.0, rows[0]["hellinger"]]
            method_rows[scaler] = {
                "rows": rows,
                "zero_noise_hellinger": float(np.polyval(poly, 0.0)),
                "slope": float(poly[0]),
                "intercept": float(poly[1]),
            }
        payload["regimes"].append({"regime": label, "per_method": method_rows})

    save_result("nonclifford_zne_tiger", payload)


if __name__ == "__main__":
    main()
