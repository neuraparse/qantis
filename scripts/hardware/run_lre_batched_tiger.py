"""QANTIS-2 E4 (batched variant): Layerwise Richardson Extrapolation
on 4-state Tiger with *all* scaled circuits submitted as a single
SamplerV2 job per backend sweep.

The original ``run_lre_4state_tiger.py`` uses ``mitiq.lre.execute_with_lre``
which submits each scaled circuit as a separate queue entry; that pushes
${6 \times 20+}$ queue entries per campaign and stalls on contended
Kingston/Aachen windows.

Here we call ``mitiq.lre.construct_circuits`` and
``mitiq.lre.combine_results`` directly so the layer-scaled circuits can
be pooled across *all* priors into one batched ``SamplerV2.run`` call.
Chunking (``num_chunks=3``) holds per-prior circuit count to 10, total
sweep to 60 circuits in one submission. Typical wall-time on Heron R2
with queue=0 is ~1 minute.
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
    run_on_aer_batched,
    run_on_heron_batched,
    save_partial,
    save_result,
)


_PRIORS = [
    ("p_uniform",           [0.25, 0.25, 0.25, 0.25]),
    ("p_mild_left",         [0.40, 0.20, 0.20, 0.20]),
    ("p_mild_right",        [0.20, 0.20, 0.20, 0.40]),
    ("p_concentrated_tl",   [0.70, 0.10, 0.10, 0.10]),
    ("p_concentrated_br",   [0.10, 0.10, 0.10, 0.70]),
    ("p_conflicting",       [0.85, 0.05, 0.05, 0.05]),
]

_DEGREE = 2
_FOLD_MULT = 3
_NUM_CHUNKS = 3  # keeps circuit count at 10 / prior


def _build_4state_tiger(prior: list[float]) -> Any:
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    from qiskit import transpile as _qk_transpile

    qc = QuantumCircuit(3, 3)
    amps = np.sqrt(np.maximum(prior, 0.0))
    norm = float(np.linalg.norm(amps)) or 1.0
    amps = amps / norm

    prep = QuantumCircuit(2)
    prep.append(StatePreparation(amps.tolist()), [0, 1])
    prep = _qk_transpile(
        prep,
        basis_gates=["cx", "rz", "sx", "x", "h", "rx", "ry", "u"],
        optimization_level=1,
    )
    qc.compose(prep, qubits=[0, 1], inplace=True)

    for _ in range(4):
        qc.rz(0.45, 0)
        qc.rx(0.3, 1)
        qc.rzz(0.4, 0, 1)
        qc.ry(0.2, 2)
        qc.rzz(0.3, 1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _zz_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values()) or 1
    parity = 0.0
    for bs, cnt in counts.items():
        bits = bs.replace(" ", "")[-3:][::-1]
        parity += cnt * (1 if (int(bits[0]) ^ int(bits[1])) == 0 else -1)
    return parity / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from mitiq.lre import construct_circuits, combine_results

    flat_circuits: list[Any] = []
    plan: list[tuple[str, int]] = []
    base_circuits: dict[str, Any] = {}

    for label, prior in _PRIORS:
        base = _build_4state_tiger(prior)
        base_circuits[label] = base
        scaled = construct_circuits(
            base,
            degree=_DEGREE,
            fold_multiplier=_FOLD_MULT,
            num_chunks=_NUM_CHUNKS,
        )
        # Mitiq folds the *unitary* part only, so the scaled circuits
        # come back without measurements. Attach a fresh 3-bit classical
        # register so SamplerV2 can produce counts.
        from qiskit.circuit import ClassicalRegister
        fixed_scaled = []
        for sc in scaled:
            sc2 = sc.copy()
            sc2.remove_final_measurements(inplace=True)
            if not sc2.cregs:
                sc2.add_register(ClassicalRegister(3, "meas"))
            sc2.measure([0, 1, 2], [0, 1, 2])
            fixed_scaled.append(sc2)
        start_idx = len(flat_circuits)
        flat_circuits.extend(fixed_scaled)
        end_idx = len(flat_circuits)
        plan.append((label, start_idx, end_idx))

    payload: dict = {
        "campaign": "E4_lre_batched_4state_tiger",
        "backend": args.backend,
        "dry_run": args.dry_run,
        "shots": args.shots,
        "num_priors": len(_PRIORS),
        "num_circuits": len(flat_circuits),
        "config": {
            "degree": _DEGREE,
            "fold_multiplier": _FOLD_MULT,
            "num_chunks": _NUM_CHUNKS,
        },
        "rows": [],
        "reference": (
            "Russo-Mari-LaRose PRX Quantum 5:040313 (2024) "
            "DOI 10.1103/PRXQuantum.5.040313"
        ),
    }
    save_partial("lre_batched_4state_tiger", payload)

    t0 = time.perf_counter()
    if args.dry_run:
        counts_list = run_on_aer_batched(flat_circuits, args.shots)
        job_id = None
    else:
        counts_list, job_id = run_on_heron_batched(
            flat_circuits,
            backend_name=args.backend,
            shots=args.shots,
            checkpoint_name="lre_batched_4state_tiger",
        )
    t_submit = time.perf_counter() - t0
    payload["job_id"] = job_id
    payload["t_submit_s"] = t_submit

    for (label, s, e), (_, prior) in zip(plan, _PRIORS):
        sub_counts = counts_list[s:e]
        expectations = [_zz_from_counts(c) for c in sub_counts]
        base_zz = expectations[0]  # scale=1.0 slot (first scaled circuit is identity)
        try:
            lre_zz = combine_results(
                expectations,
                base_circuits[label],
                degree=_DEGREE,
                fold_multiplier=_FOLD_MULT,
                num_chunks=_NUM_CHUNKS,
            )
            lre_error = None
        except Exception as exc:  # noqa: BLE001
            lre_zz = float("nan")
            lre_error = f"{type(exc).__name__}: {exc}"

        payload["rows"].append({
            "prior_label": label,
            "prior": prior,
            "base_expectation": base_zz,
            "lre_expectation": lre_zz,
            "num_scaled": e - s,
            "scaled_expectations": expectations,
            "lre_error": lre_error,
        })
        save_partial("lre_batched_4state_tiger", payload)

    save_result("lre_batched_4state_tiger", payload)


if __name__ == "__main__":
    main()
