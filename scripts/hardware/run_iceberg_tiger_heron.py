"""QANTIS-2 E5: Iceberg [[k+2, k, 2]] encoded Tiger on IBM Heron.

Runs the He-Amaro-Shaydulin-Pistoia Iceberg construction (Comm Phys
8:217, 2025, DOI 10.1038/s42005-025-02136-8, arXiv:2409.12104) on an
IBM Heron R3 QPU. He et al. reported the beyond-break-even result on
Quantinuum trapped ions; the expected outcome on superconducting
hardware is below break even due to post-selection losses, which
itself is a publishable boundary-marker datapoint for the paper's
Section 6.3 "Supporting Diagnostics" discussion.

Reuses :func:`quantum_mht.solvers.iceberg_qaoa_solver.build_iceberg_qaoa_circuit`
and :func:`postselect_iceberg_counts` so that the compiler and
post-selection logic stays in one place.

Usage
-----
Dry-run (Aer):

    python scripts/hardware/run_iceberg_tiger_heron.py --dry-run

Hardware on Pittsburgh (k_logical defaults to 2 logical data qubits):

    IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_iceberg_tiger_heron.py \
        --backend ibm_pittsburgh --shots 16384 --k-logical 2
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
sys.path.insert(0, str(_repo / "packages" / "quantum-mht" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

import numpy as np

from scripts.hardware import (  # type: ignore
    databin_get_counts,
    get_ibm_token_optional,
    ibm_channel,
    ibm_instance,
    save_result,
)
from quantum_mht.solvers.iceberg_qaoa_solver import (
    build_iceberg_qaoa_circuit,
    postselect_iceberg_counts,
)


def _build_unencoded_baseline(k_logical: int, gamma: float, beta: float) -> Any:
    """Unencoded single-layer QAOA on ``k_logical`` data qubits.

    Matches the Iceberg circuit's gate structure minus the ancilla
    ladder, so the acceptance / Hellinger comparison stays apples-to-
    apples. The ZZ coupling graph is a chain (i, i+1) which reproduces
    the He et al. 2025 Ring-Ising benchmark shape.
    """
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(k_logical, k_logical)
    for q in range(k_logical):
        qc.h(q)
    for i in range(k_logical - 1):
        qc.rzz(2.0 * gamma, i, i + 1)
    for q in range(k_logical):
        qc.rx(2.0 * beta, q)
    qc.measure(list(range(k_logical)), list(range(k_logical)))
    return qc


def _execute(
    circuit: Any, shots: int, dry_run: bool, backend_name: str,
) -> dict[str, int]:
    if dry_run:
        from qiskit_aer import AerSimulator

        return AerSimulator().run(circuit, shots=shots).result().get_counts()

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(
        token=get_ibm_token_optional(),
        channel=ibm_channel(),
        instance=ibm_instance(),
    )
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa = pm.run(circuit)
    isa._layout = None
    sampler = SamplerV2(mode=backend)
    res = sampler.run([isa], shots=shots).result()
    return databin_get_counts(res[0])


def _mode_energy(
    counts: dict[str, int], zz_pairs: list[tuple[int, int, float]], num_qubits: int,
) -> float:
    """Energy of the most-common bitstring under the QAOA cost Hamiltonian."""
    if not counts:
        return float("nan")
    best = max(counts, key=counts.get)
    bits = best.replace(" ", "")[:num_qubits][::-1]
    x = np.array([int(b) for b in bits])
    s = 1 - 2 * x  # Ising spins
    energy = sum(coeff * s[i] * s[j] for (i, j, coeff) in zz_pairs)
    return float(energy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="ibm_pittsburgh")
    parser.add_argument("--shots", type=int, default=16384)
    parser.add_argument("--k-logical", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    k = args.k_logical
    zz_pairs = [(i, i + 1, 1.0) for i in range(k - 1)]

    encoded_circuit = build_iceberg_qaoa_circuit(
        k_logical=k,
        zz_pairs=zz_pairs,
        gamma=args.gamma,
        beta=args.beta,
        framework="qiskit",
    )
    unencoded_circuit = _build_unencoded_baseline(k, args.gamma, args.beta)

    t0 = time.perf_counter()
    encoded_counts = _execute(encoded_circuit, args.shots, args.dry_run, args.backend)
    t_encoded = time.perf_counter() - t0

    accepted, acceptance_ratio = postselect_iceberg_counts(encoded_counts, k_logical=k)
    encoded_energy = _mode_energy(accepted, zz_pairs, k)

    t0 = time.perf_counter()
    bare_counts = _execute(unencoded_circuit, args.shots, args.dry_run, args.backend)
    t_bare = time.perf_counter() - t0
    bare_energy = _mode_energy(bare_counts, zz_pairs, k)

    save_result(
        "iceberg_tiger_heron",
        {
            "campaign": "E5_iceberg_tiger_heron",
            "backend": args.backend,
            "dry_run": args.dry_run,
            "shots": args.shots,
            "k_logical": k,
            "encoded": {
                "physical_qubits": k + 2,
                "acceptance_ratio": acceptance_ratio,
                "mode_energy": encoded_energy,
                "num_accepted_unique_bitstrings": len(accepted),
                "wall_time_s": t_encoded,
            },
            "unencoded": {
                "physical_qubits": k,
                "mode_energy": bare_energy,
                "wall_time_s": t_bare,
            },
            "reference": (
                "He, Amaro, Shaydulin, Pistoia, Comm Phys 8:217 (2025) "
                "(10.1038/s42005-025-02136-8); Quantinuum arXiv:2602.22211 (2026)"
            ),
        },
    )


if __name__ == "__main__":
    main()
