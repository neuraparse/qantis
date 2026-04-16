#!/usr/bin/env python3
"""Prepare a paper-ready high-qubit dry-run probe for QCE26.

The probe is intentionally conservative:
- it does not submit a QPU job,
- it reports logical width and full circuit width separately,
- it records transpilation cost on a real IBM backend target,
- and it treats any simulator result as a preview, not as hardware validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "quantum-pomdp" / "src"))

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from quantum_common.backends.ibm import IBMQuantumBackend
from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.quantum_circuits.belief_update import (
    BeliefUpdateCircuitConfig,
    QuantumBeliefUpdateCircuit,
)
from quantum_pomdp.scenarios.gps_denied import create_gps_denied_pomdp
from quantum_pomdp.scenarios.grid_navigation import create_grid_navigation_pomdp


OUTPUT_DIR = ROOT / "output" / "paper"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a high-qubit dry-run probe for the QCE26 paper.")
    parser.add_argument(
        "--scenario",
        choices=["gps_denied_6x6", "gps_denied_4x4", "grid_navigation_4x4"],
        default="gps_denied_6x6",
        help="Scenario to probe (default: gps_denied_6x6).",
    )
    parser.add_argument(
        "--backend",
        default="ibm_pittsburgh",
        help="IBM backend target for transpilation only (default: ibm_pittsburgh).",
    )
    parser.add_argument(
        "--sim-shots",
        type=int,
        default=64,
        help="Shots for the local MPS preview (default: 64).",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Skip the local MPS preview run.",
    )
    return parser.parse_args()


def _build_model(scenario: str):
    if scenario == "gps_denied_6x6":
        return create_gps_denied_pomdp(grid_size=6), 4
    if scenario == "gps_denied_4x4":
        return create_gps_denied_pomdp(grid_size=4), 4
    if scenario == "grid_navigation_4x4":
        return create_grid_navigation_pomdp(grid_size=4), 4
    raise ValueError(f"Unsupported scenario: {scenario}")


def _scenario_label(scenario: str) -> str:
    return {
        "gps_denied_6x6": "GPSDenied-6x6",
        "gps_denied_4x4": "GPSDenied-4x4",
        "grid_navigation_4x4": "GridNavigation-4x4",
    }[scenario]


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# QCE26 High-Qubit Dry-Run Probe")
    lines.append("")
    lines.append(f"- Scenario: `{report['scenario_label']}`")
    lines.append(f"- Target backend for transpilation: `{report['backend']}`")
    lines.append(f"- Rare observation index: `{report['observation']}`")
    lines.append(f"- Action index: `{report['action']}`")
    lines.append(
        f"- Evidence probability: `{report['evidence_probability']:.5f}` "
        f"(classical queries per accept ≈ `{report['classical_queries_per_accept']:.2f}`, "
        f"quantum query scale ≈ `{report['quantum_query_scale_only']:.2f}`, "
        f"asymptotic gain ≈ `{report['asymptotic_sample_efficiency_gain']:.2f}x`)"
    )
    lines.append("")
    lines.append("## Width and Abstract Circuit")
    lines.append("")
    lines.append(
        f"- states=`{report['num_states']}`, actions=`{report['num_actions']}`, observations=`{report['num_observations']}`"
    )
    lines.append(
        f"- logical width=`{report['logical_qubits']}` qubits "
        f"(state=`{report['state_qubits']}`, action=`{report['action_qubits']}`, "
        f"observation=`{report['observation_qubits']}`, reward=`4`)"
    )
    lines.append(
        f"- full circuit width including ancilla=`{report['full_circuit_qubits']}` qubits"
    )
    lines.append(
        f"- abstract depth=`{report['abstract_depth']}`, abstract ops=`{report['abstract_ops']}`"
    )
    lines.append("")
    lines.append("## IBM Target Dry-Run")
    lines.append("")
    for item in report["transpile_levels"]:
        lines.append(
            f"- opt level `{item['optimization_level']}`: ISA depth=`{item['isa_depth']}`, "
            f"ISA ops=`{item['isa_ops']}`, 2Q ops=`{item['isa_2q_ops']}`, "
            f"elapsed=`{item['elapsed_s']:.3f}s`"
        )
    lines.append("")
    lines.append("## Local MPS Preview")
    lines.append("")
    if report["mps_preview"]["ran"]:
        lines.append(
            f"- shots=`{report['mps_preview']['shots']}`, elapsed=`{report['mps_preview']['elapsed_s']:.3f}s`, "
            f"transpiled depth=`{report['mps_preview']['transpiled_depth']}`, "
            f"distinct outcomes=`{report['mps_preview']['num_count_keys']}`"
        )
        lines.append(f"- top outcomes: `{report['mps_preview']['top_counts']}`")
    else:
        lines.append(f"- skipped: `{report['mps_preview']['reason']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- This probe is a dry-run/transpile feasibility test only; it does not submit a QPU job."
    )
    lines.append(
        "- The scenario is large enough to demonstrate pathway scale, but the compiled ISA cost should be interpreted against the paper's validated small-scale operating window."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()

    model, action = _build_model(args.scenario)
    observation = model.num_observations - 1
    scenario_label = _scenario_label(args.scenario)

    belief = BeliefState(np.ones(model.num_states, dtype=float) / model.num_states)
    classical_posterior = belief.classical_update(
        action=action,
        observation=observation,
        transition_tensor=model.transition_tensor,
        observation_tensor=model.observation_tensor,
    )
    evidence_probability = model.observation_probability(belief.probabilities, action, observation)
    classical_queries = 1.0 / max(evidence_probability, 1e-12)
    quantum_scale = 1.0 / np.sqrt(max(evidence_probability, 1e-12))

    config = BeliefUpdateCircuitConfig(
        use_amplitude_amplification=False,
        include_reward_register=True,
    )
    circuit = QuantumBeliefUpdateCircuit(model, config).build(
        belief,
        action=action,
        observation=None,
    )

    backend = IBMQuantumBackend(
        channel=os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform"),
        instance=os.environ.get("IBM_QUANTUM_INSTANCE") or None,
        backend_name=args.backend,
        token=os.environ.get("IBM_QUANTUM_TOKEN") or None,
    )
    transpile_levels: list[dict[str, Any]] = []
    for level in (1, 3):
        t0 = time.perf_counter()
        isa = backend.transpile([circuit], optimization_level=level)[0]
        elapsed = time.perf_counter() - t0
        counts = isa.count_ops()
        transpile_levels.append(
            {
                "optimization_level": level,
                "isa_depth": int(isa.depth()),
                "isa_ops": int(sum(counts.values())),
                "isa_2q_ops": int(
                    counts.get("cx", 0) + counts.get("ecr", 0) + counts.get("cz", 0)
                ),
                "top_ops": sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8],
                "elapsed_s": float(elapsed),
            }
        )

    if args.skip_sim:
        mps_preview: dict[str, Any] = {"ran": False, "reason": "skipped by flag"}
    else:
        sim = AerSimulator(method="matrix_product_state")
        t0 = time.perf_counter()
        sim_circuit = transpile(circuit, sim, optimization_level=0)
        sim_result = sim.run(sim_circuit, shots=args.sim_shots).result()
        sim_counts = sim_result.get_counts()
        mps_preview = {
            "ran": True,
            "shots": args.sim_shots,
            "elapsed_s": float(time.perf_counter() - t0),
            "transpiled_depth": int(sim_circuit.depth()),
            "transpiled_ops": int(sum(sim_circuit.count_ops().values())),
            "num_count_keys": len(sim_counts),
            "top_counts": sorted(sim_counts.items(), key=lambda kv: kv[1], reverse=True)[:8],
        }

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "scenario": args.scenario,
        "scenario_label": scenario_label,
        "backend": args.backend,
        "action": action,
        "observation": observation,
        "num_states": model.num_states,
        "num_actions": model.num_actions,
        "num_observations": model.num_observations,
        "state_qubits": model.state_qubits,
        "action_qubits": model.action_qubits,
        "observation_qubits": model.observation_qubits,
        "logical_qubits": model.total_circuit_qubits,
        "full_circuit_qubits": circuit.num_qubits,
        "abstract_depth": int(circuit.depth()),
        "abstract_ops": int(sum(circuit.count_ops().values())),
        "evidence_probability": float(evidence_probability),
        "classical_queries_per_accept": float(classical_queries),
        "quantum_query_scale_only": float(quantum_scale),
        "asymptotic_sample_efficiency_gain": float(classical_queries / max(quantum_scale, 1e-12)),
        "classical_posterior_top8": classical_posterior.probabilities[:8].tolist(),
        "transpile_levels": transpile_levels,
        "mps_preview": mps_preview,
        "notes": {
            "qpu_execution": "not submitted",
            "interpretation": (
                "Dry-run feasibility probe for a large POMDP belief-update circuit; "
                "useful for pathway-to-scale discussion, not for a hardware-validation claim."
            ),
        },
    }

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    json_path = OUTPUT_DIR / f"qce26_high_qubit_probe_{timestamp}.json"
    md_path = OUTPUT_DIR / f"qce26_high_qubit_probe_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)

    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
