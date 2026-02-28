"""VALIDATION-ROADMAP Task 1.4 — ZNE error mitigation on IBM QPU.

Measures the Zero-Noise Extrapolation (ZNE) improvement on:
  (a) Bell state |Φ+> = (|00> + |11>)/√2  — ideal <ZZ> = +1
  (b) Tiger POMDP belief update circuit (if --tiger flag is set)

ZNE amplifies noise at scale factors [1.0, 1.5, 2.0, 3.0] via unitary
folding (Mitiq), then applies Richardson extrapolation to estimate the
zero-noise limit. Success: |ZNE - ideal| < |raw - ideal|.

Usage
-----
# Bell state only:
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_zne_ibm.py \\
    --backend ibm_brisbane --shots 4096

# Bell state + Tiger circuit:
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_zne_ibm.py \\
    --backend ibm_brisbane --shots 4096 --tiger

# Simulator dry run (no credentials):
python scripts/hardware/run_zne_ibm.py --dry-run

Output
------
  output/hardware/zne_ibm_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

from scripts.hardware import (
    build_tiger_pomdp,
    compute_zz_expectation,
    ibm_channel,
    ibm_instance,
    get_ibm_token_optional,
    save_result,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZNE error mitigation on IBM QPU (Task 1.4)")
    p.add_argument("--backend", default="ibm_brisbane", help="IBM backend name")
    p.add_argument("--shots", type=int, default=4096, help="Shot count")
    p.add_argument(
        "--scale-factors", nargs="+", type=float,
        default=[1.0, 3.0, 5.0],
        help="ZNE noise scale factors (converted to nearest odd integers; "
             "default [1,3,5] gives 3-point Richardson extrapolation)",
    )
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel: ibm_quantum_platform or ibm_cloud (overrides IBM_QUANTUM_CHANNEL env)",
    )
    p.add_argument(
        "--instance", default=None,
        help="IBM instance / CRN (overrides IBM_QUANTUM_INSTANCE env)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only — no IBM hardware credentials needed",
    )
    p.add_argument(
        "--tiger", action="store_true",
        help="Also run ZNE on Tiger belief update circuit",
    )
    p.add_argument(
        "--tiger-type", choices=["full", "minimal"], default="full",
        help="Tiger circuit type: 'full' (ISA ~4237, framework) or 'minimal' "
             "(ISA ~12, 2-qubit direct encoding). Default: full.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bell circuit helper
# ---------------------------------------------------------------------------

def _build_bell_circuit():
    """Two-qubit Bell state |Φ+> with measurement on both qubits."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# ZNE helpers — gate folding without Mitiq
# ---------------------------------------------------------------------------

def _fold_circuit(circuit, scale_factor: int):
    """Gate folding: circuit -> C (C† C)^((scale-1)//2).

    For odd integer scale factors, the folded circuit is logically equivalent
    to the original but has scale-fold more gates, amplifying noise by ~scale.
    This is the standard unitary-folding ZNE approach (Temme et al. 2017).

    Args:
        circuit: QuantumCircuit (may include measure_all at end)
        scale_factor: Odd integer >= 1 (e.g., 1, 3, 5)
    Returns:
        QuantumCircuit with measurements at the end
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.exceptions import CircuitError

    # Separate unitary part from measurements.
    # Only include quantum registers (no classical) so that measure_all()
    # creates a fresh 'meas' register rather than conflicting with an
    # existing empty one from the original circuit's cregs.
    qc_u = QuantumCircuit(*circuit.qregs)
    for inst in circuit.data:
        if inst.operation.name not in ("measure", "barrier"):
            qc_u.append(inst)

    n_extra = (scale_factor - 1) // 2  # number of (C† C) pairs to append
    folded = qc_u.copy()
    # Add a barrier after the original unitary to prevent transpiler
    # from cancelling gates across the fold boundary.
    if n_extra > 0:
        folded.barrier()
    for i in range(n_extra):
        try:
            inv = qc_u.inverse()
        except CircuitError:
            inv = qc_u.inverse(annotated=True)
        folded = folded.compose(inv)
        folded.barrier()
        folded = folded.compose(qc_u)
        if i < n_extra - 1:
            folded.barrier()

    folded.measure_all()
    return folded


def _richardson_extrapolate(scales: list[float], values: list[float]) -> float:
    """Richardson extrapolation to zero-noise limit.

    - 1 point: returns the single value.
    - 2 points: exact two-point Richardson: ZNE = (s2*v1 - s1*v2) / (s2 - s1).
    - 3+ points: linear least-squares fit through all (scale, value) pairs,
      extrapolated to scale=0.  More robust than two-point for noisy data.
    """
    import numpy as np

    if len(scales) < 2:
        return values[0]
    if len(scales) == 2:
        s1, s2 = scales[0], scales[1]
        v1, v2 = values[0], values[1]
        return (s2 * v1 - s1 * v2) / (s2 - s1)
    # N >= 3: linear fit, evaluate at 0
    coeffs = np.polyfit(scales, values, deg=1)
    return float(np.polyval(coeffs, 0.0))


def _run_zne_circuit(
    circuit,
    backend,
    shots: int,
    scale_factors: list[float],
    factory_type: str = "Richardson",
) -> tuple[dict[str, int], dict[str, int], dict]:
    """Run ZNE via gate folding and Richardson extrapolation.

    Builds gate-folded copies of *circuit* for each odd-integer scale factor,
    submits them in a single backend call, computes <ZZ> for each, and
    extrapolates to zero noise. Returns (raw_counts, mitigated_counts, zne_detail)
    where mitigated_counts is constructed from the extrapolated <ZZ> value.
    """
    import numpy as np
    from quantum_common.backends.base import ExecutionRequest

    # Use odd integer scales closest to requested scale_factors
    int_scales = [max(1, 2 * int(round((s - 1) / 2)) + 1) for s in scale_factors]
    unique_scales = sorted(set(int_scales))

    # Build one folded circuit per unique scale
    folded_circuits = [_fold_circuit(circuit, s) for s in unique_scales]

    # Submit all in a single backend call
    result = backend.execute(ExecutionRequest(circuits=folded_circuits, shots=shots))

    # Compute <ZZ> for each scale
    zz_by_scale = {}
    for i, s in enumerate(unique_scales):
        zz_by_scale[s] = compute_zz_expectation(result.counts[i])

    raw_counts = result.counts[0]  # scale=1 result
    raw_zz = zz_by_scale[unique_scales[0]]

    # Richardson extrapolation using all available scale factors (more robust).
    # Previously used only first two; using all N points reduces variance
    # and is the correct approach for N>=3 (linear least-squares extrapolation).
    scale_list = unique_scales
    zz_list = [zz_by_scale[s] for s in scale_list]
    mitigated_zz = _richardson_extrapolate([float(s) for s in scale_list], zz_list)

    # Construct mitigated_counts to match the expected <ZZ>
    p_corr = min(1.0, max(0.0, (1.0 + mitigated_zz) / 2))
    p_anti = 1.0 - p_corr
    n = shots
    mitigated_counts = {
        "00": int(round(n * p_corr / 2)),
        "11": int(round(n * p_corr / 2)),
        "01": int(round(n * p_anti / 2)),
        "10": int(round(n * p_anti / 2)),
    }

    zne_detail = {
        "scales_used": unique_scales,
        "zz_by_scale": {str(s): v for s, v in zz_by_scale.items()},
        "raw_zz": raw_zz,
        "mitigated_zz": float(mitigated_zz),
    }

    return raw_counts, mitigated_counts, zne_detail


# ---------------------------------------------------------------------------
# Minimal Tiger ZNE helper
# ---------------------------------------------------------------------------

def _run_zne_tiger_minimal(
    circuit,
    backend,
    shots: int,
    scale_factors: list,
    true_posterior,
    target_obs: int,
) -> dict:
    """ZNE for minimal 2-qubit Tiger circuit via gate-folding + Hellinger extrapolation.

    For the 2-qubit Tiger circuit (qubit0=state, qubit1=obs), the relevant
    observable is Hellinger distance between the post-selected state distribution
    and the classical Bayesian posterior.  We apply Richardson extrapolation to
    the Hellinger values across scale factors (not to <ZZ>) to estimate the
    zero-noise Hellinger limit.

    Measurement bitstring format after measure_all(): "obs_state" (MSB-first, i.e.
    qubit1=obs is leftmost, qubit0=state is rightmost).  Post-selection on
    obs bit == target_obs is done with _post_select_counts(counts, 1, 1, target_obs).

    Returns:
        dict with keys raw_hellinger, zne_hellinger, hellinger_by_scale, scales_used.
    """
    from quantum_common.backends.base import ExecutionRequest
    from scripts.hardware.run_tiger_ibm import _post_select_counts
    from quantum_pomdp.models.belief_state import BeliefState

    int_scales = [max(1, 2 * int(round((s - 1) / 2)) + 1) for s in scale_factors]
    unique_scales = sorted(set(int_scales))

    folded_circuits = [_fold_circuit(circuit, s) for s in unique_scales]
    result = backend.execute(ExecutionRequest(circuits=folded_circuits, shots=shots))

    hellinger_by_scale = {}
    for i, s in enumerate(unique_scales):
        ps_counts = _post_select_counts(result.counts[i], 1, 1, target_obs)
        if ps_counts:
            belief = BeliefState.from_quantum_measurement(ps_counts, 2, 1)
            hellinger_by_scale[s] = belief.hellinger_distance(true_posterior)
        else:
            hellinger_by_scale[s] = 1.0  # total post-selection failure

    raw_hellinger = hellinger_by_scale[unique_scales[0]]
    zne_hellinger = _richardson_extrapolate(
        [float(s) for s in unique_scales],
        [hellinger_by_scale[s] for s in unique_scales],
    )

    return {
        "raw_hellinger": float(raw_hellinger),
        "zne_hellinger": float(zne_hellinger),
        "hellinger_by_scale": {str(s): float(v) for s, v in hellinger_by_scale.items()},
        "scales_used": unique_scales,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"\n=== ZNE Error Mitigation — IBM QPU (Task 1.4) ===")
    print(f"  Backend: {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots: {args.shots}")
    print(f"  Scale factors: {args.scale_factors}")
    tiger_type_label = getattr(args, "tiger_type", "full") if args.tiger else "N/A"
    print(f"  Tiger circuit: {args.tiger}  (type: {tiger_type_label})")

    result_data: dict = {
        "task_ids": ["1.4"],
        "backend": "aer_simulator" if args.dry_run else args.backend,
        "shots": args.shots,
        "scale_factors": args.scale_factors,
    }

    # ------------------------------------------------------------------
    # Setup backends
    # ------------------------------------------------------------------
    from quantum_common.backends.simulator import AerSimulatorBackend
    from quantum_common.backends.base import ExecutionRequest

    aer = AerSimulatorBackend()

    if not args.dry_run:
        token = get_ibm_token_optional()
        channel = args.channel or ibm_channel()
        instance = args.instance or ibm_instance()
        from quantum_common.backends.ibm import IBMQuantumBackend
        hw_backend = IBMQuantumBackend(
            backend_name=args.backend, token=token,
            channel=channel, instance=instance,
        )
    else:
        hw_backend = aer  # Use aer as "hardware" in dry-run

    # ------------------------------------------------------------------
    # (a) Bell state |Phi+> -- ideal <ZZ> = +1
    # ------------------------------------------------------------------
    print(f"\n[bell] Building Bell circuit |Phi+> ...")
    bell_circuit = _build_bell_circuit()
    print(f"  Circuit qubits: {bell_circuit.num_qubits}")

    # Simulator ideal
    sim_result = aer.execute(ExecutionRequest(circuits=[bell_circuit], shots=args.shots))
    sim_counts = sim_result.counts[0]
    sim_zz = compute_zz_expectation(sim_counts)
    print(f"  Simulator <ZZ>: {sim_zz:.4f}  (ideal=1.0)")

    # Hardware raw + ZNE via gate folding
    print(f"  Running on {'AerSimulator' if args.dry_run else args.backend} ...")
    hw_raw_counts, hw_mitigated_counts, zne_detail = _run_zne_circuit(
        bell_circuit, hw_backend, args.shots, args.scale_factors
    )
    hw_raw_zz = zne_detail["raw_zz"]
    hw_mitigated_zz = zne_detail["mitigated_zz"]

    ideal_zz = 1.0
    zne_improved_bell = abs(hw_mitigated_zz - ideal_zz) < abs(hw_raw_zz - ideal_zz)

    print(f"  HW raw <ZZ>      : {hw_raw_zz:.4f}")
    print(f"  HW mitigated <ZZ>: {hw_mitigated_zz:.4f}")
    print(f"  ZNE improved     : {zne_improved_bell}")
    for s, v in zne_detail["zz_by_scale"].items():
        print(f"    scale={s}: <ZZ>={v:.4f}")

    result_data["bell"] = {
        "ideal_zz": ideal_zz,
        "sim_zz": sim_zz,
        "hw_raw_zz": hw_raw_zz,
        "hw_mitigated_zz": hw_mitigated_zz,
        "zne_improved": zne_improved_bell,
        "zne_detail": zne_detail,
    }

    # ------------------------------------------------------------------
    # (b) Tiger belief circuit (optional)
    # ------------------------------------------------------------------
    if args.tiger:
        from quantum_pomdp.models.belief_state import BeliefState

        pomdp = build_tiger_pomdp()
        belief = BeliefState.uniform(num_states=pomdp.num_states)
        true_posterior = belief.classical_update(
            action=0, observation=0,
            transition_tensor=pomdp.transition_tensor,
            observation_tensor=pomdp.observation_tensor,
        )

        tiger_type = getattr(args, "tiger_type", "full")

        if tiger_type == "minimal":
            # ------------------------------------------------------------------
            # Tiger minimal: 2-qubit direct encoding, ISA ~12 → well within
            # the ZNE effective regime (ISA < 100).  Uses Hellinger extrapolation.
            # ------------------------------------------------------------------
            print(f"\n[tiger-minimal] Building Tiger minimal 2-qubit ZNE circuit ...")
            from scripts.hardware.run_tiger_ibm import _build_minimal_tiger_circuit

            p_obs_given_state = [0.85, 0.15]  # P(obs=0|state=0), P(obs=0|state=1)
            tiger_circuit = _build_minimal_tiger_circuit(
                p_obs_given_state=p_obs_given_state,
                prior=belief.probabilities.tolist(),
                action=0,
                target_observation=0,
            )
            print(f"  Circuit qubits: {tiger_circuit.num_qubits}")
            print(f"  Circuit depth : {tiger_circuit.depth()}")

            # Simulator baseline
            sim_tiger = aer.execute(ExecutionRequest(circuits=[tiger_circuit], shots=args.shots))
            from scripts.hardware.run_tiger_ibm import _post_select_counts
            sim_ps = _post_select_counts(sim_tiger.counts[0], 1, 1, 0)
            if sim_ps:
                sim_tiger_belief = BeliefState.from_quantum_measurement(sim_ps, 2, 1)
                sim_hellinger = sim_tiger_belief.hellinger_distance(true_posterior)
            else:
                sim_hellinger = 1.0
            print(f"  Simulator Hellinger  : {sim_hellinger:.4f}")

            # Hardware ZNE via gate-folding + Hellinger extrapolation
            print(f"  Running ZNE on {'AerSimulator' if args.dry_run else args.backend} ...")
            zne_detail = _run_zne_tiger_minimal(
                tiger_circuit, hw_backend, args.shots, args.scale_factors,
                true_posterior, target_obs=0,
            )
            hw_hellinger_raw = zne_detail["raw_hellinger"]
            hw_hellinger_mit = zne_detail["zne_hellinger"]
            zne_improved_tiger = hw_hellinger_mit < hw_hellinger_raw

            print(f"  HW raw Hellinger     : {hw_hellinger_raw:.4f}")
            print(f"  HW ZNE Hellinger     : {hw_hellinger_mit:.4f}")
            print(f"  ZNE improved         : {zne_improved_tiger}")
            for s, v in zne_detail["hellinger_by_scale"].items():
                print(f"    scale={s}: Hellinger={v:.4f}")

            result_data["tiger_minimal"] = {
                "circuit_qubits": tiger_circuit.num_qubits,
                "circuit_depth": tiger_circuit.depth(),
                "tiger_type": "minimal",
                "sim_hellinger": float(sim_hellinger),
                "hw_raw_hellinger": float(hw_hellinger_raw),
                "hw_zne_hellinger": float(hw_hellinger_mit),
                "zne_improved": zne_improved_tiger,
                "zne_detail": zne_detail,
            }

        else:
            # ------------------------------------------------------------------
            # Tiger full: framework circuit, ISA ~4237
            # ------------------------------------------------------------------
            print(f"\n[tiger] Building Tiger belief update circuit (full framework) ...")
            from quantum_pomdp.quantum_circuits.belief_update import (
                BeliefUpdateCircuitConfig,
                QuantumBeliefUpdateCircuit,
            )

            config = BeliefUpdateCircuitConfig(
                use_amplitude_amplification=True,
                use_hardware_compatible_encoding=True,
                reward_precision_bits=2,
            )
            tiger_circuit = QuantumBeliefUpdateCircuit(pomdp=pomdp, config=config).build(
                belief=belief, action=0, observation=0
            )
            print(f"  Circuit qubits: {tiger_circuit.num_qubits}")
            print(f"  Circuit depth : {tiger_circuit.depth()}")

            # Simulator
            sim_tiger = aer.execute(ExecutionRequest(circuits=[tiger_circuit], shots=args.shots))
            sim_tiger_counts = sim_tiger.counts[0]
            sim_tiger_belief = BeliefState.from_quantum_measurement(
                sim_tiger_counts, pomdp.num_states, pomdp.state_qubits
            )
            sim_hellinger = sim_tiger_belief.hellinger_distance(true_posterior)

            # Hardware ZNE via gate folding
            print(f"  Running ZNE on {'AerSimulator' if args.dry_run else args.backend} ...")
            hw_tiger_raw, hw_tiger_mitigated, tiger_zne_detail = _run_zne_circuit(
                tiger_circuit, hw_backend, args.shots, args.scale_factors
            )
            from scripts.hardware.run_tiger_ibm import _post_select_counts
            hw_tiger_raw_ps = _post_select_counts(
                hw_tiger_raw, pomdp.state_qubits, pomdp.observation_qubits, 0
            )
            hw_tiger_mitigated_ps = _post_select_counts(
                hw_tiger_mitigated, pomdp.state_qubits, pomdp.observation_qubits, 0
            )
            hw_tiger_belief_raw = BeliefState.from_quantum_measurement(
                hw_tiger_raw_ps, pomdp.num_states, pomdp.state_qubits
            )
            hw_tiger_belief_mit = BeliefState.from_quantum_measurement(
                hw_tiger_mitigated_ps, pomdp.num_states, pomdp.state_qubits
            )
            hw_hellinger_raw = hw_tiger_belief_raw.hellinger_distance(true_posterior)
            hw_hellinger_mit = hw_tiger_belief_mit.hellinger_distance(true_posterior)
            zne_improved_tiger = hw_hellinger_mit < hw_hellinger_raw

            print(f"  Simulator Hellinger  : {sim_hellinger:.4f}")
            print(f"  HW raw Hellinger     : {hw_hellinger_raw:.4f}")
            print(f"  HW mitigated Hellinger: {hw_hellinger_mit:.4f}")
            print(f"  ZNE improved         : {zne_improved_tiger}")

            result_data["tiger"] = {
                "circuit_qubits": tiger_circuit.num_qubits,
                "circuit_depth": tiger_circuit.depth(),
                "tiger_type": "full",
                "sim_hellinger": sim_hellinger,
                "hw_raw_hellinger": hw_hellinger_raw,
                "hw_mitigated_hellinger": hw_hellinger_mit,
                "zne_improved": zne_improved_tiger,
            }
    else:
        zne_improved_tiger = None

    # ------------------------------------------------------------------
    # Pass/fail and save
    # ------------------------------------------------------------------
    # In dry-run, ZNE returns raw counts unchanged, so improvement is 0
    # Real hardware: ZNE should improve over raw
    if args.dry_run:
        passed = True  # Dry-run always passes (infrastructure check only)
        result_data["notes"] = "Dry-run: ZNE infrastructure verified, no real hardware noise"
    else:
        passed = zne_improved_bell
        if zne_improved_tiger is not None:
            passed = passed and zne_improved_tiger
        result_data["notes"] = (
            f"Bell ZNE improved: {zne_improved_bell}"
            + (f", Tiger ZNE improved: {zne_improved_tiger}" if args.tiger else "")
        )

    result_data["pass"] = passed
    save_result("zne_ibm", result_data)
    print(f"\n  PASS: {passed}")
    print(f"  Notes: {result_data['notes']}")


if __name__ == "__main__":
    main()
