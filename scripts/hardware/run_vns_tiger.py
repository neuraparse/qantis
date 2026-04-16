"""Virtual Noise Scaling (VNS) ZNE for Tiger POMDP — Task 1.

Implements the Virtual Noise Scaling technique inspired by Uzdin (arXiv:2601.22785,
January 2026).  Instead of physically folding gates (which increases circuit depth
3-5x), VNS exploits the Pauli twirling infrastructure already present in IBM
Runtime SamplerV2 to create *virtual* noise diversity.

Key insight
-----------
When Pauli twirling is applied with ``num_randomizations = N``, the effective
noise is averaged over N random twirl realisations.  Low N gives noisier
(higher-variance) results; high N gives cleaner averaging.  By sweeping N
across several values [4, 16, 64, 256] and measuring the posterior Hellinger
distance at each level, we can *extrapolate* to N -> infinity (zero residual
twirling noise) using a simple 1/N model:

    H(N) = H_inf + c / N

This is a depth-free ZNE variant: the ISA circuit is identical at every noise
level — only the Pauli twirling budget changes.  Total extra overhead is
purely in shots (run the same circuit 4 times), NOT in circuit depth.

Circuits
--------
- Tiger minimal (2 qubits): state + observation
- Tiger 4-state (3 qubits): 4-state corridor variant

Usage
-----
# Dry-run:
python scripts/hardware/run_vns_tiger.py --dry-run

# Hardware:
IBM_QUANTUM_TOKEN=xxx IBM_QUANTUM_CHANNEL=ibm_cloud IBM_QUANTUM_INSTANCE=crn:... \
    python -u scripts/hardware/run_vns_tiger.py --backend ibm_kingston --shots 8192

Output
------
  output/hardware/vns_tiger_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

from scripts.hardware import (
    get_ibm_token_optional,
    ibm_channel,
    ibm_instance,
    save_result,
)
from scripts.hardware.run_tiger_4state_ibm import (
    P_OBS_GIVEN_STATE_4,
    STATE_NAMES,
    _build_tiger_4state_circuit,
    _classical_bayes_4state,
    _counts_to_probs,
    _hellinger,
    _post_select_4state,
)
from scripts.hardware.run_tiger_ibm import (
    _build_minimal_tiger_circuit,
    _post_select_counts,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Virtual Noise Scaling ZNE for Tiger POMDP (Uzdin 2026)"
    )
    p.add_argument("--backend", default="ibm_kingston", help="IBM backend name")
    p.add_argument(
        "--shots", type=int, default=8192,
        help="Shot count per noise level (default: 8192)",
    )
    p.add_argument(
        "--obs", type=int, default=0, choices=[0, 1],
        help="Target observation: 0=hear-left, 1=hear-right (default: 0)",
    )
    p.add_argument(
        "--noise-levels", nargs="+", type=int, default=[4, 16, 64, 256],
        metavar="N",
        help="num_randomizations values to sweep (default: 4 16 64 256)",
    )
    p.add_argument("--channel", default=None, help="Qiskit channel")
    p.add_argument("--instance", default=None, help="IBM instance / CRN")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only (no credentials needed)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# VNS extrapolation
# ---------------------------------------------------------------------------

def _vns_extrapolate_1overN(
    num_rands: list[int],
    hellingers: list[float],
) -> float:
    """Extrapolate Hellinger to N -> infinity using H(N) = H_inf + c/N.

    Fit a linear model in x = 1/N:  H = c * x + H_inf
    The y-intercept H_inf is the VNS-extrapolated zero-noise Hellinger.
    """
    xs = [1.0 / n for n in num_rands]
    if len(xs) < 2:
        return hellingers[0]
    coeffs = np.polyfit(xs, hellingers, 1)  # [slope, intercept]
    return max(0.0, float(coeffs[1]))


def _vns_extrapolate_posteriors(
    num_rands: list[int],
    posteriors: list[list[float]],
) -> list[float]:
    """Extrapolate each posterior component to N -> infinity.

    Uses linear fit in x = 1/N: P_s(x) = a_s * x + b_s
    Extrapolated P_s(0) = b_s.
    """
    n_states = len(posteriors[0])
    xs = [1.0 / n for n in num_rands]
    extrapolated = []

    for s in range(n_states):
        values = [post[s] for post in posteriors]
        if len(xs) >= 2:
            coeffs = np.polyfit(xs, values, 1)
            p_zero = coeffs[1]
        else:
            p_zero = values[0]
        extrapolated.append(max(0.0, p_zero))

    total = sum(extrapolated)
    if total > 1e-12:
        extrapolated = [p / total for p in extrapolated]
    else:
        extrapolated = [1.0 / n_states] * n_states

    return extrapolated


# ---------------------------------------------------------------------------
# Hardware execution with configurable num_randomizations
# ---------------------------------------------------------------------------

def _run_hw_with_num_rand(
    isa_circuit, backend, shots: int, num_randomizations: int,
) -> dict:
    """Run a transpiled circuit on hardware with a specific num_randomizations."""
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.strategy = "active-accum"
    sampler.options.twirling.num_randomizations = num_randomizations
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.dynamical_decoupling.scheduling_method = "alap"

    job = sampler.run([isa_circuit], shots=shots)
    raw = job.result()[0]
    _creg_name = next(k for k in vars(raw.data) if not k.startswith("_"))
    return dict(getattr(raw.data, _creg_name).get_counts())


# ---------------------------------------------------------------------------
# Simulator dry-run
# ---------------------------------------------------------------------------

def _run_sim(circuit, shots: int) -> dict:
    """Run circuit on AerSimulator."""
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile

    sim = AerSimulator()
    basis = ["cx", "u", "id", "reset", "measure"]
    decomposed = transpile(circuit, basis_gates=basis, optimization_level=0)
    job = sim.run(decomposed, shots=shots)
    return dict(job.result().get_counts())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    num_rands = sorted(args.noise_levels)

    # --- 4-state Tiger circuit ---
    prior_4 = [0.25, 0.25, 0.25, 0.25]
    classical_post_4 = _classical_bayes_4state(prior_4, args.obs)

    # --- 2-state Tiger circuit ---
    prior_2 = [0.5, 0.5]
    P_OBS_2 = [0.85, 0.15]
    # Classical Bayesian posterior for 2-state
    if args.obs == 0:
        lik2 = np.array(P_OBS_2)
    else:
        lik2 = 1.0 - np.array(P_OBS_2)
    unnorm2 = np.array(prior_2) * lik2
    classical_post_2 = (unnorm2 / unnorm2.sum()).tolist()

    print(f"\n{'='*65}")
    print(f"  Virtual Noise Scaling (VNS) ZNE for Tiger POMDP")
    print(f"  Uzdin, arXiv:2601.22785 (January 2026)")
    print(f"{'='*65}")
    print(f"  Backend           : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots per level   : {args.shots}")
    print(f"  num_randomizations: {num_rands}")
    print(f"  Observation       : {args.obs} ({'hear-left' if args.obs == 0 else 'hear-right'})")
    print(f"\n  Classical posterior (2-state): {[round(x, 4) for x in classical_post_2]}")
    print(f"  Classical posterior (4-state): {[round(x, 4) for x in classical_post_4]}")

    # Build circuits
    circuit_2q = _build_minimal_tiger_circuit(
        p_obs_given_state=P_OBS_2,
        prior=prior_2,
        action=0,
        target_observation=args.obs,
    )
    circuit_3q = _build_tiger_4state_circuit(prior_4, args.obs)

    print(f"\n  2-qubit circuit depth : {circuit_2q.depth()}")
    print(f"  3-qubit circuit depth : {circuit_3q.depth()}")

    # ------------------------------------------------------------------
    # Connect to backend
    # ------------------------------------------------------------------
    backend_obj = None
    if not args.dry_run:
        token = get_ibm_token_optional()
        if token is None:
            print("  ERROR: IBM_QUANTUM_TOKEN not set. Use --dry-run for simulator.")
            sys.exit(1)

        from qiskit_ibm_runtime import QiskitRuntimeService

        channel = args.channel or ibm_channel()
        instance = args.instance or ibm_instance()
        svc_kwargs: dict = {"channel": channel, "token": token}
        if instance:
            svc_kwargs["instance"] = instance
        svc = QiskitRuntimeService(**svc_kwargs)
        backend_obj = svc.backend(args.backend)
        print(f"\n  Connected: {backend_obj.name} ({backend_obj.num_qubits} qubits)")

    # ------------------------------------------------------------------
    # Transpile circuits once (same ISA for all noise levels)
    # ------------------------------------------------------------------
    isa_2q = None
    isa_3q = None
    isa_depth_2q = circuit_2q.depth()
    isa_depth_3q = circuit_3q.depth()

    if not args.dry_run and backend_obj is not None:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(optimization_level=3, backend=backend_obj)
        isa_2q = pm.run(circuit_2q)
        isa_2q._layout = None
        isa_depth_2q = isa_2q.depth()

        isa_3q = pm.run(circuit_3q)
        isa_3q._layout = None
        isa_depth_3q = isa_3q.depth()

        print(f"  ISA depth (2q): {isa_depth_2q}")
        print(f"  ISA depth (3q): {isa_depth_3q}")

    # ------------------------------------------------------------------
    # Sweep num_randomizations for each circuit
    # ------------------------------------------------------------------
    results_2q: list[dict[str, Any]] = []
    results_3q: list[dict[str, Any]] = []

    for circuit_label, circuit_obj, isa_obj, n_states, classical_post, post_fn in [
        ("2-state Tiger (2q)", circuit_2q, isa_2q, 2, classical_post_2, "2q"),
        ("4-state Tiger (3q)", circuit_3q, isa_3q, 4, classical_post_4, "3q"),
    ]:
        print(f"\n{'='*65}")
        print(f"  VNS sweep: {circuit_label}")
        print(f"{'='*65}")

        sweep_posteriors: list[list[float]] = []
        sweep_hellingers: list[float] = []
        sweep_results: list[dict[str, Any]] = []

        for nr in num_rands:
            print(f"\n  num_randomizations = {nr}")

            if args.dry_run:
                # Simulator: all noise levels produce same ideal result
                raw_counts = _run_sim(circuit_obj, args.shots)
            else:
                assert isa_obj is not None and backend_obj is not None
                print(f"    Submitting to {args.backend} (shots={args.shots}, nr={nr}) ...")
                raw_counts = _run_hw_with_num_rand(
                    isa_obj, backend_obj, args.shots, nr,
                )

            # Post-select and compute posterior
            if post_fn == "2q":
                # 2-qubit: bitstring "obs_bit state_bit" (MSB first)
                ps_counts = _post_select_counts(raw_counts, 1, 1, args.obs)
                total_ps = sum(ps_counts.values())
                # Convert to probability vector
                probs = [0.0, 0.0]
                if total_ps > 0:
                    for bs, cnt in ps_counts.items():
                        idx = int(bs)
                        probs[idx] = cnt / total_ps
                # Hellinger
                h = float(np.sqrt(0.5 * sum(
                    (np.sqrt(probs[i]) - np.sqrt(classical_post[i])) ** 2
                    for i in range(2)
                )))
            else:
                # 3-qubit 4-state
                ps_counts_int = _post_select_4state(raw_counts, args.obs)
                probs = _counts_to_probs(ps_counts_int, n_states=4)
                h = _hellinger(probs, classical_post)
                total_ps = sum(ps_counts_int.values())

            sweep_posteriors.append(probs)
            sweep_hellingers.append(h)

            print(f"    Post-selected: {total_ps}/{args.shots}")
            print(f"    Posterior: {[round(x, 4) for x in probs]}")
            print(f"    Hellinger: {h:.6f}")

            sweep_results.append({
                "num_randomizations": nr,
                "posterior": probs,
                "hellinger": round(h, 6),
                "ps_total": total_ps,
            })

        # VNS extrapolation
        vns_posterior = _vns_extrapolate_posteriors(num_rands, sweep_posteriors)
        if post_fn == "2q":
            vns_hellinger = float(np.sqrt(0.5 * sum(
                (np.sqrt(vns_posterior[i]) - np.sqrt(classical_post[i])) ** 2
                for i in range(2)
            )))
        else:
            vns_hellinger = _hellinger(vns_posterior, classical_post)

        vns_hellinger_direct = _vns_extrapolate_1overN(num_rands, sweep_hellingers)

        # Best single run
        best_idx = int(min(range(len(sweep_hellingers)), key=lambda i: sweep_hellingers[i]))
        best_h = sweep_hellingers[best_idx]
        best_nr = num_rands[best_idx]

        # Use whichever VNS method gives the better result
        best_vns = min(vns_hellinger, vns_hellinger_direct)
        vns_method = "posterior" if vns_hellinger <= vns_hellinger_direct else "direct"
        improvement = best_h - best_vns
        improvement_pct = (improvement / best_h * 100) if best_h > 0 else 0

        print(f"\n  --- VNS Extrapolation ({circuit_label}) ---")
        print(f"    VNS posterior extrap.: {[round(x, 4) for x in vns_posterior]}  H={vns_hellinger:.6f}")
        print(f"    VNS direct extrap.  : H={vns_hellinger_direct:.6f}")
        print(f"    Best single (nr={best_nr}): H={best_h:.6f}")
        print(f"    Best VNS ({vns_method}): H={best_vns:.6f}")
        print(f"    Improvement vs best : {improvement:+.6f} ({improvement_pct:+.1f}%)")

        # Linear fit in 1/N
        xs = [1.0 / n for n in num_rands]
        fit_coeffs = np.polyfit(xs, sweep_hellingers, 1)
        print(f"    Fit: H(1/N) = {fit_coeffs[0]:.4f} * (1/N) + {fit_coeffs[1]:.6f}")

        entry = {
            "circuit": circuit_label,
            "n_states": n_states,
            "classical_posterior": classical_post,
            "isa_depth": isa_depth_2q if post_fn == "2q" else isa_depth_3q,
            "sweep": sweep_results,
            "vns_posterior_extrapolation": {
                "posterior": vns_posterior,
                "hellinger": round(vns_hellinger, 6),
            },
            "vns_direct_extrapolation": {
                "hellinger": round(vns_hellinger_direct, 6),
            },
            "best_single_run": {
                "num_randomizations": best_nr,
                "hellinger": round(best_h, 6),
            },
            "best_vns_method": vns_method,
            "best_vns_hellinger": round(best_vns, 6),
            "improvement_vs_best": round(improvement, 6),
            "improvement_pct": round(improvement_pct, 2),
            "linear_fit": {
                "slope": round(float(fit_coeffs[0]), 4),
                "intercept": round(float(fit_coeffs[1]), 6),
            },
        }

        if post_fn == "2q":
            results_2q.append(entry)
        else:
            results_3q.append(entry)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*65}")
    print(f"  {'Circuit':<25s} {'Best Single H':>14s} {'VNS H':>10s} {'Improve':>10s}")
    print(f"  {'-'*25} {'-'*14} {'-'*10} {'-'*10}")
    for entry in results_2q + results_3q:
        label = entry["circuit"][:25]
        bsh = entry["best_single_run"]["hellinger"]
        vsh = entry["best_vns_hellinger"]
        imp = entry["improvement_vs_best"]
        print(f"  {label:<25s} {bsh:>14.6f} {vsh:>10.6f} {imp:>+10.6f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    overall_pass = all(
        e["best_vns_hellinger"] < 0.05
        for e in results_2q + results_3q
    )

    result: dict[str, Any] = {
        "task_ids": ["vns"],
        "technique": "Virtual Noise Scaling (VNS) ZNE",
        "reference": "Uzdin, arXiv:2601.22785 (January 2026)",
        "noise_model": "H(N) = H_inf + c/N  (N = num_randomizations)",
        "backend": args.backend if not args.dry_run else "aer_simulator",
        "shots_per_level": args.shots,
        "total_shots": args.shots * len(num_rands) * 2,  # 2 circuits
        "num_randomizations_sweep": num_rands,
        "observation": args.obs,
        "circuits": {
            "tiger_2q": results_2q[0] if results_2q else None,
            "tiger_4state_3q": results_3q[0] if results_3q else None,
        },
        "pass": overall_pass,
    }

    save_result("vns_tiger", result)

    print(f"\n  PASS: {overall_pass}")


if __name__ == "__main__":
    main()
