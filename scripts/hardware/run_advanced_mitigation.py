"""Error Suppression Ablation Study on IBM Heron R2 (2026 Techniques).

Systematic comparison of IBM Runtime V2 error mitigation / suppression
primitives on the Tiger POMDP belief-update circuits.  This produces a
publishable ablation table showing the marginal contribution of each
technique to the final Hellinger fidelity.

Mitigation configurations tested
---------------------------------
  Config A : Bare          — no mitigation at all
  Config B : DD only       — XY4 dynamical decoupling (ALAP scheduling)
  Config C : Twirl only    — Pauli gate + measurement twirling (active-accum)
  Config D : Full stack    — DD + Twirl (gate + measure) + TREX
  Config E : Full + Hi-NR  — Config D with num_randomizations = 128

Circuits
--------
  1. Tiger minimal  (2 qubits, 1 state + 1 obs)  — ISA depth ~12
  2. Tiger 4-state  (3 qubits, 2 state + 1 obs)  — ISA depth ~30-50

Metrics
-------
  - Hellinger distance to classical Bayesian posterior
  - Post-selection yield (fraction of shots surviving obs post-selection)

Usage
-----
# Dry-run (simulator only):
python scripts/hardware/run_advanced_mitigation.py --dry-run

# Hardware run:
IBM_QUANTUM_TOKEN=xxx IBM_QUANTUM_CHANNEL=ibm_cloud IBM_QUANTUM_INSTANCE=crn:... \
    python -u scripts/hardware/run_advanced_mitigation.py \
    --backend ibm_kingston --shots 8192

Output
------
  output/hardware/ablation_mitigation_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
import time
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
# Mitigation configurations
# ---------------------------------------------------------------------------

CONFIGS: list[dict[str, Any]] = [
    {
        "name": "A: Bare",
        "label": "bare",
        "dd_enable": False,
        "twirl_gates": False,
        "twirl_measure": False,
        "num_randomizations": None,   # default
    },
    {
        "name": "B: DD only (XY4)",
        "label": "dd_only",
        "dd_enable": True,
        "twirl_gates": False,
        "twirl_measure": False,
        "num_randomizations": None,
    },
    {
        "name": "C: Twirl only",
        "label": "twirl_only",
        "dd_enable": False,
        "twirl_gates": True,
        "twirl_measure": True,
        "num_randomizations": None,
    },
    {
        "name": "D: Full stack (DD+Twirl)",
        "label": "full_stack",
        "dd_enable": True,
        "twirl_gates": True,
        "twirl_measure": True,
        "num_randomizations": None,
    },
    {
        "name": "E: Full + High NR (128)",
        "label": "full_high_nr",
        "dd_enable": True,
        "twirl_gates": True,
        "twirl_measure": True,
        "num_randomizations": 128,
    },
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Error suppression ablation study on IBM Heron R2"
    )
    p.add_argument("--backend", default="ibm_kingston", help="IBM backend name")
    p.add_argument(
        "--shots", type=int, default=8192,
        help="Shot count per config per circuit (default: 8192)",
    )
    p.add_argument(
        "--obs", type=int, default=0, choices=[0, 1],
        help="Target observation: 0=hear-left, 1=hear-right (default: 0)",
    )
    p.add_argument("--channel", default=None, help="Qiskit channel")
    p.add_argument("--instance", default=None, help="IBM instance / CRN")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only (no credentials needed)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hardware execution with configurable mitigation
# ---------------------------------------------------------------------------

def _run_hw_with_config(
    isa_circuit,
    backend,
    shots: int,
    config: dict[str, Any],
) -> dict:
    """Run a transpiled circuit on hardware with a specific mitigation config."""
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)

    # Dynamical decoupling
    if config["dd_enable"]:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.dynamical_decoupling.scheduling_method = "alap"
    else:
        sampler.options.dynamical_decoupling.enable = False

    # Pauli twirling (gates + measurement)
    if config["twirl_gates"]:
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.strategy = "active-accum"
    else:
        sampler.options.twirling.enable_gates = False

    if config["twirl_measure"]:
        sampler.options.twirling.enable_measure = True
    else:
        sampler.options.twirling.enable_measure = False

    # Num randomizations
    if config["num_randomizations"] is not None:
        sampler.options.twirling.num_randomizations = config["num_randomizations"]

    job = sampler.run([isa_circuit], shots=shots)
    raw = job.result()[0]
    _creg_name = next(k for k in vars(raw.data) if not k.startswith("_"))
    return dict(getattr(raw.data, _creg_name).get_counts())


# ---------------------------------------------------------------------------
# Simulator dry-run
# ---------------------------------------------------------------------------

def _run_sim(circuit, shots: int) -> dict:
    """Run circuit on AerSimulator (ideal, no noise)."""
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile

    sim = AerSimulator()
    basis = ["cx", "u", "id", "reset", "measure"]
    decomposed = transpile(circuit, basis_gates=basis, optimization_level=0)
    job = sim.run(decomposed, shots=shots)
    return dict(job.result().get_counts())


# ---------------------------------------------------------------------------
# Classical Bayesian reference for 2-state Tiger
# ---------------------------------------------------------------------------

def _classical_bayes_2state(prior: list[float], obs: int) -> list[float]:
    """Exact Bayesian posterior for 2-state Tiger (listen action)."""
    P_OBS_2 = [0.85, 0.15]
    if obs == 0:
        lik = np.array(P_OBS_2)
    else:
        lik = 1.0 - np.array(P_OBS_2)
    unnorm = np.array(prior) * lik
    z = unnorm.sum()
    return (unnorm / z).tolist() if z > 1e-12 else [0.5, 0.5]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # --- Build circuits ---
    prior_2 = [0.5, 0.5]
    prior_4 = [0.25, 0.25, 0.25, 0.25]
    P_OBS_2 = [0.85, 0.15]

    classical_post_2 = _classical_bayes_2state(prior_2, args.obs)
    classical_post_4 = _classical_bayes_4state(prior_4, args.obs)

    circuit_2q = _build_minimal_tiger_circuit(
        p_obs_given_state=P_OBS_2,
        prior=prior_2,
        action=0,
        target_observation=args.obs,
    )
    circuit_3q = _build_tiger_4state_circuit(prior_4, args.obs)

    print(f"\n{'='*70}")
    print(f"  Error Suppression Ablation Study — IBM Heron R2 (2026)")
    print(f"{'='*70}")
    print(f"  Backend    : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots/run  : {args.shots}")
    print(f"  Observation: {args.obs} ({'hear-left' if args.obs == 0 else 'hear-right'})")
    print(f"  Configs    : {len(CONFIGS)}")
    print(f"  Total jobs : {len(CONFIGS) * 2} (2 circuits x {len(CONFIGS)} configs)")
    print(f"\n  Classical posterior (2-state): {[round(x, 4) for x in classical_post_2]}")
    print(f"  Classical posterior (4-state): {[round(x, 4) for x in classical_post_4]}")
    print(f"  2q logical depth : {circuit_2q.depth()}")
    print(f"  3q logical depth : {circuit_3q.depth()}")

    # ------------------------------------------------------------------
    # Connect to backend
    # ------------------------------------------------------------------
    backend_obj = None
    isa_2q = None
    isa_3q = None
    isa_depth_2q = circuit_2q.depth()
    isa_depth_3q = circuit_3q.depth()

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
    # Run all configs
    # ------------------------------------------------------------------
    circuits_info = [
        {
            "label": "Tiger 2q",
            "circuit": circuit_2q,
            "isa": isa_2q,
            "n_states": 2,
            "classical_post": classical_post_2,
            "isa_depth": isa_depth_2q,
            "post_fn": "2q",
        },
        {
            "label": "Tiger 4-state 3q",
            "circuit": circuit_3q,
            "isa": isa_3q,
            "n_states": 4,
            "classical_post": classical_post_4,
            "isa_depth": isa_depth_3q,
            "post_fn": "3q",
        },
    ]

    # Structure: all_results[circuit_label][config_label] = {...}
    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for cinfo in circuits_info:
        c_label = cinfo["label"]
        all_results[c_label] = {}

        print(f"\n{'='*70}")
        print(f"  Circuit: {c_label}  (ISA depth: {cinfo['isa_depth']})")
        print(f"{'='*70}")

        for cfg in CONFIGS:
            cfg_name = cfg["name"]
            cfg_label = cfg["label"]

            print(f"\n  --- {cfg_name} ---")
            t0 = time.time()

            if args.dry_run:
                raw_counts = _run_sim(cinfo["circuit"], args.shots)
            else:
                assert cinfo["isa"] is not None and backend_obj is not None
                print(f"    Submitting to {args.backend} ...")
                raw_counts = _run_hw_with_config(
                    cinfo["isa"], backend_obj, args.shots, cfg,
                )

            elapsed = time.time() - t0

            # Post-select and compute posterior
            if cinfo["post_fn"] == "2q":
                ps_counts = _post_select_counts(raw_counts, 1, 1, args.obs)
                total_ps = sum(ps_counts.values())
                probs = [0.0, 0.0]
                if total_ps > 0:
                    for bs, cnt in ps_counts.items():
                        idx = int(bs)
                        probs[idx] = cnt / total_ps
                h = float(np.sqrt(0.5 * sum(
                    (np.sqrt(probs[i]) - np.sqrt(cinfo["classical_post"][i])) ** 2
                    for i in range(2)
                )))
            else:
                ps_counts_int = _post_select_4state(raw_counts, args.obs)
                probs = _counts_to_probs(ps_counts_int, n_states=4)
                h = _hellinger(probs, cinfo["classical_post"])
                total_ps = sum(ps_counts_int.values())

            ps_yield = total_ps / args.shots if args.shots > 0 else 0.0

            print(f"    Posterior: {[round(x, 4) for x in probs]}")
            print(f"    Hellinger: {h:.6f}")
            print(f"    PS yield : {total_ps}/{args.shots} ({ps_yield:.1%})")
            print(f"    Time     : {elapsed:.1f}s")

            all_results[c_label][cfg_label] = {
                "config_name": cfg_name,
                "posterior": probs,
                "hellinger": round(h, 6),
                "ps_total": total_ps,
                "ps_yield": round(ps_yield, 4),
                "elapsed_s": round(elapsed, 1),
            }

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'='*70}")

    # Header
    config_names_short = ["Bare", "DD", "Twirl", "Full", "Full+HiNR"]
    config_labels = [c["label"] for c in CONFIGS]

    for cinfo in circuits_info:
        c_label = cinfo["label"]
        print(f"\n  Circuit: {c_label}")
        print(f"  Classical: {[round(x, 4) for x in cinfo['classical_post']]}")
        print(f"  ISA depth: {cinfo['isa_depth']}")
        print()
        print(f"    {'Config':<22s} {'Hellinger':>10s} {'PS Yield':>10s} {'vs Bare':>10s}")
        print(f"    {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

        bare_h = all_results[c_label]["bare"]["hellinger"]

        for cfg, short in zip(CONFIGS, config_names_short):
            r = all_results[c_label][cfg["label"]]
            delta = bare_h - r["hellinger"]
            delta_str = f"{delta:+.6f}" if cfg["label"] != "bare" else "  ---"
            print(f"    {cfg['name']:<22s} {r['hellinger']:>10.6f} {r['ps_yield']:>9.1%} {delta_str:>10s}")

    # ------------------------------------------------------------------
    # Best config analysis
    # ------------------------------------------------------------------
    print(f"\n  --- Best Config per Circuit ---")
    for cinfo in circuits_info:
        c_label = cinfo["label"]
        best_label = min(
            config_labels,
            key=lambda cl: all_results[c_label][cl]["hellinger"],
        )
        best_r = all_results[c_label][best_label]
        best_cfg_name = best_r["config_name"]
        bare_h = all_results[c_label]["bare"]["hellinger"]
        reduction = bare_h - best_r["hellinger"]
        reduction_pct = (reduction / bare_h * 100) if bare_h > 0 else 0
        print(f"    {c_label}: {best_cfg_name}  H={best_r['hellinger']:.6f}  "
              f"(reduction vs bare: {reduction:+.6f}, {reduction_pct:+.1f}%)")

    # ------------------------------------------------------------------
    # Marginal contribution analysis
    # ------------------------------------------------------------------
    print(f"\n  --- Marginal Contribution (averaged across circuits) ---")

    avg_h: dict[str, float] = {}
    for cl in config_labels:
        vals = [all_results[c_label][cl]["hellinger"] for c_label in all_results]
        avg_h[cl] = sum(vals) / len(vals)

    # DD contribution: bare -> dd_only
    dd_marginal = avg_h["bare"] - avg_h["dd_only"]
    # Twirl contribution: bare -> twirl_only
    twirl_marginal = avg_h["bare"] - avg_h["twirl_only"]
    # Combined vs sum of parts: full_stack vs bare
    full_marginal = avg_h["bare"] - avg_h["full_stack"]
    # High NR contribution: full_stack -> full_high_nr
    hinr_marginal = avg_h["full_stack"] - avg_h["full_high_nr"]

    print(f"    DD alone          : {dd_marginal:+.6f}")
    print(f"    Twirling alone    : {twirl_marginal:+.6f}")
    print(f"    Full stack (DD+TW): {full_marginal:+.6f}")
    print(f"    High NR (128)     : {hinr_marginal:+.6f} (on top of full stack)")
    synergy = full_marginal - (dd_marginal + twirl_marginal)
    print(f"    Synergy (full - sum of parts): {synergy:+.6f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    overall_pass = all(
        all_results[c_label]["full_high_nr"]["hellinger"] < 0.05
        for c_label in all_results
    )

    result: dict[str, Any] = {
        "task_ids": ["ablation"],
        "technique": "Error Suppression Ablation Study (2026 IBM Runtime V2)",
        "backend": args.backend if not args.dry_run else "aer_simulator",
        "shots_per_config": args.shots,
        "total_shots": args.shots * len(CONFIGS) * 2,
        "observation": args.obs,
        "configs": [
            {
                "label": c["label"],
                "name": c["name"],
                "dd_enable": c["dd_enable"],
                "twirl_gates": c["twirl_gates"],
                "twirl_measure": c["twirl_measure"],
                "num_randomizations": c["num_randomizations"],
            }
            for c in CONFIGS
        ],
        "circuits": {},
        "marginal_contributions": {
            "dd_alone": round(dd_marginal, 6),
            "twirling_alone": round(twirl_marginal, 6),
            "full_stack": round(full_marginal, 6),
            "high_nr_marginal": round(hinr_marginal, 6),
            "synergy": round(synergy, 6),
        },
        "pass": overall_pass,
    }

    for cinfo in circuits_info:
        c_label = cinfo["label"]
        circuit_result: dict[str, Any] = {
            "n_states": cinfo["n_states"],
            "classical_posterior": cinfo["classical_post"],
            "isa_depth": cinfo["isa_depth"],
            "configs": {},
        }
        for cfg in CONFIGS:
            circuit_result["configs"][cfg["label"]] = all_results[c_label][cfg["label"]]

        # Best config for this circuit
        best_label = min(
            config_labels,
            key=lambda cl: all_results[c_label][cl]["hellinger"],
        )
        circuit_result["best_config"] = best_label
        circuit_result["best_hellinger"] = all_results[c_label][best_label]["hellinger"]
        result["circuits"][c_label] = circuit_result

    save_result("ablation_mitigation", result)
    print(f"\n  PASS: {overall_pass}")


if __name__ == "__main__":
    main()
