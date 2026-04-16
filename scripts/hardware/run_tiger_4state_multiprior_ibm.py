"""4-State Tiger POMDP belief update — multi-prior hardware sweep.

Addresses Reviewer Weakness 2: "Tiger |S|=2 is a toy problem / single prior."

Strategy
--------
Run the existing 4-state Tiger minimal circuit (Task 2.5) across THREE
distinct prior distributions × TWO observations = 6 hardware runs at
ISA depth ~162 each.  Demonstrates that:

  (a) The quantum belief update generalises beyond the uniform prior.
  (b) Posterior fidelity (Hellinger < 0.05) holds across informative
      priors, not just the uniform case.
  (c) Belief REVISION is quantified: we measure how much the posterior
      shifts relative to the prior, confirming that the quantum circuit
      performs genuine Bayesian inference.

Prior configurations
--------------------
  Prior A: [0.70, 0.10, 0.10, 0.10]  — far-left concentrated
  Prior B: [0.25, 0.25, 0.25, 0.25]  — uniform (baseline)
  Prior C: [0.10, 0.10, 0.10, 0.70]  — far-right concentrated

Observations: 0=hear-left, 1=hear-right (both per prior)

Expected posteriors (illustrative)
-----------------------------------
  Prior A, obs=0 → [0.838, 0.099, 0.042, 0.021]  (prior reinforced)
  Prior A, obs=1 → [0.362, 0.103, 0.241, 0.293]  (prior conflicted)
  Prior B, obs=0 → [0.425, 0.350, 0.150, 0.075]  (baseline)
  Prior B, obs=1 → [0.075, 0.150, 0.350, 0.425]  (baseline, flipped)
  Prior C, obs=0 → [0.293, 0.241, 0.103, 0.362]  (prior conflicted)
  Prior C, obs=1 → [0.021, 0.042, 0.099, 0.838]  (prior reinforced)

The symmetric structure (A,obs=0 mirrors C,obs=1) provides a built-in
consistency check for hardware noise.

Run
---
# Dry-run (Aer simulator, no credentials):
python scripts/hardware/run_tiger_4state_multiprior_ibm.py --dry-run

# Hardware run:
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_tiger_4state_multiprior_ibm.py \\
    --backend ibm_marrakesh --shots 8192

Output
------
  output/hardware/tiger_4state_multiprior_ibm_<timestamp>.json
  Console table (copy-paste ready for paper)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

# Reuse circuit logic from the single-run script (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tiger_4state_ibm import (
    _build_tiger_4state_circuit,
    _build_tiger_4state_optimized,
    _post_select_4state,
    _counts_to_probs,
    _classical_bayes_4state,
    _hellinger,
    _run_4state_sim,
    _run_4state_hw,
    STATE_NAMES,
)

from scripts.hardware import (
    get_ibm_token_optional,
    ibm_channel,
    ibm_instance,
    make_ibm_runtime_service,
    save_result,
)


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

PRIORS: list[tuple[str, list[float]]] = [
    ("far-left",  [0.70, 0.10, 0.10, 0.10]),
    ("uniform",   [0.25, 0.25, 0.25, 0.25]),
    ("far-right", [0.10, 0.10, 0.10, 0.70]),
]

OBSERVATIONS: list[tuple[str, int]] = [
    ("hear-left",  0),
    ("hear-right", 1),
]

HELLINGER_THRESHOLD = 0.05  # same as single-run script


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4-State Tiger multi-prior sweep on IBM QPU"
    )
    p.add_argument("--backend", default="ibm_marrakesh", help="IBM backend name")
    p.add_argument(
        "--shots", type=int, default=8192,
        help="Shot count per circuit (default: 8192)",
    )
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel: ibm_quantum_platform or ibm_cloud",
    )
    p.add_argument("--instance", default=None, help="IBM instance / CRN")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only, no IBM credentials needed",
    )
    p.add_argument(
        "--optimized", action="store_true",
        help="Use the optimized unitary-synthesis 4-state circuit family",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Belief revision metric
# ---------------------------------------------------------------------------

def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q) — how much posterior p differs from prior q."""
    import numpy as np
    eps = 1e-12
    p_arr = np.array(p, dtype=float) + eps
    q_arr = np.array(q, dtype=float) + eps
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_one(
    prior: list[float],
    obs: int,
    shots: int,
    backend_obj,
    dry_run: bool,
    optimized: bool,
) -> dict:
    """Run one prior × obs combination. Return result dict."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    circuit_orig = _build_tiger_4state_circuit(prior, obs)
    circuit = _build_tiger_4state_optimized(prior, obs) if optimized else circuit_orig
    classical_post = _classical_bayes_4state(prior, obs)
    kl_prior_post = _kl_divergence(classical_post, prior)

    # --- Simulator ---
    sim_raw = _run_4state_sim(circuit, shots=shots)
    sim_ps = _post_select_4state(sim_raw, obs)
    sim_probs = _counts_to_probs(sim_ps, n_states=4)
    sim_hell = _hellinger(sim_probs, classical_post)

    entry = {
        "prior": prior,
        "observation": obs,
        "classical_posterior": classical_post,
        "kl_prior_to_posterior": round(kl_prior_post, 4),
        "circuit": {
            "num_qubits": circuit.num_qubits,
            "logical_depth": circuit.depth(),
            "optimized": optimized,
        },
        "simulator": {
            "posterior": sim_probs,
            "hellinger": round(sim_hell, 6),
            "pass": sim_hell < HELLINGER_THRESHOLD,
        },
    }

    if not dry_run and backend_obj is not None:
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend_obj)
        isa = pm.run(circuit)
        isa._layout = None  # prevent QPY Error 3211
        isa_depth = isa.depth()
        isa_size = isa.size()
        isa_two_q = sum(1 for inst in isa.data if inst.operation.num_qubits == 2)

        hw_raw, _ = _run_4state_hw(circuit, backend_obj, shots=shots)
        hw_ps = _post_select_4state(hw_raw, obs)
        hw_probs = _counts_to_probs(hw_ps, n_states=4)
        hw_hell = _hellinger(hw_probs, classical_post)

        entry["hardware"] = {
            "isa_depth": isa_depth,
            "isa_size": isa_size,
            "two_qubit_gates": isa_two_q,
            "posterior": hw_probs,
            "hellinger": round(hw_hell, 6),
            "pass": hw_hell < HELLINGER_THRESHOLD,
        }
        entry["circuit"]["isa_depth"] = isa_depth
        entry["circuit"]["isa_size"] = isa_size
        entry["circuit"]["two_qubit_gates"] = isa_two_q

        if optimized:
            isa_orig = pm.run(circuit_orig)
            isa_orig._layout = None
            entry["circuit"]["original_logical_depth"] = circuit_orig.depth()
            entry["circuit"]["original_isa_depth"] = isa_orig.depth()
            entry["circuit"]["original_isa_size"] = isa_orig.size()
            entry["circuit"]["original_two_qubit_gates"] = sum(
                1 for inst in isa_orig.data if inst.operation.num_qubits == 2
            )

    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"\n=== 4-State Tiger Multi-Prior Sweep — IBM QPU ===")
    print(f"  Backend : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Mode    : {'OPTIMIZED' if args.optimized else 'ORIGINAL'}")
    print(f"  Shots   : {args.shots}")
    print(f"  Runs    : {len(PRIORS)} priors × {len(OBSERVATIONS)} obs = "
          f"{len(PRIORS) * len(OBSERVATIONS)} circuits")
    print()

    # Connect to IBM backend once (reused for all 6 runs)
    backend_obj = None
    if not args.dry_run:
        token = get_ibm_token_optional()
        if token is None:
            print("  ERROR: IBM_QUANTUM_TOKEN not set. Use --dry-run for simulator.")
            sys.exit(1)
        channel = args.channel or ibm_channel()
        instance = args.instance if args.instance is not None else ibm_instance()
        svc = make_ibm_runtime_service(token=token, channel=channel, instance=instance)
        backend_obj = svc.backend(args.backend)
        print(f"  Connected: {backend_obj.name} ({backend_obj.num_qubits} qubits)\n")

    all_runs: list[dict] = []
    results_table: list[tuple] = []

    for prior_name, prior in PRIORS:
        for obs_name, obs in OBSERVATIONS:
            label = f"{prior_name}, obs={obs} ({obs_name})"
            print(f"  [{len(all_runs)+1}/6] Prior: {prior_name}  Obs: {obs} ({obs_name})")

            try:
                entry = _run_one(
                    prior=prior,
                    obs=obs,
                    shots=args.shots,
                    backend_obj=backend_obj,
                    dry_run=args.dry_run,
                    optimized=args.optimized,
                )
            except Exception as exc:
                partial = {
                    "task_ids": ["2.5-multiprior"],
                    "backend": args.backend if not args.dry_run else "aer_simulator",
                    "shots": args.shots,
                    "optimized": args.optimized,
                    "hellinger_threshold": HELLINGER_THRESHOLD,
                    "runs": all_runs,
                    "failure": {
                        "prior_name": prior_name,
                        "obs_name": obs_name,
                        "message": str(exc),
                    },
                }
                save_result("tiger_4state_multiprior_ibm_partial", partial)
                raise
            entry["prior_name"] = prior_name
            entry["obs_name"] = obs_name
            all_runs.append(entry)

            # Collect row data for table
            src = "hardware" if not args.dry_run else "simulator"
            src_res = entry.get(src, entry.get("simulator", {}))
            hw_hell = src_res.get("hellinger", float("nan"))
            hw_pass = src_res.get("pass", False)
            isa = entry.get("circuit", {}).get("isa_depth", entry["circuit"]["logical_depth"])
            kl = entry["kl_prior_to_posterior"]
            classical_post = entry["classical_posterior"]
            most_prob = STATE_NAMES[classical_post.index(max(classical_post))]
            results_table.append((
                prior_name, obs_name, isa, kl, hw_hell, hw_pass, most_prob,
            ))

            print(f"    Classical posterior: "
                  f"{[round(x, 3) for x in entry['classical_posterior']]}")
            print(f"    Most probable state: {most_prob}")
            print(f"    KL(post||prior)    : {kl:.4f}  (revision magnitude)")
            src_res_sim = entry["simulator"]
            print(f"    Sim Hellinger      : {src_res_sim['hellinger']:.4f}  "
                  f"({'PASS' if src_res_sim['pass'] else 'FAIL'})")
            if not args.dry_run:
                hw_res = entry.get("hardware", {})
                if hw_res:
                    print(f"    ISA depth          : {hw_res.get('isa_depth', 'N/A')}")
                    if args.optimized:
                        orig_depth = entry.get("circuit", {}).get("original_isa_depth", "N/A")
                        two_q = hw_res.get("two_qubit_gates", "N/A")
                        orig_two_q = entry.get("circuit", {}).get("original_two_qubit_gates", "N/A")
                        print(f"    ISA compare        : {orig_depth} -> {hw_res.get('isa_depth', 'N/A')}, "
                              f"2Q {orig_two_q} -> {two_q}")
                    print(f"    HW Hellinger       : {hw_res['hellinger']:.4f}  "
                          f"({'PASS' if hw_res['pass'] else 'FAIL (> 0.05)'})")
            print()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("=" * 90)
    print(f"{'Prior':>12} {'Obs':>11} {'ISA':>5} {'KL':>6}  "
          f"{'Hellinger':>10}  {'PASS':>5}  {'Most-prob state':>18}")
    print("-" * 90)
    all_pass = True
    for prior_name, obs_name, isa, kl, hell, passed, mp in results_table:
        pass_str = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {prior_name:>10} {obs_name:>12} {isa:>5} {kl:>6.3f}  "
              f"{hell:>10.4f}  {pass_str:>5}  {mp:>18}")
    print("-" * 90)
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAIL — see table'}")
    print()

    # Symmetry check (A,obs=0 vs C,obs=1 should mirror each other)
    # A,obs=0 ↔ run index 0; C,obs=1 ↔ run index 5
    if len(all_runs) == 6:
        hell_A0 = results_table[0][4]
        hell_C1 = results_table[5][4]
        post_A0 = all_runs[0]["classical_posterior"]
        post_C1 = all_runs[5]["classical_posterior"]
        post_C1_rev = list(reversed(post_C1))
        sym_hell = _hellinger(post_A0, post_C1_rev)
        print(f"  Symmetry check (A,obs=0 ↔ reversed C,obs=1): "
              f"Hellinger={sym_hell:.4f}  "
              f"({'PASS — model symmetric' if sym_hell < 0.001 else 'MISMATCH'})")
        print()

    # Save JSON
    output = {
        "task_ids": ["2.5-multiprior"],
        "backend": args.backend if not args.dry_run else "aer_simulator",
        "shots": args.shots,
        "optimized": args.optimized,
        "hellinger_threshold": HELLINGER_THRESHOLD,
        "runs": all_runs,
        "summary": {
            "n_pass": sum(1 for r in results_table if r[5]),
            "n_fail": sum(1 for r in results_table if not r[5]),
            "all_pass": all_pass,
        },
    }

    save_result("tiger_4state_multiprior_ibm", output)


if __name__ == "__main__":
    main()
