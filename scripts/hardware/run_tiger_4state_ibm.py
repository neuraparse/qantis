"""4-State Tiger POMDP belief update on IBM QPU (Task 2.5).

Extends the 2-qubit Tiger minimal circuit to a 4-state "Corridor Tiger"
POMDP using 3 qubits (2 state + 1 observation).  Demonstrates that the
quantum belief update principle scales beyond the binary case within the
NISQ coherence budget.

Model
-----
  States : {far-left, near-left, near-right, far-right}
           encoded as |00>, |01>, |10>, |11>  (q0=MSB, q1=LSB)
  Obs    : 0=hear-left, 1=hear-right
  P(hear-left | state): 0.85, 0.70, 0.30, 0.15
  Transition (listen): identity
  Prior  : uniform [0.25, 0.25, 0.25, 0.25]

Circuit (3 qubits)
------------------
  1. UCR_Y encode 4-amplitude prior onto q0, q1
  2. Doubly-controlled R_y on q2 for each state to encode P(obs|state)
  3. Measure all 3 qubits; post-select q2 == target_observation

Expected classical posterior (uniform prior, obs=0):
  [0.425, 0.350, 0.150, 0.075]  (higher probability on left states)

Expected classical posterior (uniform prior, obs=1):
  [0.075, 0.150, 0.350, 0.425]  (higher probability on right states)

Usage
-----
# Dry-run (no credentials):
python scripts/hardware/run_tiger_4state_ibm.py --dry-run

# Hardware run:
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_tiger_4state_ibm.py \\
    --backend ibm_marrakesh --shots 8192

Output
------
  output/hardware/tiger_4state_ibm_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

from scripts.hardware import get_ibm_token_optional, save_result


# ---------------------------------------------------------------------------
# POMDP model constants
# ---------------------------------------------------------------------------

# 4-state Corridor Tiger observation model
# P(hear-left | state=s) for s in {0=far-left, 1=near-left, 2=near-right, 3=far-right}
P_OBS_GIVEN_STATE_4 = [0.85, 0.70, 0.30, 0.15]

STATE_NAMES = ["far-left", "near-left", "near-right", "far-right"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4-State Tiger POMDP belief update on IBM QPU (Task 2.5)"
    )
    p.add_argument("--backend", default="ibm_marrakesh", help="IBM backend name")
    p.add_argument(
        "--shots", type=int, default=8192,
        help="Shot count per circuit execution (default: 8192)",
    )
    p.add_argument(
        "--obs", type=int, default=0, choices=[0, 1],
        help="Target observation to post-select on: 0=hear-left, 1=hear-right (default: 0)",
    )
    p.add_argument(
        "--prior", nargs=4, type=float, default=[0.25, 0.25, 0.25, 0.25],
        metavar=("P0", "P1", "P2", "P3"),
        help="Prior belief [P(s0), P(s1), P(s2), P(s3)], must sum to 1 (default: uniform)",
    )
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel: ibm_quantum_platform or ibm_cloud",
    )
    p.add_argument(
        "--instance", default=None,
        help="IBM instance / CRN",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only, skip IBM hardware (no credentials needed)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 4-state Tiger circuit (3 qubits)
# ---------------------------------------------------------------------------

def _build_tiger_4state_circuit(
    prior_4: list[float],
    target_observation: int,
) -> "QuantumCircuit":
    """Build a 3-qubit 4-state Tiger belief-update circuit.

    Qubit layout:
        q0 : MSB of state register  (|0>=left half, |1>=right half)
        q1 : LSB of state register
        q2 : observation register   (|0>=hear-left, |1>=hear-right)

    State encoding:
        |00> = far-left (s=0),  |01> = near-left (s=1)
        |10> = near-right (s=2), |11> = far-right (s=3)

    Steps:
        1. UCR_Y: encode 4-amplitude prior onto q0, q1 using 3 rotations.
        2. Doubly-controlled R_y on q2 for each of the 4 states.
        3. Measure all 3 qubits.

    Post-select on q2 = target_observation to obtain Bayesian posterior.

    Bitstring format (Qiskit MSB-first): c2 c1 c0
        c2 = obs bit  (q2), c1 = state LSB (q1), c0 = state MSB (q0)
    State index from post-selected bitstring:
        s = 2*int(bs[2]) + int(bs[1])
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    p = [float(x) for x in prior_4]
    assert abs(sum(p) - 1.0) < 1e-6, f"Prior must sum to 1, got {sum(p)}"

    qc = QuantumCircuit(3, 3, name="tiger_4state")

    # ------------------------------------------------------------------
    # Step 1: UCR_Y — encode 4-amplitude prior onto q0, q1
    # ------------------------------------------------------------------
    # Marginals for q0:
    p01 = float(np.clip(p[0] + p[1], 1e-9, 1.0))  # P(q0=0)
    p23 = float(np.clip(p[2] + p[3], 1e-9, 1.0))  # P(q0=1)

    # R_y(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    # We want P(q0=0) = cos^2(theta/2) = p01
    theta_q0 = 2.0 * float(np.arccos(np.sqrt(np.clip(p01, 1e-9, 1.0))))
    qc.ry(theta_q0, 0)

    # Conditional rotations on q1:
    # When q0=0: P(q1=0|q0=0) = p[0]/p01
    cond_p0 = p[0] / p01
    theta_q1_given_0 = 2.0 * float(np.arccos(np.sqrt(np.clip(cond_p0, 1e-9, 1.0))))
    # Apply when q0=0: flip q0, apply CRY, flip back
    qc.x(0)
    qc.cry(theta_q1_given_0, 0, 1)
    qc.x(0)

    # When q0=1: P(q1=0|q0=1) = p[2]/p23
    cond_p2 = p[2] / p23
    theta_q1_given_1 = 2.0 * float(np.arccos(np.sqrt(np.clip(cond_p2, 1e-9, 1.0))))
    # Apply when q0=1: CRY directly
    qc.cry(theta_q1_given_1, 0, 1)

    # ------------------------------------------------------------------
    # Step 2: Observation model — doubly-controlled R_y on q2
    # ------------------------------------------------------------------
    # For each state s, apply R_y(theta_obs[s]) on q2 when (q0,q1) == binary(s)
    # theta_obs[s] such that P(obs=0|s) = cos^2(theta_obs[s]/2)
    theta_obs = [
        2.0 * float(np.arccos(np.sqrt(np.clip(p_s, 1e-9, 1.0))))
        for p_s in P_OBS_GIVEN_STATE_4
    ]

    # State 0 (q0=0, q1=0): flip both → CCRy → flip both back
    qc.x(0)
    qc.x(1)
    qc.append(RYGate(theta_obs[0]).control(2), [0, 1, 2])
    qc.x(0)
    qc.x(1)

    # State 1 (q0=0, q1=1): flip q0 → CCRy → flip back
    qc.x(0)
    qc.append(RYGate(theta_obs[1]).control(2), [0, 1, 2])
    qc.x(0)

    # State 2 (q0=1, q1=0): flip q1 → CCRy → flip back
    qc.x(1)
    qc.append(RYGate(theta_obs[2]).control(2), [0, 1, 2])
    qc.x(1)

    # State 3 (q0=1, q1=1): apply CCRy directly
    qc.append(RYGate(theta_obs[3]).control(2), [0, 1, 2])

    # ------------------------------------------------------------------
    # Step 3: Measure all qubits
    # ------------------------------------------------------------------
    # q0→c0, q1→c1, q2→c2
    # Qiskit bitstring: c2 c1 c0 (MSB first)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)

    return qc


# ---------------------------------------------------------------------------
# Post-selection and probability extraction
# ---------------------------------------------------------------------------

def _post_select_4state(counts: dict, target_obs: int) -> dict[int, int]:
    """Post-select on observation qubit (leftmost bit in Qiskit bitstring).

    Bitstring format: c2(obs) c1(state_lsb=q1) c0(state_msb=q0)
    After post-selecting on obs=c2:
        state_idx = 2*int(c0) + int(c1) = 2*int(bs[2]) + int(bs[1])
    """
    obs_char = str(target_obs)
    filtered: dict[int, int] = {}
    for bs, count in counts.items():
        bs = bs.zfill(3)  # ensure 3 characters
        if bs[0] == obs_char:
            state_idx = 2 * int(bs[2]) + int(bs[1])
            filtered[state_idx] = filtered.get(state_idx, 0) + count
    return filtered


def _counts_to_probs(counts: dict[int, int], n_states: int = 4) -> list[float]:
    """Convert integer-keyed state counts to a probability vector."""
    total = sum(counts.values())
    if total == 0:
        return [1.0 / n_states] * n_states
    return [counts.get(s, 0) / total for s in range(n_states)]


def _classical_bayes_4state(prior_4: list[float], obs: int) -> list[float]:
    """Exact Bayesian posterior for 4-state Tiger listen action."""
    import numpy as np
    p = np.array(prior_4, dtype=float)
    if obs == 0:
        likelihood = np.array(P_OBS_GIVEN_STATE_4, dtype=float)
    else:
        likelihood = 1.0 - np.array(P_OBS_GIVEN_STATE_4, dtype=float)
    unnorm = p * likelihood
    z = unnorm.sum()
    return (unnorm / z).tolist() if z > 1e-12 else [0.25] * 4


def _hellinger(p: list[float], q: list[float]) -> float:
    """Hellinger distance between two probability vectors."""
    import numpy as np
    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_arr) - np.sqrt(q_arr)) ** 2)))


# ---------------------------------------------------------------------------
# Simulation execution (Aer)
# ---------------------------------------------------------------------------

def _run_4state_sim(circuit, shots: int) -> dict:
    """Run 4-state circuit on AerSimulator, return raw counts.

    Decomposes high-level gates (ccry) into basis gates before simulation.
    """
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile

    sim = AerSimulator()
    basis = ["cx", "u", "id", "reset", "measure"]
    decomposed = transpile(circuit, basis_gates=basis, optimization_level=0)
    job = sim.run(decomposed, shots=shots)
    return dict(job.result().get_counts())


# ---------------------------------------------------------------------------
# IBM hardware execution
# ---------------------------------------------------------------------------

def _run_4state_hw(circuit, backend_obj, shots: int) -> tuple[dict, int]:
    """Run 4-state circuit on IBM QPU, return (raw_counts, isa_depth)."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend_obj)
    isa = pm.run(circuit)
    isa._layout = None  # prevent QPY Error 3211
    isa_depth = isa.depth()

    sampler = SamplerV2(mode=backend_obj)
    job = sampler.run([isa], shots=shots)
    raw = job.result()[0]
    # Register name varies by circuit ('c' for minimal, 'meas' for some)
    _creg_name = next(k for k in vars(raw.data) if not k.startswith("_"))
    return dict(getattr(raw.data, _creg_name).get_counts()), isa_depth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    prior_4 = list(args.prior)
    prior_sum = sum(prior_4)
    prior_4 = [x / prior_sum for x in prior_4]  # normalise

    print(f"\n=== 4-State Tiger POMDP Belief Update — IBM QPU (Task 2.5) ===")
    print(f"  Backend    : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots      : {args.shots}")
    print(f"  Prior      : {[round(x, 4) for x in prior_4]}")
    print(f"  Observation: {args.obs} ({'hear-left' if args.obs == 0 else 'hear-right'})")

    # Connect to IBM backend
    backend_obj = None
    if not args.dry_run:
        token = get_ibm_token_optional()
        if token is None:
            print("  ERROR: IBM_QUANTUM_TOKEN not set. Use --dry-run for simulator.")
            sys.exit(1)
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(token=token, channel="ibm_quantum_platform")
        backend_obj = svc.backend(args.backend)
        print(f"  Connected  : {backend_obj.name} ({backend_obj.num_qubits} qubits)")

    # Build circuit
    circuit = _build_tiger_4state_circuit(prior_4, args.obs)
    print(f"\n  Circuit qubits : {circuit.num_qubits}")
    print(f"  Circuit depth  : {circuit.depth()}")

    # Classical reference posterior
    classical_post = _classical_bayes_4state(prior_4, args.obs)
    print(f"\n  Classical posterior: {[round(x, 4) for x in classical_post]}")
    print(f"    (Most probable state: {STATE_NAMES[classical_post.index(max(classical_post))]})")

    result: dict = {
        "task_ids": ["2.5"],
        "backend": args.backend if not args.dry_run else "aer_simulator",
        "shots": args.shots,
        "prior": prior_4,
        "observation": args.obs,
        "states": STATE_NAMES,
        "p_obs_given_state": P_OBS_GIVEN_STATE_4,
        "circuit": {
            "num_qubits": circuit.num_qubits,
            "logical_depth": circuit.depth(),
        },
        "classical_posterior": classical_post,
    }

    # Simulator run
    print(f"\n  [Sim] running on AerSimulator ...")
    sim_raw = _run_4state_sim(circuit, shots=args.shots)
    sim_ps = _post_select_4state(sim_raw, args.obs)
    sim_probs = _counts_to_probs(sim_ps, n_states=4)
    sim_hellinger = _hellinger(sim_probs, classical_post)
    print(f"    Sim posterior : {[round(x, 4) for x in sim_probs]}")
    print(f"    Sim Hellinger : {sim_hellinger:.4f}")

    result["simulator"] = {
        "posterior": sim_probs,
        "hellinger": round(sim_hellinger, 6),
        "ps_counts": sim_ps,
        "pass": sim_hellinger < 0.05,
    }

    # Hardware run
    if not args.dry_run and backend_obj is not None:
        print(f"\n  [HW] running on {args.backend} ...")
        hw_raw, isa_depth = _run_4state_hw(circuit, backend_obj, shots=args.shots)
        hw_ps = _post_select_4state(hw_raw, args.obs)
        hw_probs = _counts_to_probs(hw_ps, n_states=4)
        hw_hellinger = _hellinger(hw_probs, classical_post)
        print(f"    ISA depth    : {isa_depth}")
        print(f"    HW posterior : {[round(x, 4) for x in hw_probs]}")
        print(f"    HW Hellinger : {hw_hellinger:.4f}  "
              f"({'PASS' if hw_hellinger < 0.05 else 'FAIL (> 0.05)'})")

        result["hardware"] = {
            "isa_depth": isa_depth,
            "posterior": hw_probs,
            "hellinger": round(hw_hellinger, 6),
            "ps_counts": hw_ps,
            "pass": hw_hellinger < 0.05,
        }
        result["circuit"]["isa_depth"] = isa_depth

    # Summary
    src = "hardware" if not args.dry_run else "simulator"
    src_result = result.get(src, result.get("simulator", {}))
    overall_pass = src_result.get("pass", False)

    print(f"\n--- Summary ---")
    print(f"  Source    : {src}")
    print(f"  Hellinger : {src_result.get('hellinger', 'N/A')}")
    print(f"  PASS      : {overall_pass}")
    result["pass"] = overall_pass

    save_result("tiger_4state_ibm", result)


if __name__ == "__main__":
    main()
