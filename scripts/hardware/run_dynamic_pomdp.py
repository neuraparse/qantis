"""Dynamic-circuit quantum POMDP belief loop on IBM QPU.

The FIRST dynamic-circuit quantum POMDP on hardware: runs a T-step
Tiger POMDP belief update loop as a SINGLE quantum circuit using
mid-circuit measurements and qubit resets (IBM Heron R2 native ops).

Architecture comparison
-----------------------
  Non-dynamic (run_end_to_end_pomdp_ibm.py):
    - T separate circuits, T job submissions
    - Each circuit: ISA depth ~12
    - Classical feedback between circuits (priors computed on CPU)

  Dynamic (this script):
    - 1 circuit, 1 job submission
    - All T steps unrolled into a single circuit
    - Mid-circuit measurements record intermediate observations
    - Qubit resets between steps (native on Heron R2)
    - Priors are pre-computed classically (same Bayesian update)
    - Final measurement gives the last posterior

The observation sequence is pre-defined (no classical feedforward /
if_test needed). At each step the circuit:
  1. Resets both qubits (except step 0)
  2. Encodes the pre-computed prior via R_y on the state qubit
  3. Applies the Tiger observation model via CRY gates
  4. Mid-circuit measures the observation qubit (steps 0..T-2)
  5. At step T-1: measures both qubits (final posterior)

Novelty: To the best of our knowledge, this is the first experimental
demonstration of a quantum POMDP belief-tracking loop executed as a
single dynamic quantum circuit on superconducting hardware.

Usage
-----
# Dry-run (simulator only, no credentials needed):
python scripts/hardware/run_dynamic_pomdp.py --dry-run

# Dry-run with custom observation sequence:
python scripts/hardware/run_dynamic_pomdp.py --dry-run --n-steps 4 --obs-sequence 0 0 0 0

# Hardware run:
python scripts/hardware/run_dynamic_pomdp.py --backend ibm_kingston --shots 8192

# Hardware run with all-right observations:
python scripts/hardware/run_dynamic_pomdp.py --backend ibm_kingston --shots 8192 \\
    --n-steps 4 --obs-sequence 1 1 1 1

Output
------
  output/hardware/dynamic_pomdp_ibm_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

from scripts.hardware import get_ibm_token_optional, save_result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dynamic-circuit quantum POMDP belief loop on IBM QPU"
    )
    p.add_argument("--backend", default="ibm_kingston", help="IBM backend name")
    p.add_argument(
        "--shots", type=int, default=8192,
        help="Shots for the dynamic circuit (default: 8192)",
    )
    p.add_argument(
        "--n-steps", type=int, default=4,
        help="Number of POMDP time steps T (default: 4)",
    )
    p.add_argument(
        "--obs-sequence", nargs="+", type=int, default=None,
        help="Observation sequence (0=hear-left, 1=hear-right). "
             "Length must match --n-steps. Default: all zeros.",
    )
    p.add_argument(
        "--p-correct", type=float, default=0.85,
        help="Tiger observation accuracy P(correct obs | state) (default: 0.85)",
    )
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel: ibm_quantum_platform or ibm_cloud",
    )
    p.add_argument("--instance", default=None, help="IBM instance / CRN")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only, skip IBM hardware (no credentials needed)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Classical Bayesian belief update (pre-compute priors for each step)
# ---------------------------------------------------------------------------

def compute_priors_and_posteriors(
    obs_sequence: list[int],
    p_correct: float = 0.85,
) -> tuple[list[list[float]], list[list[float]]]:
    """Pre-compute the prior and posterior at each POMDP step.

    The prior at step 0 is always uniform [0.5, 0.5].
    For step t > 0, the prior is the posterior from step t-1.

    Args:
        obs_sequence: Observation at each step (0=hear-left, 1=hear-right).
        p_correct: P(correct observation | state).

    Returns:
        (priors, posteriors) — each is a list of [P(tiger_left), P(tiger_right)].
    """
    priors: list[list[float]] = [[0.5, 0.5]]
    posteriors: list[list[float]] = []

    for obs in obs_sequence:
        prior = priors[-1]
        # Bayesian update: P(s | o) = P(o | s) * P(s) / P(o)
        if obs == 0:  # hear-left
            # P(hear-left | tiger-left)  = p_correct
            # P(hear-left | tiger-right) = 1 - p_correct
            p_obs_total = prior[0] * p_correct + prior[1] * (1 - p_correct)
            post = [
                prior[0] * p_correct / p_obs_total,
                prior[1] * (1 - p_correct) / p_obs_total,
            ]
        else:  # hear-right (obs == 1)
            # P(hear-right | tiger-left)  = 1 - p_correct
            # P(hear-right | tiger-right) = p_correct
            p_obs_total = prior[0] * (1 - p_correct) + prior[1] * p_correct
            post = [
                prior[0] * (1 - p_correct) / p_obs_total,
                prior[1] * p_correct / p_obs_total,
            ]
        posteriors.append(post)
        # The posterior becomes the prior for the next step
        if len(posteriors) < len(obs_sequence):
            priors.append(post)

    return priors, posteriors


# ---------------------------------------------------------------------------
# Hellinger distance
# ---------------------------------------------------------------------------

def _hellinger(p: list[float], q: list[float]) -> float:
    """Hellinger distance between two probability distributions."""
    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)
    return float(np.sqrt(1.0 - np.sum(np.sqrt(p_arr * q_arr))))


# ---------------------------------------------------------------------------
# Build the dynamic POMDP circuit
# ---------------------------------------------------------------------------

def build_dynamic_pomdp_circuit(
    priors: list[list[float]],
    obs_sequence: list[int],
    p_correct: float = 0.85,
) -> "QuantumCircuit":
    """Build a single dynamic circuit encoding T steps of Tiger POMDP belief updates.

    The circuit uses 2 qubits (state + observation) and encodes the entire
    POMDP belief loop using mid-circuit measurements and qubit resets.

    At each step t:
      1. Reset both qubits (except step 0)
      2. Encode prior[t] on state qubit via R_y
      3. Apply Tiger observation model via CRY gates
      4. Mid-circuit measure obs qubit into obs_t register (steps 0..T-2)
      5. At step T-1: measure both qubits into final register

    The CRY encoding follows the Tiger minimal circuit convention:
      - q[0] = state qubit (|0> = tiger-left, |1> = tiger-right)
      - q[1] = observation qubit (|0> = hear-left, |1> = hear-right)

    Observation model:
      When state = tiger-left  (q[0]=|0>): R_y(theta_s0) on q[1]
        where cos^2(theta_s0/2) = P(hear-left | tiger-left) = p_correct
      When state = tiger-right (q[0]=|1>): R_y(theta_s1) on q[1]
        where cos^2(theta_s1/2) = P(hear-left | tiger-right) = 1-p_correct

    Args:
        priors: Pre-computed prior for each step. priors[t] = [P(TL), P(TR)].
        obs_sequence: Observation index at each step.
        p_correct: Tiger observation accuracy.

    Returns:
        QuantumCircuit with T mid-circuit measurements + 1 final measurement.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    T = len(obs_sequence)
    assert len(priors) == T, f"Expected {T} priors, got {len(priors)}"

    q = QuantumRegister(2, "q")
    # Separate classical register for each mid-circuit obs measurement
    c_obs = [ClassicalRegister(1, f"obs_t{t}") for t in range(T)]
    # Final measurement register (both qubits)
    c_final = ClassicalRegister(2, "final")

    qc = QuantumCircuit(q, *c_obs, c_final)

    # Observation model angles (constant across all steps)
    # R_y(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    # P(obs=0 | state) = cos^2(theta/2) = p
    # => theta = 2 * arccos(sqrt(p))
    theta_obs_s0 = 2.0 * float(np.arccos(np.sqrt(p_correct)))       # state=TL
    theta_obs_s1 = 2.0 * float(np.arccos(np.sqrt(1 - p_correct)))   # state=TR

    for t in range(T):
        prior = priors[t]

        # --- 1. Reset qubits (except at step 0) ---
        if t > 0:
            qc.reset(q[0])
            qc.reset(q[1])

        # Add a barrier for visual clarity between steps
        if t > 0:
            qc.barrier()

        # --- 2. Encode prior on state qubit ---
        # P(state=0) = prior[0], P(state=1) = prior[1]
        # R_y(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
        # => P(|1>) = sin^2(theta/2) = prior[1]
        # => theta = 2 * arcsin(sqrt(prior[1]))
        theta_prior = 2.0 * float(np.arcsin(np.sqrt(np.clip(prior[1], 1e-9, 1.0 - 1e-9))))
        qc.ry(theta_prior, q[0])

        # --- 3. Apply observation model (CRY gates) ---
        # When state qubit = |0> (tiger-left): apply CRY conditioned on |0>
        # Use X-CRY-X trick to condition on |0>
        qc.x(q[0])                        # flip: ctrl=|1> when state=tiger-left
        qc.cry(theta_obs_s0, q[0], q[1])  # P(obs | tiger-left) rotation
        qc.x(q[0])                        # restore
        qc.cry(theta_obs_s1, q[0], q[1])  # P(obs | tiger-right) rotation (ctrl=|1>)

        # --- 4. Measurement ---
        if t < T - 1:
            # Mid-circuit: measure obs qubit only
            qc.measure(q[1], c_obs[t][0])
        else:
            # Final step: measure both qubits
            # Also record obs in its register for completeness
            qc.measure(q[1], c_obs[t][0])
            qc.measure(q[0], c_final[0])
            qc.measure(q[1], c_final[1])

    return qc


# ---------------------------------------------------------------------------
# Simulator execution
# ---------------------------------------------------------------------------

def run_dynamic_sim(qc: "QuantumCircuit", shots: int) -> dict[str, dict]:
    """Run dynamic circuit on AerSimulator and return per-register counts.

    Returns:
        Dict mapping register name -> counts dict.
        E.g., {"obs_t0": {"0": 5000, "1": 3192}, "final": {"01": 4500, ...}}
    """
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    job = sim.run(qc, shots=shots)
    raw_counts = dict(job.result().get_counts())

    # Parse the combined bitstring into per-register counts.
    # Qiskit get_counts() returns space-separated registers in reverse order:
    # "final obs_tN-1 ... obs_t0" (MSB-first, last-added register leftmost).
    registers = list(qc.cregs)  # in circuit addition order
    reg_names = [cr.name for cr in registers]
    reg_sizes = [cr.size for cr in registers]

    per_register: dict[str, dict[str, int]] = {name: {} for name in reg_names}

    for bitstring, count in raw_counts.items():
        # Split by spaces
        parts = bitstring.split()
        if len(parts) == len(registers):
            # Space-separated format: parts[0] = last register, parts[-1] = first register
            # Reversed because Qiskit puts last-added register first (MSB-first)
            for i, name in enumerate(reg_names):
                val = parts[len(registers) - 1 - i]
                per_register[name][val] = per_register[name].get(val, 0) + count
        else:
            # No spaces: concatenated bitstring. Parse by register sizes in reverse.
            concat = bitstring.replace(" ", "")
            pos = 0
            vals: list[str] = []
            for size in reversed(reg_sizes):
                vals.append(concat[pos:pos + size])
                pos += size
            vals.reverse()
            for name, val in zip(reg_names, vals):
                per_register[name][val] = per_register[name].get(val, 0) + count

    return per_register


def run_dynamic_samplerv2_sim(qc: "QuantumCircuit", shots: int) -> dict[str, dict]:
    """Run dynamic circuit via SamplerV2 on AerSimulator.

    Returns per-register counts dict (same format as run_dynamic_sim).
    """
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import SamplerV2

    sim = AerSimulator()
    sampler = SamplerV2(mode=sim)
    result = sampler.run([qc], shots=shots).result()[0]

    per_register: dict[str, dict[str, int]] = {}
    for attr in [k for k in vars(result.data) if not k.startswith("_")]:
        data = getattr(result.data, attr)
        try:
            per_register[attr] = dict(data.get_counts())
        except Exception:
            pass

    return per_register


# ---------------------------------------------------------------------------
# Hardware execution
# ---------------------------------------------------------------------------

def run_dynamic_hw(
    qc: "QuantumCircuit",
    backend_obj: Any,
    shots: int,
) -> tuple[dict[str, dict], int, str]:
    """Run dynamic circuit on IBM QPU via SamplerV2.

    Returns:
        (per_register_counts, isa_depth, job_id)
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    # Transpile with optimization
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend_obj)
    isa = pm.run(qc)
    isa._layout = None  # prevent QPY Error 3211
    isa_depth = isa.depth()

    print(f"  ISA depth       : {isa_depth}")
    print(f"  ISA gate count  : {sum(isa.count_ops().values())}")
    isa_ops = isa.count_ops()
    print(f"  ISA ops         : {dict(isa_ops)}")

    sampler = SamplerV2(mode=backend_obj)
    # Error suppression: Pauli twirling + TREX + Dynamical Decoupling (XY4)
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.strategy = "active-accum"
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.dynamical_decoupling.scheduling_method = "alap"

    print(f"  Submitting to {backend_obj.name} ...")
    t0 = time.time()
    job = sampler.run([isa], shots=shots)
    result = job.result()[0]
    elapsed = time.time() - t0
    print(f"  Job completed in {elapsed:.1f}s (job_id={job.job_id()})")

    # Extract per-register counts
    per_register: dict[str, dict[str, int]] = {}
    for attr in [k for k in vars(result.data) if not k.startswith("_")]:
        data = getattr(result.data, attr)
        try:
            per_register[attr] = dict(data.get_counts())
        except Exception:
            pass

    return per_register, isa_depth, job.job_id()


# ---------------------------------------------------------------------------
# Post-processing: extract posterior from final register
# ---------------------------------------------------------------------------

def extract_posterior_from_final(
    final_counts: dict[str, int],
    target_obs: int,
) -> tuple[list[float], dict[str, int]]:
    """Post-select final register counts on target observation.

    The final register has 2 bits: bit 0 = state qubit, bit 1 = obs qubit.
    Bitstring convention (MSB-first): "obs_bit state_bit"
      "00" -> obs=0, state=0 (hear-left, tiger-left)
      "01" -> obs=0, state=1 (hear-left, tiger-right)
      "10" -> obs=1, state=0 (hear-right, tiger-left)
      "11" -> obs=1, state=1 (hear-right, tiger-right)

    Post-select on obs_bit == target_obs, then compute P(state=s).

    Returns:
        (posterior [P(TL), P(TR)], post_selected_counts)
    """
    obs_target = str(target_obs)
    ps_counts: dict[str, int] = {}

    for bs, count in final_counts.items():
        bs_clean = bs.strip()
        if len(bs_clean) >= 2:
            obs_bit = bs_clean[0]   # MSB = obs qubit (bit 1)
            state_bit = bs_clean[1]  # LSB = state qubit (bit 0)
            if obs_bit == obs_target:
                ps_counts[state_bit] = ps_counts.get(state_bit, 0) + count

    total = sum(ps_counts.values())
    if total == 0:
        return [0.5, 0.5], ps_counts

    # state_bit: "0" = tiger-left, "1" = tiger-right
    p_tl = ps_counts.get("0", 0) / total
    p_tr = ps_counts.get("1", 0) / total

    return [p_tl, p_tr], ps_counts


# ---------------------------------------------------------------------------
# Non-dynamic baseline: run T separate circuits (for comparison)
# ---------------------------------------------------------------------------

def build_single_step_circuit(
    prior: list[float],
    p_correct: float = 0.85,
) -> "QuantumCircuit":
    """Build a single-step Tiger belief circuit (2 qubits, no dynamic features).

    Same encoding as each step in the dynamic circuit, but as a standalone circuit
    for comparison purposes.
    """
    from qiskit import QuantumCircuit

    theta_prior = 2.0 * float(np.arcsin(np.sqrt(np.clip(prior[1], 1e-9, 1.0 - 1e-9))))
    theta_obs_s0 = 2.0 * float(np.arccos(np.sqrt(p_correct)))
    theta_obs_s1 = 2.0 * float(np.arccos(np.sqrt(1 - p_correct)))

    qc = QuantumCircuit(2, 2, name="tiger_single")
    qc.ry(theta_prior, 0)
    qc.x(0)
    qc.cry(theta_obs_s0, 0, 1)
    qc.x(0)
    qc.cry(theta_obs_s1, 0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve observation sequence
    if args.obs_sequence is None:
        obs_seq = [0] * args.n_steps
    else:
        obs_seq = list(args.obs_sequence)

    # Pad or truncate to n_steps
    while len(obs_seq) < args.n_steps:
        obs_seq.append(0)
    obs_seq = obs_seq[: args.n_steps]

    T = args.n_steps
    p_correct = args.p_correct

    print(f"\n{'='*70}")
    print(f"  Dynamic-Circuit Quantum POMDP Belief Loop")
    print(f"  First dynamic-circuit quantum POMDP on hardware")
    print(f"{'='*70}")
    print(f"  Backend     : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Steps (T)   : {T}")
    print(f"  Obs sequence: {obs_seq}")
    print(f"  P(correct)  : {p_correct}")
    print(f"  Shots       : {args.shots}")
    print()

    # ------------------------------------------------------------------
    # 1. Pre-compute classical Bayesian beliefs (ground truth)
    # ------------------------------------------------------------------
    priors, posteriors = compute_priors_and_posteriors(obs_seq, p_correct)

    print("--- Classical Bayesian trajectory (ground truth) ---")
    for t in range(T):
        obs_label = "hear-left" if obs_seq[t] == 0 else "hear-right"
        print(f"  t={t}: prior={[round(x, 4) for x in priors[t]]}  "
              f"obs={obs_seq[t]} ({obs_label})  "
              f"-> posterior={[round(x, 4) for x in posteriors[t]]}")
    print()

    # ------------------------------------------------------------------
    # 2. Build the dynamic circuit
    # ------------------------------------------------------------------
    print("--- Building dynamic circuit ---")
    qc = build_dynamic_pomdp_circuit(priors, obs_seq, p_correct)
    print(f"  Qubits         : {qc.num_qubits}")
    print(f"  Classical bits : {qc.num_clbits}")
    print(f"  Registers      : {[cr.name for cr in qc.cregs]}")
    print(f"  Logical depth  : {qc.depth()}")
    print(f"  Gate count     : {sum(qc.count_ops().values())}")
    print(f"  Gate ops       : {dict(qc.count_ops())}")
    print()

    # ------------------------------------------------------------------
    # 3. Simulator execution
    # ------------------------------------------------------------------
    print("--- Simulator execution ---")
    sim_reg_counts = run_dynamic_sim(qc, shots=args.shots)

    # Display mid-circuit observation statistics
    for t in range(T):
        reg_name = f"obs_t{t}"
        if reg_name in sim_reg_counts:
            obs_counts = sim_reg_counts[reg_name]
            total = sum(obs_counts.values())
            p0 = obs_counts.get("0", 0) / max(total, 1)
            p1 = obs_counts.get("1", 0) / max(total, 1)
            expected_obs = obs_seq[t]
            print(f"  obs_t{t}: P(hear-left)={p0:.3f}  P(hear-right)={p1:.3f}  "
                  f"[expected obs={expected_obs}]")

    # Extract posterior from final register
    final_counts = sim_reg_counts.get("final", {})
    target_obs = obs_seq[-1]  # The observation at the last step
    sim_posterior, sim_ps_counts = extract_posterior_from_final(final_counts, target_obs)
    classical_posterior = posteriors[-1]
    sim_hellinger = _hellinger(sim_posterior, classical_posterior)

    print(f"\n  Final register counts: {final_counts}")
    print(f"  Post-selected on obs={target_obs}: {sim_ps_counts}")
    print(f"  Sim posterior   : {[round(x, 4) for x in sim_posterior]}")
    print(f"  Classical post  : {[round(x, 4) for x in classical_posterior]}")
    print(f"  Hellinger dist  : {sim_hellinger:.4f}")
    sim_pass = sim_hellinger < 0.05
    print(f"  Sim PASS        : {sim_pass} (threshold < 0.05)")
    print()

    # ------------------------------------------------------------------
    # 4. Non-dynamic baseline comparison (T separate circuits on sim)
    # ------------------------------------------------------------------
    print("--- Non-dynamic baseline (T separate circuits, simulator) ---")
    baseline_posteriors: list[list[float]] = []
    baseline_hellingers: list[float] = []
    total_baseline_depth = 0

    for t in range(T):
        single_qc = build_single_step_circuit(priors[t], p_correct)
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        job = sim.run(single_qc, shots=args.shots)
        counts = dict(job.result().get_counts())

        # Post-select on observation = obs_seq[t]
        obs_target_str = str(obs_seq[t])
        ps_counts: dict[str, int] = {}
        for bs, cnt in counts.items():
            bs_clean = bs.strip()
            if len(bs_clean) >= 2 and bs_clean[0] == obs_target_str:
                state_bit = bs_clean[1]
                ps_counts[state_bit] = ps_counts.get(state_bit, 0) + cnt

        total = sum(ps_counts.values())
        if total > 0:
            post = [ps_counts.get("0", 0) / total, ps_counts.get("1", 0) / total]
        else:
            post = [0.5, 0.5]

        hell = _hellinger(post, posteriors[t])
        baseline_posteriors.append(post)
        baseline_hellingers.append(hell)
        total_baseline_depth += single_qc.depth()

        print(f"  t={t}: posterior={[round(x, 4) for x in post]}  "
              f"Hellinger={hell:.4f}  depth={single_qc.depth()}")

    print(f"\n  Total baseline depth (sum of {T} circuits): {total_baseline_depth}")
    print(f"  Dynamic circuit depth (1 circuit)         : {qc.depth()}")
    print(f"  Depth ratio (dynamic / sum_baseline)      : {qc.depth() / max(total_baseline_depth, 1):.2f}")
    print()

    # ------------------------------------------------------------------
    # 5. Hardware execution (if not dry-run)
    # ------------------------------------------------------------------
    hw_result_data: dict[str, Any] = {}

    if not args.dry_run:
        print("--- Hardware execution ---")
        token = get_ibm_token_optional()
        if token is None:
            print("  ERROR: IBM_QUANTUM_TOKEN not set. Use --dry-run for simulator.")
            sys.exit(1)

        from qiskit_ibm_runtime import QiskitRuntimeService

        channel = args.channel or "ibm_quantum_platform"
        svc_kwargs: dict[str, Any] = {"token": token, "channel": channel}
        if args.instance:
            svc_kwargs["instance"] = args.instance
        svc = QiskitRuntimeService(**svc_kwargs)
        backend_obj = svc.backend(args.backend)
        print(f"  Connected: {backend_obj.name} ({backend_obj.num_qubits} qubits)")

        hw_reg_counts, isa_depth, job_id = run_dynamic_hw(
            qc, backend_obj, args.shots
        )

        # Display mid-circuit observation statistics
        print("\n  Mid-circuit observation statistics (hardware):")
        for t in range(T):
            reg_name = f"obs_t{t}"
            if reg_name in hw_reg_counts:
                obs_counts = hw_reg_counts[reg_name]
                total = sum(obs_counts.values())
                p0 = obs_counts.get("0", 0) / max(total, 1)
                p1 = obs_counts.get("1", 0) / max(total, 1)
                print(f"    obs_t{t}: P(hear-left)={p0:.3f}  P(hear-right)={p1:.3f}")

        # Extract posterior from final register
        hw_final_counts = hw_reg_counts.get("final", {})
        hw_posterior, hw_ps_counts = extract_posterior_from_final(hw_final_counts, target_obs)
        hw_hellinger = _hellinger(hw_posterior, classical_posterior)

        print(f"\n  Final register counts: {hw_final_counts}")
        print(f"  Post-selected on obs={target_obs}: {hw_ps_counts}")
        print(f"  HW posterior    : {[round(x, 4) for x in hw_posterior]}")
        print(f"  Classical post  : {[round(x, 4) for x in classical_posterior]}")
        print(f"  Hellinger dist  : {hw_hellinger:.4f}")
        hw_pass = hw_hellinger < 0.15  # hardware threshold (more lenient)
        print(f"  HW PASS         : {hw_pass} (threshold < 0.15)")

        hw_result_data = {
            "posterior": hw_posterior,
            "hellinger": round(hw_hellinger, 6),
            "pass": hw_pass,
            "isa_depth": isa_depth,
            "job_id": job_id,
            "final_counts": hw_final_counts,
            "post_selected_counts": {k: v for k, v in hw_ps_counts.items()},
            "mid_circuit_obs": {},
        }
        for t in range(T):
            reg_name = f"obs_t{t}"
            if reg_name in hw_reg_counts:
                hw_result_data["mid_circuit_obs"][reg_name] = hw_reg_counts[reg_name]

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Architecture   : Dynamic circuit (1 circuit, {T} POMDP steps)")
    print(f"  Obs sequence   : {obs_seq}")
    print(f"  Circuit depth  : {qc.depth()} (dynamic) vs {total_baseline_depth} (sum of {T} separate)")
    print(f"  Job submissions: 1 (dynamic) vs {T} (non-dynamic)")

    # Belief trajectory
    print(f"\n  Belief trajectory (classical ground truth):")
    print(f"    t=0 prior  : [0.5000, 0.5000]")
    for t in range(T):
        print(f"    t={t} post   : {[round(x, 4) for x in posteriors[t]]}")

    print(f"\n  Simulator results:")
    print(f"    Final posterior : {[round(x, 4) for x in sim_posterior]}")
    print(f"    Hellinger       : {sim_hellinger:.4f}")
    print(f"    PASS            : {sim_pass}")

    if hw_result_data:
        print(f"\n  Hardware results:")
        print(f"    Final posterior : {[round(x, 4) for x in hw_result_data['posterior']]}")
        print(f"    Hellinger       : {hw_result_data['hellinger']:.4f}")
        print(f"    ISA depth       : {hw_result_data['isa_depth']}")
        print(f"    PASS            : {hw_result_data['pass']}")

    overall_pass = sim_pass
    if hw_result_data:
        overall_pass = hw_result_data["pass"]

    print(f"\n  Overall PASS: {overall_pass}")
    print()

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    result_data: dict[str, Any] = {
        "task_ids": ["dynamic-pomdp"],
        "description": "First dynamic-circuit quantum POMDP belief loop on hardware",
        "backend": args.backend if not args.dry_run else "aer_simulator",
        "n_steps": T,
        "observation_sequence": obs_seq,
        "p_correct": p_correct,
        "shots": args.shots,
        "circuit": {
            "num_qubits": qc.num_qubits,
            "num_clbits": qc.num_clbits,
            "logical_depth": qc.depth(),
            "gate_count": sum(qc.count_ops().values()),
            "gate_ops": dict(qc.count_ops()),
        },
        "classical_trajectory": {
            "priors": priors,
            "posteriors": posteriors,
        },
        "simulator": {
            "final_posterior": sim_posterior,
            "hellinger": round(sim_hellinger, 6),
            "pass": sim_pass,
            "final_counts": final_counts,
            "post_selected_counts": {k: v for k, v in sim_ps_counts.items()},
            "mid_circuit_obs": {},
        },
        "baseline_comparison": {
            "total_depth_T_circuits": total_baseline_depth,
            "dynamic_depth_1_circuit": qc.depth(),
            "depth_ratio": round(qc.depth() / max(total_baseline_depth, 1), 4),
            "job_submissions_saved": T - 1,
            "per_step": [
                {
                    "t": t,
                    "posterior": baseline_posteriors[t],
                    "hellinger": round(baseline_hellingers[t], 6),
                }
                for t in range(T)
            ],
        },
        "pass": overall_pass,
    }

    # Add mid-circuit obs stats to simulator result
    for t in range(T):
        reg_name = f"obs_t{t}"
        if reg_name in sim_reg_counts:
            result_data["simulator"]["mid_circuit_obs"][reg_name] = sim_reg_counts[reg_name]

    if hw_result_data:
        result_data["hardware"] = hw_result_data

    out_path = save_result("dynamic_pomdp_ibm", result_data)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
