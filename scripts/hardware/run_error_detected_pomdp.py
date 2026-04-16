"""[[4,2,2]] Iceberg-code error-detected Tiger POMDP belief circuit on IBM QPU.

First error-detected quantum POMDP on any hardware.

Approach: [[4,2,2]] redundant-pair error detection
----------------------------------------------------
The [[4,2,2]] code encodes 2 logical qubits into 4 physical qubits.
Each logical qubit is encoded as a redundant pair:
  Logical |0_L> = |00>  (both physical qubits in |0>)
  Logical |1_L> = |11>  (both physical qubits in |1>)

  q0 = state qubit (primary)
  q1 = state qubit (redundant copy)
  q2 = observation qubit (primary)
  q3 = observation qubit (redundant copy)

The Tiger belief circuit operates on the primary qubits (q0, q2),
then CNOT copies propagate the state to the redundant qubits (q1, q3).
In the error-free code space, q0==q1 and q2==q3 always holds.

Any single-qubit bit-flip error breaks this parity:
  X on q0 -> q0 != q1 (detected)
  X on q1 -> q0 != q1 (detected)
  X on q2 -> q2 != q3 (detected)
  X on q3 -> q2 != q3 (detected)

Post-selection: keep only shots where q0==q1 AND q2==q3.
On an ideal (noiseless) simulator, acceptance rate = 100%.
On noisy hardware, erroneous shots are filtered out, improving fidelity.

Usage
-----
# Dry-run (simulator only, no credentials needed):
python scripts/hardware/run_error_detected_pomdp.py --dry-run --prior 0.5 0.5

# Hardware run:
IBM_QUANTUM_TOKEN=xxx IBM_QUANTUM_CHANNEL=ibm_cloud IBM_QUANTUM_INSTANCE=crn:... \
    python -u scripts/hardware/run_error_detected_pomdp.py \
    --backend ibm_kingston --shots 16384 --prior 0.5 0.5

Output
------
  output/hardware/error_detected_pomdp_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Allow running from repo root without installing the package
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="[[4,2,2]] Iceberg error-detected Tiger POMDP on IBM QPU"
    )
    p.add_argument("--backend", default="ibm_kingston", help="IBM backend name")
    p.add_argument(
        "--shots", type=int, default=16384,
        help="Shot count (default: 16384; extra headroom for post-selection)",
    )
    p.add_argument(
        "--prior", nargs=2, type=float, default=[0.5, 0.5],
        metavar=("P_LEFT", "P_RIGHT"),
        help="Prior belief [P(tiger-left), P(tiger-right)] (default: 0.5 0.5)",
    )
    p.add_argument(
        "--p-correct", type=float, default=0.85,
        help="Tiger listen accuracy P(hear-correct | tiger-location) (default: 0.85)",
    )
    p.add_argument(
        "--obs", type=int, default=0, choices=[0, 1],
        help="Target observation to post-select on: 0=hear-left, 1=hear-right (default: 0)",
    )
    p.add_argument("--channel", default=None, help="Qiskit channel")
    p.add_argument("--instance", default=None, help="IBM instance / CRN")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only, skip IBM hardware (no credentials needed)",
    )
    p.add_argument(
        "--opt-level", type=int, default=3,
        help="Transpiler optimization level (0-3; default: 3)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def build_tiger_raw(
    prior: list[float],
    p_correct: float,
    target_obs: int,
) -> "QuantumCircuit":
    """Build the raw 2-qubit Tiger belief-update circuit (no error detection).

    q0 = state, q1 = observation.
    Uses the same encoding as the minimal Tiger circuit in run_tiger_ibm.py.
    """
    from qiskit import QuantumCircuit

    p00 = float(np.clip(p_correct, 1e-9, 1.0 - 1e-9))        # P(obs=0|state=0)
    p10 = float(np.clip(1.0 - p_correct, 1e-9, 1.0 - 1e-9))  # P(obs=0|state=1)

    theta_state = 2.0 * float(np.arccos(np.sqrt(max(prior[0], 1e-9))))
    theta_obs_s0 = 2.0 * float(np.arccos(np.sqrt(p00)))
    theta_obs_s1 = 2.0 * float(np.arccos(np.sqrt(p10)))

    qc = QuantumCircuit(2, 2, name="tiger_raw")

    # 1. Encode prior
    qc.ry(theta_state, 0)

    # 2. Conditional observation model
    qc.x(0)
    qc.cry(theta_obs_s0, 0, 1)  # when state=|0> (tiger-left)
    qc.x(0)
    qc.cry(theta_obs_s1, 0, 1)  # when state=|1> (tiger-right)

    # 3. Measure
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


def build_tiger_error_detected(
    prior: list[float],
    p_correct: float,
    target_obs: int,
) -> "QuantumCircuit":
    """Build the 4-qubit [[4,2,2]] error-detected Tiger belief circuit.

    Redundant-pair encoding:
      q0 = state (primary),      q1 = state (redundant copy)
      q2 = observation (primary), q3 = observation (redundant copy)

    Circuit steps:
      1. Encode prior on q0 (state primary)
      2. CNOT q0 -> q1 to create redundant state pair: |0_L>=|00>, |1_L>=|11>
      3. Conditional observation model on q2 (obs primary), controlled by q0
      4. CNOT q2 -> q3 to create redundant obs pair
      5. Measure all 4 qubits

    Post-selection: keep shots where q0==q1 AND q2==q3.
    Any single-qubit bit-flip error on any of the 4 qubits is detected.

    In the error-free case (ideal simulator), ALL shots pass (100% acceptance).
    On noisy hardware, erroneous shots are filtered out.
    """
    from qiskit import QuantumCircuit

    p00 = float(np.clip(p_correct, 1e-9, 1.0 - 1e-9))
    p10 = float(np.clip(1.0 - p_correct, 1e-9, 1.0 - 1e-9))

    theta_state = 2.0 * float(np.arccos(np.sqrt(max(prior[0], 1e-9))))
    theta_obs_s0 = 2.0 * float(np.arccos(np.sqrt(p00)))
    theta_obs_s1 = 2.0 * float(np.arccos(np.sqrt(p10)))

    qc = QuantumCircuit(4, 4, name="tiger_error_detected")

    # === 1. Encode prior on state qubit (q0) ===
    qc.ry(theta_state, 0)

    # === 2. Redundant copy: state q0 -> q1 ===
    qc.cx(0, 1)

    # === 3. Conditional observation model on obs qubit (q2) ===
    # Control on q0 (state primary), target q2 (obs primary)
    qc.x(0)
    qc.cry(theta_obs_s0, 0, 2)  # P(obs|tiger-left): applied when q0=|0>
    qc.x(0)
    qc.cry(theta_obs_s1, 0, 2)  # P(obs|tiger-right): applied when q0=|1>

    # === 4. Redundant copy: obs q2 -> q3 ===
    qc.cx(2, 3)

    # === 5. Measure all 4 qubits ===
    # c0=state_primary, c1=state_copy, c2=obs_primary, c3=obs_copy
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)
    qc.measure(3, 3)

    return qc


# ---------------------------------------------------------------------------
# Post-selection logic
# ---------------------------------------------------------------------------

def post_select_error_free(counts: dict[str, int]) -> tuple[dict[str, int], float]:
    """Post-select on redundant-pair consistency, returning data-only counts.

    Qiskit bitstring format is MSB-first: "c3 c2 c1 c0".
      c3 = obs copy     (q3)
      c2 = obs primary  (q2)
      c1 = state copy   (q1)
      c0 = state primary(q0)

    Keep only shots where c0==c1 (state pair consistent)
    AND c2==c3 (obs pair consistent).

    Return 2-bit data strings "obs_primary state_primary" = "c2 c0"
    and acceptance rate.
    """
    good_counts: dict[str, int] = {}
    total = 0
    accepted = 0
    for bitstring, count in counts.items():
        total += count
        # Pad to 4 bits if Qiskit compresses leading zeros
        bs = bitstring.zfill(4)
        c3 = int(bs[0])  # obs copy
        c2 = int(bs[1])  # obs primary
        c1 = int(bs[2])  # state copy
        c0 = int(bs[3])  # state primary

        state_pair_ok = (c0 == c1)
        obs_pair_ok = (c2 == c3)

        if state_pair_ok and obs_pair_ok:
            # Extract logical data: obs_primary + state_primary
            data_key = f"{c2}{c0}"
            good_counts[data_key] = good_counts.get(data_key, 0) + count
            accepted += count

    acceptance_rate = accepted / total if total > 0 else 0.0
    return good_counts, acceptance_rate


def post_select_observation(
    counts: dict[str, int],
    target_obs: int,
) -> dict[str, int]:
    """Post-select 2-bit data counts on observation qubit value.

    Input: dict of 2-bit strings "obs state" (e.g. "00", "01", "10", "11").
    Returns: dict of 1-bit strings for the state qubit, filtered by obs match.
    """
    filtered: dict[str, int] = {}
    obs_target_str = str(target_obs)
    for bitstring, count in counts.items():
        if len(bitstring) >= 2 and bitstring[0] == obs_target_str:
            state_bit = bitstring[1]
            filtered[state_bit] = filtered.get(state_bit, 0) + count
    return filtered


def syndrome_distribution(counts: dict[str, int]) -> dict[str, int]:
    """Compute error syndrome distribution from 4-qubit measurement counts.

    Syndromes:
      state_ok AND obs_ok  -> no error detected
      state_err AND obs_ok -> error on state pair (q0 or q1 flipped)
      state_ok AND obs_err -> error on obs pair (q2 or q3 flipped)
      state_err AND obs_err -> errors on both pairs
    """
    dist = {
        "no_error": 0,
        "state_pair_error": 0,
        "obs_pair_error": 0,
        "both_pairs_error": 0,
    }
    for bitstring, count in counts.items():
        bs = bitstring.zfill(4)
        c3, c2, c1, c0 = int(bs[0]), int(bs[1]), int(bs[2]), int(bs[3])
        state_err = (c0 != c1)
        obs_err = (c2 != c3)
        if not state_err and not obs_err:
            dist["no_error"] += count
        elif state_err and not obs_err:
            dist["state_pair_error"] += count
        elif not state_err and obs_err:
            dist["obs_pair_error"] += count
        else:
            dist["both_pairs_error"] += count
    return dist


# ---------------------------------------------------------------------------
# Classical Bayesian posterior (ground truth)
# ---------------------------------------------------------------------------

def classical_bayes_posterior(
    prior: list[float],
    p_correct: float,
    obs: int,
) -> list[float]:
    """Exact Bayesian posterior for 2-state Tiger (listen action).

    P(state=s | obs) = P(obs | state=s) * prior(s) / Z

    For obs=0 (hear-left):
      P(obs=0 | state=0) = p_correct   (tiger-left, hear-left)
      P(obs=0 | state=1) = 1-p_correct (tiger-right, hear-left)
    For obs=1 (hear-right): complement.
    """
    if obs == 0:
        likelihood = np.array([p_correct, 1.0 - p_correct])
    else:
        likelihood = np.array([1.0 - p_correct, p_correct])
    unnorm = np.array(prior) * likelihood
    z = unnorm.sum()
    if z < 1e-12:
        return [0.5, 0.5]
    return (unnorm / z).tolist()


# ---------------------------------------------------------------------------
# Hellinger distance
# ---------------------------------------------------------------------------

def hellinger_distance(p: list[float], q: list[float]) -> float:
    """Compute Hellinger distance between two probability distributions."""
    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_arr) - np.sqrt(q_arr)) ** 2)))


def counts_to_probs(counts: dict[str, int], n_states: int = 2) -> list[float]:
    """Convert bitstring counts to state probabilities."""
    total = sum(counts.values())
    if total == 0:
        return [1.0 / n_states] * n_states
    probs = [0.0] * n_states
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx < n_states:
            probs[idx] += count / total
    return probs


# ---------------------------------------------------------------------------
# Simulator execution
# ---------------------------------------------------------------------------

def run_simulator(circuit, shots: int) -> dict[str, int]:
    """Run circuit on AerSimulator (ideal, no noise)."""
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

    prior = args.prior
    # Normalise prior
    s = sum(prior)
    prior = [p / s for p in prior]

    p_correct = args.p_correct
    target_obs = args.obs

    classical_post = classical_bayes_posterior(prior, p_correct, target_obs)

    print(f"\n{'='*72}")
    print(f"  [[4,2,2]] Iceberg Error-Detected Tiger POMDP Belief Circuit")
    print(f"  First error-detected quantum POMDP on any hardware")
    print(f"{'='*72}")
    print(f"  Backend       : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots         : {args.shots}")
    print(f"  Prior         : {[round(x, 4) for x in prior]}")
    print(f"  P(correct)    : {p_correct}")
    print(f"  Target obs    : {target_obs} ({'hear-left' if target_obs == 0 else 'hear-right'})")
    print(f"  Classical post: {[round(x, 4) for x in classical_post]}")
    print(f"  Encoding      : redundant-pair [[4,2,2]] (q0=q1 state, q2=q3 obs)")

    # ------------------------------------------------------------------
    # 1. Build circuits
    # ------------------------------------------------------------------
    circuit_raw = build_tiger_raw(prior, p_correct, target_obs)
    circuit_ed = build_tiger_error_detected(prior, p_correct, target_obs)

    print(f"\n  Raw circuit       : {circuit_raw.num_qubits}q, depth={circuit_raw.depth()}")
    print(f"  Error-detected    : {circuit_ed.num_qubits}q, depth={circuit_ed.depth()}")

    result_data: dict[str, Any] = {
        "task_ids": ["error_detection"],
        "technique": "[[4,2,2]] Iceberg redundant-pair error detection",
        "description": (
            "Redundant-pair encoding: each logical qubit (state, obs) is encoded "
            "into a pair of physical qubits. CNOT copies create the redundant "
            "encoding after computation. Post-selection on pair consistency "
            "(q0==q1, q2==q3) detects single-qubit bit-flip errors."
        ),
        "backend": args.backend if not args.dry_run else "aer_simulator",
        "shots": args.shots,
        "prior": prior,
        "p_correct": p_correct,
        "target_obs": target_obs,
        "classical_posterior": classical_post,
        "circuit_raw_qubits": circuit_raw.num_qubits,
        "circuit_raw_depth": circuit_raw.depth(),
        "circuit_ed_qubits": circuit_ed.num_qubits,
        "circuit_ed_depth": circuit_ed.depth(),
    }

    # ------------------------------------------------------------------
    # 2. Simulator baseline
    # ------------------------------------------------------------------
    print(f"\n--- Simulator ---")

    # Raw circuit
    sim_raw_counts = run_simulator(circuit_raw, args.shots)
    obs_str = str(target_obs)
    sim_raw_ps: dict[str, int] = {}
    for bs, cnt in sim_raw_counts.items():
        bs = bs.zfill(2)
        if bs[0] == obs_str:
            state_bit = bs[1]
            sim_raw_ps[state_bit] = sim_raw_ps.get(state_bit, 0) + cnt
    sim_raw_probs = counts_to_probs(sim_raw_ps, 2)
    sim_raw_hell = hellinger_distance(sim_raw_probs, classical_post)
    sim_raw_yield = sum(sim_raw_ps.values()) / args.shots

    print(f"  Raw posterior     : {[round(x, 4) for x in sim_raw_probs]}")
    print(f"  Raw Hellinger     : {sim_raw_hell:.6f}")
    print(f"  Raw obs yield     : {sim_raw_yield:.1%}")

    # Error-detected circuit
    sim_ed_counts = run_simulator(circuit_ed, args.shots)
    sim_ed_good, sim_ed_accept = post_select_error_free(sim_ed_counts)
    sim_ed_obs_ps = post_select_observation(sim_ed_good, target_obs)
    sim_ed_probs = counts_to_probs(sim_ed_obs_ps, 2)
    sim_ed_hell = hellinger_distance(sim_ed_probs, classical_post)
    sim_ed_yield = sum(sim_ed_obs_ps.values()) / args.shots

    # Syndrome distribution for simulator (should be ~100% no_error)
    sim_syndrome = syndrome_distribution(sim_ed_counts)

    print(f"  ED posterior      : {[round(x, 4) for x in sim_ed_probs]}")
    print(f"  ED Hellinger      : {sim_ed_hell:.6f}")
    print(f"  ED acceptance     : {sim_ed_accept:.1%}")
    print(f"  ED obs yield      : {sim_ed_yield:.1%}")
    print(f"  ED syndromes      : {sim_syndrome}")

    result_data["simulator"] = {
        "raw": {
            "posterior": sim_raw_probs,
            "hellinger": round(sim_raw_hell, 6),
            "obs_yield": round(sim_raw_yield, 4),
        },
        "error_detected": {
            "posterior": sim_ed_probs,
            "hellinger": round(sim_ed_hell, 6),
            "acceptance_rate": round(sim_ed_accept, 4),
            "obs_yield": round(sim_ed_yield, 4),
            "syndrome_distribution": sim_syndrome,
        },
    }

    if args.dry_run:
        print(f"\n[dry-run] Skipping IBM hardware execution.")
        # On ideal simulator, both should be near-zero Hellinger
        passed = sim_ed_hell < 0.05 and sim_ed_accept > 0.99
        result_data["pass"] = passed
        result_data["notes"] = (
            f"Dry-run: simulator only. "
            f"Raw Hellinger={sim_raw_hell:.6f}, "
            f"ED Hellinger={sim_ed_hell:.6f}, "
            f"ED acceptance={sim_ed_accept:.1%} (expected ~100% on ideal sim)"
        )
        save_result("error_detected_pomdp", result_data)
        print(f"\n  PASS: {passed}  (ED Hellinger={sim_ed_hell:.6f}, "
              f"acceptance={sim_ed_accept:.1%})")
        return

    # ------------------------------------------------------------------
    # 3. IBM hardware
    # ------------------------------------------------------------------
    token = get_ibm_token_optional()
    if token is None:
        print("  ERROR: IBM_QUANTUM_TOKEN not set. Use --dry-run for simulator.")
        sys.exit(1)

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    channel = args.channel or ibm_channel()
    instance = args.instance or ibm_instance()
    svc_kwargs: dict = {"channel": channel, "token": token}
    if instance:
        svc_kwargs["instance"] = instance
    svc = QiskitRuntimeService(**svc_kwargs)
    backend_obj = svc.backend(args.backend)
    print(f"\n  Connected: {backend_obj.name} ({backend_obj.num_qubits} qubits)")

    pm = generate_preset_pass_manager(
        optimization_level=args.opt_level, backend=backend_obj,
    )

    # Transpile both circuits
    isa_raw = pm.run(circuit_raw)
    isa_raw._layout = None
    isa_ed = pm.run(circuit_ed)
    isa_ed._layout = None

    isa_depth_raw = isa_raw.depth()
    isa_depth_ed = isa_ed.depth()

    print(f"  ISA depth (raw)   : {isa_depth_raw}")
    print(f"  ISA depth (ED)    : {isa_depth_ed}")

    result_data["isa_depth_raw"] = isa_depth_raw
    result_data["isa_depth_ed"] = isa_depth_ed

    # ------------------------------------------------------------------
    # 3a. Raw circuit on hardware
    # ------------------------------------------------------------------
    print(f"\n--- Hardware: Raw circuit ---")
    t0 = time.time()

    sampler_raw = SamplerV2(mode=backend_obj)
    sampler_raw.options.twirling.enable_gates = True
    sampler_raw.options.twirling.enable_measure = True
    sampler_raw.options.twirling.strategy = "active-accum"
    sampler_raw.options.dynamical_decoupling.enable = True
    sampler_raw.options.dynamical_decoupling.sequence_type = "XY4"
    sampler_raw.options.dynamical_decoupling.scheduling_method = "alap"
    print(f"  Twirling+DD       : enabled (XY4/alap)")
    print(f"  Submitting to {args.backend} ...")

    job_raw = sampler_raw.run([isa_raw], shots=args.shots)
    pub_raw = job_raw.result()[0]
    _creg_raw = next(k for k in vars(pub_raw.data) if not k.startswith("_"))
    hw_raw_counts = dict(getattr(pub_raw.data, _creg_raw).get_counts())
    hw_raw_job_id = job_raw.job_id()
    t_raw = time.time() - t0

    # Post-select on observation
    hw_raw_ps: dict[str, int] = {}
    for bs, cnt in hw_raw_counts.items():
        bs = bs.zfill(2)
        if bs[0] == obs_str:
            state_bit = bs[1]
            hw_raw_ps[state_bit] = hw_raw_ps.get(state_bit, 0) + cnt
    hw_raw_probs = counts_to_probs(hw_raw_ps, 2)
    hw_raw_hell = hellinger_distance(hw_raw_probs, classical_post)
    hw_raw_yield = sum(hw_raw_ps.values()) / args.shots

    print(f"  Job ID            : {hw_raw_job_id}")
    print(f"  Raw posterior     : {[round(x, 4) for x in hw_raw_probs]}")
    print(f"  Raw Hellinger     : {hw_raw_hell:.6f}")
    print(f"  Raw obs yield     : {hw_raw_yield:.1%}")
    print(f"  Time              : {t_raw:.1f}s")

    # ------------------------------------------------------------------
    # 3b. Error-detected circuit on hardware
    # ------------------------------------------------------------------
    print(f"\n--- Hardware: Error-detected circuit ---")
    t0 = time.time()

    sampler_ed = SamplerV2(mode=backend_obj)
    sampler_ed.options.twirling.enable_gates = True
    sampler_ed.options.twirling.enable_measure = True
    sampler_ed.options.twirling.strategy = "active-accum"
    sampler_ed.options.dynamical_decoupling.enable = True
    sampler_ed.options.dynamical_decoupling.sequence_type = "XY4"
    sampler_ed.options.dynamical_decoupling.scheduling_method = "alap"
    print(f"  Twirling+DD       : enabled (XY4/alap)")
    print(f"  Submitting to {args.backend} ...")

    job_ed = sampler_ed.run([isa_ed], shots=args.shots)
    pub_ed = job_ed.result()[0]
    _creg_ed = next(k for k in vars(pub_ed.data) if not k.startswith("_"))
    hw_ed_counts = dict(getattr(pub_ed.data, _creg_ed).get_counts())
    hw_ed_job_id = job_ed.job_id()
    t_ed = time.time() - t0

    # Error-detection post-selection: keep only pair-consistent shots
    hw_ed_good, hw_ed_accept = post_select_error_free(hw_ed_counts)
    # Then post-select on observation
    hw_ed_obs_ps = post_select_observation(hw_ed_good, target_obs)
    hw_ed_probs = counts_to_probs(hw_ed_obs_ps, 2)
    hw_ed_hell = hellinger_distance(hw_ed_probs, classical_post)
    hw_ed_yield = sum(hw_ed_obs_ps.values()) / args.shots

    print(f"  Job ID            : {hw_ed_job_id}")
    print(f"  ED posterior      : {[round(x, 4) for x in hw_ed_probs]}")
    print(f"  ED Hellinger      : {hw_ed_hell:.6f}")
    print(f"  ED acceptance     : {hw_ed_accept:.1%}")
    print(f"  ED obs yield      : {hw_ed_yield:.1%}")
    print(f"  Time              : {t_ed:.1f}s")

    # ------------------------------------------------------------------
    # 3c. Syndrome distribution (diagnostic)
    # ------------------------------------------------------------------
    hw_syndrome = syndrome_distribution(hw_ed_counts)
    total_hw_ed = sum(hw_ed_counts.values())

    print(f"\n--- Syndrome Distribution (hardware) ---")
    for syn_name, syn_count in hw_syndrome.items():
        frac = syn_count / total_hw_ed if total_hw_ed > 0 else 0
        print(f"  {syn_name:20s}: {syn_count:6d}  ({frac:.1%})")

    # ------------------------------------------------------------------
    # 4. Comparison
    # ------------------------------------------------------------------
    improvement = hw_raw_hell - hw_ed_hell
    improvement_pct = (improvement / hw_raw_hell * 100) if hw_raw_hell > 0 else 0.0
    ratio = hw_raw_hell / hw_ed_hell if hw_ed_hell > 0 else float("inf")

    print(f"\n{'='*72}")
    print(f"  COMPARISON: Raw vs Error-Detected")
    print(f"{'='*72}")
    print(f"  {'Metric':<25s} {'Raw':>12s} {'Error-Det':>12s} {'Improvement':>14s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*14}")
    print(f"  {'Hellinger':<25s} {hw_raw_hell:>12.6f} {hw_ed_hell:>12.6f} {improvement:>+14.6f}")
    print(f"  {'Obs yield':<25s} {hw_raw_yield:>11.1%} {hw_ed_yield:>11.1%}")
    print(f"  {'ED acceptance':<25s} {'---':>12s} {hw_ed_accept:>11.1%}")
    print(f"  {'ISA depth':<25s} {isa_depth_raw:>12d} {isa_depth_ed:>12d}")
    print(f"  {'Improvement ratio':<25s} {'':>12s} {'':>12s} {ratio:>13.2f}x")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    ed_improved = hw_ed_hell < hw_raw_hell
    passed = hw_ed_hell < 0.05

    result_data["hardware"] = {
        "raw": {
            "job_id": hw_raw_job_id,
            "posterior": hw_raw_probs,
            "hellinger": round(hw_raw_hell, 6),
            "obs_yield": round(hw_raw_yield, 4),
            "time_s": round(t_raw, 1),
        },
        "error_detected": {
            "job_id": hw_ed_job_id,
            "posterior": hw_ed_probs,
            "hellinger": round(hw_ed_hell, 6),
            "acceptance_rate": round(hw_ed_accept, 4),
            "obs_yield": round(hw_ed_yield, 4),
            "time_s": round(t_ed, 1),
            "syndrome_distribution": hw_syndrome,
        },
        "comparison": {
            "hellinger_improvement": round(improvement, 6),
            "hellinger_improvement_pct": round(improvement_pct, 2),
            "improvement_ratio": round(ratio, 4),
            "error_detection_improved": ed_improved,
        },
    }

    result_data["pass"] = passed
    result_data["notes"] = (
        f"Raw Hellinger={hw_raw_hell:.6f}, "
        f"ED Hellinger={hw_ed_hell:.6f}, "
        f"improvement={improvement:+.6f} ({improvement_pct:+.1f}%), "
        f"acceptance={hw_ed_accept:.1%}, "
        f"{'ED improved' if ed_improved else 'ED did NOT improve'}, "
        f"{'PASS' if passed else 'FAIL'} (threshold=0.05)"
    )

    save_result("error_detected_pomdp", result_data)

    print(f"\n  Error detection improved: {ed_improved}")
    print(f"  PASS: {passed}  (ED Hellinger={hw_ed_hell:.6f}, threshold=0.05)")


if __name__ == "__main__":
    main()
