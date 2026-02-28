"""Grover amplitude amplification for Tiger POMDP belief oracle on IBM QPU.

Demonstrates that 1 Grover iteration boosts P(rare observation) from ~17% to ~93%
when the prior is concentrated (b=[0.97, 0.03]) and the target observation is rare
(hear-right with P≈0.17).  This directly validates the core quantum speedup claim:

    Classical POMCP: O(P(e)^{-1}) samples per belief node  (need ~6 samples)
    Grover-1 step:   P(e) amplified to ~93% → ~1 sample    (5.4× amplification)

Circuit overview (2 qubits: q0=state, q1=obs)
---------------------------------------------
  Baseline:  A → measure                                (ISA ~12)
  Grover-1:  A → S_f → A† → S₀ → A → measure          (ISA ~40–60)

where:
  A    = Tiger belief oracle (encodes prior + obs model via conditional R_y)
  S_f  = Phase flip on obs=target (Z on q1 for target=1)
  A†   = Inverse oracle
  S₀   = Reflection around |00⟩ (X₀X₁ · CZ · X₀X₁)

After 1 Grover step, P(obs=target) = sin²(3θ) where sin²(θ) = P_baseline.
  P_baseline = 0.171  →  θ = 0.419 rad  →  sin²(3θ) = sin²(1.257) ≈ 0.927

Usage
-----
# Dry run (no credentials):
python scripts/hardware/run_grover_belief_ibm.py --dry-run

# Hardware:
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_grover_belief_ibm.py \\
    --backend ibm_marrakesh --shots 8192

Output
------
  output/hardware/grover_belief_ibm_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Tiger POMDP parameters (fixed for this experiment)
# ---------------------------------------------------------------------------

P_OBS_GIVEN_STATE = [0.85, 0.15]  # P(hear-left | tiger-left/right)
PRIOR             = [0.97, 0.03]  # concentrated prior: tiger-left very likely
TARGET_OBS        = 1             # hear-right (rare under this prior)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grover amplitude amplification for Tiger POMDP on IBM QPU"
    )
    p.add_argument("--backend", default="ibm_marrakesh", help="IBM backend name")
    p.add_argument("--shots", type=int, default=8192, help="Shot count per circuit")
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel: ibm_quantum_platform or ibm_cloud",
    )
    p.add_argument(
        "--instance", default=None,
        help="IBM instance / CRN",
    )
    p.add_argument(
        "--opt-level", type=int, default=3,
        help="Transpiler optimization level (0–3; default: 3)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only, skip IBM hardware (no credentials needed)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def _tiger_angles(prior: list[float], p_obs_given_state: list[float]):
    """Return R_y angles for Tiger belief oracle.

    Uses R_y(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩, so:
        P(|0⟩) = cos²(θ/2) = p  →  θ = 2·arccos(√p)
    """
    p00 = float(np.clip(p_obs_given_state[0], 1e-9, 1.0 - 1e-9))
    p10 = float(np.clip(p_obs_given_state[1], 1e-9, 1.0 - 1e-9))
    theta_state  = 2.0 * float(np.arccos(np.sqrt(max(prior[0], 1e-9))))
    theta_obs_s0 = 2.0 * float(np.arccos(np.sqrt(p00)))
    theta_obs_s1 = 2.0 * float(np.arccos(np.sqrt(p10)))
    return theta_state, theta_obs_s0, theta_obs_s1


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def _oracle_gates(qc, theta_state, theta_obs_s0, theta_obs_s1) -> None:
    """Append Tiger belief oracle A (no measurement) in place."""
    qc.ry(theta_state, 0)                    # encode prior on state qubit
    qc.x(0)                                  # flip control for state=0 arm
    qc.cry(theta_obs_s0, 0, 1)               # P(obs|tiger-left) rotation
    qc.x(0)                                  # restore
    qc.cry(theta_obs_s1, 0, 1)               # P(obs|tiger-right) rotation


def _oracle_inverse_gates(qc, theta_state, theta_obs_s0, theta_obs_s1) -> None:
    """Append A† (inverse oracle) in place."""
    qc.cry(-theta_obs_s1, 0, 1)
    qc.x(0)
    qc.cry(-theta_obs_s0, 0, 1)
    qc.x(0)
    qc.ry(-theta_state, 0)


def _build_baseline_circuit(
    prior: list[float],
    p_obs_given_state: list[float],
) -> "QuantumCircuit":
    """Oracle A only — measure joint P(state, obs) without amplification."""
    from qiskit import QuantumCircuit

    angles = _tiger_angles(prior, p_obs_given_state)
    qc = QuantumCircuit(2, 2, name="baseline")
    _oracle_gates(qc, *angles)
    qc.measure([0, 1], [0, 1])
    return qc


def _build_grover1_circuit(
    prior: list[float],
    p_obs_given_state: list[float],
    target_obs: int,
) -> "QuantumCircuit":
    """1-step Grover amplification: A → S_f → A† → S₀ → A → measure.

    After this circuit, P(obs=target_obs) ≈ sin²(3θ) where sin²(θ) = P_baseline.

    Circuit layout (2 qubits):
        q0: state register  (|0⟩ = tiger-left, |1⟩ = tiger-right)
        q1: obs register    (|0⟩ = hear-left,  |1⟩ = hear-right)

    Gate count (pre-transpile):
        A:    5 gates (ry, x, cry, x, cry)
        S_f:  1 gate  (z on q1 for target=1; x-z-x for target=0)
        A†:   5 gates (cry, x, cry, x, ry — reversed + negated)
        S₀:   5 gates (x, x, cz, x, x)
        A:    5 gates
        M:    2 gates
        Total: ~23 gates → ISA depth ~45–65 on Heron R2
    """
    from qiskit import QuantumCircuit

    angles = _tiger_angles(prior, p_obs_given_state)
    qc = QuantumCircuit(2, 2, name="grover1")

    # Step 1: Oracle A (prepare |ψ⟩ = A|00⟩ = joint belief+obs state)
    _oracle_gates(qc, *angles)

    # Step 2: S_f — phase flip on "good" subspace (obs=target_obs)
    # For target_obs=1: Z on q1 flips phase of |q1=1⟩
    # For target_obs=0: X-Z-X on q1 flips phase of |q1=0⟩
    if target_obs == 1:
        qc.z(1)
    else:
        qc.x(1)
        qc.z(1)
        qc.x(1)

    # Step 3: A† (un-prepare)
    _oracle_inverse_gates(qc, *angles)

    # Step 4: S₀ = reflection around |00⟩ = -(I - 2|00⟩⟨00|)
    # Implemented as X₀X₁ · CZ · X₀X₁
    # (CZ flips phase of |11⟩; X maps |00⟩↔|11⟩; net: |00⟩ → -|00⟩)
    # Global phase is physically irrelevant for measurement statistics.
    qc.x(0)
    qc.x(1)
    qc.cz(0, 1)
    qc.x(0)
    qc.x(1)

    # Step 5: Oracle A again (final amplified state before measurement)
    _oracle_gates(qc, *angles)

    # Step 6: Measure both qubits
    qc.measure([0, 1], [0, 1])
    return qc


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _compute_obs_probability(counts: dict, target_obs: int) -> float:
    """Compute P(obs=target_obs) from raw 2-qubit measurement counts.

    Qiskit MSB-first bitstring ordering: "c1 c0" where c1=obs (q1), c0=state (q0).
    P(obs=target) = fraction of shots where leftmost bit == str(target_obs).
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    target_char = str(target_obs)
    good = sum(cnt for bs, cnt in counts.items() if bs and bs[0] == target_char)
    return good / total


def _theoretical_amplified_prob(
    prior: list[float],
    p_obs_given_state: list[float],
    target_obs: int,
    k: int = 1,
) -> tuple[float, float]:
    """Compute theoretical P(target) before and after k Grover steps.

    Returns:
        (p_baseline, p_amplified)
    """
    if target_obs == 0:
        p_e = sum(prior[s] * p_obs_given_state[s] for s in range(len(prior)))
    else:
        p_e = sum(prior[s] * (1.0 - p_obs_given_state[s]) for s in range(len(prior)))
    theta = np.arcsin(np.sqrt(p_e))
    p_amp = float(np.sin((2 * k + 1) * theta) ** 2)
    return float(p_e), p_amp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    p_baseline, p_amplified = _theoretical_amplified_prob(
        PRIOR, P_OBS_GIVEN_STATE, TARGET_OBS, k=1
    )
    amp_factor = p_amplified / max(p_baseline, 1e-9)

    print("\n=== Grover Amplitude Amplification — Tiger POMDP Belief Oracle ===")
    print(f"  Prior b:                  {PRIOR}")
    print(f"  P(hear-left | L/R):       {P_OBS_GIVEN_STATE}")
    print(f"  Target obs:               {TARGET_OBS}  "
          f"({'hear-right' if TARGET_OBS == 1 else 'hear-left'})")
    print(f"  Theoretical P_baseline:   {p_baseline:.4f}  (rare observation)")
    print(f"  Theoretical P_amplified:  {p_amplified:.4f}  (after 1 Grover step)")
    print(f"  Theoretical amplification:{amp_factor:.1f}×")
    print(f"  Backend:                  "
          f"{'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots:                    {args.shots}")

    baseline_circuit = _build_baseline_circuit(PRIOR, P_OBS_GIVEN_STATE)
    grover_circuit   = _build_grover1_circuit(PRIOR, P_OBS_GIVEN_STATE, TARGET_OBS)

    print(f"\n  Baseline circuit depth:   {baseline_circuit.depth()}")
    print(f"  Grover-1 circuit depth:   {grover_circuit.depth()}")

    result_data: dict = {
        "task_id": "grover_belief",
        "backend": args.backend if not args.dry_run else "aer_simulator",
        "shots": args.shots,
        "prior": PRIOR,
        "p_obs_given_state": P_OBS_GIVEN_STATE,
        "target_obs": TARGET_OBS,
        "theoretical_p_baseline": p_baseline,
        "theoretical_p_amplified": p_amplified,
        "theoretical_amplification_factor": amp_factor,
        "baseline_circuit_depth": baseline_circuit.depth(),
        "grover_circuit_depth": grover_circuit.depth(),
    }

    # ------------------------------------------------------------------
    # Simulator baseline
    # ------------------------------------------------------------------
    from quantum_common.backends.simulator import AerSimulatorBackend
    from quantum_common.backends.base import ExecutionRequest

    aer = AerSimulatorBackend()
    sim_bl = aer.execute(ExecutionRequest(circuits=[baseline_circuit], shots=args.shots))
    sim_gr = aer.execute(ExecutionRequest(circuits=[grover_circuit],  shots=args.shots))

    sim_p_bl = _compute_obs_probability(sim_bl.counts[0], TARGET_OBS)
    sim_p_gr = _compute_obs_probability(sim_gr.counts[0], TARGET_OBS)
    sim_amp  = sim_p_gr / max(sim_p_bl, 1e-9)

    print(f"\n  [Simulator] Baseline P(obs={TARGET_OBS}): {sim_p_bl:.4f}  "
          f"(theory: {p_baseline:.4f})")
    print(f"  [Simulator] Grover-1 P(obs={TARGET_OBS}): {sim_p_gr:.4f}  "
          f"(theory: {p_amplified:.4f})")
    print(f"  [Simulator] Amplification factor: {sim_amp:.1f}×")

    result_data["sim_p_baseline"] = sim_p_bl
    result_data["sim_p_amplified"] = sim_p_gr
    result_data["sim_amplification_factor"] = sim_amp

    if args.dry_run:
        passed = sim_p_gr > 0.5 and sim_amp > 3.0
        result_data["pass"] = passed
        result_data["notes"] = (
            f"Dry-run: sim amplification {sim_amp:.1f}× "
            f"({'PASS' if passed else 'FAIL'} — threshold 3×, P>0.5)"
        )
        save_result("grover_belief_ibm", result_data)
        print(f"\n  [dry-run] PASS: {passed}  "
              f"(amplification={sim_amp:.1f}×, P_amplified={sim_p_gr:.4f})")
        return

    # ------------------------------------------------------------------
    # IBM hardware execution
    # ------------------------------------------------------------------
    token    = get_ibm_token_optional()
    channel  = args.channel  or ibm_channel()
    instance = args.instance or ibm_instance()

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if token:
        _svc_kw: dict = {"channel": channel, "token": token}
        if instance:
            _svc_kw["instance"] = instance
        service = QiskitRuntimeService(**_svc_kw)
    else:
        service = QiskitRuntimeService()

    backend = service.backend(args.backend)
    print(f"\n[ibm] Connected to {args.backend} ({backend.num_qubits} qubits)")

    pm = generate_preset_pass_manager(
        optimization_level=args.opt_level, backend=backend
    )

    isa_bl = pm.run(baseline_circuit)
    isa_gr = pm.run(grover_circuit)
    isa_bl._layout = None  # prevent QPY layout-register mismatch (IBM Error 3211)
    isa_gr._layout = None

    print(f"  Baseline ISA depth: {isa_bl.depth()}")
    print(f"  Grover-1 ISA depth: {isa_gr.depth()}")

    result_data["isa_depth_baseline"] = isa_bl.depth()
    result_data["isa_depth_grover1"]  = isa_gr.depth()

    sampler = SamplerV2(mode=backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.strategy = "active-accum"
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.dynamical_decoupling.scheduling_method = "alap"

    print("\n[ibm] Submitting both circuits (baseline + Grover-1) ...")
    job = sampler.run([isa_bl, isa_gr], shots=args.shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")

    results = job.result()

    def _extract_counts(pub_result) -> dict:
        creg_name = next(k for k in vars(pub_result.data) if not k.startswith("_"))
        return dict(getattr(pub_result.data, creg_name).get_counts())

    hw_counts_bl = _extract_counts(results[0])
    hw_counts_gr = _extract_counts(results[1])

    hw_p_bl = _compute_obs_probability(hw_counts_bl, TARGET_OBS)
    hw_p_gr = _compute_obs_probability(hw_counts_gr, TARGET_OBS)
    hw_amp  = hw_p_gr / max(hw_p_bl, 1e-9)

    print(f"\n  [HW] Baseline P(obs={TARGET_OBS}): {hw_p_bl:.4f}  "
          f"(theory: {p_baseline:.4f})")
    print(f"  [HW] Grover-1 P(obs={TARGET_OBS}): {hw_p_gr:.4f}  "
          f"(theory: {p_amplified:.4f})")
    print(f"  [HW] Amplification factor:  {hw_amp:.1f}×  "
          f"(theory: {amp_factor:.1f}×)")

    result_data["hw_p_baseline"]           = hw_p_bl
    result_data["hw_p_amplified"]          = hw_p_gr
    result_data["hw_amplification_factor"] = hw_amp
    result_data["hw_job_id"]               = job_id
    result_data["hw_counts_baseline"]      = hw_counts_bl
    result_data["hw_counts_grover1"]       = hw_counts_gr

    # Pass criterion: hardware amplification ≥ 3× AND P_amplified > 0.5
    passed = hw_p_gr > 0.5 and hw_amp > 3.0
    result_data["pass"] = passed
    result_data["notes"] = (
        f"HW amplification {hw_amp:.1f}× "
        f"(P_baseline={hw_p_bl:.4f}, P_amplified={hw_p_gr:.4f}) "
        f"{'PASS' if passed else 'FAIL'} — threshold: 3× and P>0.5"
    )

    save_result("grover_belief_ibm", result_data)
    print(f"\n  PASS: {passed}  "
          f"(hw_amplification={hw_amp:.1f}×, threshold=3×, P_amplified={hw_p_gr:.4f}>0.5)")


if __name__ == "__main__":
    main()
