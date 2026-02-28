"""VALIDATION-ROADMAP Task 2.3 ? BIQAE adaptive amplitude estimation on IBM QPU.

Tests Bayesian Iterative Quantum Amplitude Estimation (Li et al., 2026) on real
IBM gate-based hardware.  Uses a 1-qubit oracle with known amplitude ``a_true``
to validate adaptive Grover iteration scheduling and Bayesian posterior updates
on a physical superconducting processor.

Circuit design
--------------
Oracle A: R_y(2?)|0? = cos(?)|0? + sin(?)|1?,  a = sin?(?)
Grover Q = A ? S? ? A? ? S_?  (1-qubit derivation)
  S_? = Z     (phase flip on marked state |1?)
  S?  = Z     (reflection around |0? in 1-qubit subspace: 2|0??0|?I = Z)
  ? Q = R_y(2?) Z R_y(?2?) Z

After k Grover iterations P(|1?) = sin?((2k+1)?)  [Brassard et al., 2002].
BIQAE uses base-3 exponential schedule K_t = 3^t to approach Heisenberg-limited
oracle complexity O(1/?) [Li et al., Quantum 10:1962 (2026), arXiv:2507.23074].

Usage
-----
# Simulator only (no credentials needed):
python scripts/hardware/run_biqae_ibm.py --dry-run

# Full hardware run on IBM Torino:
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_biqae_ibm.py \\
    --backend ibm_torino --shots-per-iter 200 --max-iterations 6

# Custom amplitude + more iterations:
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_biqae_ibm.py \\
    --backend ibm_torino --amplitude 0.25 --shots-per-iter 300 --max-iterations 8

Output
------
  output/hardware/biqae_ibm_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

from scripts.hardware import ibm_channel, ibm_instance, get_ibm_token_optional, save_result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIQAE on IBM QPU (Task 2.3)")
    p.add_argument("--backend", default="ibm_torino", help="IBM backend name")
    p.add_argument(
        "--shots-per-iter", type=int, default=200,
        help="Shots per BIQAE iteration (default: 200)",
    )
    p.add_argument(
        "--max-iterations", type=int, default=6,
        help="Maximum BIQAE iterations (default: 6)",
    )
    p.add_argument(
        "--amplitude", type=float, default=0.3,
        help="True target amplitude 0 < a < 1 (default: 0.3)",
    )
    p.add_argument(
        "--k-base", type=int, default=3,
        help="Grover schedule base K_t = k_base^t (default: 3, per paper)",
    )
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel: ibm_quantum_platform or ibm_cloud",
    )
    p.add_argument(
        "--instance", default=None,
        help="IBM instance / CRN (overrides IBM_QUANTUM_INSTANCE env)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only, skip IBM hardware (no credentials needed)",
    )
    p.add_argument(
        "--prior-mean", type=float, default=None,
        help="Prior mean for Bayesian estimator (default: 0.5). "
             "Set to the expected amplitude for better convergence at extreme values.",
    )
    p.add_argument(
        "--prior-std", type=float, default=None,
        help="Prior std for Bayesian estimator (default: 0.25). "
             "Narrower prior (e.g. 0.05) helps at extreme amplitudes.",
    )
    p.add_argument(
        "--num-qubits", type=int, choices=[1, 2], default=1,
        help="Oracle dimensionality: 1 (standard 1-qubit R_y) or 2 (separable "
             "2-qubit, targets |11>, P(|11>)=a_true). Default: 1.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Oracle and Grover operator construction
# ---------------------------------------------------------------------------

def _build_oracle_and_grover(a_true: float):
    """Build 1-qubit oracle A and Grover operator Q for amplitude a_true.

    Oracle A: R_y(2?)|0? = cos(?)|0? + sin(?)|1?
    Grover Q = R_y(2?) Z R_y(-2?) Z

    After k applications of Q:  P(|1?) = sin?((2k+1)?)
    This identity follows from the eigenvalue structure of Q and is the
    standard result from Brassard et al. (2002), Sec. 2.
    """
    import numpy as np
    from qiskit import QuantumCircuit

    theta = float(np.arcsin(np.sqrt(a_true)))

    oracle = QuantumCircuit(1, name="oracle_A")
    oracle.ry(2.0 * theta, 0)

    # Grover operator: Q = A ? S? ? A? ? S_?
    # For 1 qubit: S_? = Z (phase flip |1?), S? = Z (reflection around |0?)
    # Circuit order (right-to-left): first S_?, then A?, then S?, then A
    grover = QuantumCircuit(1, name="grover_Q")
    grover.z(0)               # S_?: phase kickback on |1?
    grover.ry(-2.0 * theta, 0)  # A?
    grover.z(0)               # S?: 2|0??0| ? I = Z for 1-qubit
    grover.ry(2.0 * theta, 0)   # A

    return oracle, grover, theta


def _build_2qubit_oracle_and_grover(a_true: float):
    """Build 2-qubit separable oracle A and joint Grover operator Q for amplitude a_true.

    Oracle A = R_y(2*theta) x R_y(2*theta),  theta = arcsin(a_true^(1/4))
    Targets |11> with P(|11>) = sin^2(theta) * sin^2(theta) = sin^4(theta) = a_true.

    Grover Q = A . S_0 . A_dag . S_psi  where
      S_psi = CZ             (phase -1 on |11>,  CZ|11> = -|11>)
      A_dag = R_y(-2t, 0) x R_y(-2t, 1)
      S_0   = X x X . CZ . X x X  (phase -1 on |00>, i.e. reflection about |0^2>)
      A     = R_y(2t, 0) x R_y(2t, 1)

    After k applications: P(|11>) = sin^2((2k+1)*theta_eff)
    where sin(theta_eff) = sin^2(theta) = sqrt(a_true).
    Hence the BIQAE model P(success) = sin^2((2k+1)*arcsin(sqrt(a))) is preserved
    exactly with a = a_true — identical formula to the 1-qubit case, just
    factored across two physical qubits with a CZ entangling gate.
    """
    import numpy as np
    from qiskit import QuantumCircuit

    theta = float(np.arcsin(np.sqrt(np.sqrt(max(a_true, 1e-9)))))

    oracle = QuantumCircuit(2, name="oracle_A2")
    oracle.ry(2.0 * theta, 0)
    oracle.ry(2.0 * theta, 1)

    grover = QuantumCircuit(2, name="grover_Q2")
    # S_psi: phase -1 on |11> via CZ  (CZ|11> = -|11>, others unchanged)
    grover.cz(0, 1)
    # A_dag: R_y(-2*theta) on both qubits
    grover.ry(-2.0 * theta, 0)
    grover.ry(-2.0 * theta, 1)
    # S_0: phase -1 on |00> via X x X, CZ, X x X
    grover.x(0)
    grover.x(1)
    grover.cz(0, 1)
    grover.x(0)
    grover.x(1)
    # A: R_y(2*theta) on both qubits
    grover.ry(2.0 * theta, 0)
    grover.ry(2.0 * theta, 1)

    return oracle, grover, theta


# ---------------------------------------------------------------------------
# IBM executor factory
# ---------------------------------------------------------------------------

def _make_ibm_executor(backend_obj):
    """Return an executor callable (qc, shots) ? counts dict for IBM QPU.

    The callable is designed to match BIQAEEstimator._execute_quantum's
    expected signature: executor(circuit_with_measurements, shots) ? dict.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    pm = generate_preset_pass_manager(optimization_level=2, backend=backend_obj)

    def executor(qc, shots: int = 200) -> dict:
        # Transpile to ISA (Instruction Set Architecture) format.
        isa = pm.run(qc)
        # Clear private layout reference to avoid QPY register-name mismatch
        # (IBM Error 3211: StopIteration when server deserialises physical reg).
        isa._layout = None

        sampler = SamplerV2(mode=backend_obj)
        job = sampler.run([isa], shots=shots)
        raw = job.result()[0]
        return dict(raw.data.meas.get_counts())

    return executor


def _make_ibm_executor_2qubit(backend_obj):
    """IBM QPU executor for 2-qubit BIQAE that collapses counts to binary success.

    Success = |11> (both qubits measure 1).  All other outcomes (|00>, |01>, |10>)
    map to failure.  The returned dict {"0": n_fail, "1": n_success} is compatible
    with BIQAEEstimator._execute_quantum which counts bitstrings ending in '1'.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    pm = generate_preset_pass_manager(optimization_level=2, backend=backend_obj)

    def executor(qc, shots: int = 200) -> dict:
        isa = pm.run(qc)
        isa._layout = None
        sampler = SamplerV2(mode=backend_obj)
        job = sampler.run([isa], shots=shots)
        raw = job.result()[0]
        raw_counts = dict(raw.data.meas.get_counts())
        # Collapse: success = |11> (both qubits 1), fail = everything else.
        # Qiskit MSB-first: "11" means qubit1=1, qubit0=1 (both 1). ✓
        success = sum(v for k, v in raw_counts.items() if k.replace(" ", "") == "11")
        fail = sum(v for k, v in raw_counts.items() if k.replace(" ", "") != "11")
        return {"1": success, "0": fail}

    return executor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    import numpy as np

    is_2qubit = args.num_qubits == 2
    oracle_label = "2-qubit separable R_y(x)R_y (|11>)" if is_2qubit else "1-qubit R_y"

    print(f"\n=== BIQAE / IBM QPU (Task 2.3) ===")
    print(f"  Oracle : {oracle_label}, a_true = {args.amplitude}")
    print(f"  Backend: {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots/iter: {args.shots_per_iter}, Max iterations: {args.max_iterations}")
    print(f"  K-schedule: K_t = {args.k_base}^t  (base-3 per Li et al. 2026)")

    if is_2qubit:
        oracle, grover, theta_true = _build_2qubit_oracle_and_grover(args.amplitude)
        p11 = float(np.sin(theta_true) ** 4)
        print(f"\n  theta = {theta_true:.4f} rad  ({np.degrees(theta_true):.2f} deg)")
        print(f"  P(|11>) = sin^4(theta) = {p11:.4f}  (a_true = {args.amplitude:.4f})")
    else:
        oracle, grover, theta_true = _build_oracle_and_grover(args.amplitude)
        print(f"\n  theta_true = {theta_true:.4f} rad  "
              f"({np.degrees(theta_true):.2f} deg),  sin2(theta) = {np.sin(theta_true)**2:.4f}")
    print(f"  Oracle depth: {oracle.depth()}, Grover depth: {grover.depth()}")

    from quantum_pomdp.algorithms.biqae_estimator import BIQAEConfig, BIQAEEstimator

    config_kwargs: dict = dict(
        variant="beta",
        max_iterations=args.max_iterations,
        confidence_level=0.95,
        shots_per_iteration=args.shots_per_iter,
        k_base=args.k_base,
    )
    if args.prior_mean is not None:
        config_kwargs["prior_mean"] = args.prior_mean
    if args.prior_std is not None:
        config_kwargs["prior_std"] = args.prior_std
    config = BIQAEConfig(**config_kwargs)

    oracle_desc = (
        "2-qubit R_y(2t)xR_y(2t), Grover=CZ·Ry(-2t)^2·X^2·CZ·X^2·Ry(2t)^2, success=|11>"
        if is_2qubit else
        "1-qubit R_y(2t), Grover=Z·R_y(-2t)·Z·R_y(2t), success=|1>"
    )
    result_data: dict = {
        "task_ids": ["2.3"],
        "backend": "aer_simulator" if args.dry_run else args.backend,
        "a_true": args.amplitude,
        "theta_true": float(theta_true),
        "shots_per_iteration": args.shots_per_iter,
        "max_iterations": args.max_iterations,
        "k_base": args.k_base,
        "num_qubits": args.num_qubits,
        "oracle_qubits": args.num_qubits,
        "oracle": oracle_desc,
    }

    # ------------------------------------------------------------------
    # Classical simulation baseline (no executor ? BIQAE internal sim)
    # ------------------------------------------------------------------
    print("\n[simulator] BIQAE with internal classical simulator ...")
    estimator_sim = BIQAEEstimator(config)
    biqae_sim = estimator_sim.estimate(oracle, grover, executor=None)

    sim_error = abs(biqae_sim.amplitude_estimate - args.amplitude)
    sim_ci_contains = (
        biqae_sim.confidence_interval[0] <= args.amplitude
        <= biqae_sim.confidence_interval[1]
    )
    print(f"  Estimate  : {biqae_sim.amplitude_estimate:.4f}  (true: {args.amplitude:.4f})")
    print(f"  95% CI    : [{biqae_sim.confidence_interval[0]:.4f}, "
          f"{biqae_sim.confidence_interval[1]:.4f}]")
    print(f"  Iterations: {biqae_sim.num_iterations}")
    print(f"  Total shots: {biqae_sim.total_shots}")
    print(f"  |Error|   : {sim_error:.4f}")
    print(f"  CI ? true : {sim_ci_contains}")

    result_data["simulator"] = {
        "amplitude_estimate": biqae_sim.amplitude_estimate,
        "confidence_interval": list(biqae_sim.confidence_interval),
        "num_iterations": biqae_sim.num_iterations,
        "total_shots": biqae_sim.total_shots,
        "error": float(sim_error),
        "ci_contains_true": sim_ci_contains,
    }

    # Dry-run pass: CI must contain the true value (algorithm ran, CI is calibrated).
    # Note: internal sim uses posterior mean to generate samples (self-referential),
    # so |error| may be large; CI coverage is the correct convergence criterion.
    sim_ok = sim_ci_contains

    if args.dry_run:
        result_data["pass"] = sim_ok
        result_data["notes"] = "Dry-run: simulator only; pass=CI_contains_true"
        save_result("biqae_ibm", result_data)
        print(f"\n  PASS: {sim_ok}  (CI_contains_true={sim_ci_contains}, |error|={sim_error:.4f})")
        return

    # ------------------------------------------------------------------
    # IBM hardware run
    # ------------------------------------------------------------------
    token = get_ibm_token_optional()  # None => saved account fallback
    channel = args.channel or ibm_channel()
    instance = args.instance or ibm_instance()

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        print(f"[error] qiskit-ibm-runtime not installed: {exc}")
        raise SystemExit(1)

    if token:
        kwargs: dict = {"channel": channel, "token": token}
        if instance:
            kwargs["instance"] = instance
        service = QiskitRuntimeService(**kwargs)
    else:
        service = QiskitRuntimeService()  # saved account
    backend_obj = service.backend(args.backend)
    print(f"\n[ibm] Connected to {args.backend}  ({backend_obj.num_qubits} qubits)")

    # Inspect transpiled oracle depth for reference
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend_obj)
    oracle_for_depth = oracle.copy()
    oracle_for_depth.measure_all()
    isa_oracle = pm.run(oracle_for_depth)
    print(f"  Oracle ISA depth (with meas): {isa_oracle.depth()}  "
          f"(logical: {oracle.depth()})")

    # Grover + oracle composed for depth estimate
    composed_k1 = oracle.copy()
    composed_k1.compose(grover, inplace=True)
    composed_k1.measure_all()
    isa_k1 = pm.run(composed_k1)
    print(f"  Oracle+1?Grover ISA depth   : {isa_k1.depth()}  "
          f"(logical: {composed_k1.depth()})")

    result_data["circuit_info"] = {
        "oracle_logical_depth": oracle.depth(),
        "oracle_isa_depth": isa_oracle.depth(),
        "oracle_plus_1grover_logical_depth": composed_k1.depth(),
        "oracle_plus_1grover_isa_depth": isa_k1.depth(),
    }

    if is_2qubit:
        ibm_executor = _make_ibm_executor_2qubit(backend_obj)
    else:
        ibm_executor = _make_ibm_executor(backend_obj)

    print(f"\n[ibm] Running BIQAE on hardware  "
          f"({args.max_iterations} iterations × {args.shots_per_iter} shots) ...")
    try:
        estimator_hw = BIQAEEstimator(config)
        biqae_hw = estimator_hw.estimate(oracle, grover, executor=ibm_executor)

        hw_error = abs(biqae_hw.amplitude_estimate - args.amplitude)
        hw_ci_contains = (
            biqae_hw.confidence_interval[0] <= args.amplitude
            <= biqae_hw.confidence_interval[1]
        )

        print(f"  Estimate  : {biqae_hw.amplitude_estimate:.4f}  (true: {args.amplitude:.4f})")
        print(f"  95% CI    : [{biqae_hw.confidence_interval[0]:.4f}, "
              f"{biqae_hw.confidence_interval[1]:.4f}]")
        print(f"  Iterations: {biqae_hw.num_iterations}")
        print(f"  Total shots: {biqae_hw.total_shots}")
        print(f"  |Error|   : {hw_error:.4f}")
        print(f"  CI ? true : {hw_ci_contains}")

        result_data["hardware"] = {
            "amplitude_estimate": biqae_hw.amplitude_estimate,
            "confidence_interval": list(biqae_hw.confidence_interval),
            "num_iterations": biqae_hw.num_iterations,
            "total_shots": biqae_hw.total_shots,
            "error": float(hw_error),
            "ci_contains_true": hw_ci_contains,
        }
        result_data["hw_vs_sim_error_delta"] = float(hw_error - sim_error)
        result_data["hw_ci_contains_true"] = hw_ci_contains

        # Pass criterion: estimate within 0.15 of truth on noisy hardware
        passed = hw_error < 0.15
        result_data["pass"] = passed
        result_data["notes"] = (
            f"HW estimate {biqae_hw.amplitude_estimate:.4f} "
            f"(true {args.amplitude:.4f}, error {hw_error:.4f}, "
            f"CI_contains={hw_ci_contains})"
        )

    except Exception as exc:
        print(f"  [error] IBM BIQAE failed: {exc}")
        result_data["hardware"] = {"error": str(exc)}
        result_data["pass"] = False
        result_data["notes"] = f"Hardware execution failed: {exc}"

    save_result("biqae_ibm", result_data)
    print(f"\n  PASS: {result_data['pass']}")


if __name__ == "__main__":
    main()
