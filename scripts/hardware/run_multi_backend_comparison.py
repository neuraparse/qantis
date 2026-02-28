"""Multi-backend comparison: ibm_torino (Heron R1) vs ibm_fez / ibm_marrakesh (Heron R2).

Runs the same test suite on multiple IBM backends and prints a side-by-side
comparison table, useful for characterising Heron R1 vs R2 differences.

Tests
-----
  bell_zne       Bell state |Φ+⟩ — ZNE with scale factors [1,3,5], 4096 shots
  biqae          BIQAE adaptive estimation at a=0.50, 6 iterations, 300 shots
  fpc_transpile  FPC-QAOA p=2 transpile only (no QPU job) — reports ISA depth
  tiger_transpile Tiger belief circuit transpile only — reports ISA depth
  tiger_belief   Tiger full execution (slow, optional) — reports Hellinger

Usage
-----
  # Saved account (no env var needed):
  python scripts/hardware/run_multi_backend_comparison.py \\
      --backends ibm_torino ibm_fez --tests bell_zne biqae fpc_transpile

  # With explicit token:
  IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_multi_backend_comparison.py \\
      --backends ibm_torino ibm_fez ibm_marrakesh --tests bell_zne biqae

  # Dry run (Aer simulator for all backends):
  python scripts/hardware/run_multi_backend_comparison.py --dry-run

Output
------
  output/hardware/multi_backend_comparison_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-mht" / "src"))

from scripts.hardware import (
    build_tiger_pomdp,
    compute_zz_expectation,
    get_ibm_token_optional,
    ibm_channel,
    ibm_instance,
    make_qubo_instance,
    save_result,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_ALL_TESTS = ["bell_zne", "biqae", "fpc_transpile", "tiger_transpile", "tiger_belief"]

_BACKEND_INFO: dict[str, dict[str, Any]] = {
    "ibm_torino":    {"processor": "Heron R1", "qubits": 133, "ecr_error": "~2.5e-3"},
    "ibm_fez":       {"processor": "Heron R2", "qubits": 156, "ecr_error": "~1.5e-3"},
    "ibm_marrakesh": {"processor": "Heron R2", "qubits": 156, "ecr_error": "~1.5e-3"},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-backend IBM comparison (Heron R1 vs R2)"
    )
    p.add_argument(
        "--backends", nargs="+", default=["ibm_torino", "ibm_fez"],
        help="IBM backend names to compare",
    )
    p.add_argument(
        "--tests", nargs="+", default=["bell_zne", "biqae", "fpc_transpile"],
        choices=_ALL_TESTS,
        help=f"Tests to run: {_ALL_TESTS}",
    )
    p.add_argument("--shots", type=int, default=4096, help="Shots for bell_zne / tiger_belief")
    p.add_argument("--biqae-shots", type=int, default=300, help="Shots per BIQAE iteration")
    p.add_argument("--biqae-iters", type=int, default=6, help="BIQAE max iterations")
    p.add_argument("--scale-factors", nargs="+", type=float, default=[1.0, 3.0, 5.0],
                   help="ZNE scale factors")
    p.add_argument("--channel", default=None)
    p.add_argument("--instance", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Use Aer simulator for all backends (no QPU jobs)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ZNE helpers (copied from run_zne_ibm.py to keep script self-contained)
# ---------------------------------------------------------------------------

def _fold_circuit(circuit, scale_factor: int):
    from qiskit import QuantumCircuit
    from qiskit.circuit.exceptions import CircuitError

    qc_u = QuantumCircuit(*circuit.qregs)
    for inst in circuit.data:
        if inst.operation.name not in ("measure", "barrier"):
            qc_u.append(inst)

    n_extra = (scale_factor - 1) // 2
    folded = qc_u.copy()
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
    import numpy as np
    if len(scales) < 2:
        return values[0]
    if len(scales) == 2:
        s1, s2 = scales[0], scales[1]
        v1, v2 = values[0], values[1]
        return (s2 * v1 - s1 * v2) / (s2 - s1)
    coeffs = np.polyfit(scales, values, deg=1)
    return float(np.polyval(coeffs, 0.0))


def _build_bell_circuit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bell_zne(backend, backend_name: str, shots: int,
                  scale_factors: list[float], dry_run: bool) -> dict[str, Any]:
    """Bell state ZNE — measures <ZZ> raw and mitigated."""
    from quantum_common.backends.base import ExecutionRequest
    from quantum_common.backends.simulator import AerSimulatorBackend

    print(f"  [bell_zne] scale factors {scale_factors}, {shots} shots")
    bell = _build_bell_circuit()

    # Aer ideal
    aer = AerSimulatorBackend()
    sim_counts = aer.execute(ExecutionRequest(circuits=[bell], shots=shots)).counts[0]
    sim_zz = compute_zz_expectation(sim_counts)

    # Odd integer scales
    int_scales = sorted(set(max(1, 2 * int(round((s - 1) / 2)) + 1) for s in scale_factors))
    folded = [_fold_circuit(bell, s) for s in int_scales]
    result = backend.execute(ExecutionRequest(circuits=folded, shots=shots))
    zz_by_scale = {int_scales[i]: compute_zz_expectation(result.counts[i])
                   for i in range(len(int_scales))}

    raw_zz = zz_by_scale[int_scales[0]]
    mitigated_zz = _richardson_extrapolate(
        [float(s) for s in int_scales], [zz_by_scale[s] for s in int_scales]
    )
    improved = abs(mitigated_zz - 1.0) < abs(raw_zz - 1.0)

    print(f"    sim_zz={sim_zz:.4f}  raw={raw_zz:.4f}  mitigated={mitigated_zz:.4f}  improved={improved}")
    return {
        "sim_zz": sim_zz,
        "hw_raw_zz": raw_zz,
        "hw_mitigated_zz": mitigated_zz,
        "zne_improved": improved,
        "zne_delta": mitigated_zz - raw_zz,
        "scales_used": int_scales,
        "zz_by_scale": {str(s): v for s, v in zz_by_scale.items()},
    }


def test_biqae(backend, backend_name: str, shots_per_iter: int,
               max_iters: int, dry_run: bool) -> dict[str, Any]:
    """BIQAE at a=0.50 — 6 iterations, base-3 schedule."""
    import math

    print(f"  [biqae] a=0.50, {max_iters} iterations x {shots_per_iter} shots")
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    a_true = 0.50
    theta = math.asin(math.sqrt(a_true))

    # Oracle: R_y(2θ)|0⟩ puts amplitude a on |1⟩
    def build_oracle(k: int):
        qc = QuantumCircuit(1)
        qc.ry(2 * theta, 0)
        for _ in range(k):
            qc.ry(2 * theta, 0)  # Grover: apply oracle k times
        qc.measure_all()
        return qc

    # For BIQAE we use a direct executor approach via saved account
    if dry_run:
        from quantum_common.backends.simulator import AerSimulatorBackend
        from quantum_common.backends.base import ExecutionRequest
        aer = AerSimulatorBackend()
        def executor(qc, shots=shots_per_iter):
            return aer.execute(ExecutionRequest(circuits=[qc], shots=shots)).counts[0]
    else:
        try:
            service = QiskitRuntimeService()
        except Exception:
            # fallback: use backend's internal service
            service = None

        if service is not None:
            be_obj = service.backend(backend_name)
            pm = generate_preset_pass_manager(optimization_level=2, backend=be_obj)
            sampler = SamplerV2(mode=be_obj)

            def executor(qc, shots=shots_per_iter):
                isa = pm.run(qc)
                isa._layout = None
                job = sampler.run([isa], shots=shots)
                raw = job.result()[0]
                return dict(raw.data.meas.get_counts())
        else:
            from quantum_common.backends.base import ExecutionRequest
            def executor(qc, shots=shots_per_iter):
                return backend.execute(ExecutionRequest(circuits=[qc], shots=shots)).counts[0]

    from quantum_pomdp.algorithms.biqae_estimator import BIQAEEstimator, BIQAEConfig

    oracle = build_oracle(0)
    grover = build_oracle(1)  # one Grover step

    config = BIQAEConfig(
        variant="beta",
        max_iterations=max_iters,
        confidence_level=0.95,
        shots_per_iteration=shots_per_iter,
        k_base=3,
    )
    est = BIQAEEstimator(config)
    result = est.estimate(oracle, grover, executor=executor)

    error = abs(result.amplitude_estimate - a_true)
    ci_covers = result.confidence_interval[0] <= a_true <= result.confidence_interval[1]
    print(f"    estimate={result.amplitude_estimate:.4f}  error={error:.4f}  CI={result.confidence_interval}  covers={ci_covers}")
    return {
        "a_true": a_true,
        "estimate": result.amplitude_estimate,
        "error": error,
        "ci": list(result.confidence_interval),
        "ci_covers_true": ci_covers,
        "num_iterations": result.num_iterations,
        "total_shots": result.total_shots,
    }


def test_fpc_transpile(backend, backend_name: str, dry_run: bool) -> dict[str, Any]:
    """FPC-QAOA p=2 transpile only — reports ISA depth, no QPU job."""
    import numpy as np
    print(f"  [fpc_transpile] transpiling FPC-QAOA p=2, N=2, M=3 (11 qubits)")

    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.translators import to_ising
    from quantum_mht.solvers.fpc_qaoa_solver import digitize_schedule, _polynomial_schedule

    qaoa_depth = 2
    num_schedule_params = 3

    qubo = make_qubo_instance(n_tracks=2, n_meas=3, seed=42)

    # Build QuadraticProgram from QUBO
    qp = QuadraticProgram()
    for i in range(qubo.num_variables):
        qp.binary_var(f"x{i}")
    lin, quad = {}, {}
    for (i, j), val in qubo.Q.items():
        if i == j:
            lin[f"x{i}"] = lin.get(f"x{i}", 0.0) + val
        else:
            quad[(f"x{i}", f"x{j}")] = quad.get((f"x{i}", f"x{j}"), 0.0) + val
    qp.minimize(linear=lin, quadratic=quad)
    ising_op, _ = to_ising(qp)

    # Build QAOAAnsatz and bind FPC-schedule parameters
    gamma_coeffs = [0.0] + [float(np.pi)] * (num_schedule_params - 1)
    beta_coeffs = [float(np.pi / 4)] + [0.0] * (num_schedule_params - 1)
    gammas = digitize_schedule(_polynomial_schedule, gamma_coeffs, qaoa_depth)
    betas = digitize_schedule(_polynomial_schedule, beta_coeffs, qaoa_depth)
    initial_point = []
    for g, b in zip(gammas, betas):
        initial_point.extend([g, b])

    raw_circuit = QAOAAnsatz(ising_op, reps=qaoa_depth)
    raw_circuit = raw_circuit.assign_parameters(
        dict(zip(raw_circuit.parameters, initial_point))
    )
    raw_circuit.measure_all()

    if dry_run:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        aer_backend = AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=2, backend=aer_backend)
        transpiled = pm.run(raw_circuit)
        isa_depth = transpiled.depth()
        cx_count = transpiled.count_ops().get("cx", transpiled.count_ops().get("ecr", 0))
    else:
        transpiled_list = backend.transpile([raw_circuit], optimization_level=2)
        t = transpiled_list[0]
        isa_depth = t.depth()
        cx_count = t.count_ops().get("cx", t.count_ops().get("ecr", 0))

    print(f"    logical_depth={raw_circuit.depth()}  ISA_depth={isa_depth}  CX/ECR={cx_count}")
    return {
        "n_tracks": 2, "n_meas": 3, "qaoa_depth": 2,
        "logical_qubits": raw_circuit.num_qubits,
        "logical_depth": raw_circuit.depth(),
        "isa_depth": isa_depth,
        "cx_ecr_count": cx_count,
    }


def test_tiger_transpile(backend, backend_name: str, dry_run: bool) -> dict[str, Any]:
    """Tiger belief circuit transpile only — reports ISA depth, no QPU job."""
    print(f"  [tiger_transpile] transpiling Tiger belief circuit")
    from quantum_pomdp.models.belief_state import BeliefState
    from quantum_pomdp.quantum_circuits.belief_update import (
        BeliefUpdateCircuitConfig, QuantumBeliefUpdateCircuit,
    )

    pomdp = build_tiger_pomdp()
    belief = BeliefState.uniform(num_states=pomdp.num_states)
    config = BeliefUpdateCircuitConfig(
        use_amplitude_amplification=True,
        use_hardware_compatible_encoding=True,
        reward_precision_bits=2,
    )
    circuit = QuantumBeliefUpdateCircuit(pomdp=pomdp, config=config).build(
        belief=belief, action=0, observation=0
    )

    if dry_run:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        aer_backend = AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=2, backend=aer_backend)
        transpiled = pm.run(circuit)
        isa_depth = transpiled.depth()
        cx_count = transpiled.count_ops().get("cx", 0)
    else:
        transpiled_list = backend.transpile([circuit], optimization_level=2)
        t = transpiled_list[0]
        isa_depth = t.depth()
        cx_count = t.count_ops().get("cx", t.count_ops().get("ecr", 0))

    print(f"    logical_depth={circuit.depth()}  ISA_depth={isa_depth}  CX={cx_count}")
    return {
        "logical_qubits": circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "isa_depth": isa_depth,
        "cx_ecr_count": cx_count,
    }


def test_tiger_belief(backend, backend_name: str, shots: int, dry_run: bool) -> dict[str, Any]:
    """Tiger belief circuit full execution — measures Hellinger distance."""
    from quantum_common.backends.base import ExecutionRequest
    from quantum_common.backends.simulator import AerSimulatorBackend
    from quantum_pomdp.models.belief_state import BeliefState
    from quantum_pomdp.quantum_circuits.belief_update import (
        BeliefUpdateCircuitConfig, QuantumBeliefUpdateCircuit,
    )

    print(f"  [tiger_belief] executing Tiger circuit, {shots} shots (SLOW)")
    pomdp = build_tiger_pomdp()
    belief = BeliefState.uniform(num_states=pomdp.num_states)
    config = BeliefUpdateCircuitConfig(
        use_amplitude_amplification=True,
        use_hardware_compatible_encoding=True,
        reward_precision_bits=2,
    )
    circuit = QuantumBeliefUpdateCircuit(pomdp=pomdp, config=config).build(
        belief=belief, action=0, observation=0
    )
    true_posterior = belief.classical_update(
        action=0, observation=0,
        transition_tensor=pomdp.transition_tensor,
        observation_tensor=pomdp.observation_tensor,
    )

    # Simulator baseline
    aer = AerSimulatorBackend()
    sim_counts = aer.execute(ExecutionRequest(circuits=[circuit], shots=shots)).counts[0]
    sim_belief = BeliefState.from_quantum_measurement(sim_counts, pomdp.num_states, pomdp.state_qubits)
    sim_hellinger = sim_belief.hellinger_distance(true_posterior)

    # Hardware
    hw_counts = backend.execute(ExecutionRequest(circuits=[circuit], shots=shots)).counts[0]
    hw_belief = BeliefState.from_quantum_measurement(hw_counts, pomdp.num_states, pomdp.state_qubits)
    hw_hellinger = hw_belief.hellinger_distance(true_posterior)

    print(f"    sim_hellinger={sim_hellinger:.4f}  hw_hellinger={hw_hellinger:.4f}")
    return {
        "logical_qubits": circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "sim_hellinger": sim_hellinger,
        "hw_hellinger": hw_hellinger,
    }


# ---------------------------------------------------------------------------
# Per-backend runner
# ---------------------------------------------------------------------------

_TEST_FNS = {
    "bell_zne": test_bell_zne,
    "biqae": test_biqae,
    "fpc_transpile": test_fpc_transpile,
    "tiger_transpile": test_tiger_transpile,
    "tiger_belief": test_tiger_belief,
}


def run_backend(backend_name: str, args: argparse.Namespace) -> dict[str, Any]:
    info = _BACKEND_INFO.get(backend_name, {"processor": "?", "qubits": "?", "ecr_error": "?"})
    print(f"\n{'='*60}")
    print(f"Backend: {backend_name}  ({info['processor']}, {info['qubits']}q, ECR {info['ecr_error']})")
    print(f"{'='*60}")

    if args.dry_run:
        from quantum_common.backends.simulator import AerSimulatorBackend
        backend = AerSimulatorBackend()
    else:
        from quantum_common.backends.ibm import IBMQuantumBackend
        token = get_ibm_token_optional()  # None → saved account
        channel = args.channel or ibm_channel()
        instance = args.instance or ibm_instance()
        backend = IBMQuantumBackend(
            backend_name=backend_name,
            token=token,
            channel=channel,
            instance=instance,
        )

    results: dict[str, Any] = {"backend_info": info}
    for test in args.tests:
        print(f"\n  Running: {test}")
        try:
            if test == "bell_zne":
                results[test] = test_bell_zne(backend, backend_name, args.shots, args.scale_factors, args.dry_run)
            elif test == "biqae":
                results[test] = test_biqae(backend, backend_name, args.biqae_shots, args.biqae_iters, args.dry_run)
            elif test == "fpc_transpile":
                results[test] = test_fpc_transpile(backend, backend_name, args.dry_run)
            elif test == "tiger_transpile":
                results[test] = test_tiger_transpile(backend, backend_name, args.dry_run)
            elif test == "tiger_belief":
                results[test] = test_tiger_belief(backend, backend_name, args.shots, args.dry_run)
            results[test]["status"] = "ok"
        except Exception as e:
            print(f"    ERROR: {e}")
            results[test] = {"status": "error", "error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _fmt(val: Any, decimals: int = 4) -> str:
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    if val is None:
        return "-"
    return str(val)


def print_comparison_table(backends: list[str], all_results: dict[str, Any]) -> None:
    col_w = max(22, *(len(b) + 4 for b in backends))
    header = f"{'Metric':<35}" + "".join(f"{b:^{col_w}}" for b in backends)
    print(f"\n{'='*len(header)}")
    print("MULTI-BACKEND COMPARISON")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    # Bell ZNE
    for metric, key, path in [
        ("Bell raw <ZZ>", "bell_zne", "hw_raw_zz"),
        ("Bell mitigated <ZZ>", "bell_zne", "hw_mitigated_zz"),
        ("ZNE improvement delta<ZZ>", "bell_zne", "zne_delta"),
    ]:
        row = f"{metric:<35}"
        for b in backends:
            r = all_results.get(b, {}).get(key, {})
            row += f"{_fmt(r.get(path)):^{col_w}}"
        print(row)

    # BIQAE
    for metric, key, path in [
        ("BIQAE a=0.50 estimate", "biqae", "estimate"),
        ("BIQAE a=0.50 |error|", "biqae", "error"),
        ("BIQAE CI covers truth", "biqae", "ci_covers_true"),
    ]:
        row = f"{metric:<35}"
        for b in backends:
            r = all_results.get(b, {}).get(key, {})
            row += f"{_fmt(r.get(path)):^{col_w}}"
        print(row)

    # Transpile depths
    for metric, key, path in [
        ("FPC-QAOA p=2 ISA depth", "fpc_transpile", "isa_depth"),
        ("FPC-QAOA CX/ECR count", "fpc_transpile", "cx_ecr_count"),
        ("Tiger ISA depth", "tiger_transpile", "isa_depth"),
        ("Tiger CX/ECR count", "tiger_transpile", "cx_ecr_count"),
    ]:
        row = f"{metric:<35}"
        for b in backends:
            r = all_results.get(b, {}).get(key, {})
            row += f"{_fmt(r.get(path)):^{col_w}}"
        print(row)

    # Tiger belief
    for metric, key, path in [
        ("Tiger Hellinger (sim)", "tiger_belief", "sim_hellinger"),
        ("Tiger Hellinger (HW)", "tiger_belief", "hw_hellinger"),
    ]:
        row = f"{metric:<35}"
        for b in backends:
            r = all_results.get(b, {}).get(key, {})
            row += f"{_fmt(r.get(path)):^{col_w}}"
        print(row)

    print(f"{'='*len(header)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"\n=== QANTIS Multi-Backend Comparison ===")
    print(f"  Backends : {args.backends}")
    print(f"  Tests    : {args.tests}")
    print(f"  Shots    : {args.shots}")
    print(f"  Dry run  : {args.dry_run}")

    all_results: dict[str, Any] = {}
    for backend_name in args.backends:
        all_results[backend_name] = run_backend(backend_name, args)

    print_comparison_table(args.backends, all_results)

    # Build comparison summary dict for JSON
    comparison: dict[str, Any] = {}
    for test in args.tests:
        comparison[test] = {}
        for backend_name in args.backends:
            comparison[test][backend_name] = all_results.get(backend_name, {}).get(test, {})

    payload = {
        "task_ids": ["multi_backend"],
        "backends": args.backends,
        "tests": args.tests,
        "shots": args.shots,
        "scale_factors": args.scale_factors,
        "dry_run": args.dry_run,
        "per_backend": all_results,
        "comparison": comparison,
    }
    save_result("multi_backend_comparison", payload)


if __name__ == "__main__":
    main()
