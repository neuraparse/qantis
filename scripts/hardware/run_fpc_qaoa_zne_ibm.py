"""FPC-QAOA + ZNE on IBM QPU (Task 1.3 extension).

Applies Zero-Noise Extrapolation (gate folding + Richardson extrapolation) to
FPC-QAOA circuits, measuring the *average QUBO objective expectation value*
at multiple noise scales and extrapolating to zero noise.

Key design decisions
--------------------
* ZNE quantity: E_QUBO(λ) = Σ_x P_λ(x) · f(x), the expectation of the QUBO
  objective under the noisy measurement distribution at scale λ.  This is a
  proper expectation value that satisfies the ZNE linear model, unlike the
  argmax (best bitstring), which is not an expectation and cannot be
  extrapolated.
* Gate folding: each scale s applies C (C† C)^((s-1)//2) to the raw circuit
  before transpilation, following Temme et al. 2017.  Barriers prevent the
  compiler from cancelling fold pairs.
* Richardson extrapolation: polynomial fit through (scale, E_QUBO) pairs,
  evaluated at scale=0 to obtain the zero-noise estimate.
* Default depth: p=3 (ISA depth ≈435) — in the sweet spot where ZNE is
  expected to be beneficial (ISA depth 100–1000; below 100 noise is tiny,
  above 1000 folding amplifies noise super-linearly).

Usage
-----
# Dry-run (simulator only, no QPU):
python scripts/hardware/run_fpc_qaoa_zne_ibm.py --dry-run

# Full hardware run with ZNE:
python scripts/hardware/run_fpc_qaoa_zne_ibm.py --backend ibm_fez \\
    --depth 3 --shots 4096 --scale-factors 1 3 5

Output
------
  output/hardware/fpc_qaoa_zne_ibm_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-mht" / "src"))

from scripts.hardware import (
    ibm_channel,
    ibm_instance,
    make_qubo_instance,
    get_ibm_token_optional,
    save_result,
)

import numpy as np


# ---------------------------------------------------------------------------
# ZNE helpers (gate folding + Richardson extrapolation)
# Adapted from run_zne_ibm.py — self-contained copy for clarity.
# ---------------------------------------------------------------------------

def _fold_circuit(circuit, scale_factor: int):
    """Gate folding: circuit → C (C† C)^((scale-1)//2).

    For odd integer scale factors, the folded circuit is logically equivalent
    to the original but has scale-fold more gates, amplifying noise by ~scale.
    Barriers prevent the transpiler from cancelling gates across fold boundary.
    """
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
    """Richardson extrapolation to zero-noise limit.

    - 1 point: return the single value.
    - 2 points: two-point Richardson formula.
    - 3+ points: linear least-squares fit, evaluate at scale=0.
    """
    if len(scales) < 2:
        return values[0]
    if len(scales) == 2:
        s1, s2 = scales[0], scales[1]
        v1, v2 = values[0], values[1]
        return (s2 * v1 - s1 * v2) / (s2 - s1)
    coeffs = np.polyfit(scales, values, deg=1)
    return float(np.polyval(coeffs, 0.0))


# ---------------------------------------------------------------------------
# QUBO evaluation helpers
# ---------------------------------------------------------------------------

def _eval_qubo_obj(qubo, x_str: str) -> float:
    """Compute x^T Q x for a bitstring."""
    x = np.array([int(b) for b in x_str], dtype=np.int_)
    obj = 0.0
    for (ii, jj), q_val in qubo.Q.items():
        obj += q_val * x[ii] * x[jj]
    return float(obj)


def _eval_avg_qubo_obj(counts: dict[str, int], qubo) -> float:
    """Compute E[f(x)] = Σ_x P(x) · f(x) over measurement distribution.

    This is a proper expectation value suitable for Richardson extrapolation
    (unlike argmax, which is not an expectation and cannot be linearly
    extrapolated across noise scales).
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    avg = 0.0
    for bs, cnt in counts.items():
        avg += (cnt / total) * _eval_qubo_obj(qubo, bs)
    return avg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FPC-QAOA + ZNE on IBM QPU (Task 1.3 extension)"
    )
    p.add_argument("--backend", default="ibm_fez", help="IBM backend name")
    p.add_argument("--shots", type=int, default=4096, help="Shots per scale factor")
    p.add_argument("--depth", type=int, default=3, help="QAOA circuit depth (p)")
    p.add_argument(
        "--schedule-params", type=int, default=3,
        help="FPC schedule parameter count (k)",
    )
    p.add_argument(
        "--scale-factors", nargs="+", type=float, default=[1.0, 3.0, 5.0],
        help="ZNE noise scale factors (converted to odd integers; default [1,3,5])",
    )
    p.add_argument("--n-tracks", type=int, default=2)
    p.add_argument("--n-meas", type=int, default=3)
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel (overrides IBM_QUANTUM_CHANNEL env)",
    )
    p.add_argument("--instance", default=None)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only — no IBM hardware credentials needed",
    )
    p.add_argument(
        "--optimizer-maxiter", type=int, default=100,
        help="COBYLA max iterations for simulator warm-start",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Use odd integer scales throughout
    int_scales = sorted(set(
        max(1, 2 * int(round((s - 1) / 2)) + 1) for s in args.scale_factors
    ))

    n_vars = args.n_tracks * args.n_meas + args.n_tracks + args.n_meas
    print(f"\n=== FPC-QAOA + ZNE — IBM QPU (Task 1.3 extension) ===")
    print(f"  N={args.n_tracks} tracks, M={args.n_meas} measurements, {n_vars} QUBO vars")
    print(f"  Backend : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  QAOA p  : {args.depth},  schedule params k={args.schedule_params}")
    print(f"  Shots   : {args.shots} per scale")
    print(f"  ZNE scales: {int_scales}")

    # ------------------------------------------------------------------
    # Build QUBO + Hungarian baseline
    # ------------------------------------------------------------------
    qubo = make_qubo_instance(n_tracks=args.n_tracks, n_meas=args.n_meas, seed=42)
    from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver
    hungarian = HungarianSolver().solve(qubo)
    print(f"\n  QUBO variables     : {qubo.num_variables}")
    print(f"  Hungarian objective: {hungarian.objective_value:.4f}")

    result_data: dict = {
        "task_ids": ["1.3-zne"],
        "backend": "aer_simulator" if args.dry_run else args.backend,
        "n_tracks": args.n_tracks,
        "n_meas": args.n_meas,
        "num_variables": qubo.num_variables,
        "qaoa_depth": args.depth,
        "schedule_params": args.schedule_params,
        "shots_per_scale": args.shots,
        "scale_factors": int_scales,
        "hungarian_objective": hungarian.objective_value,
    }

    # ------------------------------------------------------------------
    # Simulator run to get warm-start parameters
    # ------------------------------------------------------------------
    print(f"\n[simulator] Running FPC-QAOA p={args.depth} for warm-start params ...")
    from quantum_mht.solvers.fpc_qaoa_solver import FPCQAOASolver

    try:
        sim_solver = FPCQAOASolver(
            depth=args.depth,
            num_schedule_params=args.schedule_params,
            shots=args.shots,
            optimizer_maxiter=args.optimizer_maxiter,
        )
        sim_result = sim_solver.solve(qubo)
        sim_ratio = sim_result.objective_value / max(abs(hungarian.objective_value), 1e-9)
        print(f"  Simulator objective  : {sim_result.objective_value:.4f}")
        print(f"  Simulator approx_ratio: {sim_ratio:.3f}")
        print(f"  Simulator solve_time : {sim_result.solve_time_s:.2f}s")

        result_data["simulator"] = {
            "objective": sim_result.objective_value,
            "approx_ratio": sim_ratio,
            "solve_time_s": sim_result.solve_time_s,
        }

        # Extract COBYLA-optimised parameter vector for hardware warm-start
        optimal_params = None
        if hasattr(sim_result, "metadata") and "optimal_params" in (sim_result.metadata or {}):
            optimal_params = sim_result.metadata["optimal_params"]
        sim_ok = True
    except Exception as exc:
        print(f"  [warning] Simulator run failed: {exc}")
        result_data["simulator"] = {"error": str(exc)}
        optimal_params = None
        sim_ok = False

    if args.dry_run:
        result_data["pass"] = sim_ok
        result_data["notes"] = "Dry-run: simulator only"
        save_result("fpc_qaoa_zne_ibm", result_data)
        print(f"\n  PASS: {result_data['pass']}")
        return

    # ------------------------------------------------------------------
    # Build raw QAOA circuit with warm-start parameters
    # ------------------------------------------------------------------
    print(f"\n[circuit] Building QAOA p={args.depth} circuit ...")
    try:
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit_optimization.translators import to_ising
        from qiskit_optimization import QuadraticProgram

        _qp = QuadraticProgram()
        for _i in range(qubo.num_variables):
            _qp.binary_var(f"x{_i}")
        _lin, _quad = {}, {}
        for (_ii, _jj), _val in qubo.Q.items():
            if _ii == _jj:
                _lin[f"x{_ii}"] = _lin.get(f"x{_ii}", 0.0) + _val
            else:
                _quad[(f"x{_ii}", f"x{_jj}")] = (
                    _quad.get((f"x{_ii}", f"x{_jj}"), 0.0) + _val
                )
        _qp.minimize(linear=_lin, quadratic=_quad)
        _ising, _ = to_ising(_qp)
        raw_circuit = QAOAAnsatz(_ising, reps=args.depth)

        # Use COBYLA warm-start params if available; else schedule seed
        if optimal_params is not None and len(optimal_params) == 2 * args.depth:
            bind_params = optimal_params
            print(f"  Using COBYLA-optimised parameters (from simulator)")
        else:
            from quantum_mht.solvers.fpc_qaoa_solver import digitize_schedule, _polynomial_schedule
            gamma_coeffs = [0.0, np.pi] + [0.0] * (args.schedule_params - 2)
            beta_coeffs = [np.pi / 4] + [0.0] * (args.schedule_params - 1)
            gammas = digitize_schedule(_polynomial_schedule, gamma_coeffs, args.depth)
            betas = digitize_schedule(_polynomial_schedule, beta_coeffs, args.depth)
            bind_params = [v for g, b in zip(gammas, betas) for v in (g, b)]
            print(f"  Using FPC schedule seed (fallback)")

        raw_circuit = raw_circuit.assign_parameters(
            dict(zip(raw_circuit.parameters, bind_params))
        )
        print(f"  Logical qubits: {raw_circuit.num_qubits}")
        print(f"  Logical depth : {raw_circuit.depth()}")
        result_data["circuit_info"] = {
            "logical_qubits": raw_circuit.num_qubits,
            "logical_depth": raw_circuit.depth(),
        }
        circuit_ok = True
    except Exception as exc:
        print(f"  [error] Circuit build failed: {exc}")
        result_data["circuit_info"] = {"error": str(exc)}
        result_data["pass"] = False
        result_data["notes"] = f"Circuit build failed: {exc}"
        save_result("fpc_qaoa_zne_ibm", result_data)
        return

    # ------------------------------------------------------------------
    # Connect to IBM backend + pass manager
    # ------------------------------------------------------------------
    token = get_ibm_token_optional()
    channel = args.channel or ibm_channel()
    instance = args.instance or ibm_instance()
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if token:
        _svc_kwargs: dict = {"channel": channel, "token": token}
        if instance:
            _svc_kwargs["instance"] = instance
        service = QiskitRuntimeService(**_svc_kwargs)
    else:
        service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

    # ------------------------------------------------------------------
    # ZNE: fold → transpile → run for each scale
    # ------------------------------------------------------------------
    print(f"\n[zne] Running FPC-QAOA at scale factors {int_scales} ...")
    scale_results: dict[int, dict] = {}
    job_ids: list[str] = []

    for scale in int_scales:
        print(f"  Scale={scale}: folding circuit (logical depth ×{scale}) ...")
        try:
            folded_raw = _fold_circuit(raw_circuit, scale)
            isa_folded = pm.run(folded_raw)
            isa_depth = isa_folded.depth()
            print(f"    ISA depth at scale={scale}: {isa_depth}")

            sampler = SamplerV2(mode=backend)
            job = sampler.run([isa_folded], shots=args.shots)
            job_id = job.job_id()
            job_ids.append(job_id)
            print(f"    Job ID: {job_id} — waiting ...")

            counts = dict(job.result()[0].data.meas.get_counts())
            avg_obj = _eval_avg_qubo_obj(counts, qubo)
            best_obj = min(_eval_qubo_obj(qubo, bs) for bs in counts)
            top_counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:5])

            scale_results[scale] = {
                "isa_depth": isa_depth,
                "job_id": job_id,
                "avg_qubo_obj": round(avg_obj, 6),
                "best_qubo_obj": round(best_obj, 6),
                "top_counts": top_counts,
            }
            print(f"    avg QUBO obj: {avg_obj:.4f}  |  best QUBO obj: {best_obj:.4f}")
        except Exception as exc:
            print(f"    [error] scale={scale} failed: {exc}")
            scale_results[scale] = {"error": str(exc)}

    result_data["scale_results"] = scale_results

    # ------------------------------------------------------------------
    # Richardson extrapolation on avg QUBO objective
    # ------------------------------------------------------------------
    valid_scales = [s for s in int_scales if "avg_qubo_obj" in scale_results.get(s, {})]
    if len(valid_scales) >= 2:
        scale_vals = [float(s) for s in valid_scales]
        avg_objs = [scale_results[s]["avg_qubo_obj"] for s in valid_scales]
        best_objs = [scale_results[s]["best_qubo_obj"] for s in valid_scales]

        zne_avg_obj = _richardson_extrapolate(scale_vals, avg_objs)
        zne_best_obj = _richardson_extrapolate(scale_vals, best_objs)

        raw_avg_obj = scale_results[int_scales[0]]["avg_qubo_obj"]
        raw_best_obj = scale_results[int_scales[0]]["best_qubo_obj"]

        hungarian_abs = max(abs(hungarian.objective_value), 1e-9)
        raw_avg_ratio = raw_avg_obj / hungarian_abs
        zne_avg_ratio = zne_avg_obj / hungarian_abs
        raw_best_ratio = raw_best_obj / hungarian_abs
        zne_best_ratio = zne_best_obj / hungarian_abs

        zne_avg_improved = abs(zne_avg_obj) > abs(raw_avg_obj)  # more negative = better
        zne_best_improved = abs(zne_best_obj) > abs(raw_best_obj)

        print(f"\n[zne] Richardson extrapolation results:")
        print(f"  RAW  avg obj: {raw_avg_obj:.4f}  (ratio: {raw_avg_ratio:.3f})")
        print(f"  ZNE  avg obj: {zne_avg_obj:.4f}  (ratio: {zne_avg_ratio:.3f})")
        print(f"  RAW  best obj: {raw_best_obj:.4f}  (ratio: {raw_best_ratio:.3f})")
        print(f"  ZNE  best obj: {zne_best_obj:.4f}  (ratio: {zne_best_ratio:.3f})")
        print(f"  Hungarian   : {hungarian.objective_value:.4f}")
        print(f"  ZNE improved avg_obj: {zne_avg_improved}")
        print(f"  ZNE improved best_obj: {zne_best_improved}")

        result_data["zne"] = {
            "valid_scales": valid_scales,
            "avg_obj_by_scale": {str(s): scale_results[s]["avg_qubo_obj"] for s in valid_scales},
            "best_obj_by_scale": {str(s): scale_results[s]["best_qubo_obj"] for s in valid_scales},
            "raw_avg_obj": round(raw_avg_obj, 6),
            "zne_avg_obj": round(zne_avg_obj, 6),
            "raw_best_obj": round(raw_best_obj, 6),
            "zne_best_obj": round(zne_best_obj, 6),
            "raw_avg_ratio": round(raw_avg_ratio, 4),
            "zne_avg_ratio": round(zne_avg_ratio, 4),
            "raw_best_ratio": round(raw_best_ratio, 4),
            "zne_best_ratio": round(zne_best_ratio, 4),
            "hungarian_objective": hungarian.objective_value,
            "zne_avg_improved": zne_avg_improved,
            "zne_best_improved": zne_best_improved,
            "job_ids": job_ids,
        }
        passed = len(valid_scales) >= 2  # at least 2 scales ran successfully
    else:
        print(f"  [warning] Not enough valid scales for Richardson extrapolation")
        result_data["zne"] = {"error": "Insufficient valid scales"}
        passed = False

    result_data["pass"] = passed
    result_data["notes"] = (
        f"ZNE avg improved: {zne_avg_improved}, ZNE best improved: {zne_best_improved}"
        if passed else "Insufficient scales"
    )
    save_result("fpc_qaoa_zne_ibm", result_data)
    print(f"\n  PASS: {passed}")


if __name__ == "__main__":
    main()
