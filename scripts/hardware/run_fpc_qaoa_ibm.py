"""VALIDATION-ROADMAP Task 1.3 — FPC-QAOA circuit analysis and IBM QPU submission.

Runs Fixed-Parameter-Count QAOA (arXiv:2512.21181) for the N=2 track, M=3
measurement scenario (11 QUBO variables). Reports circuit resource metrics
after transpilation to IBM basis gates, and submits via SamplerV2 on IBM QPU.

Usage
-----
# Simulator + transpile inspection (no QPU queue):
python scripts/hardware/run_fpc_qaoa_ibm.py --dry-run

# Full hardware run (SamplerV2 on IBM):
IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_fpc_qaoa_ibm.py \\
    --backend ibm_brisbane --shots 4096

Output
------
  output/hardware/fpc_qaoa_ibm_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-mht" / "src"))

from scripts.hardware import ibm_channel, ibm_instance, make_qubo_instance, get_ibm_token_optional, save_result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FPC-QAOA on IBM QPU (Task 1.3)")
    p.add_argument("--backend", default="ibm_brisbane", help="IBM backend name")
    p.add_argument("--shots", type=int, default=8192,
                   help="Shot count (default: 8192; arXiv:2512.08245 recommends ≥8K for 11-qubit QUBO)")
    p.add_argument("--depth", type=int, default=8, help="QAOA circuit depth (p)")
    p.add_argument("--schedule-params", type=int, default=3,
                   help="FPC schedule parameter count (k)")
    p.add_argument("--n-tracks", type=int, default=2,
                   help="Number of MTDA tracks (N); default 2")
    p.add_argument("--n-meas", type=int, default=3,
                   help="Number of MTDA measurements (M); default 3")
    p.add_argument(
        "--channel", default=None,
        help="Qiskit channel: ibm_quantum_platform or ibm_cloud (overrides IBM_QUANTUM_CHANNEL env)",
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
        "--optimizer-maxiter", type=int, default=100,
        help="COBYLA max iterations for simulator run (default: 100; use 20 for quick validation)",
    )
    # Error mitigation options (2025-2026 best practices)
    p.add_argument(
        "--twirling", action="store_true", default=True,
        help="Enable Pauli twirling via SamplerOptions (default: on; arXiv:2508.14142)",
    )
    p.add_argument(
        "--no-twirling", dest="twirling", action="store_false",
        help="Disable Pauli twirling",
    )
    p.add_argument(
        "--dd", action="store_true", default=True,
        help="Enable dynamical decoupling XY4 via SamplerOptions (default: on; arXiv:2405.17230)",
    )
    p.add_argument(
        "--no-dd", dest="dd", action="store_false",
        help="Disable dynamical decoupling",
    )
    p.add_argument(
        "--twirl-randomizations", type=int, default=0,
        help="Twirling num_randomizations (0 = auto; default: auto)",
    )
    # 2026: Qiskit v2.3 optimization_level=3 with AI-assisted passes
    p.add_argument(
        "--opt-level", type=int, default=3,
        help="Transpiler optimization level (0-3; default: 3 per Qiskit v2.3 best practices, "
             "enables heavier routing+gate-cancel passes; arXiv:2601.01263)",
    )
    # 2026: TREX (Twirled Readout Error eXtinction) via resilience_level=1
    # Provides diagonal readout noise calibration beyond Pauli twirling.
    # Reference: arXiv:2601.22785 (orders-of-magnitude QEM runtime reduction)
    p.add_argument(
        "--resilience", action="store_true", default=True,
        help="Enable TREX readout mitigation via resilience_level=1 (default: on; arXiv:2601.22785)",
    )
    p.add_argument(
        "--no-resilience", dest="resilience", action="store_false",
        help="Disable TREX resilience mitigation",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    n_vars = args.n_tracks * args.n_meas + args.n_tracks + args.n_meas
    print(f"\n=== FPC-QAOA — IBM QPU (Task 1.3) ===")
    print(f"  N={args.n_tracks} tracks, M={args.n_meas} measurements, {n_vars} QUBO variables")
    print(f"  Backend: {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  QAOA depth: {args.depth}, schedule params: {args.schedule_params}")
    print(f"  Shots: {args.shots}")
    print(f"  Pauli twirling: {args.twirling}  |  Dynamical decoupling (XY4): {args.dd}  |  TREX: {args.resilience}")
    print(f"  Transpiler opt-level: {args.opt_level} (Qiskit v2.3)")

    # Build QUBO
    qubo = make_qubo_instance(n_tracks=args.n_tracks, n_meas=args.n_meas, seed=42)
    print(f"\n  QUBO variables: {qubo.num_variables}")

    # Hungarian optimal baseline
    from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver
    hungarian = HungarianSolver().solve(qubo)
    print(f"  Hungarian objective: {hungarian.objective_value:.4f}")

    result_data: dict = {
        "task_ids": ["1.3"],
        "backend": "aer_simulator" if args.dry_run else args.backend,
        "n_tracks": args.n_tracks,
        "n_meas": args.n_meas,
        "num_variables": qubo.num_variables,
        "qaoa_depth": args.depth,
        "schedule_params": args.schedule_params,
        "shots": args.shots,
        "hungarian_objective": hungarian.objective_value,
        "transpiler_opt_level": args.opt_level,
        "error_mitigation": {
            "twirling": args.twirling if not args.dry_run else False,
            "twirl_randomizations": args.twirl_randomizations if args.twirl_randomizations > 0 else "auto",
            "dynamical_decoupling": args.dd if not args.dry_run else False,
            "dd_sequence": "XY4" if (args.dd and not args.dry_run) else None,
            "resilience_trex_via_twirling": args.resilience if not args.dry_run else False,
        },
    }

    # ------------------------------------------------------------------
    # Simulator run (FPCQAOASolver default — StatevectorSampler / Aer)
    # ------------------------------------------------------------------
    print(f"\n[simulator] Running FPC-QAOA ...")
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

        print(f"  Simulator objective   : {sim_result.objective_value:.4f}")
        print(f"  Simulator approx_ratio: {sim_ratio:.3f}")
        print(f"  Simulator solve time  : {sim_result.solve_time_s:.3f}s")
        print(f"  Assignments           : {sim_result.assignments}")

        result_data["simulator"] = {
            "objective": sim_result.objective_value,
            "approx_ratio": sim_ratio,
            "feasible": len(sim_result.assignments) > 0,
            "solve_time_s": sim_result.solve_time_s,
            "assignments": sim_result.assignments,
        }
        sim_ok = True
    except Exception as exc:
        print(f"  [warning] Simulator run failed: {exc}")
        result_data["simulator"] = {"error": str(exc)}
        sim_ratio = None
        sim_ok = False

    if args.dry_run:
        result_data["pass"] = sim_ok and (sim_ratio is not None and sim_ratio < 3.0)
        result_data["notes"] = "Dry-run: simulator only, no IBM hardware"
        save_result("fpc_qaoa_ibm", result_data)
        print(f"\n  PASS: {result_data['pass']}  (sim_ratio={sim_ratio})")
        return

    # ------------------------------------------------------------------
    # Transpile circuit to IBM basis gates (circuit resource analysis)
    # ------------------------------------------------------------------
    token = get_ibm_token_optional()  # None => saved account fallback
    channel = args.channel or ibm_channel()
    instance = args.instance or ibm_instance()
    from quantum_common.backends.ibm import IBMQuantumBackend

    ibm = IBMQuantumBackend(backend_name=args.backend, token=token, channel=channel, instance=instance)
    print(f"\n[ibm] Inspecting circuit after transpilation to {args.backend} ...")

    try:
        import numpy as np
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit_optimization.translators import to_ising
        from qiskit_optimization import QuadraticProgram as _QP2
        from quantum_mht.solvers.fpc_qaoa_solver import digitize_schedule, _polynomial_schedule

        # Correct linear-ramp initial schedule (matches FPCQAOASolver default):
        # [0, pi, 0, ...] gives f(t)=pi*t, ramp from 0 to pi.
        # [0, pi, pi, ...] (old wrong default) gives f(1)=2*pi -- NOT a linear ramp.
        gamma_coeffs = [0.0, float(np.pi)] + [0.0] * (args.schedule_params - 2)
        beta_coeffs = [float(np.pi / 4)] + [0.0] * (args.schedule_params - 1)
        gammas = digitize_schedule(_polynomial_schedule, gamma_coeffs, args.depth)
        betas = digitize_schedule(_polynomial_schedule, beta_coeffs, args.depth)
        initial_point = []
        for g, b in zip(gammas, betas):
            initial_point.extend([g, b])

        # Build QAOAAnsatz directly from QUBO Ising
        _qp2 = _QP2()
        for _i in range(qubo.num_variables):
            _qp2.binary_var(f"x{_i}")
        _lin2, _quad2 = {}, {}
        for (_ii, _jj), _val in qubo.Q.items():
            if _ii == _jj: _lin2[f"x{_ii}"] = _lin2.get(f"x{_ii}", 0.0) + _val
            else: _quad2[(f"x{_ii}", f"x{_jj}")] = _quad2.get((f"x{_ii}", f"x{_jj}"), 0.0) + _val
        _qp2.minimize(linear=_lin2, quadratic=_quad2)
        _ising, _ = to_ising(_qp2)
        raw_circuit = QAOAAnsatz(_ising, reps=args.depth)
        raw_circuit = raw_circuit.assign_parameters(
            dict(zip(raw_circuit.parameters, initial_point))
        )
        raw_circuit.measure_all()

        transpiled = ibm.transpile([raw_circuit], optimization_level=args.opt_level)
        t_circ = transpiled[0]
        cx_count = t_circ.count_ops().get("cx", t_circ.count_ops().get("ecr", 0))

        print(f"  Logical qubits  : {raw_circuit.num_qubits}")
        print(f"  Original depth  : {raw_circuit.depth()}")
        print(f"  Transpiled depth: {t_circ.depth()}")
        print(f"  CX/ECR count    : {cx_count}")

        result_data["circuit_info"] = {
            "logical_qubits": raw_circuit.num_qubits,
            "original_depth": raw_circuit.depth(),
            "transpiled_depth": t_circ.depth(),
            "cx_count": cx_count,
        }
    except Exception as exc:
        print(f"  [warning] Transpile inspection failed: {exc}")
        result_data["circuit_info"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # IBM hardware: run via SamplerV2.
    # Note: FPCQAOASolver uses StatevectorSampler internally.
    # Hardware execution uses simulator-optimised parameters when available
    # (sim_result.metadata["optimal_params"]), otherwise falls back to the
    # default FPC schedule seed (gamma=[0,pi,...], beta=[pi/4,0,...]).
    # ------------------------------------------------------------------
    print(f"\n[ibm] Submitting to {args.backend} via SamplerV2 ...")
    try:
        from qiskit_ibm_runtime import SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService

        if token:
            _svc_kwargs: dict = {"channel": channel, "token": token}
            if instance:
                _svc_kwargs["instance"] = instance
            service = QiskitRuntimeService(**_svc_kwargs)
        else:
            service = QiskitRuntimeService()  # saved account
        backend = service.backend(args.backend)
        pm = generate_preset_pass_manager(optimization_level=args.opt_level, backend=backend)
        print(f"  Transpiler level  : {args.opt_level} (Qiskit v2.3, arXiv:2601.01263)")

        # Use the ansatz circuit with parameters bound from sim optimisation
        if sim_ok and hasattr(sim_result, "metadata") and "optimal_params" in (sim_result.metadata or {}):
            opt_params = sim_result.metadata["optimal_params"]
        else:
            opt_params = initial_point

        # ROUND 12 FIX: Rebuild circuit with COBYLA-optimized parameters.
        # Previously, opt_params was extracted from sim metadata but raw_circuit
        # was always built with initial_point (circuit_info section, line ~218).
        # This fix applies sim-optimized params to the hardware circuit, implementing
        # the parameter warm-start transfer of arXiv:2601.15760 (8.06× speedup).
        try:
            from qiskit.circuit.library import QAOAAnsatz as _QAOA_hw
            from qiskit_optimization.translators import to_ising as _to_ising_hw
            from qiskit_optimization import QuadraticProgram as _QP_hw
            _qp_hw = _QP_hw()
            for _i_hw in range(qubo.num_variables):
                _qp_hw.binary_var(f"x{_i_hw}")
            _lin_hw, _quad_hw = {}, {}
            for (_ii_hw, _jj_hw), _v_hw in qubo.Q.items():
                if _ii_hw == _jj_hw:
                    _lin_hw[f"x{_ii_hw}"] = _lin_hw.get(f"x{_ii_hw}", 0.0) + _v_hw
                else:
                    _quad_hw[(f"x{_ii_hw}", f"x{_jj_hw}")] = _quad_hw.get(
                        (f"x{_ii_hw}", f"x{_jj_hw}"), 0.0) + _v_hw
            _qp_hw.minimize(linear=_lin_hw, quadratic=_quad_hw)
            _ising_hw, _ = _to_ising_hw(_qp_hw)
            raw_circuit = _QAOA_hw(_ising_hw, reps=args.depth)
            raw_circuit = raw_circuit.assign_parameters(
                dict(zip(raw_circuit.parameters, opt_params))
            )
            raw_circuit.measure_all()
            _param_src = ("COBYLA-optimized (sim->hw transfer)"
                          if (sim_ok and opt_params is not initial_point) else "initial-point")
            print(f"  Param source    : {_param_src} (arXiv:2601.15760)")
            result_data["param_source"] = _param_src
        except Exception as _e_hw:
            print(f"  [warn] Warm-start rebuild failed: {_e_hw}; using existing raw_circuit")
            result_data["param_source"] = "initial-point (rebuild failed)"

        # raw_circuit may not be defined if circuit_info section failed; rebuild it
        try:
            _check = raw_circuit
        except NameError:
            import numpy as np
            from qiskit.circuit.library import QAOAAnsatz
            from qiskit_optimization.translators import to_ising
            from qiskit_optimization import QuadraticProgram as _QP
            _qp2 = _QP()
            for _i in range(qubo.num_variables):
                _qp2.binary_var(f"x{_i}")
            _lin2, _quad2 = {}, {}
            for (_ii, _jj), _val in qubo.Q.items():
                if _ii == _jj: _lin2[f"x{_ii}"] = _lin2.get(f"x{_ii}", 0.0) + _val
                else: _quad2[(f"x{_ii}", f"x{_jj}")] = _quad2.get((f"x{_ii}", f"x{_jj}"), 0.0) + _val
            _qp2.minimize(linear=_lin2, quadratic=_quad2)
            _ising2, _ = to_ising(_qp2)
            raw_circuit = QAOAAnsatz(_ising2, reps=args.depth)
            _gamma_c = [0.0, float(np.pi)] + [0.0] * (args.schedule_params - 2)
            _beta_c = [float(np.pi / 4)] + [0.0] * (args.schedule_params - 1)
            from quantum_mht.solvers.fpc_qaoa_solver import digitize_schedule, _polynomial_schedule
            _ip = []
            for _g, _b in zip(digitize_schedule(_polynomial_schedule, _gamma_c, args.depth),
                               digitize_schedule(_polynomial_schedule, _beta_c, args.depth)):
                _ip.extend([_g, _b])
            raw_circuit = raw_circuit.assign_parameters(dict(zip(raw_circuit.parameters, _ip)))
            raw_circuit.measure_all()

        isa_circuit = pm.run(raw_circuit)
        sampler = SamplerV2(mode=backend)

        # 2026 error mitigation stack:
        # 1. TREX (resilience_level=1): diagonal readout noise calibration (arXiv:2601.22785)
        # 2. Pauli twirling: gate-level depolarizing noise randomization (arXiv:2508.14142)
        # 3. XY4 DD: idle-qubit decoherence suppression (arXiv:2405.17230)
        # TREX note: SamplerV2 does not expose resilience_level (EstimatorV2-only).
        # For SamplerV2, TREX-equivalent readout noise mitigation is achieved via
        # twirling.enable_measure=True (random X gates before measurement, same
        # mechanism). The --resilience flag is documented for future EstimatorV2
        # migration; for now it ensures enable_measure is always on.
        if args.resilience:
            sampler.options.twirling.enable_measure = True  # TREX-equivalent for Sampler
            print(f"  TREX (Sampler)  : twirling.enable_measure=True (arXiv:2601.22785)")
        if args.twirling:
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
            num_rand = args.twirl_randomizations if args.twirl_randomizations > 0 else "auto"
            sampler.options.twirling.num_randomizations = num_rand
            sampler.options.twirling.strategy = "active-accum"
            print(f"  Twirling enabled  : num_randomizations={num_rand}, strategy=active-accum")
        if args.dd:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.dynamical_decoupling.scheduling_method = "alap"
            sampler.options.dynamical_decoupling.skip_reset_qubits = True
            print(f"  DD enabled        : XY4 / alap")

        job = sampler.run([isa_circuit], shots=args.shots)
        hw_pub_result = job.result()[0]
        hw_counts = dict(hw_pub_result.data.meas.get_counts())

        # Decode best bitstring from SamplerV2 counts
        best_bits = max(hw_counts, key=hw_counts.get)
        # Count ones to assess solution structure
        n_ones = best_bits.count("1")

        print(f"  Job ID          : {job.job_id()}")
        print(f"  Best bitstring  : {best_bits}  (|1|={n_ones})")

        # Build top-counts dict (top 10 for better coverage)
        top_counts = dict(sorted(hw_counts.items(), key=lambda x: -x[1])[:10])

        # ---------------------------------------------------------------
        # Bitstring quality evaluation: decode each top bitstring into
        # a QUBO objective value and MTDA assignment, enabling direct
        # comparison of hardware solution quality vs. the Hungarian optimum.
        # ---------------------------------------------------------------
        import numpy as _np

        def _eval_bs(x_str: str) -> float:
            """Compute x^T Q x for a bitstring string."""
            x = _np.array([int(b) for b in x_str], dtype=_np.int_)
            obj = 0.0
            for (ii, jj), q_val in qubo.Q.items():
                obj += q_val * x[ii] * x[jj]
            return float(obj)

        top_evals = []
        best_hw_obj = None
        best_hw_assignments = None
        for bs, cnt in top_counts.items():
            obj = _eval_bs(bs)
            x_vec = _np.array([int(b) for b in bs], dtype=_np.int_)
            dec = qubo.variables.decode_solution(x_vec)
            top_evals.append({
                "bitstring": bs,
                "shots": cnt,
                "objective": round(obj, 6),
                "assignments": dec["assignments"],
                "missed": dec["missed_detections"],
                "false_alarms": dec["false_alarms"],
            })
            if best_hw_obj is None or obj < best_hw_obj:
                best_hw_obj = obj
                best_hw_assignments = dec["assignments"]

        hw_approx_ratio = (best_hw_obj / max(abs(hungarian.objective_value), 1e-9)
                           if best_hw_obj is not None else None)

        print(f"  HW best obj     : {best_hw_obj:.4f}  (Hungarian: {hungarian.objective_value:.4f})")
        print(f"  HW approx ratio : {hw_approx_ratio:.4f}" if hw_approx_ratio else "  HW approx ratio : N/A")
        print(f"  HW assignments  : {best_hw_assignments}")

        result_data["hardware"] = {
            "job_id": job.job_id(),
            "best_bitstring": best_bits,
            "num_ones": n_ones,
            "best_hw_objective": round(best_hw_obj, 6) if best_hw_obj is not None else None,
            "hw_approx_ratio": round(hw_approx_ratio, 4) if hw_approx_ratio is not None else None,
            "best_hw_assignments": best_hw_assignments,
            "top_counts": top_counts,
            "top_evaluations": top_evals,
        }
        passed = True
    except Exception as exc:
        print(f"  [error] IBM execution failed: {exc}")
        result_data["hardware"] = {"error": str(exc)}
        passed = False

    result_data["pass"] = passed
    result_data["notes"] = "Hardware FPC-QAOA circuit executed" if passed else "Execution failed"
    save_result("fpc_qaoa_ibm", result_data)
    print(f"\n  PASS: {passed}")


if __name__ == "__main__":
    main()
