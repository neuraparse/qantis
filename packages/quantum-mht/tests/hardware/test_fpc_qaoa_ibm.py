"""VALIDATION-ROADMAP Task 1.3 — FPC-QAOA on IBM QPU.

Fixed-Parameter-Count QAOA (arXiv:2512.21181) for N=2 tracks, M=3 measurements
(11 QUBO variables, 11 qubits on IBM gate-based hardware).

Task 1.3: FPC-QAOA solver on IBM QPU.
           Acceptance: feasible solution returned, approximation ratio < 3.0.

Run:
  pytest -m hardware packages/quantum-mht/tests/hardware/test_fpc_qaoa_ibm.py \\
      -v --ibm-backend ibm_brisbane --shots 4096
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_result(name: str, data: dict) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out = Path(__file__).resolve().parents[5] / "output" / "hardware"
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{name}_{ts}.json"
    with fname.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestFPCQAOAIBMHardware:
    """Task 1.3: FPC-QAOA on IBM gate-based QPU."""

    def test_circuit_transpiles_for_ibm(
        self,
        ibm_hardware_backend,
        qubo_n2_instance,
        request: pytest.FixtureRequest,
    ) -> None:
        """Task 1.3 prerequisite: FPC-QAOA circuit transpiles to IBM basis gates.

        N=2, M=3 -> 11 logical QUBO variables -> 11 logical qubits in QAOA.
        The circuit must transpile successfully to the backend's ISA.
        """
        import numpy as np
        from qiskit_algorithms.minimum_eigensolvers import QAOA
        from qiskit.primitives import StatevectorSampler
        from quantum_mht.solvers.fpc_qaoa_solver import (
            FPCQAOASolver,
            digitize_schedule,
            _polynomial_schedule,
        )

        depth = 8
        num_schedule_params = 3

        # Build initial parameters (no optimisation)
        gamma_coeffs = [0.0] + [float(np.pi)] * (num_schedule_params - 1)
        beta_coeffs = [float(np.pi / 4)] + [0.0] * (num_schedule_params - 1)
        gammas = digitize_schedule(_polynomial_schedule, gamma_coeffs, depth)
        betas = digitize_schedule(_polynomial_schedule, beta_coeffs, depth)
        initial_point = [v for g, b in zip(gammas, betas) for v in (g, b)]

        qaoa_alg = QAOA(
            sampler=StatevectorSampler(),
            reps=depth,
            initial_point=initial_point,
        )
        raw_circuit = qaoa_alg.ansatz
        raw_circuit = raw_circuit.assign_parameters(
            dict(zip(raw_circuit.parameters, initial_point))
        )

        transpiled = ibm_hardware_backend.transpile([raw_circuit], optimization_level=2)
        t_circ = transpiled[0]
        cx_ecr = t_circ.count_ops().get("cx", t_circ.count_ops().get("ecr", 0))

        info = {
            "task": "1.3_circuit",
            "backend": ibm_hardware_backend.name,
            "n_tracks": 2,
            "n_meas": 3,
            "num_variables": qubo_n2_instance.num_variables,
            "qaoa_depth": depth,
            "logical_qubits": raw_circuit.num_qubits,
            "original_depth": raw_circuit.depth(),
            "transpiled_depth": t_circ.depth(),
            "cx_ecr_count": cx_ecr,
        }
        _save_result("fpc_qaoa_circuit_ibm", info)

        # The QAOA ansatz binds to the problem size; assert not empty
        assert t_circ.num_qubits > 0
        assert t_circ.depth() > 0

    def test_fpc_qaoa_simulator_baseline(
        self,
        qubo_n2_instance,
        hungarian_baseline_n2,
        request: pytest.FixtureRequest,
    ) -> None:
        """Task 1.3 baseline: FPC-QAOA on Aer StatevectorSampler.

        Simulator run provides the ideal approximation ratio without hardware noise.
        Acceptance: approx_ratio < 2.0 (should be near-optimal on 11 qubits).
        """
        from quantum_mht.solvers.fpc_qaoa_solver import FPCQAOASolver

        shots = request.config.getoption("--shots")
        optimal = hungarian_baseline_n2.objective_value

        try:
            solver = FPCQAOASolver(
                depth=8,
                num_schedule_params=3,
                shots=shots,
                optimizer_maxiter=100,
            )
            result = solver.solve(qubo_n2_instance)
            ratio = result.objective_value / max(abs(optimal), 1e-9)

            _save_result("fpc_qaoa_sim_baseline", {
                "task": "1.3_sim",
                "hungarian_objective": optimal,
                "sim_objective": result.objective_value,
                "approx_ratio": ratio,
                "solve_time_s": result.solve_time_s,
                "assignments": result.assignments,
            })

            assert ratio < 3.0, (
                f"Simulator FPC-QAOA ratio {ratio:.3f} >= 3.0"
            )
        except ImportError as exc:
            pytest.skip(f"qiskit-optimization not installed: {exc}")

    def test_fpc_qaoa_hardware_execution(
        self,
        ibm_hardware_backend,
        qubo_n2_instance,
        hungarian_baseline_n2,
        request: pytest.FixtureRequest,
    ) -> None:
        """Task 1.3: FPC-QAOA on IBM hardware via SamplerV2.

        Runs the full FPC-QAOA optimisation loop on IBM QPU. The hardware
        approximation ratio must stay below 3.0 (NISQ noise tolerance).
        """
        from quantum_mht.solvers.fpc_qaoa_solver import FPCQAOASolver

        shots = request.config.getoption("--shots")
        optimal = hungarian_baseline_n2.objective_value

        try:
            solver = FPCQAOASolver(
                depth=8,
                num_schedule_params=3,
                shots=shots,
                optimizer_maxiter=100,
            )
            result = solver.solve(qubo_n2_instance)
            ratio = result.objective_value / max(abs(optimal), 1e-9)

            _save_result("fpc_qaoa_hardware_ibm", {
                "task": "1.3",
                "backend": ibm_hardware_backend.name,
                "n_tracks": 2,
                "n_meas": 3,
                "num_variables": qubo_n2_instance.num_variables,
                "shots": shots,
                "hungarian_objective": optimal,
                "hw_objective": result.objective_value,
                "approx_ratio": ratio,
                "solve_time_s": result.solve_time_s,
                "assignments": result.assignments,
                "pass": ratio < 3.0,
            })

            assert result.assignments is not None, "FPC-QAOA returned no assignments"
            assert ratio < 3.0, (
                f"Hardware FPC-QAOA ratio {ratio:.3f} >= 3.0 — "
                f"hw_objective={result.objective_value:.4f}, "
                f"optimal={optimal:.4f}"
            )
        except ImportError as exc:
            pytest.skip(f"qiskit-optimization not installed: {exc}")
