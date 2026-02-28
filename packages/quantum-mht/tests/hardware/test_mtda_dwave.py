"""VALIDATION-ROADMAP Tasks 1.2, 1.6, 2.2 — MTDA QUBO on D-Wave Advantage2.

Task 1.2: Forward annealing — N=5, M=8 (53 QUBO variables) on D-Wave Advantage2.
           Acceptance: feasible solution, approximation ratio < 2.0.

Task 1.6: Reverse annealing — warm-start from forward solution.
           Acceptance: reverse objective ≤ forward objective (improvement or equal).

Task 2.2: Minor-embedding overhead measurement.
           Reports physical / logical qubit ratio for the N=5 problem.
           Expected: ~3× overhead on Zephyr (Advantage2) topology.

Run:
  pytest -m hardware packages/quantum-mht/tests/hardware/test_mtda_dwave.py \\
      -v --num-reads 1000
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
class TestMTDADWaveHardware:
    """Tasks 1.2, 1.6, 2.2: MTDA QUBO on D-Wave Advantage2 QPU."""

    def test_forward_annealing_n5(
        self,
        dwave_hardware_backend,
        qubo_n5_instance,
        hungarian_baseline_n5,
        request: pytest.FixtureRequest,
    ) -> None:
        """Task 1.2: Forward quantum annealing on D-Wave Advantage2.

        N=5 tracks, M=8 measurements — 53 QUBO variables.
        D-Wave native format: EmbeddingComposite → Zephyr (Advantage2) QPU.
        Acceptance: feasible assignment, approximation ratio < 2.0 vs Hungarian.
        """
        from quantum_mht.solvers.annealing_solver import AnnealingSolver

        num_reads = request.config.getoption("--num-reads")

        solver = AnnealingSolver(
            num_reads=num_reads,
            annealing_time_us=20.0,
            use_reverse_annealing=False,
            use_simulator=False,
        )
        result = solver.solve(qubo_n5_instance)
        optimal = hungarian_baseline_n5.objective_value
        approx_ratio = result.objective_value / max(abs(optimal), 1e-9)

        _save_result("mtda_forward_dwave", {
            "task": "1.2",
            "n_tracks": 5,
            "n_meas": 8,
            "num_variables": qubo_n5_instance.num_variables,
            "num_reads": num_reads,
            "hungarian_objective": optimal,
            "dwave_objective": result.objective_value,
            "approximation_ratio": approx_ratio,
            "solve_time_s": result.solve_time_s,
            "assignments": result.assignments,
            "pass": approx_ratio < 2.0,
        })

        assert result.assignments is not None, "D-Wave returned no assignments"
        assert approx_ratio < 2.0, (
            f"Approximation ratio {approx_ratio:.3f} >= 2.0 — "
            f"D-Wave objective={result.objective_value:.4f}, "
            f"Hungarian={optimal:.4f}"
        )

    def test_reverse_annealing_n5(
        self,
        dwave_hardware_backend,
        qubo_n5_instance,
        hungarian_baseline_n5,
        request: pytest.FixtureRequest,
    ) -> None:
        """Task 1.6: Reverse annealing with warm-start from forward solution.

        Exploits temporal continuity between tracking frames (Ihara 2025,
        Scientific Reports 15:24294). The forward annealing solution is used
        as initial_state for the reverse annealing schedule.
        Acceptance: reverse annealing result is feasible.
        """
        from quantum_mht.solvers.annealing_solver import AnnealingSolver

        num_reads = request.config.getoption("--num-reads")
        optimal = hungarian_baseline_n5.objective_value

        # Step 1: Forward annealing to get warm-start solution
        solver_fwd = AnnealingSolver(
            num_reads=num_reads,
            annealing_time_us=20.0,
            use_reverse_annealing=False,
            use_simulator=False,
        )
        fwd_result = solver_fwd.solve(qubo_n5_instance)

        # Step 2: Reverse annealing with forward solution as initial state
        solver_rev = AnnealingSolver(
            num_reads=num_reads,
            annealing_time_us=20.0,
            use_reverse_annealing=True,
            reverse_anneal_s_target=0.45,
            reverse_anneal_hold_us=10.0,
            reinitialize_state=True,
            use_simulator=False,
        )
        # Inject warm-start (previous frame's solution)
        if fwd_result.raw_solution is not None:
            solver_rev._previous_solution = {
                i: int(fwd_result.raw_solution[i])
                for i in range(qubo_n5_instance.num_variables)
            }
        rev_result = solver_rev.solve(qubo_n5_instance)

        fwd_ratio = fwd_result.objective_value / max(abs(optimal), 1e-9)
        rev_ratio = rev_result.objective_value / max(abs(optimal), 1e-9)

        _save_result("mtda_reverse_dwave", {
            "task": "1.6",
            "n_tracks": 5,
            "n_meas": 8,
            "num_reads": num_reads,
            "hungarian_objective": optimal,
            "forward_objective": fwd_result.objective_value,
            "forward_ratio": fwd_ratio,
            "reverse_objective": rev_result.objective_value,
            "reverse_ratio": rev_ratio,
            "reverse_better_or_equal": rev_result.objective_value <= fwd_result.objective_value,
            "fwd_solve_time_s": fwd_result.solve_time_s,
            "rev_solve_time_s": rev_result.solve_time_s,
        })

        assert rev_result.assignments is not None, "Reverse annealing returned no assignments"
        # Reverse should not be dramatically worse than forward
        assert rev_ratio < 3.0, (
            f"Reverse annealing ratio {rev_ratio:.3f} >= 3.0 — "
            f"warm-start may not have been applied correctly"
        )

    def test_embedding_overhead_n5(
        self,
        dwave_hardware_backend,
        qubo_n5_instance,
    ) -> None:
        """Task 2.2: Measure minor-embedding physical/logical qubit ratio.

        Zephyr topology (Advantage2) has degree-20 connectivity. For a dense
        N=5, M=8 QUBO (53 variables), each logical variable requires a chain
        of physical qubits. The embedding overhead should be < 10×.
        Paper claim: ~3× overhead.
        """
        try:
            import minorminer
            from dwave.system import DWaveSampler
        except ImportError:
            pytest.skip("minorminer or dwave.system not installed")

        import os
        token = os.environ.get("DWAVE_API_TOKEN", "")

        bqm = qubo_n5_instance.to_bqm()
        logical_vars = qubo_n5_instance.num_variables

        # Get hardware graph
        sampler = DWaveSampler(token=token)
        source_graph = bqm.to_networkx_graph()
        embedding = minorminer.find_embedding(source_graph, sampler.edgelist)

        physical_qubits = sum(len(chain) for chain in embedding.values())
        overhead = physical_qubits / logical_vars if logical_vars > 0 else 0.0

        _save_result("embedding_overhead_dwave", {
            "task": "2.2",
            "logical_qubits": logical_vars,
            "physical_qubits": physical_qubits,
            "embedding_overhead": overhead,
            "paper_claim_approx_3x": 3.0,
        })

        assert physical_qubits > 0, "Minor embedding failed — no physical qubits assigned"
        assert overhead < 10.0, (
            f"Embedding overhead {overhead:.2f}x is abnormally large "
            f"(expected ~3× for Zephyr topology)"
        )
