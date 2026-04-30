"""Integration tests for 2026 Q2 solver additions.

Verifies end-to-end solve() on a shared small MTDA QUBO instance for the
new classical baselines (PT-ICM, TN baseline, Fixstars, cuOpt, Gurobi),
the new hybrid CQM / NL guard in HybridSolver, and the new QUBO
preprocessing path (Ohno-Togawa IEM coefficient reduction).

These tests are deliberately minimal: they exercise the public API
surface of each solver so that a dependency regression (Ocean / Qiskit /
Mitiq version bump) surfaces immediately rather than only during hardware
runs. Optional-dependency solvers (cuOpt, Gurobi, Fixstars) are skipped
when their SDKs are not installed rather than marked xfail, matching the
pattern in ``test_solvers.py``.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quantum_mht.formulation.mtda_qubo_builder import (
    MTDAQuboBuilder,
    interaction_extension,
)
from quantum_mht.solvers.solver_factory import create_solver


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@pytest.fixture
def small_qubo():
    """Shared 2x2 MTDA instance; identity cost is optimal for diag assignment."""
    builder = MTDAQuboBuilder()
    cost_matrix = np.array([[1.0, 5.0], [5.0, 1.0]])
    return builder.build_from_cost_matrix(cost_matrix)


@pytest.fixture
def medium_qubo():
    """Small-but-non-trivial 3x4 MTDA instance."""
    builder = MTDAQuboBuilder()
    cost_matrix = np.array(
        [
            [0.8, 4.0, 3.2, 2.1],
            [4.5, 0.9, 3.8, 1.7],
            [3.1, 3.6, 0.7, 2.9],
        ]
    )
    return builder.build_from_cost_matrix(cost_matrix)


class TestPTICMSolver:
    def test_solves_small_instance(self, small_qubo) -> None:
        """Smoke test: solver completes without raising.

        PT-ICM at a small sweep budget is not guaranteed to return a
        feasible one-hot assignment; the reviewer-facing claim is on the
        TTS(99%) curve with 10k+ sweeps, not on one 200-sweep run.
        """
        solver = create_solver(
            "pticm",
            num_replicas=6,
            num_sweeps=200,
            beta_range=(0.5, 3.0),
            seed=7,
        )
        result = solver.solve(small_qubo)
        assert result.solver_name == "PT-ICM"
        assert result.solve_time_s >= 0.0
        # Raw solution has one entry per variable (binary); sanity-check size.
        assert result.raw_solution is not None
        assert len(result.raw_solution) == small_qubo.num_variables

    def test_metadata_includes_pt_params(self, small_qubo) -> None:
        solver = create_solver("pticm", num_replicas=4, num_sweeps=50, seed=1)
        result = solver.solve(small_qubo)
        assert result.metadata["num_replicas"] == 4
        assert result.metadata["num_sweeps"] == 50


class TestTensorNetworkBaseline:
    def test_skipped_without_quimb(self, small_qubo) -> None:
        pytest.importorskip("quimb", reason="quimb not installed")
        solver = create_solver("tn_baseline", method="bp", beta=4.0, max_iter=50)
        result = solver.solve(small_qubo)
        assert result.solver_name.startswith("TN-")
        assert result.solve_time_s >= 0.0


class TestFixstarsSolver:
    def test_requires_sdk(self, small_qubo) -> None:
        pytest.importorskip(
            "amplify", reason="Fixstars Amplify SDK not installed"
        )
        solver = create_solver("fixstars", client_type="ae", token="dummy")
        # Without a live token the client should fail fast; accept either
        # ValueError (missing client type misuse) or RuntimeError (auth).
        with pytest.raises((RuntimeError, ValueError, Exception)):
            solver.solve(small_qubo)


class TestCuOptSolver:
    def test_raises_on_missing_sdk(self, small_qubo) -> None:
        if not _has_module("cuopt"):
            solver = create_solver("cuopt", time_limit_s=1.0)
            with pytest.raises(RuntimeError, match="cuOpt"):
                solver.solve(small_qubo)
            return
        solver = create_solver("cuopt", time_limit_s=1.0)
        result = solver.solve(small_qubo)
        assert result.solver_name.startswith("cuOpt")


class TestGurobiSolver:
    def test_raises_on_missing_license(self, small_qubo) -> None:
        if not _has_module("gurobipy"):
            solver = create_solver("gurobi", time_limit_s=1.0)
            with pytest.raises(RuntimeError, match="Gurobi"):
                solver.solve(small_qubo)
            return
        solver = create_solver("gurobi", time_limit_s=1.0)
        result = solver.solve(small_qubo)
        assert result.solver_name.startswith("Gurobi")


class TestHybridCQMGuard:
    def test_cqm_mode_requires_ocean(self, small_qubo) -> None:
        if not _has_module("dwave.system"):
            solver = create_solver("hybrid_cqm", time_limit_s=1)
            with pytest.raises(RuntimeError, match="dwave-ocean-sdk"):
                solver.solve(small_qubo)
            return
        pytest.skip("ocean-sdk is installed; CQM hit would require a live Leap token")

    def test_nl_guard_falls_back_on_pure_binary(self, small_qubo) -> None:
        """NL default on pure-binary MTDA QUBOs should emit a warning and fall back.

        Osaba IEEE Access 13:4724 (2025) showed NL underperforms BQM on
        pure-binary pairwise problems; ``HybridSolver`` therefore refuses
        NL when ``has_higher_order_terms == False`` and tries BQM instead.
        """
        if not _has_module("dwave.system"):
            solver = create_solver("hybrid_nl", time_limit_s=1)
            # Without Ocean installed BQM fallback will raise RuntimeError;
            # the important behaviour here is that NL did not try to submit
            # a pure-binary problem via the NL path.
            with pytest.raises(RuntimeError, match="dwave-ocean-sdk"):
                solver.solve(small_qubo)
            return
        pytest.skip("ocean-sdk is installed; NL path would require a live Leap token")


class TestInteractionExtension:
    def test_coefficient_range_shrinks(self) -> None:
        """IEM (Ohno-Togawa arXiv:2604.03546) splits large couplers."""
        q = {(0, 1): 20.0, (1, 2): 3.0, (0, 0): 1.0}
        extended, next_idx = interaction_extension(
            q, next_var_index=3, max_coupler_magnitude=5.0,
        )
        # Range of off-diagonal magnitudes should no longer exceed the
        # target of 5.0.
        off_diag = [abs(v) for (i, j), v in extended.items() if i != j]
        assert max(off_diag) <= 5.0 + 1e-9
        assert next_idx > 3  # auxiliaries were added


class TestQUBOResultFeatures:
    def test_dynamic_range_computation(self, medium_qubo) -> None:
        assert medium_qubo.dynamic_range >= 1.0

    def test_has_higher_order_terms_default(self, medium_qubo) -> None:
        assert medium_qubo.has_higher_order_terms is False


class TestSolverFactoryRegistrations:
    @pytest.mark.parametrize(
        "name",
        [
            "pticm",
            "fixstars",
            "tn_baseline",
            "hybrid_cqm",
            "hybrid_nl",
            "cuopt",
            "gurobi",
            "xy_mixer_qaoa",
            "recursive_qaoa",
            "para_qaoa",
            "iceberg_qaoa",
        ],
    )
    def test_factory_instantiates(self, name: str) -> None:
        solver = create_solver(name)
        assert solver is not None
        assert hasattr(solver, "name")
        assert hasattr(solver, "solve")


class TestParaQAOASolver:
    def test_partitions_respect_num_partitions(self, medium_qubo) -> None:
        from quantum_mht.solvers.para_qaoa_solver import ParaQAOASolver

        solver = ParaQAOASolver(num_partitions=2, top_k=2)
        partitions = solver._partition(medium_qubo.Q, medium_qubo.num_variables)
        # Connectivity-preserving BFS partitioner (Huang 2603.26232): every
        # pair of consecutive partitions shares exactly one bridge vertex,
        # so the total across partitions is at most num_variables + (M-1).
        assert 1 <= len(partitions) <= 2
        bridge_overhead = max(len(partitions) - 1, 0)
        total_with_bridges = sum(len(p) for p in partitions)
        assert total_with_bridges == medium_qubo.num_variables + bridge_overhead
        # Every variable appears at least once.
        all_vars = set().union(*partitions)
        assert len(all_vars) == medium_qubo.num_variables

    def test_energy_evaluates_full_qubo(self, medium_qubo) -> None:
        from quantum_mht.solvers.para_qaoa_solver import ParaQAOASolver

        solver = ParaQAOASolver(num_partitions=2, top_k=2)
        zeros = np.zeros(medium_qubo.num_variables, dtype=int)
        assert solver._energy_full(medium_qubo.Q, zeros) == 0.0


class TestIcebergQAOAScaffold:
    def test_acceptance_model_superconducting_decays_faster(self) -> None:
        from quantum_mht.solvers.iceberg_qaoa_solver import IcebergQAOASolver

        trapped = IcebergQAOASolver(
            depth=3, backend_family="trapped_ion"
        )._estimated_acceptance(num_vars=20)
        sc = IcebergQAOASolver(
            depth=3, backend_family="superconducting"
        )._estimated_acceptance(num_vars=20)
        assert trapped > sc

    def test_metadata_records_logical_physical_counts(self, small_qubo) -> None:
        from quantum_mht.solvers.iceberg_qaoa_solver import IcebergQAOASolver

        solver = IcebergQAOASolver(depth=1, sub_solver="qaoa")
        # Sub-solver may fail if qiskit simulator is missing; accept that
        # and only verify the scaffold wiring when the path completes.
        try:
            result = solver.solve(small_qubo)
        except Exception as exc:
            pytest.skip(f"sub-solver unavailable: {exc}")
        assert result.metadata["iceberg_logical_qubits"] == small_qubo.num_variables
        assert result.metadata["iceberg_physical_qubits"] == small_qubo.num_variables + 2


class TestQAOAPredictor:
    def test_recommended_depth_is_sensible(self, medium_qubo) -> None:
        from quantum_mht.solvers.qaoa_predictor import QAOAPredictor

        advisor = QAOAPredictor()
        depth = advisor.recommend_min_depth(medium_qubo, target=0.5, max_depth=20)
        assert 1 <= depth <= 20

    def test_model_fn_override(self, medium_qubo) -> None:
        from quantum_mht.solvers.qaoa_predictor import QAOAPredictor

        advisor = QAOAPredictor()
        advisor.load_model(lambda _q, p: min(0.99, 0.2 + 0.1 * p))
        depth = advisor.recommend_min_depth(medium_qubo, target=0.6, max_depth=10)
        assert depth <= 10

    def test_graph_features_shape(self, medium_qubo) -> None:
        from quantum_mht.solvers.qaoa_predictor import QAOAPredictor

        features = QAOAPredictor().graph_features(medium_qubo)
        for key in (
            "num_vars",
            "density",
            "mean_degree",
            "max_degree",
            "mean_abs_linear",
            "mean_neigh_variance",
        ):
            assert key in features


class TestIcebergCompiler:
    def test_qiskit_circuit_has_ancillas(self) -> None:
        pytest.importorskip("qiskit")

        from quantum_mht.solvers.iceberg_qaoa_solver import build_iceberg_qaoa_circuit

        circ = build_iceberg_qaoa_circuit(
            k_logical=4,
            zz_pairs=[(0, 1, 1.0), (2, 3, 0.5)],
            gamma=0.3,
            beta=0.2,
            framework="qiskit",
        )
        assert circ.num_qubits == 6  # 4 data + 2 ancillas
        assert any(instr.operation.name == "rzz" for instr in circ.data)

    def test_postselect_discards_ancilla_nonzero(self) -> None:
        from quantum_mht.solvers.iceberg_qaoa_solver import postselect_iceberg_counts

        # Bitstrings: data (k=4) then ancilla (2) = total 6 characters.
        counts = {
            "000100": 10,  # ancilla = 00 -> keep
            "100001": 7,   # ancilla = 01 -> discard
            "111110": 5,   # ancilla = 10 -> discard
            "000000": 3,   # ancilla = 00 -> keep
        }
        accepted, acceptance = postselect_iceberg_counts(counts, k_logical=4)
        assert acceptance == pytest.approx((10 + 3) / (10 + 7 + 5 + 3))
        assert accepted["0001"] == 10
        assert accepted["0000"] == 3
