"""Shared fixtures for quantum-mht hardware tests.

Provides QUBO instances, Hungarian baselines, and backend fixtures
for D-Wave Advantage2 and IBM QPU tests.

Environment variables:
  DWAVE_API_TOKEN     — D-Wave Leap cloud token (Tasks 1.2, 1.6, 2.2)
  IBM_QUANTUM_TOKEN   — IBM Quantum Network token (Task 1.3)

CLI options (--ibm-backend, --shots, --num-reads) are registered in the
root conftest.py to avoid plugin registration collisions.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ibm_hardware_backend(request: pytest.FixtureRequest):
    """IBM QPU backend; skips when IBM_QUANTUM_TOKEN not set."""
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not token:
        pytest.skip("IBM_QUANTUM_TOKEN not set")
    from quantum_common.backends.ibm import IBMQuantumBackend

    return IBMQuantumBackend(
        backend_name=request.config.getoption("--ibm-backend"),
        token=token,
    )


@pytest.fixture(scope="session")
def dwave_hardware_backend():
    """D-Wave Advantage2 backend; skips when DWAVE_API_TOKEN not set."""
    token = os.environ.get("DWAVE_API_TOKEN", "")
    if not token:
        pytest.skip("DWAVE_API_TOKEN not set")
    from quantum_common.backends.dwave import DWaveQuantumBackend

    return DWaveQuantumBackend(token=token, use_hybrid=False)


# ---------------------------------------------------------------------------
# QUBO instance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qubo_n5_instance():
    """N=5 tracks, M=8 measurements — 53 QUBO variables (with slack vars).

    Synthetic data from a fixed random seed for reproducibility.
    Variable count: N*M + N + M = 40 + 5 + 8 = 53.
    """
    from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder

    rng = np.random.default_rng(42)
    predicted = rng.standard_normal((5, 2)).astype(np.float64)
    measurements = rng.standard_normal((8, 2)).astype(np.float64)
    covariances = np.stack([np.eye(2, dtype=np.float64) * 0.5] * 5)

    builder = MTDAQuboBuilder()
    return builder.build(predicted, measurements, covariances)


@pytest.fixture(scope="session")
def qubo_n2_instance():
    """N=2 tracks, M=3 measurements — 11 QUBO variables (FPC-QAOA).

    Variable count: N*M + N + M = 6 + 2 + 3 = 11.
    Small enough for IBM gate-based QPU without circuit cutting.
    """
    from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder

    rng = np.random.default_rng(42)
    predicted = rng.standard_normal((2, 2)).astype(np.float64)
    measurements = rng.standard_normal((3, 2)).astype(np.float64)
    covariances = np.stack([np.eye(2, dtype=np.float64) * 0.5] * 2)

    builder = MTDAQuboBuilder()
    return builder.build(predicted, measurements, covariances)


@pytest.fixture(scope="session")
def hungarian_baseline_n5(qubo_n5_instance):
    """Optimal classical assignment for the N=5 QUBO instance (Hungarian).

    Provides the O(n³) optimal reference for approximation ratio calculations.
    """
    from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver

    return HungarianSolver().solve(qubo_n5_instance)


@pytest.fixture(scope="session")
def hungarian_baseline_n2(qubo_n2_instance):
    """Optimal classical assignment for the N=2 QUBO instance (Hungarian)."""
    from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver

    return HungarianSolver().solve(qubo_n2_instance)
