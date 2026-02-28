"""Shared fixtures for quantum-common hardware tests.

All fixtures skip automatically when required environment variables are absent,
ensuring the test suite degrades gracefully without hardware credentials.

Environment variables:
  IBM_QUANTUM_TOKEN   — IBM Quantum Network / Open Plan API key
  DWAVE_API_TOKEN     — D-Wave Leap cloud token

CLI options (--ibm-backend, --shots, --num-reads) are registered in the root
conftest.py to avoid plugin registration collisions across packages.
"""
from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# IBM backend fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ibm_hardware_backend(request: pytest.FixtureRequest):
    """IBM Quantum backend via Qiskit Runtime V2 (SamplerV2).

    Skips when IBM_QUANTUM_TOKEN is not set.
    Uses --ibm-backend CLI option (default: ibm_brisbane).
    """
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not token:
        pytest.skip("IBM_QUANTUM_TOKEN not set — skipping IBM hardware tests")

    from quantum_common.backends.ibm import IBMQuantumBackend

    backend_name = request.config.getoption("--ibm-backend")
    return IBMQuantumBackend(backend_name=backend_name, token=token)


# ---------------------------------------------------------------------------
# D-Wave backend fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dwave_hardware_backend():
    """D-Wave Advantage2 backend via Ocean SDK 9.x + EmbeddingComposite.

    Skips when DWAVE_API_TOKEN is not set.
    """
    token = os.environ.get("DWAVE_API_TOKEN", "")
    if not token:
        pytest.skip("DWAVE_API_TOKEN not set — skipping D-Wave hardware tests")

    from quantum_common.backends.dwave import DWaveQuantumBackend

    return DWaveQuantumBackend(token=token, use_hybrid=False)


# ---------------------------------------------------------------------------
# Aer simulator fixture (always available, used as ideal baseline)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def aer_simulator_backend():
    """Local Qiskit Aer simulator — always available, used as baseline."""
    from quantum_common.backends.simulator import AerSimulatorBackend

    return AerSimulatorBackend()
