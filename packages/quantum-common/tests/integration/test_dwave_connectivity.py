"""Integration tests for D-Wave connectivity.

These tests require DWAVE_API_TOKEN to be set.
Run with: uv run pytest -m integration packages/quantum-common/tests/integration/
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.hardware]


@pytest.fixture
def dwave_backend():
    """Create D-Wave backend (requires credentials)."""
    try:
        from quantum_common.backends.dwave import DWaveQuantumBackend
        from quantum_common.config.credentials import get_dwave_token

        token = get_dwave_token()
        if not token:
            pytest.skip("DWAVE_API_TOKEN not set")
        return DWaveQuantumBackend(token=token)
    except ImportError:
        pytest.skip("dwave-ocean-sdk not installed")


@pytest.fixture
def simple_bqm():
    """Create a simple BQM for testing."""
    try:
        import dimod

        bqm = dimod.BinaryQuadraticModel(
            {"x0": -1.0, "x1": -1.0},
            {("x0", "x1"): 2.0},
            0.0,
            dimod.BINARY,
        )
        return bqm
    except ImportError:
        pytest.skip("dimod not installed")


class TestDWaveConnectivity:
    def test_backend_available(self, dwave_backend) -> None:
        assert dwave_backend.is_available()

    def test_status_info(self, dwave_backend) -> None:
        info = dwave_backend.status_info()
        assert info.get("available") is True
        assert "num_qubits" in info or "solver" in info

    def test_simple_qubo_execution(self, dwave_backend, simple_bqm) -> None:
        """Execute a simple QUBO on D-Wave hardware."""
        from quantum_common.backends.base import ExecutionRequest

        request = ExecutionRequest(
            circuits=[simple_bqm],
            options={"num_reads": 100},
        )
        result = dwave_backend.execute(request)
        assert len(result.counts) == 1
        assert result.execution_time_s > 0

    def test_hybrid_solver(self) -> None:
        """Test D-Wave LeapHybrid solver connectivity."""
        try:
            from quantum_common.backends.dwave import DWaveQuantumBackend
            from quantum_common.config.credentials import get_dwave_token

            token = get_dwave_token()
            if not token:
                pytest.skip("DWAVE_API_TOKEN not set")
            backend = DWaveQuantumBackend(token=token, use_hybrid=True)
            info = backend.status_info()
            assert "solver" in info or "available" in info
        except ImportError:
            pytest.skip("dwave-ocean-sdk not installed")
