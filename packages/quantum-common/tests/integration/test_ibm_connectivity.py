"""Integration tests for IBM Quantum connectivity.

These tests require IBM_QUANTUM_TOKEN to be set.
Run with: uv run pytest -m integration packages/quantum-common/tests/integration/
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.hardware]


@pytest.fixture
def ibm_backend():
    """Create IBM Quantum backend (requires credentials)."""
    try:
        from quantum_common.backends.ibm import IBMQuantumBackend
        from quantum_common.config.credentials import get_ibm_token

        token = get_ibm_token()
        if not token:
            pytest.skip("IBM_QUANTUM_TOKEN not set")
        return IBMQuantumBackend(token=token)
    except ImportError:
        pytest.skip("qiskit-ibm-runtime not installed")


class TestIBMConnectivity:
    def test_backend_available(self, ibm_backend) -> None:
        assert ibm_backend.is_available()

    def test_status_info(self, ibm_backend) -> None:
        info = ibm_backend.status_info()
        assert "backend" in info
        assert info.get("operational") is True

    def test_simple_circuit_execution(self, ibm_backend) -> None:
        """Execute a simple Bell state circuit on IBM hardware."""
        from qiskit import QuantumCircuit
        from quantum_common.backends.base import ExecutionRequest

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        request = ExecutionRequest(circuits=[qc], shots=100)
        result = ibm_backend.execute(request)
        assert len(result.counts) == 1
        assert sum(result.counts[0].values()) == 100

    def test_transpilation(self, ibm_backend) -> None:
        """Test circuit transpilation to backend ISA."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        transpiled = ibm_backend.transpile([qc], optimization_level=2)
        assert len(transpiled) == 1
        assert transpiled[0].num_qubits > 0
