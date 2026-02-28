"""Tests for local simulator backends."""

import pytest


class TestAerSimulator:
    def test_is_available(self) -> None:
        try:
            from quantum_common.backends.simulator import AerSimulatorBackend
            backend = AerSimulatorBackend()
            # May or may not be available depending on installation
            assert isinstance(backend.is_available(), bool)
        except ImportError:
            pytest.skip("quantum_common.backends.simulator not ready")

    def test_status_info(self) -> None:
        try:
            from quantum_common.backends.simulator import AerSimulatorBackend
            backend = AerSimulatorBackend()
            info = backend.status_info()
            assert "backend" in info
            assert "max_qubits" in info
        except ImportError:
            pytest.skip("quantum_common.backends.simulator not ready")


class TestDWaveSimulator:
    def test_is_available(self) -> None:
        try:
            from quantum_common.backends.simulator import DWaveSimulatorBackend
            backend = DWaveSimulatorBackend()
            assert isinstance(backend.is_available(), bool)
        except ImportError:
            pytest.skip("quantum_common.backends.simulator not ready")
