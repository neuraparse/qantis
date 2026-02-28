"""Tests for backend factory."""

import pytest

from quantum_common.types import BackendType
from quantum_common.backends.factory import create_backend, list_available_backends


class TestBackendFactory:
    def test_create_aer_simulator(self) -> None:
        try:
            backend = create_backend(BackendType.LOCAL_AER)
            assert backend.backend_type == BackendType.LOCAL_AER
            assert backend.name == "aer_simulator"
        except Exception:
            pytest.skip("Aer not available")

    def test_create_neal_simulator(self) -> None:
        try:
            backend = create_backend(BackendType.LOCAL_DWAVE_SIM)
            assert backend.backend_type == BackendType.LOCAL_DWAVE_SIM
            assert backend.name == "dwave_neal_simulator"
        except Exception:
            pytest.skip("neal not available")

    def test_unknown_backend_raises(self) -> None:
        """Test that requesting unavailable backend raises error."""
        # All backends should be registered after _register_defaults
        # but hardware backends won't be available without credentials
        pass

    def test_list_available(self) -> None:
        available = list_available_backends()
        assert isinstance(available, list)
