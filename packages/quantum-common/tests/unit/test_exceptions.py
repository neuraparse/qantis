"""Tests for quantum_common.exceptions module — exception hierarchy."""

import pytest

from quantum_common.exceptions import (
    BackendError,
    BackendExecutionError,
    BackendNotAvailableError,
    BenchmarkError,
    ConfigurationError,
    FormulationError,
    JobTimeoutError,
    MitigationError,
    QuantumAutonomousError,
)


class TestExceptionHierarchy:
    """Verify inheritance relationships among project exceptions."""

    def test_base_exception_is_exception(self) -> None:
        assert issubclass(QuantumAutonomousError, Exception)

    def test_backend_error_inherits_from_base(self) -> None:
        assert issubclass(BackendError, QuantumAutonomousError)

    def test_backend_not_available_inherits_from_backend(self) -> None:
        assert issubclass(BackendNotAvailableError, BackendError)
        assert issubclass(BackendNotAvailableError, QuantumAutonomousError)

    def test_backend_execution_error_inherits_from_backend(self) -> None:
        assert issubclass(BackendExecutionError, BackendError)
        assert issubclass(BackendExecutionError, QuantumAutonomousError)

    def test_job_timeout_inherits_from_backend(self) -> None:
        assert issubclass(JobTimeoutError, BackendError)
        assert issubclass(JobTimeoutError, QuantumAutonomousError)

    def test_configuration_error_inherits_from_base(self) -> None:
        assert issubclass(ConfigurationError, QuantumAutonomousError)

    def test_mitigation_error_inherits_from_base(self) -> None:
        assert issubclass(MitigationError, QuantumAutonomousError)

    def test_benchmark_error_inherits_from_base(self) -> None:
        assert issubclass(BenchmarkError, QuantumAutonomousError)

    def test_formulation_error_inherits_from_base(self) -> None:
        assert issubclass(FormulationError, QuantumAutonomousError)


class TestExceptionRaiseAndCatch:
    """Verify that exceptions can be raised and caught with proper messages."""

    @pytest.mark.parametrize(
        "exc_class,message",
        [
            (QuantumAutonomousError, "generic project error"),
            (BackendError, "backend communication failure"),
            (BackendNotAvailableError, "IBM Quantum backend offline"),
            (BackendExecutionError, "circuit execution timed out on backend"),
            (JobTimeoutError, "job exceeded 300s limit"),
            (ConfigurationError, "missing required field 'backend_type'"),
            (MitigationError, "ZNE extrapolation failed to converge"),
            (BenchmarkError, "benchmark trial exceeded time limit"),
            (FormulationError, "QUBO matrix has invalid dimensions"),
        ],
    )
    def test_raise_with_message(self, exc_class: type, message: str) -> None:
        with pytest.raises(exc_class, match=message):
            raise exc_class(message)

    def test_exception_str(self) -> None:
        exc = BackendNotAvailableError("ibm_brisbane is offline")
        assert str(exc) == "ibm_brisbane is offline"

    def test_exception_args(self) -> None:
        exc = JobTimeoutError("timeout", 300)
        assert exc.args == ("timeout", 300)


class TestExceptionPolymorphism:
    """Verify that catching a parent exception catches children."""

    def test_catching_base_catches_backend_error(self) -> None:
        with pytest.raises(QuantumAutonomousError):
            raise BackendError("some backend failure")

    def test_catching_base_catches_backend_not_available(self) -> None:
        with pytest.raises(QuantumAutonomousError):
            raise BackendNotAvailableError("backend unavailable")

    def test_catching_backend_error_catches_execution_error(self) -> None:
        with pytest.raises(BackendError):
            raise BackendExecutionError("execution failed")

    def test_catching_backend_error_catches_timeout(self) -> None:
        with pytest.raises(BackendError):
            raise JobTimeoutError("timed out")

    def test_catching_base_catches_configuration(self) -> None:
        with pytest.raises(QuantumAutonomousError):
            raise ConfigurationError("bad config")

    def test_catching_base_catches_mitigation(self) -> None:
        with pytest.raises(QuantumAutonomousError):
            raise MitigationError("mitigation failed")

    def test_catching_base_catches_benchmark(self) -> None:
        with pytest.raises(QuantumAutonomousError):
            raise BenchmarkError("benchmark failed")

    def test_catching_base_catches_formulation(self) -> None:
        with pytest.raises(QuantumAutonomousError):
            raise FormulationError("formulation error")

    def test_catching_exception_catches_all(self) -> None:
        """All project exceptions should be catchable as plain Exception."""
        with pytest.raises(Exception):
            raise BackendExecutionError("low-level catch")

    def test_backend_not_caught_by_unrelated_sibling(self) -> None:
        """ConfigurationError should NOT catch BackendError."""
        with pytest.raises(BackendError):
            try:
                raise BackendError("backend issue")
            except ConfigurationError:
                pytest.fail("ConfigurationError should not catch BackendError")
