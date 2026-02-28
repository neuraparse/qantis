"""Backend factory with pluggable registry.

2026 Academic References — Framework Compatibility Matrix
===========================================================
This factory supports the following backend/framework combinations:

+-------------------+---------------------------+----------------------------+
| BackendType       | Framework / SDK           | Hardware Targets           |
+===================+===========================+============================+
| IBM_QUANTUM       | Qiskit v2.3 +             | Heron R3 (156q),           |
|                   | Runtime v0.36+            | Eagle R3 (127q)            |
+-------------------+---------------------------+----------------------------+
| DWAVE             | D-Wave Ocean SDK 9.x      | Advantage2 (4400+q,        |
|                   |                           | Zephyr topology)           |
+-------------------+---------------------------+----------------------------+
| AZURE_QUANTUM     | Azure Quantum SDK 2.3     | IonQ Aria-2 (25q),         |
|                   |                           | Quantinuum H2 (56q)        |
+-------------------+---------------------------+----------------------------+
| PENNYLANE         | PennyLane v0.44           | Device-agnostic             |
|                   |                           | (plugin system)            |
+-------------------+---------------------------+----------------------------+
| LOCAL_AER         | Qiskit Aer v0.17          | Local QASM/statevector     |
+-------------------+---------------------------+----------------------------+
| LOCAL_DWAVE_SIM   | dwave.samplers (Ocean 9.x)| Local simulated annealing  |
+-------------------+---------------------------+----------------------------+
| LOCAL_PENNYLANE   | PennyLane v0.44           | default.qubit simulator    |
+-------------------+---------------------------+----------------------------+

Minimum Python version: 3.10 (for PEP 604 union types used throughout).
All backends conform to the ``QuantumBackend`` Protocol defined in base.py.
"""

from __future__ import annotations

from typing import Any, Callable
import logging

from quantum_common.types import BackendType
from quantum_common.exceptions import BackendError
from quantum_common.backends.base import QuantumBackend

logger = logging.getLogger(__name__)

# Global registry — maps BackendType enum to lazy-import factory callables.
# Lazy imports avoid pulling in heavy SDK dependencies until actually needed.
_REGISTRY: dict[BackendType, Callable[..., QuantumBackend]] = {}


def register_backend(backend_type: BackendType) -> Callable:
    """Decorator to register a backend constructor."""

    def decorator(
        factory_fn: Callable[..., QuantumBackend],
    ) -> Callable[..., QuantumBackend]:
        _REGISTRY[backend_type] = factory_fn
        return factory_fn

    return decorator


def create_backend(backend_type: BackendType, **kwargs: Any) -> QuantumBackend:
    """Create a backend instance from the registry."""
    if backend_type not in _REGISTRY:
        _register_defaults()
    if backend_type not in _REGISTRY:
        available = [bt.value for bt in _REGISTRY]
        raise BackendError(
            f"No backend registered for {backend_type.value}. Available: {available}"
        )
    return _REGISTRY[backend_type](**kwargs)


def list_available_backends() -> list[BackendType]:
    """List all backend types that have registered factories."""
    _register_defaults()
    available = []
    for bt, factory_fn in _REGISTRY.items():
        try:
            backend = factory_fn()
            if backend.is_available():
                available.append(bt)
        except Exception:
            continue
    return available


def _register_defaults() -> None:
    """Register default simulator backends (lazy import)."""
    if BackendType.LOCAL_AER not in _REGISTRY:

        @register_backend(BackendType.LOCAL_AER)
        def _create_aer(**kwargs: Any) -> QuantumBackend:
            from quantum_common.backends.simulator import AerSimulatorBackend

            return AerSimulatorBackend(**kwargs)

    if BackendType.LOCAL_DWAVE_SIM not in _REGISTRY:

        @register_backend(BackendType.LOCAL_DWAVE_SIM)
        def _create_neal(**kwargs: Any) -> QuantumBackend:
            from quantum_common.backends.simulator import DWaveSimulatorBackend

            return DWaveSimulatorBackend(**kwargs)

    if BackendType.IBM_QUANTUM not in _REGISTRY:

        @register_backend(BackendType.IBM_QUANTUM)
        def _create_ibm(**kwargs: Any) -> QuantumBackend:
            from quantum_common.backends.ibm import IBMQuantumBackend

            return IBMQuantumBackend(**kwargs)

    if BackendType.DWAVE not in _REGISTRY:

        @register_backend(BackendType.DWAVE)
        def _create_dwave(**kwargs: Any) -> QuantumBackend:
            from quantum_common.backends.dwave import DWaveQuantumBackend

            return DWaveQuantumBackend(**kwargs)

    if BackendType.PENNYLANE not in _REGISTRY:

        @register_backend(BackendType.PENNYLANE)
        def _create_pennylane(**kwargs: Any) -> QuantumBackend:
            from quantum_common.backends.pennylane import PennyLaneBackend

            return PennyLaneBackend(**kwargs)

    if BackendType.AZURE_QUANTUM not in _REGISTRY:

        @register_backend(BackendType.AZURE_QUANTUM)
        def _create_azure(**kwargs: Any) -> QuantumBackend:
            from quantum_common.backends.azure import AzureQuantumBackend

            return AzureQuantumBackend(**kwargs)
