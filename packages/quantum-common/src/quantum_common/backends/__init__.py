"""Quantum backend abstraction layer.

Exports all public classes for the backends package.
"""

from quantum_common.backends.base import (
    ExecutionRequest,
    ExecutionResult,
    QuantumBackend,
)
from quantum_common.backends.simulator import (
    AerSimulatorBackend,
    DWaveSimulatorBackend,
)
from quantum_common.backends.factory import (
    create_backend,
    list_available_backends,
    register_backend,
)
from quantum_common.backends.ibm import IBMQuantumBackend
from quantum_common.backends.dwave import DWaveQuantumBackend
from quantum_common.backends.pennylane import PennyLaneBackend
from quantum_common.backends.azure import AzureQuantumBackend

__all__ = [
    "ExecutionRequest",
    "ExecutionResult",
    "QuantumBackend",
    "AerSimulatorBackend",
    "DWaveSimulatorBackend",
    "IBMQuantumBackend",
    "DWaveQuantumBackend",
    "PennyLaneBackend",
    "AzureQuantumBackend",
    "create_backend",
    "list_available_backends",
    "register_backend",
]
