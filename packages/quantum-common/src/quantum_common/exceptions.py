"""Custom exception hierarchy for the quantum autonomous systems project.

2026 Academic References — Framework Compatibility
===================================================
Exception types in this module cover failure modes across the following
quantum framework versions (as of Jan 2026):

- Qiskit v2.3 Runtime errors (SamplerV2/EstimatorV2 primitives)
- D-Wave Ocean SDK 9.x solver connectivity and embedding failures
- PennyLane v0.44 device and QNode execution errors
- Azure Quantum SDK 2.3 workspace and target resolution errors
- Mitiq v0.44+ mitigation pipeline failures (ZNE, PEC, readout correction)

Error mitigation references:
- Cai et al., "Quantum Error Mitigation", Rev. Mod. Phys. 95, 045005 (2023).
  DOI: 10.1103/RevModPhys.95.045005 — categorizes composable error sources.
"""


class QuantumAutonomousError(Exception):
    """Base exception for all project-specific errors."""


class BackendError(QuantumAutonomousError):
    """Errors related to quantum backend operations."""


class BackendNotAvailableError(BackendError):
    """Raised when a requested backend is not configured or reachable."""


class BackendExecutionError(BackendError):
    """Raised when circuit execution fails on a backend."""


class JobTimeoutError(BackendError):
    """Raised when a job exceeds the maximum wait time."""


class ConfigurationError(QuantumAutonomousError):
    """Errors related to configuration loading or validation."""


class MitigationError(QuantumAutonomousError):
    """Errors during error mitigation pipeline execution."""


class BenchmarkError(QuantumAutonomousError):
    """Errors during benchmark execution or comparison."""


class FormulationError(QuantumAutonomousError):
    """Errors in QUBO or circuit formulation."""
