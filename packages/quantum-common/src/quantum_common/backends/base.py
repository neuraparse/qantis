"""Backend abstraction layer using structural subtyping.

2026 Academic References — Backend Protocol Design
====================================================
The ``QuantumBackend`` Protocol in this module provides a unified execution
interface across heterogeneous quantum hardware, inspired by the primitive-based
execution model introduced in:

- **Qiskit v2.3** (Jan 2026): SamplerV2 and EstimatorV2 primitives unify
  gate-based execution. The ``execute()`` method here mirrors SamplerV2's
  ``run()`` contract (circuits + shots -> counts/quasi-dists).
  See: https://docs.quantum.ibm.com/api/qiskit/primitives

- **D-Wave Ocean SDK 9.x**: Unified ``sample()`` execution model across
  DWaveSampler, LeapHybridSampler, and ``dwave.samplers`` local solvers.
  Advantage2 provides 4400+ qubits on Zephyr topology.
  See: https://docs.ocean.dwavesys.com

- **PennyLane v0.44** (Jan 2026): QNode execution model with automatic
  differentiation and device-agnostic compilation.

- **Azure Quantum SDK 2.3**: Provider-backend execution model targeting
  IonQ Aria-2 (25 qubits) and Quantinuum H2 (56 qubits).

The Protocol pattern (PEP 544 structural subtyping) allows backends to
conform without inheritance, enabling clean separation between gate-based
and annealing paradigms.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from quantum_common.types import BackendType, QuantumParadigm, JobStatus


@dataclass
class ExecutionRequest:
    """Unified execution request for gate-based and annealing backends."""

    circuits: list[Any]  # QuantumCircuit or BinaryQuadraticModel
    # Default 4096 shots. Note: Qiskit v2.3 SamplerV2 defaults to 10K shots
    # per PUB. See arXiv:2512.08245 (Dec 2025) for analysis showing 10K shots
    # captures only ~23% of the state space for typical combinatorial problems.
    shots: int = 4096
    options: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Unified execution result."""

    counts: list[dict[str, int]]
    raw_results: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_s: float = 0.0
    status: JobStatus = JobStatus.COMPLETED

    @property
    def quasi_dists(self) -> list[dict[str, float]]:
        """Convert counts to quasi-probability distributions.

        Quasi-probability distributions may contain negative values when
        error mitigation (e.g., PEC) is applied. See:
        - Temme et al., "Error Mitigation for Short-Depth Quantum Circuits",
          PRL 119, 180509 (2017). DOI: 10.1103/PhysRevLett.119.180509
        """
        dists = []
        for c in self.counts:
            total = sum(c.values())
            dists.append({k: v / total for k, v in c.items()} if total > 0 else {})
        return dists


@runtime_checkable
class QuantumBackend(Protocol):
    """Protocol for quantum backends (structural subtyping)."""

    @property
    def backend_type(self) -> BackendType: ...

    @property
    def paradigm(self) -> QuantumParadigm: ...

    @property
    def name(self) -> str: ...

    @property
    def max_qubits(self) -> int: ...

    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...

    def transpile(
        self, circuits: list[Any], optimization_level: int = 1
    ) -> list[Any]: ...

    def is_available(self) -> bool: ...

    def status_info(self) -> dict[str, Any]: ...
