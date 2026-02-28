"""Azure Quantum backend.

2026 Academic References — Azure Quantum SDK 2.3
==================================================
- **Azure Quantum SDK 2.3**: Unified workspace model for multi-provider
  access. Supports Qiskit, Cirq, and Q# circuit submission via
  ``azure-quantum`` Python package.
  See: https://learn.microsoft.com/azure/quantum

- **IonQ Aria-2** (25 algorithmic qubits): Trapped-ion architecture with
  all-to-all connectivity. Average single-qubit gate fidelity >99.5%,
  two-qubit gate fidelity ~99.4%. Available as ``ionq.aria-2`` target.

- **Quantinuum H2** (56 qubits): Trapped-ion QCCD architecture with
  mid-circuit measurement, qubit reuse, and conditional logic. Achieved
  Quantum Volume 2^20. Available as ``quantinuum.qpu.h2`` target.

- **Provider target names**:
  - ``ionq.simulator``: IonQ cloud simulator (up to 29 qubits)
  - ``ionq.aria-2``: IonQ Aria-2 hardware
  - ``quantinuum.sim.h1-1e``: Quantinuum H1-1 emulator
  - ``quantinuum.qpu.h2``: Quantinuum H2 hardware
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

from quantum_common.types import BackendType, QuantumParadigm
from quantum_common.exceptions import BackendError
from quantum_common.backends.base import ExecutionRequest, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class AzureQuantumBackend:
    """Azure Quantum backend supporting IonQ, Quantinuum, etc.

    Azure Quantum SDK 2.3 provides a unified workspace model for submitting
    circuits to IonQ Aria-2 (25 qubits), Quantinuum H2 (56 qubits), and
    cloud simulators via the ``azure.quantum`` Python package.
    """

    resource_id: str | None = None
    location: str = "eastus"
    # Default target: IonQ cloud simulator (up to 29 qubits).
    # Production targets: "ionq.aria-2" (25q), "quantinuum.qpu.h2" (56q)
    target_name: str = "ionq.simulator"
    _max_qubits: int = 29

    def __post_init__(self) -> None:
        if self.resource_id is None:
            import os

            self.resource_id = os.environ.get("AZURE_QUANTUM_RESOURCE_ID")

    @property
    def backend_type(self) -> BackendType:
        return BackendType.AZURE_QUANTUM

    @property
    def paradigm(self) -> QuantumParadigm:
        return QuantumParadigm.GATE_BASED

    @property
    def name(self) -> str:
        return f"azure:{self.target_name}"

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    def _get_workspace(self) -> Any:
        try:
            from azure.quantum import Workspace
        except ImportError as e:
            raise BackendError("azure-quantum not installed") from e
        if not self.resource_id:
            raise BackendError("AZURE_QUANTUM_RESOURCE_ID not configured")
        return Workspace(resource_id=self.resource_id, location=self.location)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        import time

        try:
            from azure.quantum.qiskit import AzureQuantumProvider
        except ImportError as e:
            raise BackendError("azure-quantum not installed") from e

        workspace = self._get_workspace()
        provider = AzureQuantumProvider(workspace=workspace)
        backend = provider.get_backend(self.target_name)

        t0 = time.perf_counter()
        all_counts: list[dict[str, int]] = []
        raw_results: list[Any] = []

        for circuit in request.circuits:
            job = backend.run(circuit, shots=request.shots)
            result = job.result()
            counts = result.get_counts()
            all_counts.append(dict(counts))
            raw_results.append(result)

        elapsed = time.perf_counter() - t0
        return ExecutionResult(
            counts=all_counts,
            raw_results=raw_results,
            metadata={"target": self.target_name, "shots": request.shots},
            execution_time_s=elapsed,
        )

    def transpile(
        self, circuits: list[Any], optimization_level: int = 1
    ) -> list[Any]:
        from qiskit import transpile

        return [transpile(c, optimization_level=optimization_level) for c in circuits]

    def is_available(self) -> bool:
        try:
            self._get_workspace()
            return True
        except Exception:
            return False

    def status_info(self) -> dict[str, Any]:
        try:
            workspace = self._get_workspace()
            return {
                "target": self.target_name,
                "available": True,
                "location": self.location,
            }
        except Exception as e:
            return {
                "target": self.target_name,
                "available": False,
                "error": str(e),
            }
