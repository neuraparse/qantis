"""PennyLane hardware-agnostic quantum backend.

2026 Academic References — PennyLane v0.44
============================================
- **PennyLane v0.44** (Jan 2026): Major release introducing QRAM templates
  (BBQRAM, SelectOnlyQRAM, HybridQRAM) for quantum data loading, and
  MultiplexerStatePreparation for efficient amplitude encoding.
  See: https://docs.pennylane.ai/en/stable/

- **QRAM Templates**: ``qml.BBQRAM`` (bucket-brigade), ``qml.SelectOnlyQRAM``,
  and ``qml.HybridQRAM`` provide logarithmic-depth data loading circuits.
  Useful for quantum machine learning and optimization workflows.

- **MultiplexerStatePreparation**: Encodes arbitrary amplitude vectors into
  quantum states using multiplexed rotation gates. Provides O(n) depth for
  n-qubit state preparation.

- **Device-agnostic execution**: QNode compilation handles device-specific
  gate decomposition and optimization automatically. Supports ``default.qubit``,
  ``lightning.qubit``, ``qiskit.aer``, and hardware devices via plugins.

- **algo_error function**: FTQC algorithm error estimation for resource
  planning.

- **IQP circuits and algorithms**: ``qre.Qubitization``, ``qre.QSP``,
  ``qre.UnaryIterationQPE``.

- **NumPy 2.0 requirement**: NumPy <2.0 maintenance support deprecated in
  v0.44, dropping in v0.45.
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
class PennyLaneBackend:
    """PennyLane device wrapper for hardware-agnostic execution.

    PennyLane v0.44 (Jan 2026) supports QRAM templates and
    MultiplexerStatePreparation for advanced circuit construction.
    Devices are resolved at runtime via the plugin system.
    """

    device_name: str = "default.qubit"
    device_kwargs: dict[str, Any] | None = None
    _max_qubits: int = 25

    @property
    def backend_type(self) -> BackendType:
        return BackendType.PENNYLANE

    @property
    def paradigm(self) -> QuantumParadigm:
        return QuantumParadigm.GATE_BASED

    @property
    def name(self) -> str:
        return f"pennylane:{self.device_name}"

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    def _get_device(self, wires: int) -> Any:
        try:
            import pennylane as qml
        except ImportError as e:
            raise BackendError("pennylane not installed") from e
        kwargs = self.device_kwargs or {}
        return qml.device(self.device_name, wires=wires, **kwargs)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute PennyLane QNode callables.

        request.circuits should contain tuples of (qnode_fn, args, kwargs)
        or pre-built QNode objects.
        """
        import time

        t0 = time.perf_counter()
        all_counts: list[dict[str, int]] = []
        raw_results: list[Any] = []

        for item in request.circuits:
            if callable(item):
                result = item()
            elif isinstance(item, tuple) and len(item) >= 2:
                fn, args = item[0], item[1]
                kwargs = item[2] if len(item) > 2 else {}
                result = fn(*args, **kwargs)
            else:
                raise BackendError(
                    f"Unsupported PennyLane circuit format: {type(item)}"
                )
            raw_results.append(result)
            # Convert result to counts-like format if possible
            if isinstance(result, dict):
                all_counts.append({str(k): int(v) for k, v in result.items()})
            else:
                all_counts.append({"result": 1})

        elapsed = time.perf_counter() - t0
        return ExecutionResult(
            counts=all_counts,
            raw_results=raw_results,
            metadata={"device": self.device_name},
            execution_time_s=elapsed,
        )

    def transpile(
        self, circuits: list[Any], optimization_level: int = 1
    ) -> list[Any]:
        # PennyLane v0.44 handles compilation internally via QNode transforms.
        # Device-specific gate decomposition is applied at execution time.
        return circuits

    def is_available(self) -> bool:
        try:
            import pennylane as qml

            qml.device(self.device_name, wires=1)
            return True
        except Exception:
            return False

    def status_info(self) -> dict[str, Any]:
        return {
            "device": self.device_name,
            "available": self.is_available(),
            "max_qubits": self._max_qubits,
        }
