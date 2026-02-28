"""Local quantum simulators for development and testing.

2026 Academic References — Local Simulators
=============================================
- **Qiskit Aer v0.17**: Statevector, QASM, and density-matrix simulators.
  Compatible with Qiskit v2.3 circuit IR. Supports GPU acceleration via
  cuStateVec for up to ~35 qubits. ``AerSimulator`` is the unified entry point
  replacing deprecated ``QasmSimulator`` and ``StatevectorSimulator``.
  See: https://qiskit.github.io/qiskit-aer/

- **D-Wave ``dwave.samplers``** (Ocean SDK 9.x): Local simulated annealing
  solver. Namespace migrated from ``neal.SimulatedAnnealingSampler`` to
  ``dwave.samplers.SimulatedAnnealingSampler``. The ``neal`` package is
  still importable as a fallback but is deprecated.
  See: https://docs.ocean.dwavesys.com/en/stable/docs_samplers/

- The fallback import pattern below (try ``dwave.samplers`` first, then
  ``neal``) ensures backward compatibility during the Ocean SDK 9.x migration.

- **Production targets**: Local Aer + neal simulators for development and
  CI testing. Production workloads target D-Wave Advantage2 (4400+ qubits,
  Zephyr 20-way) and IBM Heron R3 (156 qubits, heavy-hex).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

from quantum_common.types import BackendType, QuantumParadigm
from quantum_common.exceptions import BackendError
from quantum_common.backends.base import ExecutionRequest, ExecutionResult, QuantumBackend

logger = logging.getLogger(__name__)


@dataclass
class AerSimulatorBackend:
    """Local Qiskit Aer simulator backend.

    Uses Aer v0.17 ``AerSimulator`` for local QASM simulation.
    Supports noise model injection for realistic hardware emulation.
    """

    _max_qubits: int = 25
    _name: str = "aer_simulator"

    @property
    def backend_type(self) -> BackendType:
        return BackendType.LOCAL_AER

    @property
    def paradigm(self) -> QuantumParadigm:
        return QuantumParadigm.GATE_BASED

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        try:
            from qiskit_aer import AerSimulator
        except ImportError as e:
            raise BackendError("qiskit-aer not installed") from e

        sim = AerSimulator()
        import time

        t0 = time.perf_counter()
        all_counts: list[dict[str, int]] = []
        raw_results: list[Any] = []

        for circuit in request.circuits:
            from qiskit import transpile

            transpiled = transpile(circuit, sim, optimization_level=1)
            result = sim.run(transpiled, shots=request.shots).result()
            counts = result.get_counts()
            all_counts.append(dict(counts) if not isinstance(counts, dict) else counts)
            raw_results.append(result)

        elapsed = time.perf_counter() - t0
        return ExecutionResult(
            counts=all_counts,
            raw_results=raw_results,
            metadata={"simulator": "aer", "shots": request.shots},
            execution_time_s=elapsed,
        )

    def transpile(
        self, circuits: list[Any], optimization_level: int = 1
    ) -> list[Any]:
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile
        except ImportError as e:
            raise BackendError("qiskit-aer not installed") from e
        sim = AerSimulator()
        return [
            transpile(c, sim, optimization_level=optimization_level) for c in circuits
        ]

    def is_available(self) -> bool:
        try:
            from qiskit_aer import AerSimulator

            AerSimulator()
            return True
        except Exception:
            return False

    def status_info(self) -> dict[str, Any]:
        return {
            "backend": self._name,
            "available": self.is_available(),
            "max_qubits": self._max_qubits,
        }


@dataclass
class DWaveSimulatorBackend:
    """Local D-Wave simulated annealing backend.

    Uses ``dwave.samplers.SimulatedAnnealingSampler`` (Ocean SDK 9.x) with
    fallback to deprecated ``neal.SimulatedAnnealingSampler`` for backward
    compatibility. See ``dwave.samplers`` namespace migration notes in
    Ocean SDK 9.x release documentation.
    """

    _num_reads: int = 1000
    _max_qubits: int = 5000
    _name: str = "dwave_neal_simulator"

    @property
    def backend_type(self) -> BackendType:
        return BackendType.LOCAL_DWAVE_SIM

    @property
    def paradigm(self) -> QuantumParadigm:
        return QuantumParadigm.ANNEALING

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        # Ocean SDK 9.x namespace migration: prefer dwave.samplers over neal.
        # See: https://docs.ocean.dwavesys.com/en/stable/docs_samplers/
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError:
            try:
                # Fallback to deprecated neal package for pre-9.x compatibility
                import neal
                SimulatedAnnealingSampler = neal.SimulatedAnnealingSampler
            except ImportError as e:
                raise BackendError("dwave-ocean-sdk not installed") from e

        import time

        t0 = time.perf_counter()
        sampler = SimulatedAnnealingSampler()
        num_reads = request.options.get("num_reads", self._num_reads)
        all_counts: list[dict[str, int]] = []
        raw_results: list[Any] = []

        for bqm in request.circuits:
            sampleset = sampler.sample(bqm, num_reads=num_reads)
            # Convert to bitstring counts
            counts: dict[str, int] = {}
            for sample, energy, num_occ in sampleset.record:
                key = "".join(str(int(b)) for b in sample)
                counts[key] = counts.get(key, 0) + int(num_occ)
            all_counts.append(counts)
            raw_results.append(sampleset)

        elapsed = time.perf_counter() - t0
        return ExecutionResult(
            counts=all_counts,
            raw_results=raw_results,
            metadata={"simulator": "neal", "num_reads": num_reads},
            execution_time_s=elapsed,
        )

    def transpile(
        self, circuits: list[Any], optimization_level: int = 1
    ) -> list[Any]:
        return circuits  # No transpilation needed for annealing

    def is_available(self) -> bool:
        try:
            from dwave.samplers import SimulatedAnnealingSampler
            return True
        except ImportError:
            try:
                import neal
                return True
            except ImportError:
                return False

    def status_info(self) -> dict[str, Any]:
        return {
            "backend": self._name,
            "available": self.is_available(),
            "max_qubits": self._max_qubits,
        }
