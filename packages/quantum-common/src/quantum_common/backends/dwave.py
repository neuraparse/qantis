"""D-Wave quantum annealing backend.

2026 Academic References — D-Wave Ocean SDK 9.x
=================================================
- **D-Wave Ocean SDK 9.x**: Unified API for Advantage2 system access.
  Key namespace migration: ``neal.SimulatedAnnealingSampler`` is now
  ``dwave.samplers.SimulatedAnnealingSampler``. The ``neal`` package is
  deprecated but still importable for backward compatibility.
  See: https://docs.ocean.dwavesys.com

- **Advantage2** (2025-2026): 4400+ qubits on Zephyr topology (degree-20
  connectivity graph). Significant improvement over Advantage (5000+ qubits,
  Pegasus topology, degree-15) for densely-connected QUBO problems.
  Zephyr provides ~40% higher connectivity per qubit.

- **EmbeddingComposite**: Automatic minor-embedding for mapping logical
  QUBO variables onto the physical Zephyr graph. Chain strength is
  auto-tuned in Ocean 9.x via ``uniform_torque_compensation``.

- **LeapHybridSampler**: Hybrid quantum-classical solver for problems
  exceeding native QPU connectivity. Supports BQMs up to ~1M variables
  via decomposition. LeapHybrid supports up to 2 million
  variables/constraints via cloud decomposition.

- **Advantage2 GA (May 2025)**: 75% noise reduction, 2x coherence
  improvement over Advantage.

- **2026 Roadmap**: Advantage2 Performance Update processor with novel
  annealing protocols announced at Qubits 2026 (Jan 27-28, Boca Raton).
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
class DWaveQuantumBackend:
    """D-Wave Advantage/Advantage2 backend.

    Supports both direct QPU sampling (DWaveSampler + EmbeddingComposite)
    and hybrid quantum-classical solving (LeapHybridSampler) via Ocean SDK 9.x.
    Advantage2 provides 4400+ qubits on Zephyr topology.
    """

    solver_name: str | None = None
    use_hybrid: bool = False
    token: str | None = None
    num_reads: int = 1000
    annealing_time_us: float = 20.0
    _max_qubits: int = 5000

    def __post_init__(self) -> None:
        if self.token is None:
            from quantum_common.config.credentials import get_dwave_token

            self.token = get_dwave_token()

    @property
    def backend_type(self) -> BackendType:
        return BackendType.DWAVE

    @property
    def paradigm(self) -> QuantumParadigm:
        return QuantumParadigm.ANNEALING

    @property
    def name(self) -> str:
        suffix = "(hybrid)" if self.use_hybrid else ""
        return f"dwave:{self.solver_name or 'auto'}{suffix}"

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    def _get_sampler(self) -> Any:
        try:
            if self.use_hybrid:
                # LeapHybridSampler: hybrid quantum-classical solver for
                # large BQMs. Ocean SDK 9.x provides improved decomposition.
                from dwave.system import LeapHybridSampler

                return LeapHybridSampler(token=self.token)
            else:
                # DWaveSampler + EmbeddingComposite: direct QPU access with
                # automatic minor-embedding onto Zephyr (Advantage2) or
                # Pegasus (Advantage) topology.
                from dwave.system import DWaveSampler, EmbeddingComposite

                sampler = DWaveSampler(solver=self.solver_name, token=self.token)
                self._max_qubits = sampler.properties.get("num_qubits", 5000)
                return EmbeddingComposite(sampler)
        except ImportError as e:
            raise BackendError("dwave-ocean-sdk not installed") from e
        except Exception as e:
            raise BackendError(f"D-Wave connection failed: {e}") from e

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        import time

        sampler = self._get_sampler()
        num_reads = request.options.get("num_reads", self.num_reads)

        t0 = time.perf_counter()
        all_counts: list[dict[str, int]] = []
        raw_results: list[Any] = []

        for bqm in request.circuits:
            if self.use_hybrid:
                sampleset = sampler.sample(bqm)
            else:
                sampleset = sampler.sample(
                    bqm,
                    num_reads=num_reads,
                    annealing_time=self.annealing_time_us,
                )
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
            metadata={
                "solver": self.solver_name,
                "hybrid": self.use_hybrid,
                "num_reads": num_reads,
            },
            execution_time_s=elapsed,
        )

    def transpile(
        self, circuits: list[Any], optimization_level: int = 1
    ) -> list[Any]:
        return circuits  # No transpilation for annealing problems

    def is_available(self) -> bool:
        try:
            self._get_sampler()
            return True
        except Exception:
            return False

    def status_info(self) -> dict[str, Any]:
        try:
            if self.use_hybrid:
                from dwave.system import LeapHybridSampler

                sampler = LeapHybridSampler(token=self.token)
                return {"solver": "hybrid", "available": True}
            else:
                from dwave.system import DWaveSampler

                sampler = DWaveSampler(solver=self.solver_name, token=self.token)
                props = sampler.properties
                return {
                    "solver": props.get("chip_id", self.solver_name),
                    "num_qubits": props.get("num_qubits"),
                    "topology": props.get("topology", {}).get("type"),
                    "available": True,
                }
        except Exception as e:
            return {"error": str(e), "available": False}
