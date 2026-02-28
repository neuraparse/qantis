"""IBM Quantum backend using Qiskit Runtime V2 Primitives.

2026 Academic References — IBM Quantum / Qiskit Runtime
=========================================================
- **Qiskit v2.3** (Jan 2026): SamplerV2 and EstimatorV2 are the default
  primitives. SamplerV2.run() accepts ``shots`` per PUB (Primitive Unified Bloc).
  New features include PauliProductMeasurement, gridsynth_rz() gate synthesis.
  See: https://docs.quantum.ibm.com/api/qiskit/primitives

- **Qiskit Runtime v0.36+**: Session-based execution model for reduced
  queue latency. ``QiskitRuntimeService`` manages authentication and backend
  selection. Deprecated ``Sampler``/``Estimator`` in favour of V2 variants.

- **IBM Heron R1** (ibm_torino, 133 qubits): Heavy-hex topology, tunable
  couplers, median ECR error ~2.5e-3. Our primary validated backend.
- **IBM Heron R2** (ibm_fez / ibm_marrakesh, 156 qubits): TLS mitigation,
  improved EPLG stability (~1.5e-3 ECR error). Released July 2024.
  See: IBM Quantum Hardware Roadmap 2025-2026.

- **arXiv:2512.08245** (Dec 2025): Analysis showing SamplerV2 default of
  10K shots captures only ~23% of the state space for typical combinatorial
  optimization problems. Recommends adaptive shot allocation strategies.

- **Session Management**: Qiskit Runtime Sessions batch multiple jobs under
  a single quantum allocation window, reducing idle time between circuits.
  Critical for iterative VQE/QAOA loops.

- **PauliProductMeasurement**: Joint multi-qubit projective measurements
  for efficient observable estimation.

- **CommutativeOptimization**: New transpiler pass exploiting gate
  commutativity for depth reduction.

- **LitinskiTransformation**: Extended to measurements in v2.3, enabling
  end-to-end Pauli-based computation for fault-tolerance.

- **C API expansion**: Custom transpiler passes with QkDag and QkTarget
  objects; Rust-based VF2Layout/VF2PostLayout speedups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

from quantum_common.types import BackendType, QuantumParadigm
from quantum_common.exceptions import BackendError
from quantum_common.backends.base import ExecutionRequest, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class IBMQuantumBackend:
    """IBM Quantum backend via Qiskit Runtime V2.

    Targets IBM Heron R1 (ibm_torino, 133 qubits), Heron R2 (ibm_fez /
    ibm_marrakesh, 156 qubits), and Eagle R3 (127 qubits) processors.
    Uses SamplerV2 for measurement-based execution with configurable shot counts.

    Authentication: provide a token, set IBM_QUANTUM_TOKEN env var, or rely
    on a saved account (~/.qiskit/qiskit-ibm.json via save_account()).
    """

    channel: str = "ibm_quantum_platform"
    instance: str | None = None
    backend_name: str = "ibm_brisbane"
    token: str | None = None
    _max_qubits: int = 127

    def __post_init__(self) -> None:
        # Try credentials module; None is acceptable — _get_service() will fall
        # back to the saved account (~/.qiskit/qiskit-ibm.json) in that case.
        if self.token is None:
            try:
                from quantum_common.config.credentials import get_ibm_token
                self.token = get_ibm_token()
            except Exception:
                self.token = None  # no token configured — saved account fallback

    @property
    def backend_type(self) -> BackendType:
        return BackendType.IBM_QUANTUM

    @property
    def paradigm(self) -> QuantumParadigm:
        return QuantumParadigm.GATE_BASED

    @property
    def name(self) -> str:
        return f"ibm:{self.backend_name}"

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    def _get_service(self) -> Any:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as e:
            raise BackendError("qiskit-ibm-runtime not installed") from e
        if self.token:
            # Explicit token provided — use it with the configured channel.
            kwargs: dict[str, Any] = {"channel": self.channel, "token": self.token}
            if self.instance is not None:
                kwargs["instance"] = self.instance
            return QiskitRuntimeService(**kwargs)
        # No token — attempt to load saved account (~/.qiskit/qiskit-ibm.json).
        # This is the recommended path when credentials are pre-saved via
        # QiskitRuntimeService.save_account(channel=..., token=..., overwrite=True).
        try:
            return QiskitRuntimeService()
        except Exception as e:
            raise BackendError(
                "No IBM token configured and no saved account found. "
                "Set IBM_QUANTUM_TOKEN env var or call "
                "QiskitRuntimeService.save_account(channel=..., token=...)."
                f"  Underlying error: {e}"
            ) from e

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        import time

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        except ImportError as e:
            raise BackendError("qiskit-ibm-runtime not installed") from e

        service = self._get_service()
        backend = service.backend(self.backend_name)
        self._max_qubits = backend.num_qubits

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # Qiskit v2.3: generate_preset_pass_manager replaces legacy transpile()
        # for ISA (Instruction Set Architecture) circuit preparation.
        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

        # After transpilation, the ISA circuit's layout maps the original named
        # registers (e.g., 's_t', 'a_t') to physical qubits in register 'q'.
        # QPY serialization then references those original register names, but
        # IBM's server deserializes only knowing the physical register 'q',
        # causing StopIteration → Error 3211. Fix: clear the layout so QPY
        # serializes without register-name references.
        # Measurement classical-register assignments are already correct in the
        # transpiled circuit, so result interpretation is unaffected.
        def _transpile_no_layout(c):
            isa = pm.run(c)
            # `layout` is a read-only property; clear the private backing store.
            isa._layout = None
            return isa

        isa_circuits = [_transpile_no_layout(c) for c in request.circuits]

        t0 = time.perf_counter()
        # SamplerV2 (Qiskit Runtime v0.36+): primitive-based execution.
        # Default shot count is 10K per PUB — see arXiv:2512.08245 for
        # state-space coverage analysis. We override via request.shots.
        sampler = SamplerV2(mode=backend)
        job = sampler.run(isa_circuits, shots=request.shots)
        result = job.result()

        all_counts: list[dict[str, int]] = []
        # SamplerV2 returns PubResult objects; .data.meas holds the classical
        # register as a BitArray. get_counts() converts to {bitstring: count}.
        for pub_result in result:
            counts = pub_result.data.meas.get_counts()
            all_counts.append(dict(counts))

        elapsed = time.perf_counter() - t0
        return ExecutionResult(
            counts=all_counts,
            raw_results=[result],
            metadata={
                "backend": self.backend_name,
                "job_id": job.job_id(),
                "shots": request.shots,
            },
            execution_time_s=elapsed,
        )

    def transpile(
        self, circuits: list[Any], optimization_level: int = 2
    ) -> list[Any]:
        service = self._get_service()
        backend = service.backend(self.backend_name)
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend
        )
        return [pm.run(c) for c in circuits]

    def is_available(self) -> bool:
        try:
            service = self._get_service()
            backend = service.backend(self.backend_name)
            return backend.status().operational
        except Exception:
            return False

    def status_info(self) -> dict[str, Any]:
        try:
            service = self._get_service()
            backend = service.backend(self.backend_name)
            status = backend.status()
            return {
                "backend": self.backend_name,
                "operational": status.operational,
                "pending_jobs": status.pending_jobs,
                "max_qubits": backend.num_qubits,
            }
        except Exception as e:
            return {"backend": self.backend_name, "error": str(e)}
