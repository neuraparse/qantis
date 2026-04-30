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


def _databin_get_counts(pub_result: Any) -> dict[str, int]:
    """Walk a SamplerV2 DataBin to the first get_counts-capable field.

    qiskit-ibm-runtime 0.42 renamed / reshaped the DataBin so the legacy
    ``data.meas`` attribute is no longer stable; the transpiled circuit
    may expose the classical register under a different name depending
    on layout. This helper is the single place where we paper over the
    difference.
    """
    data = getattr(pub_result, "data", pub_result)
    for name in ("meas", "c", "meas_c", "classical", "cr"):
        field_obj = getattr(data, name, None)
        if field_obj is not None and hasattr(field_obj, "get_counts"):
            return dict(field_obj.get_counts())
    for name in dir(data):
        if name.startswith("_"):
            continue
        field_obj = getattr(data, name)
        if hasattr(field_obj, "get_counts"):
            return dict(field_obj.get_counts())
    raise RuntimeError(
        f"SamplerV2 DataBin exposes no classical field with get_counts; "
        f"attributes: {[n for n in dir(data) if not n.startswith('_')]}"
    )

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
        # SamplerV2 returns PubResult objects; DataBin field name depends on
        # transpile layout (qiskit-ibm-runtime 0.42+) so walk to the first
        # get_counts-capable field.
        for pub_result in result:
            all_counts.append(_databin_get_counts(pub_result))

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

    # ------------------------------------------------------------------
    # Qiskit Runtime V2 execution modes
    # ------------------------------------------------------------------

    def run_iterative(
        self,
        ansatz_fn: Any,
        cost_op: Any,
        theta0: Any,
        optimizer: Any,
        shots: int = 4096,
        enable_dd: bool = True,
    ) -> Any:
        """Iterative QAOA / QBRL loop inside a Runtime ``Session``.

        qiskit-ibm-runtime Q4 2025 - Q2 2026 docs continue to recommend
        :class:`Session` (not :class:`Batch`) for optimization loops
        where each objective evaluation depends on the previous one.
        The session preserves QPU allocation between classical steps.
        """
        try:
            from qiskit_ibm_runtime import Session, EstimatorV2
        except ImportError as exc:
            raise BackendError("qiskit-ibm-runtime not installed") from exc

        service = self._get_service()
        backend = service.backend(self.backend_name)
        with Session(backend=backend) as session:
            estimator = EstimatorV2(mode=session)
            estimator.options.default_shots = int(shots)
            if enable_dd:
                try:
                    estimator.options.dynamical_decoupling.enable = True
                except Exception:
                    logger.debug("EstimatorV2 DD option unavailable on this runtime")

            def objective(theta: Any) -> float:
                pub = (ansatz_fn(theta), [cost_op])
                job = estimator.run([pub])
                return float(job.result()[0].data.evs[0])

            return optimizer.minimize(objective, theta0)

    def run_batch(
        self,
        circuits: list[Any],
        shots: int = 4096,
    ) -> ExecutionResult:
        """Submit a non-adaptive batch of circuits under Runtime ``Batch``.

        Prefer this for BIQAE posterior sweeps and any other workload
        where every PUB is known upfront. Batch packs submissions into
        a single queue slot while bypassing Session's iterative machinery.
        """
        import time

        try:
            from qiskit_ibm_runtime import Batch, SamplerV2
        except ImportError as exc:
            raise BackendError("qiskit-ibm-runtime not installed") from exc

        service = self._get_service()
        backend = service.backend(self.backend_name)

        t0 = time.perf_counter()
        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            job = sampler.run(circuits, shots=shots)
            result = job.result()

        all_counts: list[dict[str, int]] = []
        for pub_result in result:
            all_counts.append(_databin_get_counts(pub_result))

        elapsed = time.perf_counter() - t0
        return ExecutionResult(
            counts=all_counts,
            raw_results=[result],
            metadata={
                "backend": self.backend_name,
                "mode": "batch",
                "job_id": job.job_id(),
                "shots": shots,
            },
            execution_time_s=elapsed,
        )

    # ------------------------------------------------------------------
    # Q-CTRL Fire Opal (Qiskit Function, Premium / Flex plan gated)
    # ------------------------------------------------------------------

    def execute_fire_opal(
        self,
        request: ExecutionRequest,
        primitive: str = "sampler",
        optimization_level: int = 1,
    ) -> ExecutionResult:
        """Route circuits through Q-CTRL's Fire Opal Qiskit Function.

        Fire Opal stacks pulse-level error-suppression (DD, measurement
        twirling, tuned gate replacements) in front of SamplerV2 /
        EstimatorV2 and was demonstrated to reach 75-qubit verifiable
        entanglement on ibm_fez in Edmunds et al. PRX Quantum 6, 020331
        (2025), DOI 10.1103/PRXQuantum.6.020331. Requires an IBM
        Premium / Flex plan with the ``q-ctrl/performance-management``
        function installed; raises :class:`BackendError` otherwise.
        """
        import time

        try:
            from qiskit_ibm_catalog import QiskitFunctionsCatalog  # type: ignore
        except ImportError as exc:
            raise BackendError(
                "qiskit-ibm-catalog not installed (required for Fire Opal)"
            ) from exc

        catalog = QiskitFunctionsCatalog(channel=self.channel, token=self.token)
        try:
            perf = catalog.load("q-ctrl/performance-management")
        except Exception as exc:
            raise BackendError(
                "Fire Opal Qiskit Function requires an IBM Premium or Flex plan; "
                f"load failed: {exc}"
            ) from exc

        t0 = time.perf_counter()
        job = perf.run(
            pubs=[(c,) for c in request.circuits],
            backend_name=self.backend_name,
            shots=request.shots,
            primitive=primitive,
        )
        result = job.result()

        all_counts: list[dict[str, int]] = []
        for pub_result in result:
            try:
                all_counts.append(_databin_get_counts(pub_result))
            except Exception:
                all_counts.append({})

        elapsed = time.perf_counter() - t0
        return ExecutionResult(
            counts=all_counts,
            raw_results=[result],
            metadata={
                "backend": self.backend_name,
                "mode": "fire_opal",
                "optimization_level": optimization_level,
                "shots": request.shots,
            },
            execution_time_s=elapsed,
        )
