"""Shared helpers for QANTIS hardware validation scripts.

Utilities used by run_tiger_ibm.py, run_mtda_dwave.py,
run_fpc_qaoa_ibm.py, and run_zne_ibm.py.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(obj: Any) -> Any:
    """JSON serialiser hook for numpy / non-native types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def save_partial(name: str, data: dict[str, Any]) -> Path:
    """Write a checkpoint JSON to output/hardware mid-experiment.

    Use inside long-running Pittsburgh loops so that a blocking
    ``SamplerV2.run().result()`` hang (seen 2026-04-19) does not
    destroy accumulated counts. The same
    ``<name>_partial.json`` is overwritten on each call so tailing
    scripts always see the latest state.
    """
    path = _output_dir() / f"{name}_partial.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    return path


def _fetch_result_robust(
    job_id: str,
    poll_timeout: float = 120.0,
    max_total_wait: float = 7200.0,
) -> Any:
    """Robust replacement for ``job.result()`` in qiskit-ibm-runtime 0.42-0.46.

    The stock ``SamplerV2`` job.result() is a blocking polling loop over
    ``job.status()``. We repeatedly observed the client-side instance
    hanging for hours after the server reported ``DONE`` (the 2026-04-19
    campaign lost 3 experiments to this). The fix is to re-fetch the
    job through a fresh :class:`QiskitRuntimeService` every
    ``poll_timeout`` seconds so stale sockets and stuck polling loops
    are bypassed. Total wall time is bounded by ``max_total_wait``.
    """
    import time

    from qiskit_ibm_runtime import QiskitRuntimeService

    start = time.time()
    last_status = None
    while True:
        elapsed = time.time() - start
        if elapsed > max_total_wait:
            raise TimeoutError(
                f"job {job_id} did not reach a final state within "
                f"{max_total_wait:.0f}s"
            )
        try:
            svc = QiskitRuntimeService()  # fresh client each iteration
            j = svc.job(job_id)
            status = str(j.status())
            if status != last_status:
                print(f"[heron] job {job_id} status={status} "
                      f"elapsed={elapsed:.0f}s")
                last_status = status
            if status == "DONE":
                return j.result(timeout=max(poll_timeout, 60.0))
            if status in ("ERROR", "CANCELLED"):
                raise RuntimeError(f"job {job_id} ended in {status}")
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[heron] transient poll error for {job_id}: "
                  f"{type(exc).__name__}: {exc}; retrying")
        time.sleep(min(poll_timeout, 30.0))


def run_on_heron_batched(
    circuits: list[Any],
    backend_name: str,
    shots: int,
    optimization_level: int = 2,
    checkpoint_name: str | None = None,
    poll_timeout: float = 120.0,
    max_total_wait: float = 7200.0,
) -> tuple[list[dict[str, int]], str]:
    """Submit many circuits to an IBM backend in a single Sampler run.

    Builds a single :class:`SamplerV2` job with all circuits in one PUB
    list so the backend dequeues them as a single workload (one queue
    wait instead of ``len(circuits)``). The result is fetched via
    :func:`_fetch_result_robust` which re-opens a fresh
    :class:`QiskitRuntimeService` every ``poll_timeout`` seconds so
    stale-socket hangs (observed repeatedly on 2026-04-19 for
    qiskit-ibm-runtime 0.42-0.46) do not leak into the campaign.
    ``max_total_wait`` bounds total wall time (default 2 hours).

    Returns ``(counts_list, job_id)`` where ``counts_list`` is aligned
    1:1 with the input circuit order.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService()  # saved account
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend
    )
    isa_list = [pm.run(c) for c in circuits]
    for isa in isa_list:
        isa._layout = None

    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_list, shots=shots)
    job_id = job.job_id()
    print(f"[heron] submitted job {job_id} with {len(circuits)} circuits")

    if checkpoint_name:
        save_partial(checkpoint_name, {
            "campaign": checkpoint_name,
            "backend": backend_name,
            "submitted_job_id": job_id,
            "num_circuits": len(circuits),
            "shots": shots,
            "submitted_utc": datetime.now(tz=timezone.utc).isoformat(),
            "note": (
                "Partial checkpoint written before polling .result(). "
                "If the wrapper still hangs, recover via "
                "`QiskitRuntimeService().job(<job_id>).result()`."
            ),
        })

    result = _fetch_result_robust(
        job_id, poll_timeout=poll_timeout, max_total_wait=max_total_wait,
    )
    counts_list: list[dict[str, int]] = []
    for pub in result:
        try:
            counts_list.append(databin_get_counts(pub))
        except Exception as exc:
            counts_list.append({"_error": f"{type(exc).__name__}: {exc}"})

    return counts_list, job_id


def run_on_aer_batched(
    circuits: list[Any], shots: int
) -> list[dict[str, int]]:
    """Aer-equivalent of :func:`run_on_heron_batched` for dry-runs.

    Transpiles to the Aer basis so controlled higher-level gates (e.g.
    controlled-Hadamard used by the corridor-Tiger transition unitary)
    decompose into Aer-executable primitives.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    counts_list = []
    for circ in circuits:
        try:
            res = sim.run(circ, shots=shots).result()
            counts_list.append(dict(res.get_counts()))
        except Exception:
            # Aer cannot execute non-standard gate names (e.g. ``cch``)
            # natively; transpile to the Aer default basis and retry.
            t = transpile(
                circ,
                basis_gates=["cx", "ccx", "cz", "rz", "sx", "x", "h", "ry", "rx",
                             "u", "measure", "reset"],
                optimization_level=1,
            )
            res = sim.run(t, shots=shots).result()
            counts_list.append(dict(res.get_counts()))
    return counts_list


def databin_get_counts(pub_result: Any) -> dict[str, int]:
    """Robustly pull ``{bitstring: count}`` out of a SamplerV2 PubResult.

    qiskit-ibm-runtime renamed the DataBin field across 0.41 -> 0.42 and
    the transpiler may also relabel the classical register based on the
    circuit layout. Walk the DataBin and return the first field that
    exposes a ``get_counts`` method.
    """
    data = getattr(pub_result, "data", pub_result)
    for name in ("meas", "c", "meas_c", "classical", "cr"):
        field = getattr(data, name, None)
        if field is not None and hasattr(field, "get_counts"):
            return dict(field.get_counts())
    for name in dir(data):
        if name.startswith("_"):
            continue
        field = getattr(data, name)
        if hasattr(field, "get_counts"):
            return dict(field.get_counts())
    raise RuntimeError(
        f"SamplerV2 DataBin exposes no classical field with get_counts; "
        f"attributes: {[n for n in dir(data) if not n.startswith('_')]}"
    )

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def _output_dir() -> Path:
    """Return (and create) the hardware results directory."""
    root = Path(__file__).resolve().parents[2]
    d = root / "output" / "hardware"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(name: str, data: dict[str, Any]) -> Path:
    """Serialise *data* to JSON and write to output/hardware/<name>_TIMESTAMP.json.

    Returns the path of the written file.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    fname = _output_dir() / f"{name}_{timestamp}.json"
    with fname.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    print(f"[saved] {fname}")
    return fname


# ---------------------------------------------------------------------------
# Quantum circuit helpers
# ---------------------------------------------------------------------------

def compute_zz_expectation(counts: dict[str, int]) -> float:
    """Compute <ZZ> expectation value from a 2-qubit bitstring count dict.

    <ZZ> = sum_bitstring  P(bitstring) * (-1)^(b0 XOR b1)

    For a Bell state |Φ+> = (|00> + |11>)/√2 the ideal value is +1.0.
    Hardware noise drives the estimate below 1.0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    zz = 0.0
    for bitstring, count in counts.items():
        # Qiskit orders bits right-to-left; take the two least-significant.
        b0 = int(bitstring[-1])
        b1 = int(bitstring[-2]) if len(bitstring) >= 2 else 0
        parity = (-1) ** (b0 ^ b1)
        zz += (count / total) * parity
    return float(zz)


# ---------------------------------------------------------------------------
# Tiger POMDP factory
# ---------------------------------------------------------------------------

def build_tiger_pomdp() -> Any:
    """Return the canonical Tiger POMDP (Kaelbling et al., 1998).

    |S|=2 (tiger-left/right), |A|=3 (listen, open-left, open-right),
    |Ω|=2 (hear-left, hear-right).  P(hear-correct | listen) = 0.85.
    γ = 0.95.
    """
    from quantum_pomdp.scenarios.tiger_problem import create_tiger_pomdp

    return create_tiger_pomdp(
        listen_accuracy=0.85,
        listen_cost=-1.0,
        tiger_penalty=-100.0,
        treasure_reward=10.0,
        discount_factor=0.95,
    )


# ---------------------------------------------------------------------------
# QUBO instance factory
# ---------------------------------------------------------------------------

def make_qubo_instance(
    n_tracks: int,
    n_meas: int,
    seed: int = 42,
) -> Any:
    """Build an MTDA QUBO from synthetic tracking data.

    Args:
        n_tracks: Number of active tracks (N).
        n_meas:   Number of measurements per frame (M).
        seed:     NumPy random seed for reproducibility.

    Returns:
        QUBOResult with Q matrix and variable map.
    """
    from quantum_mht.formulation.mtda_qubo_builder import MTDAQuboBuilder

    rng = np.random.default_rng(seed)
    predicted = rng.standard_normal((n_tracks, 2)).astype(np.float64)
    measurements = rng.standard_normal((n_meas, 2)).astype(np.float64)
    covariances = np.stack([np.eye(2, dtype=np.float64) * 0.5] * n_tracks)

    builder = MTDAQuboBuilder()
    return builder.build(predicted, measurements, covariances)


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def require_ibm_token() -> str:
    """Return IBM_QUANTUM_TOKEN from environment or raise SystemExit."""
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not token:
        print("[error] Set IBM_QUANTUM_TOKEN environment variable.")
        raise SystemExit(1)
    return token


def get_ibm_token_optional() -> str | None:
    """Return IBM_QUANTUM_TOKEN from environment, or None if not set.

    When None is returned, IBMQuantumBackend will fall back to the saved
    account (~/.qiskit/qiskit-ibm.json) loaded via
    QiskitRuntimeService.save_account().
    """
    return os.environ.get("IBM_QUANTUM_TOKEN") or None


def ibm_channel() -> str:
    """Return IBM_QUANTUM_CHANNEL from env (default: ibm_quantum_platform)."""
    return os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")


def ibm_instance() -> str | None:
    """Return IBM_QUANTUM_INSTANCE from env (None if not set)."""
    return os.environ.get("IBM_QUANTUM_INSTANCE") or None


def make_ibm_runtime_service(
    *,
    token: str | None = None,
    channel: str | None = None,
    instance: str | None = None,
) -> Any:
    """Create a QiskitRuntimeService using CLI args or env defaults.

    The helper keeps all hardware runners on the same credential path so we can
    switch between saved accounts and explicit ``token/channel/instance``
    settings without patching individual scripts.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    resolved_token = token if token is not None else get_ibm_token_optional()
    resolved_channel = channel or ibm_channel()
    resolved_instance = instance if instance is not None else ibm_instance()

    kwargs: dict[str, Any] = {}
    if resolved_token:
        kwargs["token"] = resolved_token
    if resolved_channel:
        kwargs["channel"] = resolved_channel
    if resolved_instance:
        kwargs["instance"] = resolved_instance

    if kwargs:
        return QiskitRuntimeService(**kwargs)
    return QiskitRuntimeService()


def require_dwave_token() -> str:
    """Return DWAVE_API_TOKEN from environment or raise SystemExit."""
    token = os.environ.get("DWAVE_API_TOKEN", "")
    if not token:
        print("[error] Set DWAVE_API_TOKEN environment variable.")
        raise SystemExit(1)
    return token
