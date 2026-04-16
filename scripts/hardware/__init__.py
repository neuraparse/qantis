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

    # Make numpy scalars JSON-serialisable
    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with fname.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_default)

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
