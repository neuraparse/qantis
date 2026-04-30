"""Pydantic models + JSON schema for QANTIS benchmark reports.

Provides the reproducibility manifest expected by reviewers in 2025-2026:

    - ``EmbeddingReport``: the chain-length / chain-break / timing fields
      that arXiv:2504.13376 (Mar 2026) correlates with solution error.
    - ``BenchmarkRecord``: per-run artifact -- problem metadata, backend
      calibration snapshot, TTS metrics, optional embedding report,
      reproducibility block (git SHA, container digest, seed).
    - ``qantis_bench_schema()``: returns the JSON schema as a dict,
      suitable for persisting to a ``qantis_bench.schema.json`` file in
      the repo or shipping alongside benchmark outputs.

Usage: construct a ``BenchmarkRecord`` from a solver run, serialize with
``.model_dump_json()``, and persist next to the ``dwave-inspector`` JSON
when applicable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EmbeddingReport(BaseModel):
    """Required D-Wave embedding-quality fields per arXiv:2504.13376 (Mar 2026)."""

    logical_vars: int = Field(..., ge=0)
    physical_qubits: int = Field(..., ge=0)
    embedded_to_logical_ratio: float = Field(..., ge=0.0)
    chain_length_mean: float = Field(..., ge=0.0)
    chain_length_max: int = Field(..., ge=0)
    chain_length_p95: float = Field(..., ge=0.0)
    chain_length_stddev: float = Field(..., ge=0.0)
    chain_strength: float | None = None
    chain_break_fraction_mean: float = Field(default=0.0, ge=0.0, le=1.0)
    anneal_schedule: list[tuple[float, float]] | None = None
    annealing_time_us: float | None = None
    num_reads: int | None = None
    qpu_access_time_us: float | None = None
    qpu_sampling_time_us: float | None = None
    embedding_runtime_s: float | None = None


class ProblemMeta(BaseModel):
    name: str
    instance_id: str
    size: int = Field(..., gt=0)
    qubo_density: float = Field(..., ge=0.0, le=1.0)
    difficulty_tier: str | None = None


class BackendMeta(BaseModel):
    vendor: str
    device: str
    topology: str | None = None
    calibration_timestamp: datetime | None = None


class Metrics(BaseModel):
    time_to_solution_s: float = Field(..., ge=0.0)
    energy_best: float
    energy_mean: float
    approximation_ratio: float = Field(..., ge=0.0)
    success_probability: float = Field(..., ge=0.0, le=1.0)
    hellinger_fidelity: float | None = None


class Reproducibility(BaseModel):
    git_sha: str
    container_digest: str | None = None
    seed: int
    python_version: str
    ocean_version: str | None = None
    qiskit_version: str | None = None


class BenchmarkRecord(BaseModel):
    """Per-run artifact emitted by the QANTIS benchmark harness."""

    run_id: str
    timestamp_utc: datetime
    problem: ProblemMeta
    backend: BackendMeta
    metrics: Metrics
    embedding: EmbeddingReport | None = None
    repro: Reproducibility


def qantis_bench_schema() -> dict[str, Any]:
    """Return the JSON schema for :class:`BenchmarkRecord`.

    Ship this alongside benchmark outputs so downstream tools
    (reviewer-side validators, MITRE / QED-C harnesses, CI gates) can
    validate the manifest without pulling the code in.
    """
    return BenchmarkRecord.model_json_schema()
