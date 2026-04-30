"""Smoke tests for 2026 benchmarking additions.

Covers:
    - ``BenchmarkProtocol`` TTS(99%) math + scaling fit + Wilcoxon.
    - ``EmbeddingReport`` + ``BenchmarkRecord`` pydantic round-trip.
    - ``qantis_bench_schema()`` produces a usable JSON schema.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from quantum_common.benchmarking.protocol import (
    BenchmarkProtocol,
    InstanceRun,
)
from quantum_common.benchmarking.reporting import (
    BackendMeta,
    BenchmarkRecord,
    EmbeddingReport,
    Metrics,
    ProblemMeta,
    Reproducibility,
    qantis_bench_schema,
)


class TestBenchmarkProtocolMath:
    def test_tts_formula(self) -> None:
        protocol = BenchmarkProtocol()
        # p_hit = 0.5, wall_time = 1s, target 0.99 -> TTS = log(0.01)/log(0.5)
        expected = 1.0 * math.log(1.0 - 0.99) / math.log(1.0 - 0.5)
        assert protocol.tts(0.5, 1.0) == pytest.approx(expected, rel=1e-9)

    def test_tts_clamps_zero_and_one(self) -> None:
        protocol = BenchmarkProtocol()
        # p_hit exactly 1 would blow up log(0); clamp to near-1 inside.
        assert protocol.tts(1.0, 1.0) > 0
        assert protocol.tts(0.0, 1.0) > 0

    def test_effective_tts_penalizes_chain_breaks(self) -> None:
        protocol = BenchmarkProtocol()
        base_tts = 10.0
        # cbf=0 => eff = base; cbf=0.5, L_max=1 => eff = 20.0
        assert protocol.effective_tts(base_tts, 1, 0.0) == pytest.approx(10.0)
        assert protocol.effective_tts(base_tts, 1, 0.5) == pytest.approx(20.0)
        assert protocol.effective_tts(base_tts, 4, 0.25) > base_tts


class TestSummaryAndScaling:
    def _runs(self, size: int, p: float, wall: float, count: int = 10) -> list[InstanceRun]:
        return [
            InstanceRun(
                instance_id=f"inst_{size}_{i}",
                size=size,
                hit=(i < int(round(p * count))),
                energy=-1.0,
                ref_energy=-1.0,
                wall_time_s=wall,
                chain_length_max=1,
                chain_break_fraction=0.0,
            )
            for i in range(count)
        ]

    def test_summarize_instance_hit_probability(self) -> None:
        protocol = BenchmarkProtocol()
        summary = protocol.summarize_instance(self._runs(size=10, p=0.3, wall=0.1))
        assert 0.2 < summary.p_hit < 0.4
        assert summary.median_wall_time_s == pytest.approx(0.1)
        assert summary.tts_s > 0

    def test_scaling_fit_positive_slope(self) -> None:
        protocol = BenchmarkProtocol(sizes=(10, 20, 40), seed=11)
        medians = {10: 1.0, 20: 10.0, 40: 1000.0}
        fit = protocol.scaling_fit(medians)
        assert fit["slope"] > 0
        assert fit["r2"] > 0.8
        ci = fit["ci95"]
        assert ci[0] <= fit["slope"] <= ci[1]

    def test_pairwise_significance_detects_dominance(self) -> None:
        protocol = BenchmarkProtocol()
        slow = [10.0 + 0.1 * i for i in range(30)]
        fast = [v * 0.5 for v in slow]
        result = protocol.pairwise_significance(slow, fast)
        assert result["p_value"] <= 0.01
        assert result["significant_p01"] is True

    def test_pairwise_short_series_abstains(self) -> None:
        protocol = BenchmarkProtocol()
        result = protocol.pairwise_significance([1.0, 2.0], [3.0, 4.0])
        assert result["significant_p01"] is False


class TestEmbeddingReport:
    def test_round_trip_json(self) -> None:
        report = EmbeddingReport(
            logical_vars=175,
            physical_qubits=525,
            embedded_to_logical_ratio=3.0,
            chain_length_mean=2.8,
            chain_length_max=4,
            chain_length_p95=3.7,
            chain_length_stddev=0.6,
            chain_strength=1.5,
            chain_break_fraction_mean=0.02,
            annealing_time_us=20.0,
            num_reads=1000,
        )
        payload = report.model_dump_json()
        restored = EmbeddingReport.model_validate_json(payload)
        assert restored.physical_qubits == 525
        assert restored.chain_length_max == 4

    def test_bounds_enforced(self) -> None:
        with pytest.raises(Exception):
            EmbeddingReport(
                logical_vars=-1,
                physical_qubits=0,
                embedded_to_logical_ratio=0.0,
                chain_length_mean=0.0,
                chain_length_max=0,
                chain_length_p95=0.0,
                chain_length_stddev=0.0,
            )


class TestBenchmarkRecord:
    def test_record_round_trip(self) -> None:
        record = BenchmarkRecord(
            run_id="r-123",
            timestamp_utc=datetime.now(timezone.utc),
            problem=ProblemMeta(
                name="mtda_small",
                instance_id="mtda-2x2-seed7",
                size=2,
                qubo_density=0.5,
            ),
            backend=BackendMeta(vendor="dwave", device="Advantage2", topology="zephyr"),
            metrics=Metrics(
                time_to_solution_s=0.12,
                energy_best=-1.0,
                energy_mean=-0.8,
                approximation_ratio=1.0,
                success_probability=0.95,
            ),
            embedding=EmbeddingReport(
                logical_vars=8,
                physical_qubits=24,
                embedded_to_logical_ratio=3.0,
                chain_length_mean=3.0,
                chain_length_max=3,
                chain_length_p95=3.0,
                chain_length_stddev=0.0,
            ),
            repro=Reproducibility(
                git_sha="abc123",
                seed=42,
                python_version="3.14",
            ),
        )
        payload = record.model_dump_json()
        restored = BenchmarkRecord.model_validate_json(payload)
        assert restored.run_id == "r-123"
        assert restored.embedding.physical_qubits == 24


class TestJsonSchema:
    def test_schema_has_required_fields(self) -> None:
        schema = qantis_bench_schema()
        # pydantic v2 emits $defs and properties at the top level for models
        props = schema.get("properties", {})
        for field in ("run_id", "problem", "metrics", "repro"):
            assert field in props, f"missing required schema field: {field}"
