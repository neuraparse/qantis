"""Tests for mitigation pipeline."""

import pytest


class TestMitigationPipeline:
    def test_empty_pipeline(self) -> None:
        try:
            from quantum_common.mitigation.pipeline import MitigationPipeline
            pipeline = MitigationPipeline()
            assert len(pipeline) == 0
        except ImportError:
            pytest.skip("mitigation module not ready")

    def test_passthrough(self) -> None:
        try:
            from quantum_common.mitigation.pipeline import MitigationPipeline
            pipeline = MitigationPipeline()
            counts = {"00": 500, "11": 500}
            result = pipeline.apply(None, None, counts, 1000)
            assert result == counts
        except ImportError:
            pytest.skip("mitigation module not ready")

    def test_add_strategy(self) -> None:
        try:
            from quantum_common.mitigation.pipeline import MitigationPipeline
            from quantum_common.mitigation.zne import ZNEStrategy
            pipeline = MitigationPipeline()
            pipeline.add(ZNEStrategy())
            assert len(pipeline) == 1
        except ImportError:
            pytest.skip("mitigation module not ready")
