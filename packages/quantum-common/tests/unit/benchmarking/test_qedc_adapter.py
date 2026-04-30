"""Smoke tests for QEDCAdapter + MHT extension."""
from __future__ import annotations

from quantum_common.benchmarking.qedc_adapter import (
    QEDCAdapter,
    QEDCInstance,
)


class TestMetadataMode:
    def test_iter_returns_catalogue(self) -> None:
        adapter = QEDCAdapter(suite_root=None)
        instances = list(adapter.iter_benchmarks())
        assert all(isinstance(inst, QEDCInstance) for inst in instances)
        assert any(inst.suite == "amplitude_estimation" for inst in instances)
        assert any(inst.suite == "vqe" for inst in instances)

    def test_filter_by_family(self) -> None:
        adapter = QEDCAdapter()
        maxcut_only = list(adapter.iter_benchmarks(["maxcut"]))
        assert all(inst.suite == "maxcut" for inst in maxcut_only)

    def test_mht_extension_scales(self) -> None:
        adapter = QEDCAdapter()
        mht = list(adapter.iter_mht_extension())
        assert len(mht) == 5
        sizes = [inst.size for inst in mht]
        assert sizes == sorted(sizes)
        assert mht[0].parameters["n_tracks"] <= mht[-1].parameters["n_tracks"]

    def test_missing_root_falls_back_to_metadata(self, tmp_path) -> None:
        adapter = QEDCAdapter(suite_root=str(tmp_path / "does-not-exist"))
        # Missing path -> metadata mode.
        assert adapter.mode == "metadata-only"
        assert list(adapter.iter_benchmarks())
