"""Smoke tests for 2026 mitigation additions (LRE, VD, PEA, Runtime readout)."""
from __future__ import annotations

import pytest

from quantum_common.mitigation import (
    DDStrategy,
    LREStrategy,
    PEAStrategy,
    PECStrategy,
    RuntimeReadoutStrategy,
    VDStrategy,
    ZNEStrategy,
)


class TestNameProperties:
    def test_zne_name(self) -> None:
        assert "Richardson" in ZNEStrategy().name

    def test_lre_name(self) -> None:
        assert "LRE" in LREStrategy().name

    def test_vd_name(self) -> None:
        assert "VD" in VDStrategy().name

    def test_pea_name(self) -> None:
        assert "PEA" in PEAStrategy().name

    def test_runtime_readout_name(self) -> None:
        assert "Runtime" in RuntimeReadoutStrategy().name


class TestApplyContracts:
    @pytest.mark.parametrize(
        "strategy",
        [LREStrategy(), VDStrategy(), PEAStrategy(), PECStrategy(), ZNEStrategy()],
    )
    def test_apply_is_not_a_counts_transform(self, strategy) -> None:
        with pytest.raises(NotImplementedError):
            strategy.apply(None, None, {"00": 1}, 1)


class TestRuntimeConfigure:
    def test_configure_primitive_is_safe_on_none(self) -> None:
        # Passing a bare object without options must not raise.
        RuntimeReadoutStrategy().configure_primitive(object())
        PECStrategy().configure_primitive(object())
