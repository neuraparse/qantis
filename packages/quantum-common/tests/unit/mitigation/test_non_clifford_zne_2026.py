"""Smoke tests for the Ezzell 2026 non-Clifford ZNE scaler."""
from __future__ import annotations

import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit.circuit import QuantumCircuit  # noqa: E402

from quantum_common.mitigation.zne import non_clifford_scale_method  # noqa: E402


class TestNonCliffordScaler:
    def test_identity_at_scale_one(self) -> None:
        qc = QuantumCircuit(1)
        qc.rx(0.5, 0)
        qc.rz(0.8, 0)
        out = non_clifford_scale_method(qc, 1.0)
        assert len(out.data) == len(qc.data)

    def test_scale_inserts_rotation_pairs(self) -> None:
        qc = QuantumCircuit(2)
        qc.rx(0.5, 0)
        qc.cx(0, 1)
        qc.rz(0.8, 1)
        out = non_clifford_scale_method(qc, 3.0)
        # Each non-Clifford rotation gains an identity pair (two gates).
        assert len(out.data) == len(qc.data) + 4

    def test_cx_not_duplicated(self) -> None:
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        out = non_clifford_scale_method(qc, 5.0)
        assert len(out.data) == len(qc.data)
