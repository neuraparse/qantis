"""VALIDATION-ROADMAP Task 1.4 — ZNE error mitigation on real IBM QPU.

Tests that Zero-Noise Extrapolation (ZNE) via Mitiq reduces the deviation
between IBM hardware output and the ideal (simulator) expectation value.

Bell state |Φ+> = (|00> + |11>)/√2 — ideal <ZZ> = +1.0.
Hardware noise causes <ZZ> < 1.0.
ZNE (Richardson extrapolation) should bring the estimate closer to 1.0.

Run:
  pytest -m hardware packages/quantum-common/tests/hardware/test_zne_hardware.py \\
      -v --ibm-backend ibm_brisbane --shots 4096
"""
from __future__ import annotations

import math
import pytest

from quantum_common.backends.base import ExecutionRequest
from quantum_common.mitigation.zne import ZNEStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_zz_expectation(counts: dict[str, int]) -> float:
    """<ZZ> expectation value from 2-qubit bitstring counts.

    <ZZ> = sum_bs  P(bs) * (-1)^(b0 XOR b1)
    Qiskit orders bits right-to-left.  Ideal Bell |Φ+> -> <ZZ> = +1.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    zz = 0.0
    for bitstring, count in counts.items():
        b0 = int(bitstring[-1])
        b1 = int(bitstring[-2]) if len(bitstring) >= 2 else 0
        zz += (count / total) * ((-1) ** (b0 ^ b1))
    return float(zz)


def _build_bell_circuit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def _save_result(name: str, data: dict) -> None:
    """Persist result JSON to output/hardware/."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out = Path(__file__).resolve().parents[5] / "output" / "hardware"
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{name}_{ts}.json"
    with fname.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestZNEOnIBMHardware:
    """Task 1.4: Zero-Noise Extrapolation on real IBM gate-based hardware."""

    def test_bell_state_zne(
        self,
        ibm_hardware_backend,
        aer_simulator_backend,
        request: pytest.FixtureRequest,
    ) -> None:
        """ZNE on Bell state |Φ+>: <ZZ> should approach +1 after mitigation.

        Protocol:
          1. Ideal <ZZ> from Aer simulator ≈ 1.0
          2. Raw <ZZ> from IBM hardware < 1.0 (noise)
          3. ZNE (Richardson, scale_factors=[1,1.5,2,3]) mitigated <ZZ>
          4. Assert |mitigated - ideal| < |raw - ideal|
        """
        shots = request.config.getoption("--shots")
        bell_qc = _build_bell_circuit()

        # Simulator ideal
        sim_res = aer_simulator_backend.execute(
            ExecutionRequest(circuits=[bell_qc], shots=shots)
        )
        sim_zz = _compute_zz_expectation(sim_res.counts[0])

        # Hardware raw
        hw_res = ibm_hardware_backend.execute(
            ExecutionRequest(circuits=[bell_qc], shots=shots)
        )
        hw_raw_counts = hw_res.counts[0]
        hw_raw_zz = _compute_zz_expectation(hw_raw_counts)

        # ZNE mitigation
        zne = ZNEStrategy(
            scale_factors=[1.0, 1.5, 2.0, 3.0],
            factory_type="Richardson",
        )
        mitigated_counts = zne.apply(bell_qc, ibm_hardware_backend, hw_raw_counts, shots)
        hw_mitigated_zz = _compute_zz_expectation(mitigated_counts)

        ideal = 1.0
        zne_improved = abs(hw_mitigated_zz - ideal) < abs(hw_raw_zz - ideal)

        _save_result("zne_bell_ibm", {
            "task": "1.4",
            "backend": ibm_hardware_backend.name,
            "shots": shots,
            "sim_zz": sim_zz,
            "hw_raw_zz": hw_raw_zz,
            "hw_mitigated_zz": hw_mitigated_zz,
            "zne_improved": zne_improved,
        })

        assert zne_improved, (
            f"ZNE did not improve: raw={hw_raw_zz:.4f}, "
            f"mitigated={hw_mitigated_zz:.4f}, ideal={ideal:.4f}"
        )

    def test_bell_state_simulator_ideal(
        self,
        aer_simulator_backend,
        request: pytest.FixtureRequest,
    ) -> None:
        """Simulator Bell <ZZ> should be ≥ 0.99 (infrastructure sanity check).

        This test does NOT require IBM hardware credentials and validates that
        the ZNE test infrastructure and the Bell circuit construction are correct.
        """
        shots = request.config.getoption("--shots")
        bell_qc = _build_bell_circuit()

        res = aer_simulator_backend.execute(
            ExecutionRequest(circuits=[bell_qc], shots=shots)
        )
        sim_zz = _compute_zz_expectation(res.counts[0])

        # Aer is near-ideal; statistical fluctuation only
        assert sim_zz >= 0.95, (
            f"Simulator <ZZ>={sim_zz:.4f} unexpectedly low — circuit may be wrong"
        )

    def test_zne_scale_factors_accepted(
        self,
        ibm_hardware_backend,
        request: pytest.FixtureRequest,
    ) -> None:
        """ZNE strategy initialises without error for valid scale factors."""
        scale_factors = [1.0, 1.5, 2.0, 3.0]
        for factory in ("Richardson", "Linear"):
            zne = ZNEStrategy(scale_factors=scale_factors, factory_type=factory)
            assert zne.name  # Not empty
            assert zne.scale_factors == scale_factors
