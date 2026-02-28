"""Tests for quantum_common.mitigation.noise module — NoiseProfile."""

import pytest

from quantum_common.mitigation.noise import NoiseProfile


class TestNoiseProfileCreation:
    """Verify NoiseProfile construction with default and custom values."""

    def test_default_values(self) -> None:
        profile = NoiseProfile(backend_name="test_backend")
        assert profile.backend_name == "test_backend"
        assert profile.single_qubit_error == 0.0
        assert profile.two_qubit_error == 0.0
        assert profile.readout_error == 0.0
        assert profile.t1_us == 0.0
        assert profile.t2_us == 0.0
        assert profile.gate_times_ns == {}

    def test_custom_error_rates(self) -> None:
        profile = NoiseProfile(
            backend_name="ibm_heron",
            single_qubit_error=1e-4,
            two_qubit_error=2.5e-3,
            readout_error=0.015,
            t1_us=300.0,
            t2_us=200.0,
        )
        assert profile.single_qubit_error == pytest.approx(1e-4)
        assert profile.two_qubit_error == pytest.approx(2.5e-3)
        assert profile.readout_error == pytest.approx(0.015)
        assert profile.t1_us == pytest.approx(300.0)
        assert profile.t2_us == pytest.approx(200.0)

    def test_custom_gate_times(self) -> None:
        gate_times = {"sx": 35.0, "cx": 300.0, "rz": 0.0}
        profile = NoiseProfile(backend_name="test", gate_times_ns=gate_times)
        assert profile.gate_times_ns["sx"] == pytest.approx(35.0)
        assert profile.gate_times_ns["cx"] == pytest.approx(300.0)

    def test_gate_times_default_independent(self) -> None:
        """Verify mutable default gate_times_ns is independent across instances."""
        p1 = NoiseProfile(backend_name="a")
        p2 = NoiseProfile(backend_name="b")
        p1.gate_times_ns["x"] = 50.0
        assert "x" not in p2.gate_times_ns


class TestEstimatedCircuitFidelity:
    """Verify estimated_circuit_fidelity property."""

    def test_perfect_fidelity_when_no_errors(self) -> None:
        profile = NoiseProfile(backend_name="ideal")
        assert profile.estimated_circuit_fidelity == pytest.approx(1.0)

    def test_fidelity_with_single_qubit_error_only(self) -> None:
        profile = NoiseProfile(backend_name="test", single_qubit_error=0.001)
        expected = (1 - 0.001) * (1 - 0.0)
        assert profile.estimated_circuit_fidelity == pytest.approx(expected)

    def test_fidelity_with_two_qubit_error_only(self) -> None:
        profile = NoiseProfile(backend_name="test", two_qubit_error=0.01)
        expected = (1 - 0.0) * (1 - 0.01)
        assert profile.estimated_circuit_fidelity == pytest.approx(expected)

    def test_fidelity_with_both_errors(self) -> None:
        profile = NoiseProfile(
            backend_name="noisy",
            single_qubit_error=0.001,
            two_qubit_error=0.01,
        )
        expected = (1 - 0.001) * (1 - 0.01)
        assert profile.estimated_circuit_fidelity == pytest.approx(expected)

    def test_fidelity_between_zero_and_one(self) -> None:
        profile = NoiseProfile(
            backend_name="noisy",
            single_qubit_error=0.1,
            two_qubit_error=0.2,
        )
        fidelity = profile.estimated_circuit_fidelity
        assert 0.0 < fidelity < 1.0

    def test_high_error_still_positive(self) -> None:
        profile = NoiseProfile(
            backend_name="very_noisy",
            single_qubit_error=0.5,
            two_qubit_error=0.5,
        )
        assert profile.estimated_circuit_fidelity == pytest.approx(0.25)


class TestSuggestMitigation:
    """Verify suggest_mitigation returns appropriate strategies based on noise levels."""

    def test_no_mitigation_needed_for_low_noise(self) -> None:
        profile = NoiseProfile(
            backend_name="quiet",
            single_qubit_error=1e-5,
            two_qubit_error=1e-4,
            readout_error=0.001,
        )
        suggestions = profile.suggest_mitigation()
        assert suggestions == ["none_needed"]

    def test_readout_mitigation_suggested(self) -> None:
        profile = NoiseProfile(backend_name="test", readout_error=0.02)
        suggestions = profile.suggest_mitigation()
        assert "readout_mitigation" in suggestions

    def test_zne_suggested_for_moderate_two_qubit_error(self) -> None:
        profile = NoiseProfile(backend_name="test", two_qubit_error=0.006)
        suggestions = profile.suggest_mitigation()
        assert "zne" in suggestions

    def test_pec_suggested_for_high_two_qubit_error(self) -> None:
        profile = NoiseProfile(backend_name="test", two_qubit_error=0.02)
        suggestions = profile.suggest_mitigation()
        assert "pec" in suggestions
        assert "zne" in suggestions  # ZNE threshold is lower, so also included

    def test_high_noise_suggests_more_strategies_than_low(self) -> None:
        low_noise = NoiseProfile(
            backend_name="quiet",
            readout_error=0.005,
            two_qubit_error=0.001,
        )
        high_noise = NoiseProfile(
            backend_name="loud",
            readout_error=0.05,
            two_qubit_error=0.02,
        )
        low_suggestions = low_noise.suggest_mitigation()
        high_suggestions = high_noise.suggest_mitigation()
        assert len(high_suggestions) > len(low_suggestions)

    def test_all_mitigations_suggested_for_very_noisy_backend(self) -> None:
        profile = NoiseProfile(
            backend_name="very_noisy",
            readout_error=0.1,
            two_qubit_error=0.05,
        )
        suggestions = profile.suggest_mitigation()
        assert "readout_mitigation" in suggestions
        assert "zne" in suggestions
        assert "pec" in suggestions
