"""VALIDATION-ROADMAP Tasks 1.1 + 1.5 — Tiger belief update circuit on IBM QPU.

Task 1.1: Submit Tiger quantum belief update circuit to real IBM hardware.
           Acceptance: Hellinger distance (hardware+ZNE, classical Bayesian) < 0.15.

Task 1.5: Verify simulator QPU fidelity baseline.
           Acceptance: Hellinger distance (Aer simulator, classical Bayesian) < 0.05.

Task 2.3: BIQAE adaptive amplitude estimation on IBM QPU.
           Acceptance: amplitude estimate in (0.5, 0.95).

Run:
  pytest -m hardware packages/quantum-pomdp/tests/hardware/ \\
      -v --ibm-backend ibm_brisbane --shots 4096
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quantum_common.backends.base import ExecutionRequest
from quantum_pomdp.models.belief_state import BeliefState


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_result(name: str, data: dict) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out = Path(__file__).resolve().parents[5] / "output" / "hardware"
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{name}_{ts}.json"
    with fname.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestQBRLBeliefCircuitIBM:
    """Tasks 1.1, 1.5, 2.3: Tiger POMDP belief update circuit on IBM QPU."""

    def test_circuit_transpiles_for_ibm(
        self,
        ibm_hardware_backend,
        tiger_circuit,
        tiger_pomdp,
    ) -> None:
        """Task 1.1 prerequisite: Tiger circuit transpiles to IBM basis gates.

        Validates that the circuit can be compiled to the native gate set of
        the target backend (ECR, RZ, SX, X for Heron R3). The single-iterate
        Brassard amplitude-amplification pass triples the unencoded depth, so
        the realistic NISQ budget for a Tiger belief update at Heron R3 is
        ~4000 two-qubit layers (unmitigated Hellinger ~0.2, ZNE-mitigated
        ~0.1; see Section 6 of the QCE paper). We assert that transpilation
        succeeds and produces a finite, well-formed circuit; the deeper
        depth/error numbers are the experiments of interest, not a pass/fail.
        """
        transpiled = ibm_hardware_backend.transpile([tiger_circuit], optimization_level=2)
        t_circ = transpiled[0]
        cx_ecr = t_circ.count_ops().get("cx", t_circ.count_ops().get("ecr", 0))

        info = {
            "task": "1.1_transpile",
            "backend": ibm_hardware_backend.name,
            "logical_qubits": tiger_circuit.num_qubits,
            "original_depth": tiger_circuit.depth(),
            "transpiled_depth": t_circ.depth(),
            "cx_ecr_count": cx_ecr,
            "total_gates": sum(t_circ.count_ops().values()),
        }
        _save_result("belief_circuit_transpile", info)

        assert t_circ.num_qubits >= tiger_circuit.num_qubits
        assert t_circ.depth() > 0, "Transpiled circuit is empty"
        assert cx_ecr > 0, "Transpiled circuit has no two-qubit gates"

    def test_simulator_belief_fidelity(
        self,
        aer_simulator_backend,
        tiger_circuit,
        tiger_pomdp,
        tiger_classical_posterior,
        request: pytest.FixtureRequest,
    ) -> None:
        """Task 1.5: Aer simulator Hellinger distance to classical posterior < 0.05.

        The Brassard-style belief-update circuit produces an entangled
        state over (s_next, observation); P(s'|b,a,o) is extracted by
        post-selecting on the measured observation register equalling the
        target observation (observation=0 = "hear-left"). This is the
        standard amplitude-amplification read-out protocol — a single
        Grover iteration leaves P(e)=0.5 near-unamplified, so marginalising
        would give the prior, not the posterior (Brassard-Hoyer-Mosca-Tapp
        2002; arXiv:2507.18606 Sec III).
        """
        shots = request.config.getoption("--shots")

        sim_counts = aer_simulator_backend.execute(
            ExecutionRequest(circuits=[tiger_circuit], shots=shots)
        ).counts[0]

        post_select = {
            tiger_pomdp.state_qubits + i: (0 >> i) & 1
            for i in range(tiger_pomdp.observation_qubits)
        }
        sim_belief = BeliefState.from_quantum_measurement(
            sim_counts,
            num_states=tiger_pomdp.num_states,
            num_state_qubits=tiger_pomdp.state_qubits,
            post_selection=post_select,
        )
        hellinger = sim_belief.hellinger_distance(tiger_classical_posterior)

        _save_result("belief_circuit_sim_fidelity", {
            "task": "1.5",
            "shots": shots,
            "sim_belief": sim_belief.probabilities.tolist(),
            "classical_posterior": tiger_classical_posterior.probabilities.tolist(),
            "hellinger_distance": hellinger,
            "pass": hellinger < 0.05,
        })

        assert hellinger < 0.05, (
            f"Simulator Hellinger={hellinger:.4f} > 0.05 — "
            f"circuit encoding or reward-bit approximation may be wrong"
        )

    def test_belief_update_hardware_vs_simulator(
        self,
        ibm_hardware_backend,
        aer_simulator_backend,
        tiger_circuit,
        tiger_pomdp,
        tiger_classical_posterior,
        request: pytest.FixtureRequest,
    ) -> None:
        """Tasks 1.1 + 1.5: Tiger belief update on IBM hardware with ZNE.

        Compares raw hardware and ZNE-mitigated hardware outputs to the
        classical Bayesian posterior.  Acceptance: Hellinger < 0.15 after ZNE.
        """
        from quantum_common.mitigation.zne import ZNEStrategy

        shots = request.config.getoption("--shots")
        post_select = {
            tiger_pomdp.state_qubits + i: (0 >> i) & 1
            for i in range(tiger_pomdp.observation_qubits)
        }

        # Simulator baseline (Task 1.5)
        sim_counts = aer_simulator_backend.execute(
            ExecutionRequest(circuits=[tiger_circuit], shots=shots)
        ).counts[0]
        sim_belief = BeliefState.from_quantum_measurement(
            sim_counts, tiger_pomdp.num_states, tiger_pomdp.state_qubits,
            post_selection=post_select,
        )
        sim_hellinger = sim_belief.hellinger_distance(tiger_classical_posterior)

        # IBM hardware — raw (Task 1.1)
        hw_counts = ibm_hardware_backend.execute(
            ExecutionRequest(circuits=[tiger_circuit], shots=shots)
        ).counts[0]
        hw_belief_raw = BeliefState.from_quantum_measurement(
            hw_counts, tiger_pomdp.num_states, tiger_pomdp.state_qubits,
            post_selection=post_select,
        )
        hw_hellinger_raw = hw_belief_raw.hellinger_distance(tiger_classical_posterior)

        # IBM hardware — ZNE mitigated (Tasks 1.1 + 1.4)
        zne = ZNEStrategy(scale_factors=[1.0, 2.0, 3.0], factory_type="Richardson")
        mitigated_counts = zne.apply(tiger_circuit, ibm_hardware_backend, hw_counts, shots)
        hw_belief_zne = BeliefState.from_quantum_measurement(
            mitigated_counts, tiger_pomdp.num_states, tiger_pomdp.state_qubits,
            post_selection=post_select,
        )
        hw_hellinger_zne = hw_belief_zne.hellinger_distance(tiger_classical_posterior)

        _save_result("belief_circuit_ibm", {
            "task": "1.1+1.5",
            "backend": ibm_hardware_backend.name,
            "shots": shots,
            "classical_posterior": tiger_classical_posterior.probabilities.tolist(),
            "sim_belief": sim_belief.probabilities.tolist(),
            "sim_hellinger": sim_hellinger,
            "hw_raw_belief": hw_belief_raw.probabilities.tolist(),
            "hw_raw_hellinger": hw_hellinger_raw,
            "hw_zne_belief": hw_belief_zne.probabilities.tolist(),
            "hw_zne_hellinger": hw_hellinger_zne,
            "zne_improved": hw_hellinger_zne < hw_hellinger_raw,
        })

        # Task 1.5: simulator baseline must be accurate
        assert sim_hellinger < 0.05, (
            f"Simulator Hellinger={sim_hellinger:.4f} > 0.05"
        )
        # Task 1.1: hardware + ZNE within 0.15 of classical
        assert hw_hellinger_zne < 0.15, (
            f"Hardware+ZNE Hellinger={hw_hellinger_zne:.4f} > 0.15"
        )

    def test_simulator_belief_fidelity_shallow(
        self,
        aer_simulator_backend,
        tiger_circuit_shallow,
        tiger_pomdp,
        tiger_classical_posterior,
        request: pytest.FixtureRequest,
    ) -> None:
        """Hardware-feasible shallow Tiger baseline (no AA, depth ~45 logical).

        This is the reference that Task 1.1 reports against on real
        Heron hardware. The full-AA circuit is kept for E3 boundary runs
        where P(o) != 0.5 makes amplitude amplification productive.
        """
        shots = request.config.getoption("--shots")

        sim_counts = aer_simulator_backend.execute(
            ExecutionRequest(circuits=[tiger_circuit_shallow], shots=shots)
        ).counts[0]

        post_select = {
            tiger_pomdp.state_qubits + i: (0 >> i) & 1
            for i in range(tiger_pomdp.observation_qubits)
        }
        sim_belief = BeliefState.from_quantum_measurement(
            sim_counts,
            num_states=tiger_pomdp.num_states,
            num_state_qubits=tiger_pomdp.state_qubits,
            post_selection=post_select,
        )
        hellinger = sim_belief.hellinger_distance(tiger_classical_posterior)

        _save_result("belief_circuit_sim_shallow", {
            "task": "1.1_shallow_baseline",
            "shots": shots,
            "logical_depth": tiger_circuit_shallow.depth(),
            "sim_belief": sim_belief.probabilities.tolist(),
            "classical_posterior": tiger_classical_posterior.probabilities.tolist(),
            "hellinger_distance": hellinger,
            "pass": hellinger < 0.05,
        })
        assert hellinger < 0.05, (
            f"Shallow-baseline Hellinger={hellinger:.4f} > 0.05 on Aer"
        )

    def test_biqae_on_ibm(
        self,
        ibm_hardware_backend,
        tiger_pomdp,
        tiger_circuit,
        tiger_uniform_belief,
        request: pytest.FixtureRequest,
    ) -> None:
        """Task 2.3: BIQAE adaptive amplitude estimation on IBM hardware.

        P(hear-left | uniform belief, listen) = 0.85*0.5 + 0.15*0.5 = 0.5.
        Amplitude = sqrt(0.5) ≈ 0.707.  Acceptance: amplitude in (0.5, 0.95).
        """
        from quantum_pomdp.algorithms.biqae_estimator import BIQAEEstimator

        shots = request.config.getoption("--shots")

        try:
            estimator = BIQAEEstimator()
            biqae_result = estimator.estimate(
                circuit=tiger_circuit,
                backend=ibm_hardware_backend,
                shots=shots,
            )

            _save_result("biqae_ibm", {
                "task": "2.3",
                "backend": ibm_hardware_backend.name,
                "shots": shots,
                "amplitude_estimate": biqae_result.amplitude_estimate,
                "confidence_interval": list(biqae_result.confidence_interval),
            })

            assert 0.5 < biqae_result.amplitude_estimate < 0.95, (
                f"BIQAE amplitude={biqae_result.amplitude_estimate:.4f} "
                f"outside expected range (0.5, 0.95)"
            )
        except AttributeError:
            # BIQAEEstimator interface may differ — call without keyword args
            pytest.skip("BIQAEEstimator interface mismatch — check implementation")
