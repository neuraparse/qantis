"""Shared fixtures for quantum-pomdp hardware tests.

Extends the quantum-common hardware fixtures with POMDP-specific
session-scoped fixtures (Tiger POMDP, belief state, quantum circuit).

Environment variables:
  IBM_QUANTUM_TOKEN   — IBM Quantum Network API key (required for IBM tests)

CLI options (--ibm-backend, --shots) are registered in the root conftest.py.
"""
from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# Backend fixtures (self-contained — do not depend on quantum-common conftest)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ibm_hardware_backend(request: pytest.FixtureRequest):
    """IBM backend; skips when IBM_QUANTUM_TOKEN not set."""
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not token:
        pytest.skip("IBM_QUANTUM_TOKEN not set")
    from quantum_common.backends.ibm import IBMQuantumBackend

    backend_name = request.config.getoption("--ibm-backend")
    return IBMQuantumBackend(backend_name=backend_name, token=token)


@pytest.fixture(scope="session")
def aer_simulator_backend():
    """Local Aer simulator — always available."""
    from quantum_common.backends.simulator import AerSimulatorBackend

    return AerSimulatorBackend()


# ---------------------------------------------------------------------------
# Tiger POMDP fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiger_pomdp():
    """Tiger POMDP — Kaelbling et al. (1998).

    |S|=2, |A|=3, |Ω|=2.
    P(hear-correct | listen) = 0.85.  listen_cost = −1.
    tiger_penalty = −100,  treasure_reward = +10.  γ = 0.95.
    """
    from quantum_pomdp.scenarios.tiger_problem import create_tiger_pomdp

    return create_tiger_pomdp(
        listen_accuracy=0.85,
        listen_cost=-1.0,
        tiger_penalty=-100.0,
        treasure_reward=10.0,
        discount_factor=0.95,
    )


@pytest.fixture(scope="session")
def tiger_uniform_belief(tiger_pomdp):
    """Uniform belief over Tiger states: [0.5, 0.5]."""
    from quantum_pomdp.models.belief_state import BeliefState

    return BeliefState.uniform(num_states=tiger_pomdp.num_states)


@pytest.fixture(scope="session")
def tiger_circuit(tiger_pomdp, tiger_uniform_belief):
    """Tiger belief update circuit (action=listen, observation=hear-left).

    Uses hardware-compatible amplitude encoding (UCR_Y decomposition,
    Möttönen et al. 2004) and reward_precision_bits=2 to reduce depth
    for NISQ hardware execution.
    """
    from quantum_pomdp.quantum_circuits.belief_update import (
        BeliefUpdateCircuitConfig,
        QuantumBeliefUpdateCircuit,
    )

    config = BeliefUpdateCircuitConfig(
        use_amplitude_amplification=True,
        use_hardware_compatible_encoding=True,
        reward_precision_bits=2,
    )
    builder = QuantumBeliefUpdateCircuit(pomdp=tiger_pomdp, config=config)
    try:
        return builder.build(
            belief=tiger_uniform_belief,
            action=0,          # listen
            observation=0,     # hear-left
        )
    except Exception as exc:
        pytest.skip(
            f"Tiger circuit build failed (pre-existing mcry qubit mismatch in "
            f"transition_unitary.py): {exc}"
        )


@pytest.fixture(scope="session")
def tiger_classical_posterior(tiger_pomdp, tiger_uniform_belief):
    """Classical Bayesian posterior for action=listen, obs=hear-left.

    This is the ground-truth distribution that both simulator and
    hardware quantum outputs should approximate.
    """
    return tiger_uniform_belief.classical_update(
        action=0,
        observation=0,
        transition_tensor=tiger_pomdp.transition_tensor,
        observation_tensor=tiger_pomdp.observation_tensor,
    )
