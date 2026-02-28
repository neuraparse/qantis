"""Root conftest: shared fixtures available to all packages."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Hardware CLI options — registered once here to avoid conftest plugin collision
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    """Hardware test CLI options (used by packages/*/tests/hardware/)."""
    parser.addoption(
        "--ibm-backend",
        default="ibm_brisbane",
        help="IBM Quantum backend name for hardware tests (default: ibm_brisbane)",
    )
    parser.addoption(
        "--shots",
        type=int,
        default=4096,
        help="Shot count per circuit for IBM hardware tests (default: 4096)",
    )
    parser.addoption(
        "--num-reads",
        type=int,
        default=1000,
        help="D-Wave num_reads for hardware tests (default: 1000)",
    )


@pytest.fixture
def deterministic_seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(deterministic_seed: int) -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(deterministic_seed)


# --- quantum-common fixtures ---

@pytest.fixture
def sample_config():
    """Minimal experiment configuration for testing."""
    from quantum_common.config.schema import BackendConfig, ExperimentConfig
    return ExperimentConfig(
        name="test_experiment",
        backend=BackendConfig(backend_type="local_aer", use_simulator=True),
    )


# --- quantum-pomdp fixtures ---

@pytest.fixture
def tiger_pomdp():
    """Classic Tiger POMDP: 2 states, 3 actions, 2 observations."""
    from quantum_pomdp.models.pomdp import POMDPModel
    listen_acc = 0.85
    T = np.zeros((3, 2, 2))
    T[0] = np.eye(2)
    T[1] = np.array([[0.5, 0.5], [0.5, 0.5]])
    T[2] = np.array([[0.5, 0.5], [0.5, 0.5]])
    O = np.zeros((3, 2, 2))
    O[0] = np.array([[listen_acc, 1 - listen_acc], [1 - listen_acc, listen_acc]])
    O[1] = np.array([[0.5, 0.5], [0.5, 0.5]])
    O[2] = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([[-1.0, -100.0, 10.0], [-1.0, 10.0, -100.0]])
    return POMDPModel(
        num_states=2, num_actions=3, num_observations=2,
        transition_tensor=T, observation_tensor=O, reward_matrix=R,
        discount_factor=0.95, initial_belief=np.array([0.5, 0.5]),
    )


@pytest.fixture
def uniform_belief():
    """Uniform belief over 2 states."""
    from quantum_pomdp.models.belief_state import BeliefState
    return BeliefState.uniform(2)
