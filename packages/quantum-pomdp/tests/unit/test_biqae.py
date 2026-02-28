"""Tests for Bayesian Iterative Quantum Amplitude Estimation (BIQAE)."""

import numpy as np
import pytest

from quantum_pomdp.algorithms.biqae_estimator import (
    BIQAEConfig,
    BIQAEResult,
    BIQAEEstimator,
)


class TestBIQAEConfig:
    def test_defaults(self) -> None:
        config = BIQAEConfig()
        assert config.variant == "beta"
        assert config.max_iterations == 10
        assert config.confidence_level == 0.95
        assert config.prior_mean == 0.5
        assert config.prior_std == 0.25
        assert config.shots_per_iteration == 100
        assert config.k_base == 3
        assert config.seed == 42

    def test_custom_values(self) -> None:
        config = BIQAEConfig(
            variant="normal",
            max_iterations=20,
            confidence_level=0.99,
            prior_mean=0.3,
            prior_std=0.1,
            shots_per_iteration=200,
            k_base=2,
            seed=123,
        )
        assert config.variant == "normal"
        assert config.max_iterations == 20
        assert config.confidence_level == 0.99
        assert config.prior_mean == 0.3
        assert config.prior_std == 0.1
        assert config.shots_per_iteration == 200
        assert config.k_base == 2
        assert config.seed == 123


class TestBIQAEResult:
    def test_creation_and_field_access(self) -> None:
        result = BIQAEResult(
            amplitude_estimate=0.42,
            confidence_interval=(0.35, 0.49),
            num_iterations=5,
            total_shots=500,
            posterior_mean=0.42,
            posterior_std=0.03,
        )
        assert result.amplitude_estimate == 0.42
        assert result.confidence_interval == (0.35, 0.49)
        assert result.num_iterations == 5
        assert result.total_shots == 500
        assert result.posterior_mean == 0.42
        assert result.posterior_std == 0.03

    def test_confidence_interval_ordering(self) -> None:
        result = BIQAEResult(
            amplitude_estimate=0.5,
            confidence_interval=(0.4, 0.6),
            num_iterations=3,
            total_shots=300,
            posterior_mean=0.5,
            posterior_std=0.05,
        )
        lower, upper = result.confidence_interval
        assert lower <= result.amplitude_estimate <= upper


class TestBIQAEEstimator:
    def test_creation_with_defaults(self) -> None:
        estimator = BIQAEEstimator()
        assert estimator.config.variant == "beta"
        assert estimator.config.max_iterations == 10

    def test_creation_with_custom_config(self) -> None:
        config = BIQAEConfig(max_iterations=5, seed=99)
        estimator = BIQAEEstimator(config)
        assert estimator.config.max_iterations == 5
        assert estimator.config.seed == 99

    def test_select_grover_iterations_returns_non_negative(self) -> None:
        estimator = BIQAEEstimator()
        # Create a uniform posterior and theta grid
        grid_size = 200
        theta_grid = np.linspace(0.001, np.pi / 2 - 0.001, grid_size)
        posterior = np.ones(grid_size) / grid_size
        for iteration in range(5):
            k = estimator._select_grover_iterations(posterior, theta_grid, iteration)
            assert isinstance(k, (int, np.integer))
            assert k >= 0

    def test_select_grover_iterations_increases_with_iteration(self) -> None:
        """Base-3 schedule: k should generally increase with iteration index."""
        estimator = BIQAEEstimator()
        grid_size = 200
        theta_grid = np.linspace(0.001, np.pi / 2 - 0.001, grid_size)
        # Narrow posterior (low std) so safety check does not clip
        amplitude_grid = np.sin(theta_grid) ** 2
        posterior = np.exp(-0.5 * ((amplitude_grid - 0.5) / 0.01) ** 2)
        posterior /= posterior.sum()

        k_values = [
            estimator._select_grover_iterations(posterior, theta_grid, i)
            for i in range(4)
        ]
        # With narrow posterior, k should be non-decreasing
        for i in range(len(k_values) - 1):
            assert k_values[i] <= k_values[i + 1]

    def test_bayesian_update_produces_valid_posterior(self) -> None:
        estimator = BIQAEEstimator()
        grid_size = 200
        theta_grid = np.linspace(0.001, np.pi / 2 - 0.001, grid_size)
        prior = np.ones(grid_size) / grid_size

        posterior = estimator._bayesian_update(
            prior=prior,
            theta_grid=theta_grid,
            k=1,
            successes=50,
            total=100,
        )
        assert posterior.shape == (grid_size,)
        assert np.isclose(posterior.sum(), 1.0, atol=1e-6)
        assert np.all(posterior >= 0)

    def test_bayesian_update_concentrates_posterior(self) -> None:
        """After update with data, posterior should be more concentrated than prior."""
        estimator = BIQAEEstimator()
        grid_size = 200
        theta_grid = np.linspace(0.001, np.pi / 2 - 0.001, grid_size)
        amplitude_grid = np.sin(theta_grid) ** 2
        prior = np.ones(grid_size) / grid_size

        prior_std = float(np.sqrt(np.sum((amplitude_grid - amplitude_grid.mean()) ** 2 * prior)))

        posterior = estimator._bayesian_update(
            prior=prior,
            theta_grid=theta_grid,
            k=1,
            successes=70,
            total=100,
        )
        posterior_mean = float(np.sum(amplitude_grid * posterior))
        posterior_std = float(np.sqrt(np.sum((amplitude_grid - posterior_mean) ** 2 * posterior)))

        assert posterior_std < prior_std

    def test_bayesian_update_with_zero_successes(self) -> None:
        estimator = BIQAEEstimator()
        grid_size = 200
        theta_grid = np.linspace(0.001, np.pi / 2 - 0.001, grid_size)
        prior = np.ones(grid_size) / grid_size

        posterior = estimator._bayesian_update(
            prior=prior,
            theta_grid=theta_grid,
            k=1,
            successes=0,
            total=100,
        )
        assert np.isclose(posterior.sum(), 1.0, atol=1e-6)
        assert np.all(posterior >= 0)

    def test_estimate_classical_simulation(self) -> None:
        """Test estimate() in pure classical simulation mode (no executor)."""
        config = BIQAEConfig(max_iterations=5, shots_per_iteration=100, seed=42)
        estimator = BIQAEEstimator(config)

        result = estimator.estimate(oracle_circuit=None, grover_operator=None, executor=None)

        assert isinstance(result, BIQAEResult)
        assert 0.0 <= result.amplitude_estimate <= 1.0
        assert result.num_iterations > 0
        assert result.total_shots > 0
        assert result.posterior_std >= 0
        lower, upper = result.confidence_interval
        assert lower <= upper

    def test_estimate_with_mock_executor(self) -> None:
        """Test estimate() with a mock executor returning simulated counts."""
        try:
            from qiskit.circuit import QuantumCircuit
        except ImportError:
            pytest.skip("qiskit not installed")

        oracle_qc = QuantumCircuit(1)
        oracle_qc.h(0)

        grover_qc = QuantumCircuit(1)
        grover_qc.z(0)

        def mock_executor(circuit, shots=100):
            """Mock executor that simulates a known amplitude (~0.5)."""
            rng = np.random.default_rng(42)
            success_count = int(rng.binomial(shots, 0.5))
            return {"0": shots - success_count, "1": success_count}

        config = BIQAEConfig(max_iterations=5, shots_per_iteration=100, seed=42)
        estimator = BIQAEEstimator(config)

        result = estimator.estimate(
            oracle_circuit=oracle_qc,
            grover_operator=grover_qc,
            executor=mock_executor,
        )

        assert isinstance(result, BIQAEResult)
        assert 0.0 <= result.amplitude_estimate <= 1.0

    def test_estimate_reproducibility(self) -> None:
        """Two runs with the same seed should give the same result."""
        config1 = BIQAEConfig(max_iterations=3, shots_per_iteration=50, seed=42)
        config2 = BIQAEConfig(max_iterations=3, shots_per_iteration=50, seed=42)

        est1 = BIQAEEstimator(config1)
        est2 = BIQAEEstimator(config2)

        r1 = est1.estimate(oracle_circuit=None)
        r2 = est2.estimate(oracle_circuit=None)

        assert np.isclose(r1.amplitude_estimate, r2.amplitude_estimate)
        assert r1.num_iterations == r2.num_iterations
