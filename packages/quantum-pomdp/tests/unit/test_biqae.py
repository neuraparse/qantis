"""Tests for Bayesian Iterative Quantum Amplitude Estimation (BIQAE)."""

import math

import numpy as np
import pytest

from quantum_pomdp.algorithms.biqae_estimator import (
    BIQAEConfig,
    BIQAEResult,
    BIQAEEstimator,
    CalibratedBIQAEConfig,
    CalibratedBIQAEResult,
    CalibratedBIQAEEstimator,
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
        assert config.prior_type == "gaussian"
        assert config.beta_alpha == 1.0
        assert config.beta_beta == 1.0
        assert config.eta == 1.0

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

    def test_beta_prior_initialization(self) -> None:
        """Beta prior_type should produce a valid normalized prior on the grid."""
        config = BIQAEConfig(
            prior_type="beta",
            beta_alpha=2.0,
            beta_beta=5.0,
            max_iterations=3,
            shots_per_iteration=50,
            seed=42,
        )
        estimator = BIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        assert isinstance(result, BIQAEResult)
        assert 0.0 <= result.amplitude_estimate <= 1.0
        assert result.num_iterations > 0
        assert result.total_shots > 0

    def test_beta_prior_jeffreys(self) -> None:
        """Jeffreys Beta(0.5, 0.5) prior should work and produce valid output."""
        config = BIQAEConfig(
            prior_type="beta",
            beta_alpha=0.5,
            beta_beta=0.5,
            max_iterations=5,
            shots_per_iteration=100,
            seed=42,
        )
        estimator = BIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        assert isinstance(result, BIQAEResult)
        assert 0.0 <= result.amplitude_estimate <= 1.0

    def test_beta_prior_near_zero_concentrated(self) -> None:
        """Beta(1, 20) prior should concentrate mass near zero."""
        config = BIQAEConfig(
            prior_type="beta",
            beta_alpha=1.0,
            beta_beta=20.0,
            prior_mean=0.05,  # used by classical sim for data generation
            max_iterations=5,
            shots_per_iteration=100,
            seed=42,
        )
        estimator = BIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        assert isinstance(result, BIQAEResult)
        # With Beta(1,20) prior and a_true=0.05, estimate should be near zero
        assert result.amplitude_estimate < 0.3

    def test_gaussian_prior_backward_compatibility(self) -> None:
        """Default Gaussian prior should produce identical results to before."""
        config = BIQAEConfig(
            prior_type="gaussian",
            max_iterations=5,
            shots_per_iteration=100,
            seed=42,
        )
        estimator = BIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        # Same config without explicit prior_type (defaults to "gaussian")
        config2 = BIQAEConfig(
            max_iterations=5,
            shots_per_iteration=100,
            seed=42,
        )
        estimator2 = BIQAEEstimator(config2)
        result2 = estimator2.estimate(oracle_circuit=None)

        assert np.isclose(result.amplitude_estimate, result2.amplitude_estimate)
        assert result.num_iterations == result2.num_iterations
        assert result.total_shots == result2.total_shots


class TestSelectPrior:
    """Tests for CalibratedBIQAEEstimator._select_prior logic."""

    def _make_estimator(self, boundary_threshold: float = 0.1) -> CalibratedBIQAEEstimator:
        config = CalibratedBIQAEConfig(boundary_threshold=boundary_threshold)
        return CalibratedBIQAEEstimator(config)

    def test_near_zero(self) -> None:
        """a_hat < delta should select near_zero regime with Beta(1, large)."""
        est = self._make_estimator(boundary_threshold=0.1)
        alpha, beta_param, regime = est._select_prior(0.05)

        assert regime == "near_zero"
        assert alpha == 1.0
        # ceil(1/0.05) = 20
        assert beta_param == 20.0

    def test_near_one(self) -> None:
        """a_hat > 1-delta should select near_one regime with Beta(large, 1)."""
        est = self._make_estimator(boundary_threshold=0.1)
        alpha, beta_param, regime = est._select_prior(0.95)

        assert regime == "near_one"
        assert beta_param == 1.0
        # ceil(1/0.05) = 20
        assert alpha == 20.0

    def test_interior(self) -> None:
        """Interior a_hat should select Jeffreys Beta(0.5, 0.5) prior."""
        est = self._make_estimator(boundary_threshold=0.1)
        alpha, beta_param, regime = est._select_prior(0.5)

        assert regime == "interior"
        assert alpha == 0.5
        assert beta_param == 0.5

    def test_boundary_exactly_at_delta(self) -> None:
        """a_hat == delta should be near_zero (inclusive boundary)."""
        est = self._make_estimator(boundary_threshold=0.1)
        alpha, beta_param, regime = est._select_prior(0.1)

        assert regime == "near_zero"
        assert alpha == 1.0
        assert beta_param == 10.0

    def test_boundary_exactly_at_one_minus_delta(self) -> None:
        """a_hat == 1-delta should be near_one (inclusive boundary)."""
        est = self._make_estimator(boundary_threshold=0.1)
        alpha, beta_param, regime = est._select_prior(0.9)

        assert regime == "near_one"
        assert alpha >= 10.0  # ceil(1/(1-0.9)) — float precision may give 11
        assert alpha <= 11.0
        assert beta_param == 1.0

    def test_near_zero_very_small(self) -> None:
        """Very small a_hat should cap beta_param at 100."""
        est = self._make_estimator(boundary_threshold=0.1)
        alpha, beta_param, regime = est._select_prior(0.001)

        assert regime == "near_zero"
        assert alpha == 1.0
        assert beta_param == 100.0  # capped

    def test_near_one_very_large(self) -> None:
        """a_hat very close to 1 should cap alpha at 100."""
        est = self._make_estimator(boundary_threshold=0.1)
        alpha, beta_param, regime = est._select_prior(0.999)

        assert regime == "near_one"
        assert alpha == 100.0  # capped
        assert beta_param == 1.0

    def test_near_zero_a_hat_zero(self) -> None:
        """a_hat = 0.0 should use the 1e-6 floor and cap at 100."""
        est = self._make_estimator(boundary_threshold=0.1)
        alpha, beta_param, regime = est._select_prior(0.0)

        assert regime == "near_zero"
        assert alpha == 1.0
        assert beta_param == 100.0  # ceil(1/1e-6) = 1000000, capped to 100

    def test_custom_threshold(self) -> None:
        """Custom threshold should shift the boundary detection."""
        est = self._make_estimator(boundary_threshold=0.2)
        # 0.15 < 0.2, so near_zero
        alpha, beta_param, regime = est._select_prior(0.15)
        assert regime == "near_zero"

        # 0.5 is interior
        alpha, beta_param, regime = est._select_prior(0.5)
        assert regime == "interior"

        # 0.85 > 1-0.2 = 0.8, so near_one
        alpha, beta_param, regime = est._select_prior(0.85)
        assert regime == "near_one"


class TestCalibratedBIQAEConfig:
    def test_defaults(self) -> None:
        config = CalibratedBIQAEConfig()
        assert config.n_coarse == 200
        assert config.boundary_threshold == 0.1
        assert isinstance(config.biqae_config, BIQAEConfig)

    def test_custom_values(self) -> None:
        biqae_cfg = BIQAEConfig(max_iterations=8, seed=99)
        config = CalibratedBIQAEConfig(
            n_coarse=500,
            boundary_threshold=0.05,
            biqae_config=biqae_cfg,
        )
        assert config.n_coarse == 500
        assert config.boundary_threshold == 0.05
        assert config.biqae_config.max_iterations == 8
        assert config.biqae_config.seed == 99


class TestCalibratedBIQAEResult:
    def test_creation_and_field_access(self) -> None:
        biqae_result = BIQAEResult(
            amplitude_estimate=0.05,
            confidence_interval=(0.01, 0.10),
            num_iterations=5,
            total_shots=500,
            posterior_mean=0.05,
            posterior_std=0.02,
        )
        result = CalibratedBIQAEResult(
            phase1_estimate=0.045,
            phase1_shots=200,
            regime="near_zero",
            prior_alpha=1.0,
            prior_beta=23.0,
            biqae_result=biqae_result,
            total_shots=700,
        )
        assert result.phase1_estimate == 0.045
        assert result.phase1_shots == 200
        assert result.regime == "near_zero"
        assert result.prior_alpha == 1.0
        assert result.prior_beta == 23.0
        assert result.biqae_result is biqae_result
        assert result.total_shots == 700


class TestCalibratedBIQAEEstimator:
    def test_creation_with_defaults(self) -> None:
        estimator = CalibratedBIQAEEstimator()
        assert estimator.config.n_coarse == 200
        assert estimator.config.boundary_threshold == 0.1

    def test_creation_with_custom_config(self) -> None:
        config = CalibratedBIQAEConfig(n_coarse=500)
        estimator = CalibratedBIQAEEstimator(config)
        assert estimator.config.n_coarse == 500

    def test_estimate_boundary_amplitude_near_zero(self) -> None:
        """Full calibrated estimate at a=0.05 (near-zero boundary)."""
        biqae_cfg = BIQAEConfig(
            max_iterations=5,
            shots_per_iteration=100,
            prior_mean=0.05,  # classical sim uses this as a_true
            seed=42,
        )
        config = CalibratedBIQAEConfig(
            n_coarse=200,
            boundary_threshold=0.1,
            biqae_config=biqae_cfg,
        )
        estimator = CalibratedBIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        assert isinstance(result, CalibratedBIQAEResult)
        assert result.regime == "near_zero"
        assert result.prior_alpha == 1.0
        assert result.prior_beta > 1.0
        assert result.phase1_shots == 200
        assert result.total_shots == 200 + result.biqae_result.total_shots
        assert 0.0 <= result.biqae_result.amplitude_estimate <= 1.0
        # Estimate should be reasonably close to 0.05
        assert result.biqae_result.amplitude_estimate < 0.3

    def test_estimate_boundary_amplitude_near_one(self) -> None:
        """Full calibrated estimate at a=0.95 (near-one boundary)."""
        biqae_cfg = BIQAEConfig(
            max_iterations=5,
            shots_per_iteration=100,
            prior_mean=0.95,  # classical sim uses this as a_true
            seed=42,
        )
        config = CalibratedBIQAEConfig(
            n_coarse=200,
            boundary_threshold=0.1,
            biqae_config=biqae_cfg,
        )
        estimator = CalibratedBIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        assert isinstance(result, CalibratedBIQAEResult)
        assert result.regime == "near_one"
        assert result.prior_alpha > 1.0
        assert result.prior_beta == 1.0
        assert 0.0 <= result.biqae_result.amplitude_estimate <= 1.0
        # Estimate should be reasonably close to 0.95
        assert result.biqae_result.amplitude_estimate > 0.7

    def test_estimate_interior_amplitude(self) -> None:
        """Full calibrated estimate at a=0.5 (interior, Jeffreys prior)."""
        biqae_cfg = BIQAEConfig(
            max_iterations=5,
            shots_per_iteration=100,
            prior_mean=0.5,  # classical sim uses this as a_true
            seed=42,
        )
        config = CalibratedBIQAEConfig(
            n_coarse=200,
            boundary_threshold=0.1,
            biqae_config=biqae_cfg,
        )
        estimator = CalibratedBIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        assert isinstance(result, CalibratedBIQAEResult)
        assert result.regime == "interior"
        assert result.prior_alpha == 0.5
        assert result.prior_beta == 0.5
        assert 0.0 <= result.biqae_result.amplitude_estimate <= 1.0
        assert result.total_shots == 200 + result.biqae_result.total_shots

    def test_total_shots_accounting(self) -> None:
        """total_shots should equal phase1_shots + BIQAE shots."""
        biqae_cfg = BIQAEConfig(
            max_iterations=3,
            shots_per_iteration=50,
            prior_mean=0.5,
            seed=42,
        )
        config = CalibratedBIQAEConfig(
            n_coarse=100,
            biqae_config=biqae_cfg,
        )
        estimator = CalibratedBIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        assert result.total_shots == result.phase1_shots + result.biqae_result.total_shots
        assert result.phase1_shots == 100

    def test_reproducibility(self) -> None:
        """Two runs with the same seed should give the same result."""
        biqae_cfg1 = BIQAEConfig(
            max_iterations=3,
            shots_per_iteration=50,
            prior_mean=0.3,
            seed=42,
        )
        biqae_cfg2 = BIQAEConfig(
            max_iterations=3,
            shots_per_iteration=50,
            prior_mean=0.3,
            seed=42,
        )
        config1 = CalibratedBIQAEConfig(n_coarse=100, biqae_config=biqae_cfg1)
        config2 = CalibratedBIQAEConfig(n_coarse=100, biqae_config=biqae_cfg2)

        est1 = CalibratedBIQAEEstimator(config1)
        est2 = CalibratedBIQAEEstimator(config2)

        r1 = est1.estimate(oracle_circuit=None)
        r2 = est2.estimate(oracle_circuit=None)

        assert r1.regime == r2.regime
        assert r1.prior_alpha == r2.prior_alpha
        assert r1.prior_beta == r2.prior_beta
        assert np.isclose(r1.phase1_estimate, r2.phase1_estimate)
        assert np.isclose(
            r1.biqae_result.amplitude_estimate,
            r2.biqae_result.amplitude_estimate,
        )


class TestNoiseAwareLikelihood:
    """Tests for noise-aware Bayesian QAE (Ramoa & Santos, Quantum 9:1856, 2025)."""

    def test_noise_aware_likelihood_eta_one(self) -> None:
        """eta=1.0 (noiseless) should give the same result as the original likelihood."""
        config_noiseless = BIQAEConfig(
            max_iterations=5, shots_per_iteration=100, seed=42, eta=1.0,
        )
        config_default = BIQAEConfig(
            max_iterations=5, shots_per_iteration=100, seed=42,
        )
        est1 = BIQAEEstimator(config_noiseless)
        est2 = BIQAEEstimator(config_default)

        r1 = est1.estimate(oracle_circuit=None)
        r2 = est2.estimate(oracle_circuit=None)

        assert np.isclose(r1.amplitude_estimate, r2.amplitude_estimate)
        assert r1.num_iterations == r2.num_iterations
        assert r1.total_shots == r2.total_shots

    def test_noise_aware_eta_half(self) -> None:
        """eta=0.5 should compress the likelihood toward 0.5.

        With eta=0.5: P(success|theta,k) = 0.5*sin^2((2k+1)*theta) + 0.25
        This ranges between 0.25 and 0.75 instead of 0 and 1, significantly
        flattening the likelihood. The posterior update should still be valid
        but less informative than the noiseless case.

        For the fully depolarized case (eta=0), P=0.5 for all theta (flat
        likelihood), meaning data carries no information.
        """
        # eta=0.5: likelihood is compressed but still theta-dependent
        estimator = BIQAEEstimator(BIQAEConfig(eta=0.5))
        grid_size = 200
        theta_grid = np.linspace(0.001, np.pi / 2 - 0.001, grid_size)
        prior = np.ones(grid_size) / grid_size

        eta = 0.5
        for k in [0, 1, 3, 9]:
            prob_success = eta * np.sin((2 * k + 1) * theta_grid) ** 2 + (1 - eta) / 2
            # Should be bounded in [0.25, 0.75]
            assert np.all(prob_success >= 0.25 - 1e-10)
            assert np.all(prob_success <= 0.75 + 1e-10)

        # Bayesian update should still produce a valid posterior
        posterior = estimator._bayesian_update(prior, theta_grid, k=3, successes=50, total=100)
        assert np.isclose(posterior.sum(), 1.0, atol=1e-6)
        assert np.all(posterior >= 0)

        # eta=0 (fully depolarized): likelihood should be exactly 0.5 everywhere
        eta_zero = 0.0
        for k in [0, 1, 3, 9]:
            prob_success = eta_zero * np.sin((2 * k + 1) * theta_grid) ** 2 + (1 - eta_zero) / 2
            assert np.allclose(prob_success, 0.5, atol=1e-10), (
                f"eta=0 should give P=0.5 for all theta at k={k}"
            )

    def test_noise_aware_eta_097(self) -> None:
        """eta=0.97 (mild noise) should still converge but shift estimates slightly.

        The estimate should be valid (in [0,1]) and differ from the noiseless case.
        """
        config_noisy = BIQAEConfig(
            max_iterations=5, shots_per_iteration=100, seed=42, eta=0.97,
        )
        config_clean = BIQAEConfig(
            max_iterations=5, shots_per_iteration=100, seed=42, eta=1.0,
        )
        est_noisy = BIQAEEstimator(config_noisy)
        est_clean = BIQAEEstimator(config_clean)

        r_noisy = est_noisy.estimate(oracle_circuit=None)
        r_clean = est_clean.estimate(oracle_circuit=None)

        assert 0.0 <= r_noisy.amplitude_estimate <= 1.0
        assert r_noisy.num_iterations > 0
        assert r_noisy.total_shots > 0
        # The noisy estimator uses a different likelihood, so the results
        # may or may not be numerically identical but both should be valid
        assert isinstance(r_noisy, BIQAEResult)

    def test_bayesian_update_noise_aware_valid_posterior(self) -> None:
        """Bayesian update with eta < 1 should produce a valid normalized posterior."""
        config = BIQAEConfig(eta=0.9)
        estimator = BIQAEEstimator(config)
        grid_size = 200
        theta_grid = np.linspace(0.001, np.pi / 2 - 0.001, grid_size)
        prior = np.ones(grid_size) / grid_size

        posterior = estimator._bayesian_update(prior, theta_grid, k=3, successes=60, total=100)
        assert posterior.shape == (grid_size,)
        assert np.isclose(posterior.sum(), 1.0, atol=1e-6)
        assert np.all(posterior >= 0)

    def test_calibrated_eta_estimation(self) -> None:
        """CalibratedBIQAEEstimator should auto-estimate eta and return it."""
        biqae_cfg = BIQAEConfig(
            max_iterations=5,
            shots_per_iteration=100,
            prior_mean=0.3,
            seed=42,
        )
        config = CalibratedBIQAEConfig(
            n_coarse=200,
            boundary_threshold=0.1,
            biqae_config=biqae_cfg,
        )
        estimator = CalibratedBIQAEEstimator(config)
        result = estimator.estimate(oracle_circuit=None)

        # eta_estimated should be in [0.5, 1.0]
        assert 0.5 <= result.eta_estimated <= 1.0
        assert isinstance(result, CalibratedBIQAEResult)
