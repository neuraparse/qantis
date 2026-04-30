"""Smoke tests for QuantumMCMCSampler + selector recommendation."""
from __future__ import annotations

import numpy as np
import pytest

from quantum_pomdp.quantum_circuits.quantum_mcmc import (
    QuantumMCMCSampler,
    recommend_sampler,
)


@pytest.fixture
def posterior_problem():
    # 3-state POMDP slice: deterministic transition, peaked likelihood.
    transition = np.eye(3)
    likelihood = np.array([0.05, 0.05, 0.9])
    prior = np.array([0.4, 0.3, 0.3])
    return transition, likelihood, prior


class TestClassicalPath:
    def test_sample_shape_and_support(self, posterior_problem) -> None:
        sampler = QuantumMCMCSampler(
            backend_family="classical", num_walk_steps=4, num_samples=64,
        )
        transition, likelihood, prior = posterior_problem
        samples = sampler.sample_posterior(transition, likelihood, prior, seed=7)
        assert samples.shape == (64,)
        # Mass should cluster around state index 2.
        assert (samples == 2).mean() > 0.5

    def test_zero_support_posterior_fallback(self, posterior_problem) -> None:
        sampler = QuantumMCMCSampler(backend_family="classical", num_samples=16)
        transition, _, prior = posterior_problem
        dead = np.zeros_like(prior)
        samples = sampler.sample_posterior(transition, dead, prior, seed=1)
        assert samples.shape == (16,)


class TestQuantumWalkSimulation:
    def test_trapped_ion_path_mixes_around_stationary(self, posterior_problem) -> None:
        sampler = QuantumMCMCSampler(
            backend_family="trapped_ion", num_walk_steps=6, num_samples=256,
        )
        transition, likelihood, prior = posterior_problem
        samples = sampler.sample_posterior(transition, likelihood, prior, seed=11)
        assert (samples == 2).mean() > 0.4


class TestSelector:
    def test_recommends_qmcmc_for_rare_events(self) -> None:
        assert recommend_sampler(0.02, "trapped_ion") == "qmcmc"

    def test_recommends_grover_fpaa_for_moderate(self) -> None:
        assert recommend_sampler(0.3, "trapped_ion") == "grover_fpaa"

    def test_recommends_grover_for_common(self) -> None:
        assert recommend_sampler(0.7, "trapped_ion") == "grover"
