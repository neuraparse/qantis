"""Tests for quantum_common.benchmarking.reproducibility module — SeedManager."""

import numpy as np
import pytest

from quantum_common.benchmarking.reproducibility import SeedManager


class TestSeedManagerCreation:
    """Verify SeedManager construction and defaults."""

    def test_default_seed(self) -> None:
        sm = SeedManager()
        assert sm.base_seed == 42

    def test_custom_seed(self) -> None:
        sm = SeedManager(base_seed=123)
        assert sm.base_seed == 123

    def test_counter_starts_at_zero(self) -> None:
        sm = SeedManager()
        assert sm._counter == 0


class TestGetSeed:
    """Verify get_seed returns deterministic and distinct seeds."""

    def test_same_label_same_seed(self) -> None:
        sm = SeedManager(base_seed=42)
        seed1 = sm.get_seed("experiment_1")
        seed2 = sm.get_seed("experiment_1")
        assert seed1 == seed2

    def test_different_labels_different_seeds(self) -> None:
        sm = SeedManager(base_seed=42)
        seed_a = sm.get_seed("label_a")
        seed_b = sm.get_seed("label_b")
        assert seed_a != seed_b

    def test_seed_is_integer(self) -> None:
        sm = SeedManager(base_seed=42)
        seed = sm.get_seed("test")
        assert isinstance(seed, int)

    def test_empty_label_increments_counter(self) -> None:
        sm = SeedManager(base_seed=100)
        seed1 = sm.get_seed("")
        seed2 = sm.get_seed("")
        # Seeds should be different because counter increments
        assert seed1 != seed2
        assert seed1 == 101  # base_seed + 1
        assert seed2 == 102  # base_seed + 2

    def test_labeled_seed_does_not_increment_counter(self) -> None:
        sm = SeedManager(base_seed=42)
        sm.get_seed("some_label")
        assert sm._counter == 0  # Counter unchanged for labeled seeds

    def test_unlabeled_seed_increments_counter(self) -> None:
        sm = SeedManager(base_seed=42)
        sm.get_seed("")
        assert sm._counter == 1
        sm.get_seed("")
        assert sm._counter == 2

    def test_seed_positive(self) -> None:
        sm = SeedManager(base_seed=42)
        seed = sm.get_seed("any_label")
        assert seed >= 0


class TestGetRng:
    """Verify get_rng returns a numpy Generator."""

    def test_returns_numpy_generator(self) -> None:
        sm = SeedManager(base_seed=42)
        rng = sm.get_rng("test")
        assert isinstance(rng, np.random.Generator)

    def test_same_label_produces_same_sequence(self) -> None:
        sm1 = SeedManager(base_seed=42)
        sm2 = SeedManager(base_seed=42)

        rng1 = sm1.get_rng("experiment")
        rng2 = sm2.get_rng("experiment")

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]
        assert values1 == values2

    def test_different_labels_produce_different_sequences(self) -> None:
        sm = SeedManager(base_seed=42)
        rng_a = sm.get_rng("label_a")
        rng_b = sm.get_rng("label_b")

        values_a = [rng_a.random() for _ in range(10)]
        values_b = [rng_b.random() for _ in range(10)]
        assert values_a != values_b


class TestReset:
    """Verify reset resets the counter."""

    def test_reset_sets_counter_to_zero(self) -> None:
        sm = SeedManager(base_seed=42)
        sm.get_seed("")  # Increment counter
        sm.get_seed("")  # Increment counter
        assert sm._counter == 2

        sm.reset()
        assert sm._counter == 0

    def test_reset_allows_same_sequence(self) -> None:
        sm = SeedManager(base_seed=42)
        seeds_before = [sm.get_seed("") for _ in range(5)]

        sm.reset()
        seeds_after = [sm.get_seed("") for _ in range(5)]

        assert seeds_before == seeds_after


class TestReproducibility:
    """Verify that two SeedManagers with the same base_seed produce identical results."""

    def test_same_base_seed_same_labeled_seeds(self) -> None:
        sm1 = SeedManager(base_seed=999)
        sm2 = SeedManager(base_seed=999)

        for label in ["alpha", "beta", "gamma", "delta"]:
            assert sm1.get_seed(label) == sm2.get_seed(label)

    def test_same_base_seed_same_unlabeled_sequence(self) -> None:
        sm1 = SeedManager(base_seed=7)
        sm2 = SeedManager(base_seed=7)

        seeds1 = [sm1.get_seed("") for _ in range(10)]
        seeds2 = [sm2.get_seed("") for _ in range(10)]
        assert seeds1 == seeds2

    def test_different_base_seed_different_labeled_seeds(self) -> None:
        sm1 = SeedManager(base_seed=42)
        sm2 = SeedManager(base_seed=99)

        assert sm1.get_seed("same_label") != sm2.get_seed("same_label")

    def test_rng_reproducibility(self) -> None:
        """Two SeedManagers with same base_seed produce identical random streams."""
        sm1 = SeedManager(base_seed=42)
        sm2 = SeedManager(base_seed=42)

        rng1 = sm1.get_rng("monte_carlo")
        rng2 = sm2.get_rng("monte_carlo")

        arr1 = rng1.standard_normal(100)
        arr2 = rng2.standard_normal(100)
        np.testing.assert_array_equal(arr1, arr2)
