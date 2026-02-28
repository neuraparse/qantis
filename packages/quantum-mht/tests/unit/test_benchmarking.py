"""Tests for quantum_mht.benchmarking module.

Validates MOTMetrics, MHTBenchmarkRunner, and ScenarioSuite for
MHT evaluation using CLEAR MOT metrics (Bernardin & Stiefelhagen 2008).
"""

import numpy as np
import pytest

from quantum_mht.benchmarking.benchmark_runner import MOTMetrics, MHTBenchmarkRunner
from quantum_mht.benchmarking.scenario_suite import ScenarioSuite


class TestMOTMetrics:
    """Test MOTMetrics data class."""

    def test_creation_with_defaults(self) -> None:
        """MOTMetrics should initialize with default values."""
        m = MOTMetrics()
        assert m.mota == 0.0
        assert m.motp == 0.0
        assert m.num_switches == 0
        assert m.num_false_positives == 0
        assert m.num_misses == 0
        assert m.num_objects == 0
        assert m.num_detections == 0

    def test_creation_with_custom_values(self) -> None:
        """MOTMetrics should accept custom values."""
        m = MOTMetrics(
            mota=0.85,
            motp=1.5,
            num_switches=3,
            num_false_positives=10,
            num_misses=5,
            num_objects=100,
            num_detections=95,
        )
        assert m.mota == 0.85
        assert m.motp == 1.5
        assert m.num_switches == 3
        assert m.num_false_positives == 10
        assert m.num_misses == 5
        assert m.num_objects == 100
        assert m.num_detections == 95


class TestMOTMetricsPrecision:
    """Test MOTMetrics.precision property."""

    def test_perfect_precision(self) -> None:
        """With no false positives, precision should be 1.0."""
        m = MOTMetrics(num_detections=50, num_false_positives=0)
        assert m.precision == 1.0

    def test_partial_precision(self) -> None:
        """Precision = (detections - FP) / detections."""
        m = MOTMetrics(num_detections=100, num_false_positives=20)
        assert np.isclose(m.precision, 0.8)

    def test_zero_detections(self) -> None:
        """Precision with 0 detections should be 0.0."""
        m = MOTMetrics(num_detections=0, num_false_positives=0)
        assert m.precision == 0.0

    def test_all_false_positives(self) -> None:
        """If all detections are FP, precision should be 0.0."""
        m = MOTMetrics(num_detections=10, num_false_positives=10)
        assert np.isclose(m.precision, 0.0)


class TestMOTMetricsRecall:
    """Test MOTMetrics.recall property."""

    def test_perfect_recall(self) -> None:
        """With no misses, recall should be 1.0."""
        m = MOTMetrics(num_objects=50, num_misses=0)
        assert m.recall == 1.0

    def test_partial_recall(self) -> None:
        """Recall = (objects - misses) / objects."""
        m = MOTMetrics(num_objects=100, num_misses=30)
        assert np.isclose(m.recall, 0.7)

    def test_zero_objects(self) -> None:
        """Recall with 0 objects should be 0.0."""
        m = MOTMetrics(num_objects=0, num_misses=0)
        assert m.recall == 0.0

    def test_all_missed(self) -> None:
        """If all objects are missed, recall should be 0.0."""
        m = MOTMetrics(num_objects=10, num_misses=10)
        assert np.isclose(m.recall, 0.0)


class TestMHTBenchmarkRunner:
    """Test MHTBenchmarkRunner creation and methods."""

    def test_creation(self) -> None:
        """MHTBenchmarkRunner should initialize with empty results list."""
        runner = MHTBenchmarkRunner()
        assert isinstance(runner.results, list)
        assert len(runner.results) == 0

    def test_compute_mota_perfect_tracking(self) -> None:
        """Perfect tracking (predictions match ground truth) should give MOTA = 1.0."""
        runner = MHTBenchmarkRunner()
        # Ground truth and predictions match exactly in count
        ground_truth = [[1, 2, 3], [4, 5, 6]]
        predictions = [[1, 2, 3], [4, 5, 6]]
        mota = runner.compute_mota(ground_truth, predictions)
        assert np.isclose(mota, 1.0)

    def test_compute_mota_no_predictions(self) -> None:
        """No predictions means all missed: MOTA = 1 - FN/GT."""
        runner = MHTBenchmarkRunner()
        ground_truth = [[1, 2, 3], [4, 5]]
        predictions = [[], []]
        mota = runner.compute_mota(ground_truth, predictions)
        # total_gt = 5, fn = 5, fp = 0 => MOTA = 1 - 5/5 = 0.0
        assert np.isclose(mota, 0.0)

    def test_compute_mota_extra_predictions(self) -> None:
        """Extra predictions should reduce MOTA via false positives."""
        runner = MHTBenchmarkRunner()
        ground_truth = [[1, 2], [3]]
        predictions = [[1, 2, 3, 4], [3, 4, 5]]
        mota = runner.compute_mota(ground_truth, predictions)
        # total_gt = 3, fp = 2 + 2 = 4, fn = 0 => MOTA = 1 - 4/3 < 0
        assert mota < 1.0

    def test_compute_mota_empty_ground_truth(self) -> None:
        """Empty ground truth should return 0.0."""
        runner = MHTBenchmarkRunner()
        mota = runner.compute_mota([], [])
        assert mota == 0.0

    def test_compute_mota_returns_float(self) -> None:
        """compute_mota should return a float."""
        runner = MHTBenchmarkRunner()
        ground_truth = [[1, 2]]
        predictions = [[1, 2]]
        mota = runner.compute_mota(ground_truth, predictions)
        assert isinstance(mota, float)

    def test_compute_mota_in_expected_range(self) -> None:
        """MOTA should be <= 1.0 for any input."""
        runner = MHTBenchmarkRunner()
        ground_truth = [[1, 2, 3]]
        predictions = [[1, 2]]
        mota = runner.compute_mota(ground_truth, predictions)
        assert mota <= 1.0

    def test_compute_motp_simple(self) -> None:
        """MOTP should be the mean of provided distances."""
        runner = MHTBenchmarkRunner()
        distances = [1.0, 2.0, 3.0, 4.0]
        motp = runner.compute_motp(distances)
        assert np.isclose(motp, 2.5)

    def test_compute_motp_empty(self) -> None:
        """MOTP with empty distances should return 0.0."""
        runner = MHTBenchmarkRunner()
        motp = runner.compute_motp([])
        assert motp == 0.0

    def test_compute_motp_single_distance(self) -> None:
        """MOTP with one distance should return that distance."""
        runner = MHTBenchmarkRunner()
        motp = runner.compute_motp([3.14])
        assert np.isclose(motp, 3.14)


class TestScenarioSuite:
    """Test ScenarioSuite predefined benchmarks."""

    def test_creation_defaults(self) -> None:
        """ScenarioSuite should have sensible defaults."""
        suite = ScenarioSuite()
        assert suite.name == "default"
        assert suite.num_steps == 50

    def test_custom_creation(self) -> None:
        """ScenarioSuite should accept custom name and num_steps."""
        suite = ScenarioSuite(name="custom", num_steps=100)
        assert suite.name == "custom"
        assert suite.num_steps == 100

    def test_get_scenarios_returns_dict(self) -> None:
        """get_scenarios should return a dict of scenario_name -> SimulationWorld."""
        suite = ScenarioSuite()
        scenarios = suite.get_scenarios()
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

    def test_get_scenarios_contains_expected_keys(self) -> None:
        """get_scenarios should contain crossing, clutter, and swarm scenarios."""
        suite = ScenarioSuite()
        scenarios = suite.get_scenarios()
        assert "crossing_5" in scenarios
        assert "crossing_10" in scenarios
        assert "dense_clutter" in scenarios
        assert "swarm_3x5" in scenarios

    def test_scenarios_are_simulation_worlds(self) -> None:
        """Each scenario should be a SimulationWorld instance."""
        from quantum_mht.simulation.world import SimulationWorld

        suite = ScenarioSuite()
        scenarios = suite.get_scenarios()
        for name, world in scenarios.items():
            assert isinstance(world, SimulationWorld), f"{name} is not a SimulationWorld"

    def test_scenarios_have_targets(self) -> None:
        """Each scenario should have at least one target."""
        suite = ScenarioSuite()
        scenarios = suite.get_scenarios()
        for name, world in scenarios.items():
            assert len(world.targets) > 0, f"{name} has no targets"

    def test_scenarios_runnable(self) -> None:
        """Each scenario should be able to run a simulation step."""
        rng = np.random.default_rng(42)
        suite = ScenarioSuite()
        scenarios = suite.get_scenarios()
        for name, world in scenarios.items():
            scan = world.step(rng)
            assert hasattr(scan, "measurements"), f"{name} step did not produce scan"
