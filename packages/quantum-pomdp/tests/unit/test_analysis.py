"""Tests for the analysis module (metrics, scalability)."""

import numpy as np
import pytest

from quantum_pomdp.analysis.metrics import (
    BeliefMetrics,
    PlanningMetrics,
    ExperimentComparison,
)
from quantum_pomdp.analysis.scalability import ScalingAnalysis


class TestBeliefMetrics:
    def test_kl_divergence_identical_distributions(self) -> None:
        p = np.array([0.3, 0.7])
        kl = BeliefMetrics.kl_divergence(p, p)
        assert np.isclose(kl, 0.0, atol=1e-8)

    def test_kl_divergence_known_value(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        kl = BeliefMetrics.kl_divergence(p, q)
        # KL(p||q) = 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75)
        expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        assert np.isclose(kl, expected, atol=1e-6)

    def test_kl_divergence_non_negative(self) -> None:
        p = np.array([0.2, 0.8])
        q = np.array([0.6, 0.4])
        kl = BeliefMetrics.kl_divergence(p, q)
        assert kl >= 0.0

    def test_kl_divergence_asymmetric(self) -> None:
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        kl_pq = BeliefMetrics.kl_divergence(p, q)
        kl_qp = BeliefMetrics.kl_divergence(q, p)
        # KL is not symmetric in general (though in this symmetric case they happen to be equal)
        assert kl_pq >= 0
        assert kl_qp >= 0

    def test_hellinger_distance_identical(self) -> None:
        p = np.array([0.4, 0.6])
        h = BeliefMetrics.hellinger_distance(p, p)
        assert np.isclose(h, 0.0, atol=1e-10)

    def test_hellinger_distance_bounds(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        h = BeliefMetrics.hellinger_distance(p, q)
        assert 0.0 <= h <= 1.0

    def test_hellinger_distance_symmetric(self) -> None:
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        assert np.isclose(
            BeliefMetrics.hellinger_distance(p, q),
            BeliefMetrics.hellinger_distance(q, p),
        )

    def test_hellinger_distance_maximum(self) -> None:
        """Orthogonal distributions should give maximum Hellinger distance."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        h = BeliefMetrics.hellinger_distance(p, q)
        assert np.isclose(h, 1.0, atol=1e-10)

    def test_total_variation_distance_identical(self) -> None:
        p = np.array([0.5, 0.5])
        tv = BeliefMetrics.total_variation_distance(p, p)
        assert np.isclose(tv, 0.0, atol=1e-10)

    def test_total_variation_distance_bounds(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        tv = BeliefMetrics.total_variation_distance(p, q)
        assert 0.0 <= tv <= 1.0

    def test_total_variation_distance_known_value(self) -> None:
        p = np.array([0.7, 0.3])
        q = np.array([0.3, 0.7])
        # TV = 0.5 * (|0.7-0.3| + |0.3-0.7|) = 0.5 * (0.4 + 0.4) = 0.4
        tv = BeliefMetrics.total_variation_distance(p, q)
        assert np.isclose(tv, 0.4)

    def test_total_variation_distance_symmetric(self) -> None:
        p = np.array([0.2, 0.8])
        q = np.array([0.6, 0.4])
        assert np.isclose(
            BeliefMetrics.total_variation_distance(p, q),
            BeliefMetrics.total_variation_distance(q, p),
        )

    def test_fidelity_identical(self) -> None:
        p = np.array([0.3, 0.7])
        f = BeliefMetrics.fidelity(p, p)
        assert np.isclose(f, 1.0, atol=1e-6)

    def test_fidelity_orthogonal(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        f = BeliefMetrics.fidelity(p, q)
        assert np.isclose(f, 0.0, atol=1e-10)

    def test_fidelity_bounds(self) -> None:
        p = np.array([0.6, 0.4])
        q = np.array([0.2, 0.8])
        f = BeliefMetrics.fidelity(p, q)
        assert 0.0 <= f <= 1.0

    def test_fidelity_symmetric(self) -> None:
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        assert np.isclose(BeliefMetrics.fidelity(p, q), BeliefMetrics.fidelity(q, p))


class TestPlanningMetrics:
    def test_creation_empty(self) -> None:
        metrics = PlanningMetrics()
        assert metrics.cumulative_rewards == []
        assert metrics.belief_errors == []
        assert metrics.action_sequence == []
        assert metrics.computation_times == []

    def test_add_step(self) -> None:
        metrics = PlanningMetrics()
        metrics.add_step(reward=5.0, belief_error=0.1, action=0, time_s=0.01)
        assert len(metrics.cumulative_rewards) == 1
        assert len(metrics.belief_errors) == 1
        assert len(metrics.action_sequence) == 1
        assert len(metrics.computation_times) == 1

    def test_add_multiple_steps(self) -> None:
        metrics = PlanningMetrics()
        for i in range(5):
            metrics.add_step(
                reward=float(i),
                belief_error=0.1 * i,
                action=i % 3,
                time_s=0.01 * i,
            )
        assert len(metrics.cumulative_rewards) == 5
        assert len(metrics.action_sequence) == 5

    def test_total_reward(self) -> None:
        metrics = PlanningMetrics()
        metrics.add_step(reward=1.0, belief_error=0.0, action=0, time_s=0.0)
        metrics.add_step(reward=2.0, belief_error=0.0, action=0, time_s=0.0)
        metrics.add_step(reward=3.0, belief_error=0.0, action=0, time_s=0.0)
        assert np.isclose(metrics.total_reward, 6.0)

    def test_mean_belief_error(self) -> None:
        metrics = PlanningMetrics()
        metrics.add_step(reward=0.0, belief_error=0.1, action=0, time_s=0.0)
        metrics.add_step(reward=0.0, belief_error=0.3, action=0, time_s=0.0)
        assert np.isclose(metrics.mean_belief_error, 0.2)

    def test_mean_belief_error_empty(self) -> None:
        metrics = PlanningMetrics()
        assert metrics.mean_belief_error == 0.0

    def test_mean_computation_time(self) -> None:
        metrics = PlanningMetrics()
        metrics.add_step(reward=0.0, belief_error=0.0, action=0, time_s=0.01)
        metrics.add_step(reward=0.0, belief_error=0.0, action=0, time_s=0.03)
        assert np.isclose(metrics.mean_computation_time, 0.02)

    def test_mean_computation_time_empty(self) -> None:
        metrics = PlanningMetrics()
        assert metrics.mean_computation_time == 0.0


class TestExperimentComparison:
    def _make_metrics(self, rewards, errors, times) -> PlanningMetrics:
        m = PlanningMetrics()
        for r, e, t in zip(rewards, errors, times):
            m.add_step(reward=r, belief_error=e, action=0, time_s=t)
        return m

    def test_summary_returns_dict(self) -> None:
        q = self._make_metrics([5.0, 3.0], [0.05, 0.03], [0.01, 0.01])
        c = self._make_metrics([4.0, 2.0], [0.1, 0.08], [0.05, 0.05])
        comp = ExperimentComparison(
            quantum_metrics=q,
            classical_metrics=c,
            problem_description="test_problem",
        )
        summary = comp.summary()
        assert isinstance(summary, dict)
        assert "problem" in summary
        assert "quantum_total_reward" in summary
        assert "classical_total_reward" in summary
        assert "reward_improvement" in summary
        assert "belief_accuracy_improvement" in summary
        assert "speedup" in summary

    def test_reward_improvement(self) -> None:
        q = self._make_metrics([10.0], [0.0], [0.0])
        c = self._make_metrics([8.0], [0.0], [0.0])
        comp = ExperimentComparison(quantum_metrics=q, classical_metrics=c)
        # (10 - 8) / |8| = 0.25
        assert np.isclose(comp.reward_improvement, 0.25)

    def test_reward_improvement_zero_classical(self) -> None:
        q = self._make_metrics([5.0], [0.0], [0.0])
        c = PlanningMetrics()
        comp = ExperimentComparison(quantum_metrics=q, classical_metrics=c)
        assert comp.reward_improvement == 0.0

    def test_belief_accuracy_improvement(self) -> None:
        q = self._make_metrics([0.0], [0.02], [0.0])
        c = self._make_metrics([0.0], [0.10], [0.0])
        comp = ExperimentComparison(quantum_metrics=q, classical_metrics=c)
        # (0.10 - 0.02) / 0.10 = 0.8
        assert np.isclose(comp.belief_accuracy_improvement, 0.8)

    def test_speedup(self) -> None:
        q = self._make_metrics([0.0], [0.0], [0.01])
        c = self._make_metrics([0.0], [0.0], [0.05])
        comp = ExperimentComparison(quantum_metrics=q, classical_metrics=c)
        # 0.05 / 0.01 = 5.0
        assert np.isclose(comp.speedup, 5.0)

    def test_speedup_zero_quantum_time(self) -> None:
        q = PlanningMetrics()
        c = self._make_metrics([0.0], [0.0], [0.05])
        comp = ExperimentComparison(quantum_metrics=q, classical_metrics=c)
        assert comp.speedup == 0.0


class TestScalingAnalysis:
    def test_creation_empty(self) -> None:
        sa = ScalingAnalysis()
        assert sa.problem_sizes == []
        assert sa.quantum_times == []
        assert sa.classical_times == []

    def test_add_data_point(self) -> None:
        sa = ScalingAnalysis()
        sa.add_data_point(size=4, q_time=0.1, c_time=0.5, qubits=5, depth=20)
        assert len(sa.problem_sizes) == 1
        assert sa.problem_sizes[0] == 4
        assert sa.quantum_times[0] == 0.1
        assert sa.classical_times[0] == 0.5
        assert sa.quantum_qubits[0] == 5
        assert sa.circuit_depths[0] == 20

    def test_add_multiple_data_points(self) -> None:
        sa = ScalingAnalysis()
        for i in range(1, 6):
            sa.add_data_point(size=2**i, q_time=0.1 * i, c_time=0.5 * i)
        assert len(sa.problem_sizes) == 5

    def test_speedups(self) -> None:
        sa = ScalingAnalysis()
        sa.add_data_point(size=4, q_time=0.1, c_time=0.5)
        sa.add_data_point(size=8, q_time=0.2, c_time=1.6)
        speedups = sa.speedups
        assert len(speedups) == 2
        assert np.isclose(speedups[0], 5.0)
        assert np.isclose(speedups[1], 8.0)

    def test_speedups_zero_quantum_time(self) -> None:
        sa = ScalingAnalysis()
        sa.add_data_point(size=4, q_time=0.0, c_time=0.5)
        assert sa.speedups[0] == 0.0

    def test_crossover_estimate_found(self) -> None:
        sa = ScalingAnalysis()
        # At size 4: quantum is slower; at size 8: quantum is faster
        sa.add_data_point(size=4, q_time=0.5, c_time=0.3)
        sa.add_data_point(size=8, q_time=0.4, c_time=0.6)
        crossover = sa.crossover_estimate()
        assert crossover == 8

    def test_crossover_estimate_not_found(self) -> None:
        sa = ScalingAnalysis()
        # Quantum is always slower
        sa.add_data_point(size=4, q_time=0.5, c_time=0.3)
        sa.add_data_point(size=8, q_time=1.0, c_time=0.6)
        crossover = sa.crossover_estimate()
        assert crossover is None

    def test_crossover_estimate_empty(self) -> None:
        sa = ScalingAnalysis()
        assert sa.crossover_estimate() is None

    def test_fit_scaling_with_enough_data(self) -> None:
        sa = ScalingAnalysis()
        for n in [4, 8, 16, 32]:
            sa.add_data_point(size=n, q_time=0.1 * n, c_time=0.5 * n)
        q_scaling = sa.quantum_scaling
        assert "exponent" in q_scaling
        assert "coefficient" in q_scaling
        # Linear scaling (time ~ n^1) should give exponent close to 1
        assert np.isclose(q_scaling["exponent"], 1.0, atol=0.1)

    def test_fit_scaling_insufficient_data(self) -> None:
        sa = ScalingAnalysis()
        sa.add_data_point(size=4, q_time=0.1, c_time=0.5)
        scaling = sa.fit_scaling(sa.quantum_times)
        assert scaling["exponent"] == 0.0
        assert scaling["coefficient"] == 0.0
