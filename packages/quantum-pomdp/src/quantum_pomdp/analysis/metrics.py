"""Metrics for quantum vs classical POMDP comparison.

Academic References:
    Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
    Observable Stochastic Domains", AI 101:99-134 (1998).
    -- POMDP planning quality metrics: cumulative reward and belief
       accuracy are standard evaluation criteria.

    arXiv:2507.18606 - Quantum vs classical comparison metrics used
    for QBRL evaluation. Key metrics include:
    * Cumulative reward (planning quality)
    * Belief accuracy (KL divergence, Hellinger distance)
    * Computation time (quantum speedup measurement)

    Belief state distance metrics:
    * KL divergence: D_KL(p||q) = sum p_i log(p_i/q_i)
      -- Asymmetric measure of information loss when q approximates p.
    * Hellinger distance: H(p,q) = sqrt(0.5 * sum (sqrt(p_i) - sqrt(q_i))^2)
      -- Symmetric, bounded [0,1], related to quantum state fidelity.
    * Total variation distance: TV(p,q) = 0.5 * sum |p_i - q_i|
      -- Maximum difference in probability of any event.
    * Classical fidelity: F(p,q) = (sum sqrt(p_i * q_i))^2
      -- Bhattacharyya coefficient squared; related to quantum state
         fidelity for diagonal density matrices.

    For quantum state fidelity on non-diagonal states, see:
    Jozsa, "Fidelity for Mixed Quantum States", J. Mod. Opt. 41,
    2315-2323 (1994).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

@dataclass
class BeliefMetrics:
    """Metrics for comparing belief state estimates.

    Standard information-theoretic distances for evaluating quantum vs
    classical belief accuracy (arXiv:2507.18606, experimental evaluation).
    """

    @staticmethod
    def kl_divergence(p: NDArray[np.float64], q: NDArray[np.float64], epsilon: float = 1e-10) -> float:
        """KL(p || q) = sum p_i * log(p_i / q_i)."""
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)
        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def hellinger_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
        """Hellinger distance between two distributions."""
        return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

    @staticmethod
    def total_variation_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
        """Total variation distance."""
        return float(0.5 * np.sum(np.abs(p - q)))

    @staticmethod
    def fidelity(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
        """Classical fidelity (Bhattacharyya coefficient)."""
        return float(np.sum(np.sqrt(p * q)))

@dataclass
class PlanningMetrics:
    """Metrics for POMDP planning quality."""
    cumulative_rewards: list[float] = field(default_factory=list)
    belief_errors: list[float] = field(default_factory=list)
    action_sequence: list[int] = field(default_factory=list)
    computation_times: list[float] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(self.cumulative_rewards)

    @property
    def mean_belief_error(self) -> float:
        return float(np.mean(self.belief_errors)) if self.belief_errors else 0.0

    @property
    def mean_computation_time(self) -> float:
        return float(np.mean(self.computation_times)) if self.computation_times else 0.0

    def add_step(self, reward: float, belief_error: float, action: int, time_s: float) -> None:
        self.cumulative_rewards.append(reward)
        self.belief_errors.append(belief_error)
        self.action_sequence.append(action)
        self.computation_times.append(time_s)

@dataclass
class ExperimentComparison:
    """Compare quantum vs classical experiment results.

    Quantifies the quantum advantage from arXiv:2507.18606:
    * reward_improvement: relative cumulative reward gain
    * belief_accuracy_improvement: relative belief error reduction
    * speedup: wall-clock time ratio (classical / quantum)
    """
    quantum_metrics: PlanningMetrics
    classical_metrics: PlanningMetrics
    problem_description: str = ""

    @property
    def reward_improvement(self) -> float:
        if self.classical_metrics.total_reward == 0:
            return 0.0
        return (self.quantum_metrics.total_reward - self.classical_metrics.total_reward) / abs(self.classical_metrics.total_reward)

    @property
    def belief_accuracy_improvement(self) -> float:
        c_err = self.classical_metrics.mean_belief_error
        q_err = self.quantum_metrics.mean_belief_error
        if c_err == 0:
            return 0.0
        return (c_err - q_err) / c_err

    @property
    def speedup(self) -> float:
        q_time = self.quantum_metrics.mean_computation_time
        c_time = self.classical_metrics.mean_computation_time
        if q_time == 0:
            return 0.0
        return c_time / q_time

    def summary(self) -> dict[str, Any]:
        return {
            "problem": self.problem_description,
            "quantum_total_reward": self.quantum_metrics.total_reward,
            "classical_total_reward": self.classical_metrics.total_reward,
            "reward_improvement": self.reward_improvement,
            "belief_accuracy_improvement": self.belief_accuracy_improvement,
            "speedup": self.speedup,
        }
