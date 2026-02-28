"""Tests for POMDP models."""

import numpy as np
import pytest

from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.models.pomdp import POMDPModel


class TestPOMDPModel:
    def test_tiger_dimensions(self, tiger_pomdp: POMDPModel) -> None:
        assert tiger_pomdp.num_states == 2
        assert tiger_pomdp.num_actions == 3
        assert tiger_pomdp.num_observations == 2

    def test_qubit_counts(self, tiger_pomdp: POMDPModel) -> None:
        assert tiger_pomdp.state_qubits == 1
        assert tiger_pomdp.action_qubits == 2
        assert tiger_pomdp.observation_qubits == 1

    def test_invalid_transition_raises(self) -> None:
        T = np.array([[[0.5, 0.3], [0.5, 0.5]]])  # doesn't sum to 1
        O = np.array([[[0.5, 0.5], [0.5, 0.5]]])
        R = np.array([[1.0], [1.0]])

        with pytest.raises(ValueError, match="don't sum to 1"):
            POMDPModel(
                num_states=2, num_actions=1, num_observations=2,
                transition_tensor=T, observation_tensor=O, reward_matrix=R,
            )

    def test_expected_reward(self, tiger_pomdp: POMDPModel) -> None:
        belief = np.array([0.5, 0.5])
        # Listen action (index 0): E[r] = 0.5*(-1) + 0.5*(-1) = -1
        reward = tiger_pomdp.expected_reward(belief, 0)
        assert np.isclose(reward, -1.0)

    def test_observation_probability(self, tiger_pomdp: POMDPModel) -> None:
        belief = np.array([0.5, 0.5])
        # Listen with uniform belief: P(hear-left) should be 0.5
        prob = tiger_pomdp.observation_probability(belief, 0, 0)
        assert np.isclose(prob, 0.5)


class TestBeliefState:
    def test_uniform(self) -> None:
        b = BeliefState.uniform(4)
        assert np.allclose(b.probabilities, [0.25, 0.25, 0.25, 0.25])

    def test_from_state(self) -> None:
        b = BeliefState.from_state(1, 3)
        assert np.isclose(b.probabilities[1], 1.0)
        assert np.isclose(b.probabilities[0], 0.0)

    def test_most_likely_state(self) -> None:
        b = BeliefState(np.array([0.1, 0.7, 0.2]))
        assert b.most_likely_state == 1

    def test_classical_update(self, tiger_pomdp: POMDPModel) -> None:
        b = BeliefState.uniform(2)
        b_new = b.classical_update(
            action=0, observation=0,
            transition_tensor=tiger_pomdp.transition_tensor,
            observation_tensor=tiger_pomdp.observation_tensor,
        )
        # After hearing left with listen, tiger-left should be more likely
        assert b_new.probabilities[0] > b_new.probabilities[1]

    def test_to_amplitudes(self) -> None:
        b = BeliefState(np.array([0.25, 0.75]))
        amps = b.to_amplitudes()
        assert np.isclose(np.abs(amps[0]) ** 2, 0.25)
        assert np.isclose(np.abs(amps[1]) ** 2, 0.75)

    def test_kl_divergence_self_is_zero(self) -> None:
        b = BeliefState(np.array([0.3, 0.7]))
        assert np.isclose(b.kl_divergence(b), 0.0, atol=1e-10)

    def test_hellinger_bounds(self) -> None:
        b1 = BeliefState(np.array([1.0, 0.0]))
        b2 = BeliefState(np.array([0.0, 1.0]))
        assert b1.hellinger_distance(b2) <= 1.0
        assert b1.hellinger_distance(b1) < 1e-10

    def test_entropy(self) -> None:
        b_uniform = BeliefState.uniform(2)
        b_certain = BeliefState.from_state(0, 2)
        assert b_uniform.entropy() > b_certain.entropy()
        assert np.isclose(b_uniform.entropy(), 1.0)  # log2(2)

    def test_from_quantum_measurement(self) -> None:
        counts = {"0": 700, "1": 300}
        b = BeliefState.from_quantum_measurement(counts, num_states=2, num_state_qubits=1)
        assert np.isclose(b.probabilities[0], 0.7)
        assert np.isclose(b.probabilities[1], 0.3)


# ---------------------------------------------------------------------------
# Parametrized tests — BeliefState
# ---------------------------------------------------------------------------

class TestBeliefStateParametrized:
    """Comprehensive parametrized coverage for BeliefState."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 32])
    def test_uniform_sum_to_one(self, n: int) -> None:
        b = BeliefState.uniform(n)
        assert np.isclose(b.probabilities.sum(), 1.0)
        assert len(b.probabilities) == n
        assert np.allclose(b.probabilities, 1.0 / n)

    @pytest.mark.parametrize("n", [2, 3, 4, 8])
    def test_uniform_max_entropy(self, n: int) -> None:
        """Uniform belief has maximum entropy = log2(n)."""
        b = BeliefState.uniform(n)
        expected_entropy = np.log2(n)
        assert np.isclose(b.entropy(), expected_entropy, rtol=1e-6)

    @pytest.mark.parametrize("n,state", [
        (2, 0), (2, 1),
        (3, 0), (3, 1), (3, 2),
        (5, 4),
        (10, 7),
    ])
    def test_from_state_deterministic(self, n: int, state: int) -> None:
        b = BeliefState.from_state(state, n)
        assert np.isclose(b.probabilities[state], 1.0)
        assert np.isclose(b.probabilities.sum(), 1.0)
        for i in range(n):
            if i != state:
                assert np.isclose(b.probabilities[i], 0.0)

    @pytest.mark.parametrize("n,state", [(2, 0), (2, 1), (4, 2)])
    def test_from_state_most_likely(self, n: int, state: int) -> None:
        b = BeliefState.from_state(state, n)
        assert b.most_likely_state == state

    @pytest.mark.parametrize("n,state", [(2, 0), (2, 1)])
    def test_from_state_zero_entropy(self, n: int, state: int) -> None:
        """Deterministic belief has zero entropy."""
        b = BeliefState.from_state(state, n)
        assert np.isclose(b.entropy(), 0.0, atol=1e-10)

    @pytest.mark.parametrize("probs", [
        [0.5, 0.5],
        [0.8, 0.2],
        [0.1, 0.3, 0.6],
        [0.25, 0.25, 0.25, 0.25],
        [0.9, 0.05, 0.05],
    ])
    def test_probabilities_sum_to_one(self, probs: list) -> None:
        b = BeliefState(np.array(probs))
        assert np.isclose(b.probabilities.sum(), 1.0)

    def test_unnormalized_input_is_normalized(self) -> None:
        """BeliefState normalizes inputs that don't sum to 1."""
        b = BeliefState(np.array([2.0, 2.0]))
        assert np.isclose(b.probabilities.sum(), 1.0)
        assert np.allclose(b.probabilities, [0.5, 0.5])

    def test_unnormalized_larger_input(self) -> None:
        b = BeliefState(np.array([3.0, 1.0]))
        assert np.isclose(b.probabilities[0], 0.75)
        assert np.isclose(b.probabilities[1], 0.25)

    @pytest.mark.parametrize("action,observation,expect_first_larger", [
        (0, 0, True),   # listen + hear-left → tiger-left more likely
        (0, 1, False),  # listen + hear-right → tiger-right more likely
    ])
    def test_classical_update_tiger_listen(
        self, tiger_pomdp, action, observation, expect_first_larger
    ) -> None:
        b = BeliefState.uniform(2)
        b_new = b.classical_update(
            action=action,
            observation=observation,
            transition_tensor=tiger_pomdp.transition_tensor,
            observation_tensor=tiger_pomdp.observation_tensor,
        )
        assert np.isclose(b_new.probabilities.sum(), 1.0)
        if expect_first_larger:
            assert b_new.probabilities[0] > b_new.probabilities[1]
        else:
            assert b_new.probabilities[1] > b_new.probabilities[0]

    @pytest.mark.parametrize("action", [1, 2])
    def test_classical_update_open_resets_to_uniform(self, tiger_pomdp, action) -> None:
        """After open action, belief resets to uniform."""
        b = BeliefState(np.array([0.9, 0.1]))
        for obs in [0, 1]:
            b_new = b.classical_update(
                action=action,
                observation=obs,
                transition_tensor=tiger_pomdp.transition_tensor,
                observation_tensor=tiger_pomdp.observation_tensor,
            )
            assert np.isclose(b_new.probabilities.sum(), 1.0)
            assert np.allclose(b_new.probabilities, [0.5, 0.5], atol=0.01)

    @pytest.mark.parametrize("n_qubits,counts,expected_probs", [
        (1, {"0": 700, "1": 300}, [0.7, 0.3]),
        (1, {"0": 1000, "1": 0}, [1.0, 0.0]),
        (1, {"0": 500, "1": 500}, [0.5, 0.5]),
        (2, {"00": 250, "01": 250, "10": 250, "11": 250}, [0.25, 0.25, 0.25, 0.25]),
    ])
    def test_from_quantum_measurement_parametrized(
        self, n_qubits, counts, expected_probs
    ) -> None:
        n_states = 2 ** n_qubits
        b = BeliefState.from_quantum_measurement(
            counts, num_states=n_states, num_state_qubits=n_qubits
        )
        assert np.isclose(b.probabilities.sum(), 1.0)
        for i, ep in enumerate(expected_probs):
            assert np.isclose(b.probabilities[i], ep, atol=1e-9)

    def test_from_quantum_measurement_empty_counts(self) -> None:
        """Empty counts → uniform belief (fallback)."""
        b = BeliefState.from_quantum_measurement({}, num_states=2, num_state_qubits=1)
        assert np.isclose(b.probabilities.sum(), 1.0)
        assert np.allclose(b.probabilities, [0.5, 0.5])

    def test_from_quantum_measurement_zero_shots(self) -> None:
        """All-zero counts → uniform belief (fallback)."""
        b = BeliefState.from_quantum_measurement({"0": 0, "1": 0}, num_states=2, num_state_qubits=1)
        assert np.isclose(b.probabilities.sum(), 1.0)


class TestBeliefStateMetrics:
    """Parametrized metric edge cases."""

    def test_hellinger_identical_is_zero(self) -> None:
        b = BeliefState(np.array([0.3, 0.7]))
        assert np.isclose(b.hellinger_distance(b), 0.0, atol=1e-10)

    def test_hellinger_orthogonal_is_one(self) -> None:
        b1 = BeliefState.from_state(0, 2)
        b2 = BeliefState.from_state(1, 2)
        assert np.isclose(b1.hellinger_distance(b2), 1.0)

    @pytest.mark.parametrize("p1,p2", [
        ([0.5, 0.5], [0.5, 0.5]),
        ([0.8, 0.2], [0.3, 0.7]),
        ([1.0, 0.0], [0.5, 0.5]),
        ([0.25, 0.75], [0.75, 0.25]),
    ])
    def test_hellinger_in_bounds(self, p1, p2) -> None:
        b1 = BeliefState(np.array(p1))
        b2 = BeliefState(np.array(p2))
        h = b1.hellinger_distance(b2)
        assert 0.0 <= h <= 1.0

    @pytest.mark.parametrize("p1,p2", [
        ([0.5, 0.5], [0.5, 0.5]),
        ([0.8, 0.2], [0.3, 0.7]),
    ])
    def test_hellinger_symmetric(self, p1, p2) -> None:
        b1 = BeliefState(np.array(p1))
        b2 = BeliefState(np.array(p2))
        assert np.isclose(b1.hellinger_distance(b2), b2.hellinger_distance(b1))

    def test_total_variation_identical_is_zero(self) -> None:
        b = BeliefState(np.array([0.4, 0.6]))
        assert np.isclose(b.total_variation_distance(b), 0.0, atol=1e-10)

    def test_total_variation_orthogonal_is_one(self) -> None:
        b1 = BeliefState.from_state(0, 2)
        b2 = BeliefState.from_state(1, 2)
        assert np.isclose(b1.total_variation_distance(b2), 1.0)

    @pytest.mark.parametrize("probs", [
        [0.5, 0.5], [0.3, 0.7], [0.1, 0.2, 0.7],
    ])
    def test_tv_in_bounds(self, probs) -> None:
        b1 = BeliefState(np.array(probs))
        b2 = BeliefState.uniform(len(probs))
        tv = b1.total_variation_distance(b2)
        assert 0.0 <= tv <= 1.0

    def test_fidelity_identical_is_one(self) -> None:
        b = BeliefState(np.array([0.4, 0.6]))
        assert np.isclose(b.fidelity(b), 1.0, atol=1e-10)

    def test_fidelity_orthogonal_is_zero(self) -> None:
        b1 = BeliefState.from_state(0, 2)
        b2 = BeliefState.from_state(1, 2)
        assert np.isclose(b1.fidelity(b2), 0.0, atol=1e-10)

    @pytest.mark.parametrize("probs", [
        [0.5, 0.5], [0.8, 0.2], [0.25, 0.25, 0.25, 0.25],
    ])
    def test_fidelity_in_bounds(self, probs) -> None:
        b1 = BeliefState(np.array(probs))
        b2 = BeliefState.uniform(len(probs))
        f = b1.fidelity(b2)
        assert 0.0 <= f <= 1.0

    def test_kl_divergence_asymmetric(self) -> None:
        b1 = BeliefState(np.array([0.3, 0.7]))
        b2 = BeliefState(np.array([0.6, 0.4]))
        # KL divergence is not generally symmetric
        kl_12 = b1.kl_divergence(b2)
        kl_21 = b2.kl_divergence(b1)
        assert kl_12 >= 0.0
        assert kl_21 >= 0.0

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_to_amplitudes_norm(self, n: int) -> None:
        """Amplitude vector should have norm 1 (quantum state normalization)."""
        b = BeliefState.uniform(n)
        amps = b.to_amplitudes()
        norm_sq = sum(abs(a) ** 2 for a in amps)
        assert np.isclose(norm_sq, 1.0, rtol=1e-6)

    def test_num_states_property(self) -> None:
        for n in [2, 3, 5, 10]:
            b = BeliefState.uniform(n)
            assert b.num_states == n
