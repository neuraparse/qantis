"""Belief state representation for POMDPs with quantum encoding support.

In the quantum circuit, the belief state is encoded as amplitudes:
|psi_b> = sum_s sqrt(b(s)) |s>

Academic References:
    arXiv:2507.18606, Sec 3.1 - Quantum belief state encoding using
    amplitude encoding on the S_t register (k1 qubits).

    Shende, Bullock, Markov, "Synthesis of Quantum Logic Circuits",
    IEEE Trans. CAD 25(6) (2006).
    -- State preparation complexity is O(2^n) CNOT gates for arbitrary
       n-qubit amplitudes. This bounds the depth of encode_belief().

    Moettonen, Vartiainen, Bergholm, Salomaa, PRL 93, 130502 (2004).
    -- Uniformly Controlled Rotations (UCR_Y) decomposition used in
       encode_belief_with_rotations() for hardware-friendly encoding.

    PennyLane v0.44 (Jan 2026): MultiplexerStatePreparation template
    provides an alternative optimized implementation of amplitude encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BeliefState:
    """Probability distribution over POMDP states.

    The classical belief update (Bayesian filter) per Kaelbling et al. (1998):
    b_{t+1}(s') = P(o|s',a) * sum_s P(s'|s,a) * b(s) / P(o|b,a)

    For quantum encoding, amplitudes are sqrt(b(s)) -- see Shende et al. (2006)
    for state preparation complexity O(2^n) where n = ceil(log2(|S|)).
    """

    probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not np.isclose(self.probabilities.sum(), 1.0, atol=1e-6):
            self.probabilities = self.probabilities / self.probabilities.sum()

    @classmethod
    def uniform(cls, num_states: int) -> BeliefState:
        """Create uniform belief over all states."""
        return cls(np.ones(num_states) / num_states)

    @classmethod
    def from_state(cls, state: int, num_states: int) -> BeliefState:
        """Create deterministic belief at a specific state."""
        probs = np.zeros(num_states)
        probs[state] = 1.0
        return cls(probs)

    @classmethod
    def from_quantum_measurement(
        cls,
        counts: dict[str, int],
        num_states: int,
        num_state_qubits: int,
        post_selection: dict[int, int] | None = None,
    ) -> BeliefState:
        """Reconstruct belief state from quantum circuit measurement results.

        The QBRL belief-update circuit (arXiv:2507.18606 Fig. 3) stores
        the next-state register in classical bits ``[0, num_state_qubits)``
        and the observation register in the remaining classical bits.
        Qiskit's ``counts`` dict uses the big-endian string convention, so
        ``bitstring[-1]`` is ``classical[0]`` and ``bitstring[0]`` is the
        highest-indexed classical bit. This method therefore reads the
        state bits from the **right** of the bitstring.

        When the circuit includes Brassard-Hoyer-Mosca-Tapp amplitude
        amplification for a specific observation (G^k(o)), the evidence
        subspace is amplified but not fully collapsed; passing
        ``post_selection={obs_bit_index: expected_value, ...}`` (with
        bit indices specified as *classical-register indices*, i.e.
        ``num_state_qubits`` is the first observation bit) keeps only
        the shots whose observation register matches the desired value,
        recovering the exact conditional ``P(s'|b,a,o)``.
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return cls.uniform(num_states)

        probs = np.zeros(num_states)
        for bitstring, count in counts.items():
            bs = bitstring.replace(" ", "")

            if post_selection is not None:
                if not all(
                    bs[-1 - c_idx] == str(expected)
                    for c_idx, expected in post_selection.items()
                ):
                    continue

            state_bits = bs[-num_state_qubits:] if num_state_qubits > 0 else ""
            state_idx = int(state_bits, 2) if state_bits else 0
            if state_idx < num_states:
                probs[state_idx] += count

        total = probs.sum()
        if total < 1e-12:
            return cls.uniform(num_states)

        return cls(probs / total)

    @property
    def num_states(self) -> int:
        return len(self.probabilities)

    @property
    def most_likely_state(self) -> int:
        return int(np.argmax(self.probabilities))

    def classical_update(
        self,
        action: int,
        observation: int,
        transition_tensor: NDArray[np.float64],
        observation_tensor: NDArray[np.float64],
    ) -> BeliefState:
        """Standard Bayesian belief update (classical baseline).

        b'(s') = P(o|s',a) * sum_s T(s'|s,a) * b(s) / normalizer

        Per Kaelbling, Littman, Cassandra, "Planning and Acting in Partially
        Observable Stochastic Domains", AI 101:99-134 (1998), Eq. 4.
        Classical complexity: O(|S|^2) per update. The quantum analogue
        (arXiv:2507.18606, Sec III) achieves O(P(e)^{-1/2}) via amplitude
        amplification, a quadratic speedup over rejection sampling.
        """
        t_a = transition_tensor[action]  # (S, S')
        o_a = observation_tensor[action]  # (S', Omega)

        predicted = t_a.T @ self.probabilities  # sum over s
        updated = o_a[:, observation] * predicted  # element-wise
        normalizer = updated.sum()

        if normalizer < 1e-12:
            return BeliefState.uniform(self.num_states)

        return BeliefState(updated / normalizer)

    def to_amplitudes(self) -> NDArray[np.complex128]:
        """Convert to quantum amplitudes: sqrt(b(s)) for state preparation.

        Amplitude encoding: |psi_b> = sum_s sqrt(b(s))|s>.
        Complexity: O(2^n) gates (Shende, Bullock, Markov, IEEE TCAD 2006).
        """
        return np.sqrt(np.maximum(self.probabilities, 0.0)).astype(np.complex128)

    def kl_divergence(self, other: BeliefState) -> float:
        """KL(self || other) for accuracy comparison."""
        p = np.clip(self.probabilities, 1e-12, 1.0)
        q = np.clip(other.probabilities, 1e-12, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def hellinger_distance(self, other: BeliefState) -> float:
        """Hellinger distance H(self, other) in [0, 1]."""
        sqrt_p = np.sqrt(self.probabilities)
        sqrt_q = np.sqrt(other.probabilities)
        return float(np.sqrt(0.5 * np.sum((sqrt_p - sqrt_q) ** 2)))

    def total_variation_distance(self, other: BeliefState) -> float:
        """Total variation distance TV(self, other) in [0, 1]."""
        return float(0.5 * np.sum(np.abs(self.probabilities - other.probabilities)))

    def entropy(self) -> float:
        """Shannon entropy of belief distribution."""
        p = self.probabilities[self.probabilities > 1e-12]
        return float(-np.sum(p * np.log2(p)))

    def fidelity(self, other: BeliefState) -> float:
        """Classical fidelity F(p, q) = (sum sqrt(p*q))^2."""
        return float(np.sum(np.sqrt(self.probabilities * other.probabilities)) ** 2)
