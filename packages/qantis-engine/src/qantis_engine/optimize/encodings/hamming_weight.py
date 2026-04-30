"""Hamming-weight-preserving encoding (arXiv:2601.01516, Hao et al., Jan 2026).

Confines quantum evolution to bitstrings of fixed Hamming weight ``k`` over
``n`` qubits. The feasible subspace has size ``C(n, k)``; for one-hot
constraints (k=1) this reduces to the standard XY-mixer subspace.

This encoder does not by itself emit a circuit — it specifies the feasible
subspace and the projection map. Solvers compose this with their preferred
mixer (XY ring, XY complete, hybrid XY-X).
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HammingWeightEncoding:
    """Encoding for ``{x in {0,1}^n : sum(x) == k}``."""

    n_qubits: int
    target_weight: int

    def __post_init__(self) -> None:
        if not 0 <= self.target_weight <= self.n_qubits:
            raise ValueError(
                f"target_weight {self.target_weight} not in [0, {self.n_qubits}]"
            )

    @property
    def feasible_count(self) -> int:
        return math.comb(self.n_qubits, self.target_weight)

    def num_qubits(self) -> int:
        return self.n_qubits

    def feasible_states(self) -> Iterable[NDArray[np.int_]]:
        for indices in combinations(range(self.n_qubits), self.target_weight):
            x = np.zeros(self.n_qubits, dtype=np.int_)
            x[list(indices)] = 1
            yield x

    def project(self, bitstring: NDArray[np.int_]) -> NDArray[np.int_]:
        """Closest Hamming-weight-feasible bitstring to ``bitstring`` (Hamming distance).

        The closest bitstring with weight ``k`` is obtained by either flipping
        the smallest-evidence 1s to 0s (if weight too high) or the largest-
        evidence 0s to 1s (if too low). Without an external evidence vector we
        flip arbitrary positions; callers with confidences should use
        ``project_with_scores``.
        """
        x = np.asarray(bitstring, dtype=np.int_).copy()
        current_weight = int(x.sum())
        delta = self.target_weight - current_weight
        if delta == 0:
            return x
        if delta > 0:
            zero_idx = np.flatnonzero(x == 0)
            x[zero_idx[:delta]] = 1
        else:
            one_idx = np.flatnonzero(x == 1)
            x[one_idx[: -delta]] = 0
        return x

    def project_with_scores(
        self, bitstring: NDArray[np.int_], scores: NDArray[np.float64]
    ) -> NDArray[np.int_]:
        """Closest feasible bitstring favouring positions with high ``scores``.

        Used by feasibility repair: ``scores[i]`` is the marginal probability
        / amplitude estimate that ``x_i = 1`` (e.g. from a sampler histogram).
        """
        x = np.asarray(bitstring, dtype=np.int_).copy()
        scores = np.asarray(scores, dtype=np.float64)
        current_weight = int(x.sum())
        delta = self.target_weight - current_weight
        if delta == 0:
            return x
        if delta > 0:
            zero_idx = np.flatnonzero(x == 0)
            order = np.argsort(-scores[zero_idx])
            x[zero_idx[order[:delta]]] = 1
        else:
            one_idx = np.flatnonzero(x == 1)
            order = np.argsort(scores[one_idx])
            x[one_idx[order[: -delta]]] = 0
        return x

    def is_feasible(self, bitstring: NDArray[np.int_]) -> bool:
        return int(np.sum(bitstring)) == self.target_weight

    def initial_state(self) -> NDArray[np.complex128]:
        """Uniform superposition over the feasible (Dicke) subspace."""
        amp = 1.0 / math.sqrt(self.feasible_count)
        psi = np.zeros(2**self.n_qubits, dtype=np.complex128)
        for x in self.feasible_states():
            idx = int("".join(str(b) for b in x), 2)
            psi[idx] = amp
        return psi
