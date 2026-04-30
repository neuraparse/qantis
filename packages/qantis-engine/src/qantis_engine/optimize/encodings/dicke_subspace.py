"""Generalized Dicke subspace encoding (arXiv:2601.16396, Min, Seo, Heo, 2026).

Generalized Dicke states ``|D^k_n⟩`` are uniform superpositions of all
``C(n, k)`` Hamming-weight-k bitstrings — the natural feasible-set
initialization for one-hot / weight-constrained problems. They are prepared
by the Bartschi-Eidenbenz unitary in ``O(k(n-k))`` two-qubit gates, with depth
``O(n)``. This module provides the *abstract* encoder; the actual circuit is
constructed by the solver layer (see ``optimize.solvers.feasible_sampler``).

Use cases for QANTIS:
    - 5G CBRS-style channel allocation with global cardinality constraints
    - portfolio cardinality constraints
    - sparse measurement scheduling

This encoder is the most lightweight feasibility constraint in the toolkit:
it enforces ``sum(x) == k`` only, with no per-block one-hot. For MTDA-style
problems with row+column one-hot, compose multiple Dicke subspaces (one per
row) — equivalent to the XY-ring mixer applied per row.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DickeSubspaceEncoding:
    """``|D^k_n⟩`` Dicke-state subspace over n qubits."""

    n_qubits: int
    weight: int

    def __post_init__(self) -> None:
        if not 0 <= self.weight <= self.n_qubits:
            raise ValueError(
                f"weight {self.weight} not in [0, {self.n_qubits}]"
            )

    @property
    def feasible_count(self) -> int:
        return math.comb(self.n_qubits, self.weight)

    @property
    def preparation_depth(self) -> int:
        return self.weight * (self.n_qubits - self.weight)

    def num_qubits(self) -> int:
        return self.n_qubits

    def feasible_states(self) -> Iterable[NDArray[np.int_]]:
        for indices in combinations(range(self.n_qubits), self.weight):
            x = np.zeros(self.n_qubits, dtype=np.int_)
            x[list(indices)] = 1
            yield x

    def is_feasible(self, bitstring: NDArray[np.int_]) -> bool:
        return int(np.sum(bitstring)) == self.weight

    def initial_state(self) -> NDArray[np.complex128]:
        """Statevector of the generalized Dicke state ``|D^k_n⟩``."""
        amp = 1.0 / math.sqrt(self.feasible_count)
        psi = np.zeros(2**self.n_qubits, dtype=np.complex128)
        for indices in combinations(range(self.n_qubits), self.weight):
            idx = 0
            for q in indices:
                idx |= 1 << q
            psi[idx] = amp
        return psi

    def project(self, bitstring: NDArray[np.int_]) -> NDArray[np.int_]:
        x = np.asarray(bitstring, dtype=np.int_).copy()
        current = int(x.sum())
        delta = self.weight - current
        if delta == 0:
            return x
        if delta > 0:
            zero_idx = np.flatnonzero(x == 0)
            x[zero_idx[:delta]] = 1
        else:
            one_idx = np.flatnonzero(x == 1)
            x[one_idx[: -delta]] = 0
        return x
