"""Colored-permutation encoding (arXiv:2604.04570, Onah & Michielsen, Apr 2026).

Encodes a partial assignment of N customers to K vehicles as K disjoint
partial permutations whose union equals the full identity permutation. Each
"color" is one vehicle's tour; flow conservation is preserved natively (each
customer appears in exactly one vehicle's tour, in exactly one position).

For QANTIS-Optimize this is the right encoder for capacitated assignment
problems where:
    - each task must be assigned to exactly one of K resources
    - each resource has its own ordered slot sequence (capacity)
    - missed-task / false-alarm slack maps to a designated "null" color

The encoder reduces the qubit count vs naive ``N*K`` one-hot by sharing the
position registers across colors; the saving is asymptotically the same as
PermutationOracleEncoding but with cross-color constraints.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ColoredPermutationEncoding:
    """K disjoint partial permutations covering ``[0, N)``."""

    n_tasks: int
    n_colors: int

    @property
    def color_bits(self) -> int:
        return max(1, math.ceil(math.log2(self.n_colors))) if self.n_colors > 1 else 1

    def num_qubits(self) -> int:
        return self.n_tasks * self.color_bits

    def encode(self, coloring: NDArray[np.int_]) -> NDArray[np.int_]:
        """Encode an assignment ``coloring[t] in [0, K)`` of tasks to colors."""
        if coloring.shape != (self.n_tasks,):
            raise ValueError(f"expected ({self.n_tasks},), got {coloring.shape}")
        bits = np.zeros(self.num_qubits(), dtype=np.int_)
        for t in range(self.n_tasks):
            v = int(coloring[t])
            if not 0 <= v < self.n_colors:
                raise ValueError(f"task {t} color {v} out of [0, {self.n_colors})")
            for b in range(self.color_bits):
                bits[t * self.color_bits + b] = (v >> b) & 1
        return bits

    def decode(self, bitstring: NDArray[np.int_]) -> NDArray[np.int_]:
        out = np.zeros(self.n_tasks, dtype=np.int_)
        for t in range(self.n_tasks):
            value = 0
            for b in range(self.color_bits):
                value |= int(bitstring[t * self.color_bits + b]) << b
            out[t] = value
        return out

    def is_feasible(
        self,
        bitstring: NDArray[np.int_],
        capacities: NDArray[np.int_] | None = None,
    ) -> bool:
        coloring = self.decode(bitstring)
        if coloring.min() < 0 or coloring.max() >= self.n_colors:
            return False
        if capacities is None:
            return True
        counts = np.bincount(coloring, minlength=self.n_colors)
        return bool(np.all(counts <= capacities))

    def project(
        self,
        bitstring: NDArray[np.int_],
        capacities: NDArray[np.int_] | None = None,
    ) -> NDArray[np.int_]:
        """Wrap colors into [0, K) and rebalance over-capacity colors."""
        coloring = self.decode(bitstring) % self.n_colors
        if capacities is not None:
            counts = np.bincount(coloring, minlength=self.n_colors)
            for c in range(self.n_colors):
                if counts[c] <= capacities[c]:
                    continue
                excess = int(counts[c] - capacities[c])
                idx = np.flatnonzero(coloring == c)
                receivers = [
                    cc for cc in range(self.n_colors)
                    if counts[cc] < capacities[cc]
                ]
                for k in range(min(excess, len(idx))):
                    if not receivers:
                        break
                    coloring[idx[k]] = receivers[0]
                    counts[receivers[0]] += 1
                    if counts[receivers[0]] >= capacities[receivers[0]]:
                        receivers.pop(0)
                    counts[c] -= 1
        return self.encode(coloring)

    def feasible_states(self) -> Iterable[NDArray[np.int_]]:
        if self.n_tasks > 6 or self.n_colors > 6:
            raise NotImplementedError(
                "feasible_states only available for small instances "
                f"(got n_tasks={self.n_tasks}, n_colors={self.n_colors})"
            )
        from itertools import product
        for coloring in product(range(self.n_colors), repeat=self.n_tasks):
            yield self.encode(np.array(coloring, dtype=np.int_))
