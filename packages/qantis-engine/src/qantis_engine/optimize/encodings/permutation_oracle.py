"""Valid-permutation oracle encoding (arXiv:2603.21283, Stasik & Fuchs, Mar 2026).

Encodes a permutation ``pi: [n] -> [n]`` directly using a time-register
architecture: position-t register holds the index pi(t) in binary. The
feasible set is exactly the n! permutations; an oracle gate flags valid
configurations (each index appears exactly once).

Qubit cost: ``n * ceil(log2(n))`` for the permutation; one ancilla for the
validity flag. Compared to the one-hot ``n^2`` qubit encoding, this saves a
factor of ``n / log2(n)`` qubits — at n=8 this is 24 vs 64 qubits.

Used by QANTIS-Optimize for assignment / routing / TSP-style problems where
permutation feasibility is exact (no missed-detection / false-alarm slack).
For MTDA with slack, use ColoredPermutationEncoding instead.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PermutationOracleEncoding:
    """Encoding for permutations of ``[0, n)`` using ``n * ceil(log2(n))`` qubits."""

    n: int

    @property
    def position_bits(self) -> int:
        return max(1, math.ceil(math.log2(self.n))) if self.n > 1 else 1

    def num_qubits(self) -> int:
        return self.n * self.position_bits + 1

    @property
    def feasible_count(self) -> int:
        return math.factorial(self.n)

    def encode(self, perm: NDArray[np.int_]) -> NDArray[np.int_]:
        """Encode permutation ``perm`` into the bit register layout."""
        if perm.shape != (self.n,):
            raise ValueError(f"expected shape ({self.n},), got {perm.shape}")
        if sorted(int(v) for v in perm) != list(range(self.n)):
            raise ValueError(f"not a permutation of [0, {self.n}): {perm}")
        bits = np.zeros(self.num_qubits(), dtype=np.int_)
        for t in range(self.n):
            value = int(perm[t])
            for b in range(self.position_bits):
                bits[t * self.position_bits + b] = (value >> b) & 1
        bits[-1] = 1
        return bits

    def decode(self, bitstring: NDArray[np.int_]) -> NDArray[np.int_]:
        """Decode bitstring -> integer array of length ``n`` (no validity check)."""
        out = np.zeros(self.n, dtype=np.int_)
        for t in range(self.n):
            value = 0
            for b in range(self.position_bits):
                value |= int(bitstring[t * self.position_bits + b]) << b
            out[t] = value
        return out

    def is_feasible(self, bitstring: NDArray[np.int_]) -> bool:
        decoded = self.decode(bitstring)
        if decoded.min() < 0 or decoded.max() >= self.n:
            return False
        return sorted(int(v) for v in decoded) == list(range(self.n))

    def project(self, bitstring: NDArray[np.int_]) -> NDArray[np.int_]:
        """Repair to nearest valid permutation by greedy slot reassignment."""
        decoded = self.decode(bitstring)
        n = self.n
        used: set[int] = set()
        out = np.full(n, -1, dtype=np.int_)
        for t in range(n):
            v = int(decoded[t]) % n
            if v in used:
                continue
            out[t] = v
            used.add(v)
        missing = [i for i in range(n) if i not in used]
        empties = [t for t in range(n) if out[t] == -1]
        for t, v in zip(empties, missing, strict=False):
            out[t] = v
        return self.encode(out)

    def feasible_states(self) -> Iterable[NDArray[np.int_]]:
        for perm in permutations(range(self.n)):
            yield self.encode(np.array(perm, dtype=np.int_))
