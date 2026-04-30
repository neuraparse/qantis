"""Tests for feasible-subspace encodings."""
from __future__ import annotations

import math

import numpy as np

from qantis_engine.optimize.encodings import (
    ColoredPermutationEncoding,
    DickeSubspaceEncoding,
    HammingWeightEncoding,
    PermutationOracleEncoding,
)


def test_hamming_weight_feasible_count() -> None:
    enc = HammingWeightEncoding(n_qubits=6, target_weight=3)
    assert enc.feasible_count == math.comb(6, 3)
    states = list(enc.feasible_states())
    assert len(states) == math.comb(6, 3)
    for x in states:
        assert int(x.sum()) == 3


def test_hamming_weight_project_increases_weight() -> None:
    enc = HammingWeightEncoding(n_qubits=8, target_weight=4)
    x = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    p = enc.project(x)
    assert int(p.sum()) == 4


def test_hamming_weight_project_with_scores_picks_high_score_zeros() -> None:
    enc = HammingWeightEncoding(n_qubits=6, target_weight=2)
    x = np.zeros(6, dtype=np.int_)
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.3, 0.2])
    p = enc.project_with_scores(x, scores)
    assert int(p.sum()) == 2
    assert p[1] == 1
    assert p[3] == 1


def test_hamming_weight_initial_state_is_dicke() -> None:
    enc = HammingWeightEncoding(n_qubits=4, target_weight=2)
    psi = enc.initial_state()
    assert psi.shape == (16,)
    norm = float(np.linalg.norm(psi))
    assert abs(norm - 1.0) < 1e-9
    weights = [bin(i).count("1") for i in range(16)]
    for i, w in enumerate(weights):
        if w != 2:
            assert abs(psi[i]) < 1e-12


def test_permutation_oracle_encode_decode_roundtrip() -> None:
    enc = PermutationOracleEncoding(n=4)
    perm = np.array([2, 0, 3, 1], dtype=np.int_)
    bits = enc.encode(perm)
    decoded = enc.decode(bits)
    assert np.array_equal(decoded, perm)
    assert enc.is_feasible(bits)


def test_permutation_oracle_project_repairs_collisions() -> None:
    enc = PermutationOracleEncoding(n=4)
    bits = np.zeros(enc.num_qubits(), dtype=np.int_)
    for t, value in enumerate([0, 0, 1, 2]):
        for b in range(enc.position_bits):
            bits[t * enc.position_bits + b] = (value >> b) & 1
    bits[-1] = 1
    repaired = enc.project(bits)
    decoded = enc.decode(repaired)
    assert sorted(int(v) for v in decoded) == [0, 1, 2, 3]


def test_colored_permutation_capacity_repair() -> None:
    enc = ColoredPermutationEncoding(n_tasks=6, n_colors=3)
    bits = enc.encode(np.array([0, 0, 0, 0, 1, 2], dtype=np.int_))
    capacities = np.array([2, 2, 2], dtype=np.int_)
    repaired = enc.project(bits, capacities=capacities)
    coloring = enc.decode(repaired)
    counts = np.bincount(coloring, minlength=3)
    assert bool(np.all(counts <= capacities))


def test_dicke_subspace_initial_state_norm_and_support() -> None:
    enc = DickeSubspaceEncoding(n_qubits=5, weight=3)
    psi = enc.initial_state()
    assert abs(float(np.linalg.norm(psi)) - 1.0) < 1e-9
    for i, amp in enumerate(psi):
        if abs(amp) > 1e-12:
            assert bin(i).count("1") == 3
