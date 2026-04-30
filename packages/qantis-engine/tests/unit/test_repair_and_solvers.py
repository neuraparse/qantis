"""Tests for repair and solver modules."""
from __future__ import annotations

import numpy as np

from qantis_engine.optimize.encodings.hamming_weight import HammingWeightEncoding
from qantis_engine.optimize.repair import (
    AssignmentRepair,
    PermutationRepair,
    swap_search,
    two_opt,
)
from qantis_engine.optimize.solvers import (
    HammingWeightGibbsSampler,
    HybridBnBSolver,
)


def test_assignment_repair_produces_feasible() -> None:
    repair = AssignmentRepair(n_tracks=3, n_meas=4)
    scores = np.array(
        [[0.9, 0.1, 0.1, 0.1],
         [0.1, 0.8, 0.05, 0.05],
         [0.05, 0.05, 0.85, 0.05]],
        dtype=np.float64,
    )
    assigns, missed, false_a = repair.repair(scores=scores)
    track_set = {a[0] for a in assigns}
    meas_set = {a[1] for a in assigns}
    assert len(track_set) == len(assigns)
    assert len(meas_set) == len(assigns)
    assert (0, 0) in assigns
    assert (1, 1) in assigns
    assert (2, 2) in assigns


def test_permutation_repair_handles_duplicates() -> None:
    rep = PermutationRepair(n=5)
    raw = np.array([1, 1, 1, 1, 1], dtype=np.int_)
    out = rep.repair(raw)
    assert sorted(int(v) for v in out) == [0, 1, 2, 3, 4]


def test_two_opt_improves_simple_tour() -> None:
    cities = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    def cost(tour: np.ndarray) -> float:
        d = 0.0
        n = len(tour)
        for i in range(n):
            a = cities[tour[i]]
            b = cities[tour[(i + 1) % n]]
            d += float(np.linalg.norm(a - b))
        return d

    bad = np.array([0, 2, 1, 3], dtype=np.int_)
    improved = two_opt(bad, cost)
    assert cost(improved) <= cost(bad) + 1e-12


def test_swap_search_improves_assignment() -> None:
    weights = np.array([3.0, 1.0, 2.0])

    def cost(assignment: np.ndarray) -> float:
        return float(np.sum(weights[assignment.astype(np.int_)]))

    bad = np.array([0, 0, 0], dtype=np.int_)
    improved = swap_search(bad, cost)
    assert cost(improved) <= cost(bad) + 1e-12


def test_hamming_weight_gibbs_emits_feasible_samples() -> None:
    enc = HammingWeightEncoding(n_qubits=8, target_weight=3)
    sampler = HammingWeightGibbsSampler(encoding=enc, beta=1.0, n_chain_steps=50)

    def cost_fn(x: np.ndarray) -> float:
        weights = np.arange(8, dtype=np.float64)
        return float(weights @ x)

    res = sampler.sample(cost_fn=cost_fn, n_samples=4, seed=7)
    assert res.samples.shape == (4, 8)
    for x in res.samples:
        assert int(x.sum()) == 3
    assert res.energies.shape == (4,)


def test_hybrid_bnb_returns_feasible_assignment() -> None:
    cost_matrix = np.array(
        [[1.0, 5.0, 5.0],
         [5.0, 1.0, 5.0],
         [5.0, 5.0, 1.0]],
        dtype=np.float64,
    )
    enc = HammingWeightEncoding(n_qubits=9, target_weight=3)
    sampler = HammingWeightGibbsSampler(encoding=enc, beta=2.0, n_chain_steps=40)
    bnb = HybridBnBSolver(
        n_tracks=3, n_meas=3, sampler=sampler, deadline_s=0.5, max_nodes=4,
        sampler_per_node=4,
    )
    res = bnb.solve(cost_matrix=cost_matrix, seed=0)
    assert res.incumbent.shape == (3,)
    chosen = sorted(int(c) for c in res.incumbent if 0 <= c < 3)
    assert chosen == [0, 1, 2]
    assert res.incumbent_cost <= 3.0 + 1e-9
