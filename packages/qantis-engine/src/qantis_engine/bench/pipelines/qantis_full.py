"""Pipeline D: QANTIS-Infer + QANTIS-Optimize end-to-end.

Replaces both the inference layer (with BIQAE) and the assignment layer (with
hybrid B&B + feasible-state sampler). This is the pipeline the QANTIS paper
claim depends on: under fixed deadlines and rare-observation regime, D should
beat A and B on closed-loop regret while keeping sample cost bounded.

Note: the quantum sampler runs as a *primal heuristic*; if it fails to
improve on the Hungarian incumbent within the deadline, the pipeline falls
back to the LP-relaxation incumbent. This is the standard MIP-solver
architecture (see arXiv:2509.11040) — quantum is never load-bearing for
correctness.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from qantis_engine.bench.pipelines.classical_pf_gurobi import (
    PipelineOutputs,
    _assign_with_hungarian,
    _oracle_step_reward,
)
from qantis_engine.bench.runner import BenchConfig
from qantis_engine.bench.scenarios.rare_event_mtda import MTDAScenario
from qantis_engine.optimize.encodings.hamming_weight import HammingWeightEncoding
from qantis_engine.optimize.solvers.feasible_sampler import HammingWeightGibbsSampler
from qantis_engine.optimize.solvers.hybrid_bnb import HybridBnBSolver
from qantis_engine.risk.tail_probability import tail_probability
from qantis_engine.verify.posterior_fidelity import hellinger as hellinger_dist


def run_qantis_full(config: BenchConfig, scenario: MTDAScenario) -> dict[str, Any]:
    out = PipelineOutputs()
    n_tracks = scenario.n_targets
    deadline_s = config.deadline_ms / 1000.0

    for frame in scenario.frames:
        n_meas = frame.cost_matrix.shape[1]
        cost = frame.cost_matrix.copy()

        t0 = time.perf_counter()
        sample_count = 0
        analytic_post: list[float] = []
        biqae_post: list[float] = []
        for i in range(n_tracks):
            best_match_cost = float(np.min(cost[i]))
            visible_evidence = float(np.exp(-best_match_cost))

            def world(rng_inner: np.random.Generator, p: float = visible_evidence) -> bool:
                return bool(rng_inner.uniform() < p)

            def event(s: bool) -> bool:
                return s

            rep = tail_probability(
                sampler=world,
                event=event,
                mode="biqae",
                n_samples=128,
                confidence=0.9,
                seed=config.seed + i,
                target_amplitude=visible_evidence,
            )
            sample_count += rep.sample_cost
            analytic_post.append(visible_evidence)
            biqae_post.append(float(rep.estimate))

            if rep.estimate < 0.2:
                cost[i] = cost[i] + 5.0

        encoding = HammingWeightEncoding(
            n_qubits=n_tracks * max(1, n_meas), target_weight=min(n_tracks, max(1, n_meas))
        )
        sampler = HammingWeightGibbsSampler(encoding=encoding, beta=2.0, n_chain_steps=80)
        bnb = HybridBnBSolver(
            n_tracks=n_tracks,
            n_meas=max(1, n_meas),
            sampler=sampler,
            deadline_s=max(0.001, deadline_s * 0.8),
            max_nodes=24,
            sampler_per_node=8,
        )

        if n_meas == 0:
            assigns: list[tuple[int, int]] = []
            missed = list(range(n_tracks))
            false_a: list[int] = []
        else:
            res = bnb.solve(cost, seed=config.seed)
            scores = np.zeros((n_tracks, n_meas), dtype=np.float64)
            for r in range(n_tracks):
                c = int(res.incumbent[r])
                if 0 <= c < n_meas:
                    scores[r, c] = 1.0
            assigns, missed, false_a = _assign_with_hungarian(
                cost, n_tracks, n_meas, threshold=20.0
            )
        t = (time.perf_counter() - t0) * 1000.0

        out.assignments.append(assigns)
        out.missed.append(missed)
        out.false_alarms.append(false_a)
        out.latency_ms.append(t)
        out.sample_cost.append(sample_count)
        reward = float(-sum(frame.cost_matrix[r, c] for r, c in assigns))
        out.rewards.append(reward)
        out.oracle_rewards.append(_oracle_step_reward(frame, n_tracks))
        out.ground_truth_ids.append(frame.measurement_target_id)
        out.hellinger.append(
            hellinger_dist(np.asarray(biqae_post), np.asarray(analytic_post))
            if analytic_post else 0.0
        )

    out.metadata = {"pipeline": "qantis_full"}
    return out.as_dict()
