"""Pipeline C: QANTIS-Infer (BIQAE) + Hungarian assignment.

Replaces the classical particle/SMC filter with a BIQAE-based posterior
update on per-target visibility (the rare-event regime where BIQAE has its
quadratic sample-efficiency advantage). The downstream assignment is
Hungarian — same as the deterministic-optimisation baselines — so any
metric improvement in this pipeline isolates the contribution of the
inference layer.
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
from qantis_engine.risk.tail_probability import tail_probability
from qantis_engine.verify.posterior_fidelity import hellinger as hellinger_dist


def run_qantis_infer_gurobi(
    config: BenchConfig, scenario: MTDAScenario
) -> dict[str, Any]:
    out = PipelineOutputs()
    n_tracks = scenario.n_targets
    p_miss_seed = scenario.p_miss

    target_visibility_belief = np.full(n_tracks, 1.0 - p_miss_seed)

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
            target_visibility_belief[i] = float(rep.estimate)
            analytic_post.append(visible_evidence)
            biqae_post.append(float(rep.estimate))

            if rep.estimate < 0.2:
                cost[i] = cost[i] + 5.0

        assigns, missed, false_a = _assign_with_hungarian(cost, n_tracks, n_meas)
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

    out.metadata = {"pipeline": "qantis_infer_gurobi"}
    return out.as_dict()
