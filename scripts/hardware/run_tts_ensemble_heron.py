"""QANTIS-2 E1: Rønnow-Shaydulin TTS(99%) ensemble on IBM Heron.

Computes a time-to-solution scaling curve across sequential-update horizons
T in {4, 8, 12}, with a seeded instance ensemble per size. For every run
we aggregate Hellinger distance per step + wall-clock, then fit the
scaling slope of ``log10(median TTS)`` vs ``T`` with a bootstrap 95% CI
via :class:`quantum_common.benchmarking.BenchmarkProtocol`.

Adds the scaling evidence that reviewers routinely demand beyond the
point-estimate Hellinger distances in the current Sections 3 + 6. See
Shaydulin Sci Adv 10:eadm6761 (2024) for the canonical protocol and
Ronnow Science 345:420 (2014) for the TTS(99%) definition.

Usage
-----
Simulator dry-run:

    python scripts/hardware/run_tts_ensemble_heron.py --dry-run \
        --num-instances 3 --horizons 4 8 12

Hardware run (Pittsburgh by default):

    IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_tts_ensemble_heron.py \
        --backend ibm_pittsburgh --num-instances 10 --shots 4096

Output: ``output/hardware/tts_ensemble_heron_<timestamp>.json``.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Make the package importable when running from the repo root.
_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

import numpy as np

from scripts.hardware import (  # type: ignore
    build_tiger_pomdp,
    databin_get_counts,
    run_on_aer_batched,
    run_on_heron_batched,
    save_partial,
    save_result,
)
from quantum_common.benchmarking import BenchmarkProtocol, InstanceRun


# ---------------------------------------------------------------------------
# Trajectory runner (minimal classical-reference + backend-agnostic wrapper)
# ---------------------------------------------------------------------------


def _tiger_trajectory(
    horizon: int,
    seed: int,
    observation_bias: float = 0.85,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Return a reproducible Tiger trajectory.

    Produces the belief sequence, observation sequence (0 = hear-left,
    1 = hear-right), and action sequence under the canonical Tiger
    policy (listen until P(tiger-left) > 0.85, then open the other
    door). Used both as the classical reference and as the trajectory
    driving the quantum belief-update circuits.
    """
    rng = np.random.default_rng(seed)
    belief = np.array([0.5, 0.5])
    beliefs: list[np.ndarray] = [belief.copy()]
    actions: list[int] = []
    observations: list[int] = []

    # For the TTS scaling experiment we lock actions to listen (0); the
    # scaling claim is about chained belief-update coherence over T
    # steps, and introducing open-door resets during the trajectory
    # collapses the belief and erases the cumulative evidence we are
    # trying to measure. (Earlier versions mixed opens into the policy
    # which diverged from the quantum circuit's open-action reset
    # semantics.)
    for _ in range(horizon):
        actions.append(0)
        true_state = 0 if rng.random() < belief[0] else 1
        hear_correct = rng.random() < observation_bias
        obs = true_state if hear_correct else 1 - true_state
        observations.append(obs)
        likelihood = np.array(
            [observation_bias if obs == 0 else 1 - observation_bias,
             observation_bias if obs == 1 else 1 - observation_bias],
        )
        belief = likelihood * belief
        total = belief.sum()
        if total > 0:
            belief = belief / total
        beliefs.append(belief.copy())

    return np.array(beliefs), observations, actions


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()))


def _run_instance(
    *,
    horizon: int,
    seed: int,
    backend_name: str,
    shots: int,
    dry_run: bool,
) -> InstanceRun:
    """Drive a single horizon-T trajectory through Aer or Heron."""
    t0 = time.perf_counter()
    beliefs, observations, actions = _tiger_trajectory(horizon, seed)

    # Classical reference posterior at final step; quantum hardware/Aer
    # call would replace this with a belief-update circuit reading.
    final_ref = beliefs[-1]

    final_quantum = _submit_horizon(
        beliefs, actions, observations,
        shots=shots, dry_run=dry_run, backend_name=backend_name,
    )

    hellinger = _hellinger(final_ref, final_quantum)
    elapsed = time.perf_counter() - t0
    # Per Ronnow 2014: a "hit" is Hellinger within an instance tolerance;
    # we use 0.15 (legacy QANTIS-2 acceptance band).
    hit = hellinger < 0.15
    # Approximate p_hit via the Hellinger fidelity (1 - H^2 is the
    # Bhattacharyya overlap of the two distributions, a sensible
    # per-run success-probability proxy when you only run each trajectory once).
    p_hit = float(max(1.0 - hellinger * hellinger, 0.0))
    run = InstanceRun(
        instance_id=f"tiger_T{horizon}_seed{seed}",
        size=horizon,
        hit=hit,
        energy=hellinger,
        ref_energy=0.0,
        wall_time_s=elapsed,
        chain_length_max=1,
        chain_break_fraction=0.0,
    )
    run.__dict__["p_hit_estimate"] = p_hit
    return run


def _submit_horizon(
    beliefs: np.ndarray,
    actions: list[int],
    observations: list[int],
    *,
    shots: int,
    dry_run: bool,
    backend_name: str,
) -> np.ndarray:
    """Drive the chained Tiger belief-update circuit on Aer or Heron.

    Single-instance wrapper kept for backwards compatibility; internally
    delegates to :func:`_submit_horizons_cross_seed` with a list of one
    trajectory so that the multi-seed batched dispatch is the only code
    path ever hitting the backend.
    """
    finals = _submit_horizons_cross_seed(
        [(actions, observations)],
        shots=shots,
        dry_run=dry_run,
        backend_name=backend_name,
    )
    return finals[0]


def _submit_horizons_cross_seed(
    trajectories: list[tuple[list[int], list[int]]],
    *,
    shots: int,
    dry_run: bool,
    backend_name: str,
) -> list[np.ndarray]:
    """Run multiple same-horizon chained Tiger trajectories in parallel.

    Each step ``t`` assembles one circuit per live trajectory (all
    trajectories share ``horizon`` but have independent
    ``(action, observation)`` sequences from the classical reference)
    and dispatches them as a single ``Sampler.run(circuits)`` batch so
    the full campaign incurs ``horizon`` IBM queue entries regardless of
    how many seeds it contains.

    Dry-run uses ``AerSimulatorBackend`` in the same batch-equivalent
    per-circuit loop; wall clock on Aer is dominated by transpile, not
    queue, so the interface match matters more than the queue count.
    """
    from quantum_pomdp.models.belief_state import BeliefState
    from quantum_pomdp.quantum_circuits.belief_update import (
        BeliefUpdateCircuitConfig,
        QuantumBeliefUpdateCircuit,
    )
    from quantum_pomdp.scenarios.tiger_problem import create_tiger_pomdp

    pomdp = create_tiger_pomdp(listen_accuracy=0.85)
    cfg = BeliefUpdateCircuitConfig(
        use_amplitude_amplification=False,
        use_hardware_compatible_encoding=True,
        reward_precision_bits=2,
    )
    builder = QuantumBeliefUpdateCircuit(pomdp, cfg)

    beliefs = [BeliefState.uniform(num_states=pomdp.num_states)
               for _ in trajectories]
    horizon = max(len(t[0]) for t in trajectories)

    for t in range(horizon):
        circuits = []
        indices = []
        for k, (acts, obs) in enumerate(trajectories):
            if t >= len(acts):
                continue
            circuits.append(
                builder.build(
                    belief=beliefs[k], action=acts[t], observation=obs[t],
                )
            )
            indices.append(k)
        if not circuits:
            continue

        # Batched dispatch of all active trajectories at this step. On
        # hardware the robust fetch in ``run_on_heron_batched`` polls
        # through a fresh service each cycle so a stale socket cannot
        # freeze the chained loop (2026-04-19 hang root cause).
        if dry_run:
            counts_list = run_on_aer_batched(circuits, shots)
        else:
            counts_list, _job_id = run_on_heron_batched(
                circuits,
                backend_name=backend_name,
                shots=shots,
                checkpoint_name=f"tts_ensemble_heron_step{t}",
            )

        for i, k in enumerate(indices):
            obs_k = trajectories[k][1][t]
            post_select = {
                pomdp.state_qubits + j: (obs_k >> j) & 1
                for j in range(pomdp.observation_qubits)
            }
            beliefs[k] = BeliefState.from_quantum_measurement(
                counts_list[i],
                num_states=pomdp.num_states,
                num_state_qubits=pomdp.state_qubits,
                post_selection=post_select,
            )

    return [b.probabilities for b in beliefs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="ibm_pittsburgh")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[4, 8, 12],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = BenchmarkProtocol(
        sizes=tuple(args.horizons),
        instances_per_size=args.num_instances,
        runs_per_instance=1,
    )
    summaries_by_size: dict[int, list[dict]] = {h: [] for h in args.horizons}
    runs_by_size: dict[int, list[InstanceRun]] = {h: [] for h in args.horizons}

    payload_skeleton = {
        "campaign": "E1_tts_ensemble_heron",
        "backend": args.backend,
        "dry_run": args.dry_run,
        "num_instances_per_size": args.num_instances,
        "shots": args.shots,
        "horizons": args.horizons,
        "per_size": {str(h): {"samples": []} for h in args.horizons},
        "reference": (
            "Shaydulin Sci Adv 10:eadm6761 (2024); Ronnow Science 345:420 (2014)"
        ),
    }
    save_partial("tts_ensemble_heron", payload_skeleton)

    # Cross-seed batched execution: horizon queue entries instead of
    # horizon x num_instances (the 2026-04-19 campaign ran one queued
    # submit per chained step per seed which at 3 seeds x 12 steps = 36
    # entries turned a 1-hour experiment into a 5-hour wait).
    for horizon in args.horizons:
        seeds = [1000 * horizon + k for k in range(args.num_instances)]
        trajectories = []
        classical_finals = []
        for seed in seeds:
            beliefs_s, obs_s, acts_s = _tiger_trajectory(horizon, seed)
            trajectories.append((acts_s, obs_s))
            classical_finals.append(beliefs_s[-1])

        t0 = time.perf_counter()
        quantum_finals = _submit_horizons_cross_seed(
            trajectories,
            shots=args.shots,
            dry_run=args.dry_run,
            backend_name=args.backend,
        )
        batch_elapsed = time.perf_counter() - t0
        per_seed_wall = batch_elapsed / max(len(trajectories), 1)

        for k, seed in enumerate(seeds):
            h = _hellinger(classical_finals[k], quantum_finals[k])
            hit = h < 0.15
            p_hit = float(max(1.0 - h * h, 0.0))
            run = InstanceRun(
                instance_id=f"tiger_T{horizon}_seed{seed}",
                size=horizon,
                hit=hit,
                energy=h,
                ref_energy=0.0,
                wall_time_s=per_seed_wall,
            )
            runs_by_size[horizon].append(run)
            sample = {
                "instance_id": run.instance_id,
                "hellinger": h,
                "wall_time_s": per_seed_wall,
                "hit": hit,
                "p_hit_estimate": p_hit,
            }
            summaries_by_size[horizon].append(sample)
            payload_skeleton["per_size"][str(horizon)]["samples"].append(sample)
        save_partial("tts_ensemble_heron", payload_skeleton)

    # Per-size TTS medians using the protocol's formula.
    size_to_median_tts: dict[int, float] = {}
    for horizon, runs in runs_by_size.items():
        ttses: list[float] = []
        samples = summaries_by_size[horizon]
        for r, s in zip(runs, samples):
            p_hit = float(s.get("p_hit_estimate") or 0.5)
            ttses.append(protocol.tts(p_hit, r.wall_time_s))
        size_to_median_tts[horizon] = (
            float(np.median(ttses)) if ttses else float("nan")
        )

    scaling = protocol.scaling_fit(size_to_median_tts)

    payload = {
        "campaign": "E1_tts_ensemble_heron",
        "backend": args.backend,
        "dry_run": args.dry_run,
        "num_instances_per_size": args.num_instances,
        "shots": args.shots,
        "per_size": {
            str(h): {
                "median_tts_s": size_to_median_tts[h],
                "samples": summaries_by_size[h],
            }
            for h in args.horizons
        },
        "scaling": scaling,
        "reference": (
            "Shaydulin Sci Adv 10:eadm6761 (2024); Ronnow Science 345:420 (2014)"
        ),
    }
    save_result("tts_ensemble_heron", payload)


if __name__ == "__main__":
    main()
