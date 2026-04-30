"""QANTIS-2 scaling evidence: TTS(99%) slope across |S| in {2, 4, 8}.

Promotes the paper's scaling claim from "trend" to "evidence" by reporting
the Ronnow-Shaydulin (Sci Adv 10:eadm6761, 2024) bootstrap CI slope on
three Tiger-family instance families:

  * |S|=2: Classic Tiger POMDP (Kaelbling et al. 1998).
  * |S|=4: Corridor-Tiger-4 (graded accuracy, left/right door split at 2).
  * |S|=8: Corridor-Tiger-8 (graded accuracy, split at 4).

For every size we submit N instances (default 10), collect per-instance
median TTS(99%) on a fixed horizon length, and fit
``log10 TTS`` vs ``log2 |S|`` with 100-bootstrap 95% CI. A slope +/- CI
disjoint from zero is the headline scaling claim.

Output artifact: ``output/hardware/tts_scaling_s248_<timestamp>.json``
with ``sizes``, ``per_size_tts_seconds``, ``scaling_slope`` + CI,
``reference_line``, and the full configuration block for provenance.

Dry-run (simulator only):
    python scripts/hardware/run_tts_scaling_s248.py --dry-run

Hardware:
    IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_tts_scaling_s248.py \
        --backend ibm_pittsburgh --shots 4096 --num-instances 10
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

import numpy as np

from scripts.hardware import get_ibm_token_optional, save_result  # type: ignore
from quantum_pomdp.models.belief_state import BeliefState
from quantum_pomdp.quantum_circuits.belief_update import (
    BeliefUpdateCircuitConfig,
    QuantumBeliefUpdateCircuit,
)
from quantum_pomdp.scenarios.corridor_tiger_8state import (
    create_corridor_tiger_4state,
    create_corridor_tiger_8state,
)
from quantum_pomdp.scenarios.tiger_problem import create_tiger_pomdp


# ---------------------------------------------------------------------------
# Instance factories -- yield a (POMDPModel, seed) pair for one run
# ---------------------------------------------------------------------------

def _s2_instance(seed: int):
    rng = np.random.default_rng(seed)
    acc = float(np.clip(0.85 + rng.normal(0, 0.03), 0.6, 0.95))
    return create_tiger_pomdp(listen_accuracy=acc), seed


def _s4_instance(seed: int):
    rng = np.random.default_rng(seed)
    # Jitter the listen-accuracy gradient endpoints to create a family.
    a_max = float(np.clip(0.95 + rng.normal(0, 0.02), 0.85, 0.99))
    a_min = float(np.clip(0.05 + rng.normal(0, 0.02), 0.01, 0.15))
    from quantum_pomdp.scenarios.corridor_tiger_8state import (
        create_corridor_tiger_pomdp,
    )
    return create_corridor_tiger_pomdp(
        num_cells=4, accuracy_max=a_max, accuracy_min=a_min,
    ), seed


def _s8_instance(seed: int):
    rng = np.random.default_rng(seed)
    a_max = float(np.clip(0.95 + rng.normal(0, 0.015), 0.88, 0.99))
    a_min = float(np.clip(0.05 + rng.normal(0, 0.015), 0.01, 0.12))
    from quantum_pomdp.scenarios.corridor_tiger_8state import (
        create_corridor_tiger_pomdp,
    )
    return create_corridor_tiger_pomdp(
        num_cells=8, accuracy_max=a_max, accuracy_min=a_min,
    ), seed


INSTANCE_FACTORIES = {2: _s2_instance, 4: _s4_instance, 8: _s8_instance}


# ---------------------------------------------------------------------------
# One-instance TTS measurement
# ---------------------------------------------------------------------------

def _single_instance_tts(
    pomdp,
    backend,
    shots: int,
    horizon: int,
    target_hellinger: float = 0.05,
) -> dict:
    """Run ``horizon`` chained belief updates and report TTS(99%).

    TTS(99%) := runtime / log(1 - success_prob)^(-1) * log(1 - 0.99)
    where success_prob is the fraction of steps whose Hellinger to the
    classical posterior stays below ``target_hellinger``.
    """
    from quantum_common.backends.base import ExecutionRequest

    belief = BeliefState.uniform(num_states=pomdp.num_states)
    cfg = BeliefUpdateCircuitConfig(
        use_amplitude_amplification=False,
        use_hardware_compatible_encoding=False,
        reward_precision_bits=2,
    )
    builder = QuantumBeliefUpdateCircuit(pomdp, cfg)

    hellingers: list[float] = []
    start = time.perf_counter()

    for t in range(horizon):
        classical_next = belief.classical_update(
            action=0, observation=0,
            transition_tensor=pomdp.transition_tensor,
            observation_tensor=pomdp.observation_tensor,
        )
        qc = builder.build(belief=belief, action=0, observation=0)
        counts = backend.execute(
            ExecutionRequest(circuits=[qc], shots=shots)
        ).counts[0]
        post = {
            pomdp.state_qubits + i: 0
            for i in range(pomdp.observation_qubits)
        }
        quantum_next = BeliefState.from_quantum_measurement(
            counts,
            num_states=pomdp.num_states,
            num_state_qubits=pomdp.state_qubits,
            post_selection=post,
        )
        H = quantum_next.hellinger_distance(classical_next)
        hellingers.append(H)
        belief = quantum_next

    elapsed = time.perf_counter() - start
    successes = sum(1 for h in hellingers if h < target_hellinger)
    p_success = max(successes / horizon, 1e-6)
    if p_success >= 1.0 - 1e-9:
        tts99 = elapsed
    else:
        tts99 = elapsed * math.log(1 - 0.99) / math.log(1 - p_success)

    return {
        "horizon": horizon,
        "elapsed_s": elapsed,
        "tts99_s": tts99,
        "max_hellinger": float(max(hellingers)),
        "mean_hellinger": float(sum(hellingers) / len(hellingers)),
        "hellinger_trace": [float(h) for h in hellingers],
    }


# ---------------------------------------------------------------------------
# Bootstrap scaling slope
# ---------------------------------------------------------------------------

def _bootstrap_slope(
    sizes: list[int], tts_per_size: list[list[float]], n_boot: int = 100, rng_seed: int = 2024
) -> dict:
    """Fit log10(TTS) = slope * log2(|S|) + intercept; 95% bootstrap CI."""
    sizes_arr = np.array(sizes, dtype=float)
    x = np.log2(sizes_arr)
    slopes = []
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_boot):
        ys = []
        for s_idx, samples in enumerate(tts_per_size):
            picks = rng.choice(samples, size=len(samples), replace=True)
            ys.append(float(np.median(np.log10(picks))))
        m, _ = np.polyfit(x, ys, 1)
        slopes.append(float(m))
    slopes_arr = np.array(slopes)
    return {
        "median_slope": float(np.median(slopes_arr)),
        "ci95_low": float(np.quantile(slopes_arr, 0.025)),
        "ci95_high": float(np.quantile(slopes_arr, 0.975)),
        "bootstrap_samples": int(n_boot),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="QANTIS-2 TTS(99%) scaling evidence across |S| in {2, 4, 8}"
    )
    parser.add_argument("--backend", default="aer_simulator")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--num-instances", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 8])
    args = parser.parse_args()

    if args.dry_run or args.backend == "aer_simulator":
        from quantum_common.backends.simulator import AerSimulatorBackend

        backend = AerSimulatorBackend()
        backend_name = "aer_simulator"
    else:
        # Token lookup order: explicit env -> saved-account. IBMQuantumBackend
        # handles the fallback internally when token=None.
        from quantum_common.backends.ibm import IBMQuantumBackend

        token = get_ibm_token_optional()
        backend = IBMQuantumBackend(backend_name=args.backend, token=token)
        backend_name = args.backend

    payload = {
        "script": "run_tts_scaling_s248.py",
        "backend": backend_name,
        "shots": args.shots,
        "horizon": args.horizon,
        "num_instances": args.num_instances,
        "sizes": args.sizes,
        "methodology": "Ronnow-Shaydulin Sci Adv 10:eadm6761 (2024)",
        "per_size": {},
    }

    tts_samples_per_size: list[list[float]] = []
    for size in args.sizes:
        if size not in INSTANCE_FACTORIES:
            print(f"[tts_scaling] skipping size {size}: no instance factory")
            continue
        factory = INSTANCE_FACTORIES[size]
        instance_results = []
        size_samples: list[float] = []
        for i in range(args.num_instances):
            pomdp, seed = factory(seed=1000 * size + i)
            res = _single_instance_tts(
                pomdp=pomdp,
                backend=backend,
                shots=args.shots,
                horizon=args.horizon,
            )
            res["seed"] = int(seed)
            instance_results.append(res)
            size_samples.append(res["tts99_s"])
            print(
                f"[tts_scaling] |S|={size} seed={seed} TTS99={res['tts99_s']:.3f}s "
                f"meanH={res['mean_hellinger']:.4f}"
            )
        tts_samples_per_size.append(size_samples)
        payload["per_size"][str(size)] = {
            "median_tts_s": float(np.median(size_samples)),
            "min_tts_s": float(np.min(size_samples)),
            "max_tts_s": float(np.max(size_samples)),
            "instances": instance_results,
        }

    if len(tts_samples_per_size) >= 2:
        scaling = _bootstrap_slope(
            sizes=args.sizes[: len(tts_samples_per_size)],
            tts_per_size=tts_samples_per_size,
            n_boot=args.bootstrap,
        )
        scaling["ci95_disjoint_from_zero"] = bool(
            scaling["ci95_low"] > 0 or scaling["ci95_high"] < 0
        )
    else:
        scaling = {"note": "fewer than 2 sizes collected -- slope not fitted"}

    payload["scaling"] = scaling

    save_result("tts_scaling_s248", payload)
    print("[tts_scaling] scaling slope:", scaling)
    return 0


if __name__ == "__main__":
    sys.exit(main())
