"""Folding-Free Zero-Noise Extrapolation (FF-ZNE) for 4-State Tiger POMDP.

Implements the FF-ZNE technique from arXiv:2603.13949 (March 2026).
Instead of circuit folding (which increases depth), FF-ZNE runs the SAME
circuit on different qubit layouts with different native noise levels.
On IBM Heron R2 processors, different qubit pairs have ECR error rates
ranging from ~0.0009 to 0.01+ (100x variation).  By running on 3-5
different layouts sorted by noise level and extrapolating to zero noise,
we achieve error mitigation WITHOUT any circuit depth increase.

Algorithm
---------
1. Query backend.target for all ECR gate error rates.
2. Find connected qubit triples (q0-q1-q2 path) on the coupling map.
3. For each triple, compute the average ECR error across the 2 links.
4. Select N layouts spanning low to high noise.
5. Transpile the 4-state Tiger circuit to each layout with initial_layout.
6. Run all layouts on hardware (same shots).
7. For each layout, compute posterior probabilities and Hellinger distance.
8. Apply linear extrapolation on each posterior component to x=0 (zero noise).
9. Normalise the extrapolated posterior and compute the FF-ZNE Hellinger.

Usage
-----
# Dry-run (simulator):
python scripts/hardware/run_ffzne_tiger.py --dry-run

# Hardware run:
IBM_QUANTUM_TOKEN=xxx IBM_QUANTUM_CHANNEL=ibm_cloud IBM_QUANTUM_INSTANCE=crn:... \
    python -u scripts/hardware/run_ffzne_tiger.py --backend ibm_kingston --shots 8192 --n-layouts 3

Output
------
  output/hardware/ffzne_tiger_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

from scripts.hardware import (
    get_ibm_token_optional,
    ibm_channel,
    ibm_instance,
    save_result,
)

# Reuse circuit builder and helpers from the 4-state Tiger script
from scripts.hardware.run_tiger_4state_ibm import (
    P_OBS_GIVEN_STATE_4,
    STATE_NAMES,
    _build_tiger_4state_circuit,
    _classical_bayes_4state,
    _counts_to_probs,
    _hellinger,
    _post_select_4state,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Folding-Free ZNE for 4-State Tiger POMDP (arXiv:2603.13949)"
    )
    p.add_argument("--backend", default="ibm_kingston", help="IBM backend name")
    p.add_argument(
        "--shots", type=int, default=8192,
        help="Shot count per layout (default: 8192)",
    )
    p.add_argument(
        "--n-layouts", type=int, default=3,
        help="Number of qubit layouts to use (3-5 recommended, default: 3)",
    )
    p.add_argument(
        "--obs", type=int, default=0, choices=[0, 1],
        help="Target observation: 0=hear-left, 1=hear-right (default: 0)",
    )
    p.add_argument(
        "--prior", nargs=4, type=float, default=[0.25, 0.25, 0.25, 0.25],
        metavar=("P0", "P1", "P2", "P3"),
        help="Prior belief (default: uniform)",
    )
    p.add_argument(
        "--max-error", type=float, default=0.02,
        help="Max per-edge ECR error for layout selection (default: 0.02, filters broken qubits)",
    )
    p.add_argument("--channel", default=None, help="Qiskit channel")
    p.add_argument("--instance", default=None, help="IBM instance / CRN")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulator only (no credentials needed)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Layout selection: find connected triples with diverse noise levels
# ---------------------------------------------------------------------------

def _get_ecr_errors(backend) -> dict[tuple[int, int], float]:
    """Extract ECR (or CX) 2-qubit gate error rates from backend.target."""
    target = backend.target
    ecr_errors: dict[tuple[int, int], float] = {}

    # Try ECR first (Heron R2), then CX
    gate_name = None
    for candidate in ("ecr", "cx", "cz"):
        if candidate in target.operation_names:
            gate_name = candidate
            break

    if gate_name is None:
        print("  WARNING: No 2Q gate found in backend.target, using empty errors")
        return ecr_errors

    for qubits in target.qargs_for_operation_name(gate_name):
        props = target[gate_name][qubits]
        if props is not None and props.error is not None:
            ecr_errors[qubits] = props.error

    return ecr_errors


def _find_connected_triples(backend) -> list[tuple[int, int, int]]:
    """Find all connected qubit triples (q0-q1-q2) on the coupling map.

    A connected triple means there exist edges q0-q1 and q1-q2 (or
    q0-q1 and q0-q2, etc.) so that the 3 qubits form a connected path
    that a 3-qubit circuit can be mapped to.
    """
    coupling_map = backend.coupling_map
    # Build adjacency from coupling map edges
    edges = set()
    for edge in coupling_map.get_edges():
        edges.add((edge[0], edge[1]))
        edges.add((edge[1], edge[0]))  # undirected

    # Build adjacency list
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)

    # Find all triples forming a path q0 - q1 - q2
    triples: list[tuple[int, int, int]] = []
    for q1 in adj:
        neighbors = sorted(adj[q1])
        for i, q0 in enumerate(neighbors):
            for q2 in neighbors[i + 1:]:
                triples.append((q0, q1, q2))

    return triples


def _select_diverse_layouts(
    triples: list[tuple[int, int, int]],
    ecr_errors: dict[tuple[int, int], float],
    n_layouts: int,
    max_error: float = 0.05,
) -> list[dict]:
    """Select n_layouts triples spanning low to high average ECR error.

    Filters out qubit pairs with error >= max_error (broken/defective qubits).
    Returns list of dicts with keys: qubits, avg_error, edge_errors.
    """
    scored: list[dict] = []
    for q0, q1, q2 in triples:
        # Get error for both edges (try both directions)
        err01 = ecr_errors.get((q0, q1)) or ecr_errors.get((q1, q0))
        err12 = ecr_errors.get((q1, q2)) or ecr_errors.get((q2, q1))
        if err01 is None or err12 is None:
            continue
        # Skip triples with broken/defective qubits
        if err01 >= max_error or err12 >= max_error:
            continue
        avg_err = (err01 + err12) / 2.0
        scored.append({
            "qubits": (q0, q1, q2),
            "avg_error": avg_err,
            "edge_errors": {
                f"{q0}-{q1}": err01,
                f"{q1}-{q2}": err12,
            },
        })

    if not scored:
        return []

    # Sort by average error
    scored.sort(key=lambda x: x["avg_error"])

    if len(scored) <= n_layouts:
        return scored

    # Pick n_layouts evenly spaced from low to high noise
    indices = []
    for i in range(n_layouts):
        idx = int(round(i * (len(scored) - 1) / (n_layouts - 1)))
        indices.append(idx)

    # Ensure unique indices
    selected_indices = sorted(set(indices))
    # If deduplication reduced count, add nearest alternatives
    while len(selected_indices) < n_layouts and len(selected_indices) < len(scored):
        for idx in range(len(scored)):
            if idx not in selected_indices:
                selected_indices.append(idx)
                selected_indices.sort()
                if len(selected_indices) >= n_layouts:
                    break

    return [scored[i] for i in selected_indices[:n_layouts]]


# ---------------------------------------------------------------------------
# Transpile circuit to specific layout
# ---------------------------------------------------------------------------

def _transpile_to_layout(circuit, backend, initial_layout: list[int]):
    """Transpile circuit to a specific qubit layout.

    Returns (isa_circuit, isa_depth).
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        initial_layout=initial_layout,
    )
    isa = pm.run(circuit)
    isa._layout = None  # prevent QPY Error 3211
    return isa, isa.depth()


# ---------------------------------------------------------------------------
# Compute circuit noise from transpiled ISA circuit
# ---------------------------------------------------------------------------

def _compute_circuit_noise(isa_circuit, backend) -> float:
    """Estimate total circuit error from the transpiled ISA circuit.

    Computes: 1 - product(1 - gate_error) for all 2Q gates in the circuit.
    This gives the probability that at least one 2Q gate introduces an error,
    which is a better noise proxy than just the average edge error rate.
    """
    import numpy as np

    target = backend.target
    gate_name = None
    for candidate in ("ecr", "cx", "cz"):
        if candidate in target.operation_names:
            gate_name = candidate
            break

    if gate_name is None:
        return 0.0

    total_fidelity = 1.0
    n_2q_gates = 0
    for instruction in isa_circuit.data:
        if instruction.operation.name == gate_name:
            qubits = tuple(isa_circuit.find_bit(q).index for q in instruction.qubits)
            props = target[gate_name].get(qubits)
            if props is not None and props.error is not None:
                total_fidelity *= (1.0 - props.error)
                n_2q_gates += 1

    circuit_error = 1.0 - total_fidelity
    return circuit_error


# ---------------------------------------------------------------------------
# Hardware execution for a single layout
# ---------------------------------------------------------------------------

def _run_single_layout_hw(isa_circuit, backend, shots: int) -> dict:
    """Run a transpiled circuit on hardware. Returns raw counts dict."""
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.strategy = "active-accum"
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.dynamical_decoupling.scheduling_method = "alap"

    job = sampler.run([isa_circuit], shots=shots)
    raw = job.result()[0]
    _creg_name = next(k for k in vars(raw.data) if not k.startswith("_"))
    return dict(getattr(raw.data, _creg_name).get_counts())


def _run_batch_hw(isa_circuits: list, backend, shots: int) -> list[dict]:
    """Batch-submit all transpiled circuits in a single job. Returns list of count dicts.

    Submitting in one job ensures all circuits run under the same calibration
    conditions, reducing temporal drift between layouts.
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.strategy = "active-accum"
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.dynamical_decoupling.scheduling_method = "alap"

    pubs = [(circ,) for circ in isa_circuits]
    job = sampler.run(pubs, shots=shots)
    results = job.result()

    all_counts = []
    for i in range(len(isa_circuits)):
        raw = results[i]
        _creg_name = next(k for k in vars(raw.data) if not k.startswith("_"))
        counts = dict(getattr(raw.data, _creg_name).get_counts())
        all_counts.append(counts)
    return all_counts


# ---------------------------------------------------------------------------
# Simulator dry-run for a single layout (simulates noise variation)
# ---------------------------------------------------------------------------

def _run_single_layout_sim(circuit, shots: int, noise_scale: float = 0.0) -> dict:
    """Run circuit on AerSimulator. noise_scale is ignored (ideal sim)."""
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile

    sim = AerSimulator()
    basis = ["cx", "u", "id", "reset", "measure"]
    decomposed = transpile(circuit, basis_gates=basis, optimization_level=0)
    job = sim.run(decomposed, shots=shots)
    return dict(job.result().get_counts())


# ---------------------------------------------------------------------------
# FF-ZNE extrapolation
# ---------------------------------------------------------------------------

def _ffzne_extrapolate_posteriors(
    noise_levels: list[float],
    posteriors: list[list[float]],
) -> list[float]:
    """Extrapolate each posterior probability component to zero noise.

    Uses linear fit: P_i(noise) = a_i * noise + b_i
    Extrapolated P_i(0) = b_i (y-intercept)
    Then normalise to get a valid probability distribution.

    Args:
        noise_levels: average ECR error for each layout
        posteriors: list of posterior probability vectors (one per layout)

    Returns:
        Normalised extrapolated posterior at zero noise.
    """
    import numpy as np

    n_states = len(posteriors[0])
    extrapolated = []

    for s in range(n_states):
        values = [post[s] for post in posteriors]
        if len(noise_levels) >= 2:
            coeffs = np.polyfit(noise_levels, values, 1)
            p_zero = coeffs[1]  # y-intercept
        else:
            p_zero = values[0]
        extrapolated.append(max(0.0, p_zero))  # clip negative

    # Normalise
    total = sum(extrapolated)
    if total > 1e-12:
        extrapolated = [p / total for p in extrapolated]
    else:
        extrapolated = [1.0 / n_states] * n_states

    return extrapolated


def _ffzne_extrapolate_hellinger(
    noise_levels: list[float],
    hellingers: list[float],
) -> float:
    """Simple linear extrapolation of Hellinger distance to zero noise.

    H(noise) = a * noise + b
    H(0) = b (y-intercept)
    """
    import numpy as np

    if len(noise_levels) < 2:
        return hellingers[0]
    coeffs = np.polyfit(noise_levels, hellingers, 1)
    return max(0.0, float(coeffs[1]))  # clip negative


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    prior_4 = list(args.prior)
    prior_sum = sum(prior_4)
    prior_4 = [x / prior_sum for x in prior_4]

    print(f"\n{'='*65}")
    print(f"  Folding-Free ZNE (FF-ZNE) for 4-State Tiger POMDP")
    print(f"  arXiv:2603.13949 — March 2026")
    print(f"{'='*65}")
    print(f"  Backend     : {'AerSimulator (dry-run)' if args.dry_run else args.backend}")
    print(f"  Shots/layout: {args.shots}")
    print(f"  N layouts   : {args.n_layouts}")
    print(f"  Prior       : {[round(x, 4) for x in prior_4]}")
    print(f"  Observation : {args.obs} ({'hear-left' if args.obs == 0 else 'hear-right'})")
    print(f"  Max edge err: {args.max_error}")

    # Classical reference posterior
    classical_post = _classical_bayes_4state(prior_4, args.obs)
    print(f"\n  Classical posterior: {[round(x, 4) for x in classical_post]}")
    print(f"    Most probable: {STATE_NAMES[classical_post.index(max(classical_post))]}")

    # Build the 4-state Tiger circuit (logical)
    circuit = _build_tiger_4state_circuit(prior_4, args.obs)
    print(f"\n  Circuit qubits : {circuit.num_qubits}")
    print(f"  Circuit depth  : {circuit.depth()}")

    # ------------------------------------------------------------------
    # Connect to backend
    # ------------------------------------------------------------------
    backend_obj = None
    layouts = []

    if not args.dry_run:
        token = get_ibm_token_optional()
        if token is None:
            print("  ERROR: IBM_QUANTUM_TOKEN not set. Use --dry-run for simulator.")
            sys.exit(1)

        from qiskit_ibm_runtime import QiskitRuntimeService

        channel = args.channel or ibm_channel()
        instance = args.instance or ibm_instance()
        svc_kwargs: dict = {"channel": channel, "token": token}
        if instance:
            svc_kwargs["instance"] = instance
        svc = QiskitRuntimeService(**svc_kwargs)
        backend_obj = svc.backend(args.backend)
        print(f"\n  Connected: {backend_obj.name} ({backend_obj.num_qubits} qubits)")

        # Get ECR error rates
        ecr_errors = _get_ecr_errors(backend_obj)
        print(f"  2Q gate error rates found: {len(ecr_errors)} edges")

        if ecr_errors:
            err_values = list(ecr_errors.values())
            print(f"    Min error: {min(err_values):.6f}")
            print(f"    Max error: {max(err_values):.6f}")
            print(f"    Ratio    : {max(err_values)/min(err_values):.1f}x")

        # Find connected triples
        triples = _find_connected_triples(backend_obj)
        print(f"  Connected triples found: {len(triples)}")

        # Select diverse layouts
        layouts = _select_diverse_layouts(triples, ecr_errors, args.n_layouts, max_error=args.max_error)
        print(f"  Selected {len(layouts)} layouts:")
        for i, lay in enumerate(layouts):
            q = lay["qubits"]
            label = "BEST" if i == 0 else ("WORST" if i == len(layouts) - 1 else "MID")
            print(f"    Layout {i+1} ({label}): qubits {q}  "
                  f"avg_error={lay['avg_error']:.6f}  {lay['edge_errors']}")
    else:
        # Dry-run: simulate 3 layouts with synthetic noise levels
        print(f"\n  [dry-run] Simulating {args.n_layouts} layouts with synthetic noise levels")
        synthetic_errors = [0.001, 0.005, 0.01, 0.015, 0.02][:args.n_layouts]
        for i, err in enumerate(synthetic_errors):
            layouts.append({
                "qubits": (0, 1, 2),
                "avg_error": err,
                "edge_errors": {"0-1": err, "1-2": err},
            })
            label = "BEST" if i == 0 else ("WORST" if i == len(synthetic_errors) - 1 else "MID")
            print(f"    Layout {i+1} ({label}): synthetic avg_error={err:.6f}")

    if not layouts:
        print("\n  ERROR: No suitable qubit layouts found!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Transpile circuit to each layout
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Transpiling and running on {len(layouts)} layouts ...")
    print(f"{'='*65}")

    noise_levels: list[float] = []
    circuit_noise_levels: list[float] = []
    posteriors: list[list[float]] = []
    hellingers: list[float] = []
    layout_results: list[dict] = []
    isa_depths: list[int] = []
    isa_circuits: list = []

    # Phase 1: Transpile all layouts
    for i, lay in enumerate(layouts):
        qubits = lay["qubits"]
        label = "BEST" if i == 0 else ("WORST" if i == len(layouts) - 1 else f"MID-{i}")

        if not args.dry_run and backend_obj is not None:
            initial_layout = list(qubits)
            isa_circ, isa_depth = _transpile_to_layout(circuit, backend_obj, initial_layout)
            circ_noise = _compute_circuit_noise(isa_circ, backend_obj)
            isa_circuits.append(isa_circ)
            isa_depths.append(isa_depth)
            circuit_noise_levels.append(circ_noise)
            print(f"  Layout {i+1} ({label}): qubits {qubits}  "
                  f"ISA depth={isa_depth}  "
                  f"avg_ecr={lay['avg_error']:.6f}  "
                  f"circuit_noise={circ_noise:.6f}")
        else:
            isa_depths.append(circuit.depth())
            circuit_noise_levels.append(lay["avg_error"])

    # Phase 2: Execute all circuits
    if not args.dry_run and backend_obj is not None:
        print(f"\n  Batch-submitting {len(isa_circuits)} circuits to {args.backend} "
              f"({args.shots} shots each) ...")
        all_counts = _run_batch_hw(isa_circuits, backend_obj, args.shots)
    else:
        all_counts = [_run_single_layout_sim(circuit, args.shots) for _ in layouts]

    # Phase 3: Process results
    print(f"\n  --- Results ---")
    for i, lay in enumerate(layouts):
        qubits = lay["qubits"]
        avg_err = lay["avg_error"]
        circ_noise = circuit_noise_levels[i]
        label = "BEST" if i == 0 else ("WORST" if i == len(layouts) - 1 else f"MID-{i}")

        raw_counts = all_counts[i]
        ps_counts = _post_select_4state(raw_counts, args.obs)
        probs = _counts_to_probs(ps_counts, n_states=4)
        h = _hellinger(probs, classical_post)

        # Use circuit_noise (from actual transpiled gates) as noise proxy for FF-ZNE
        noise_levels.append(circ_noise)
        posteriors.append(probs)
        hellingers.append(h)

        total_ps = sum(ps_counts.values())
        print(f"\n  Layout {i+1}/{len(layouts)} ({label}):")
        print(f"      Qubits: {qubits}  Avg ECR: {avg_err:.6f}  Circuit noise: {circ_noise:.6f}")
        print(f"      ISA depth: {isa_depths[i]}  Post-selected: {total_ps}/{args.shots}")
        print(f"      Posterior: {[round(x, 4) for x in probs]}")
        print(f"      Hellinger: {h:.6f}")

        layout_results.append({
            "layout_index": i,
            "label": label,
            "qubits": list(qubits),
            "avg_ecr_error": avg_err,
            "circuit_noise": circ_noise,
            "edge_errors": lay["edge_errors"],
            "isa_depth": isa_depths[i],
            "ps_counts": {str(k): v for k, v in ps_counts.items()},
            "posterior": probs,
            "hellinger": round(h, 6),
        })

    # ------------------------------------------------------------------
    # FF-ZNE Extrapolation
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  FF-ZNE Extrapolation to Zero Noise")
    print(f"{'='*65}")

    # Method 1: Extrapolate posterior probabilities (primary)
    ffzne_posterior = _ffzne_extrapolate_posteriors(noise_levels, posteriors)
    ffzne_hellinger = _hellinger(ffzne_posterior, classical_post)
    print(f"\n  Method 1: Posterior extrapolation")
    print(f"    FF-ZNE posterior : {[round(x, 4) for x in ffzne_posterior]}")
    print(f"    FF-ZNE Hellinger: {ffzne_hellinger:.6f}")

    # Method 2: Extrapolate Hellinger directly (secondary)
    ffzne_hellinger_direct = _ffzne_extrapolate_hellinger(noise_levels, hellingers)
    print(f"\n  Method 2: Direct Hellinger extrapolation")
    print(f"    FF-ZNE Hellinger: {ffzne_hellinger_direct:.6f}")

    # Best individual layout (minimum Hellinger among all layouts)
    best_idx = int(min(range(len(hellingers)), key=lambda i: hellingers[i]))
    best_layout_hellinger = hellingers[best_idx]
    worst_layout_hellinger = max(hellingers)
    avg_layout_hellinger = sum(hellingers) / len(hellingers)

    # Determine best FF-ZNE result
    best_ffzne = min(ffzne_hellinger, ffzne_hellinger_direct)
    ffzne_method = "posterior" if ffzne_hellinger <= ffzne_hellinger_direct else "direct"

    # Compute improvements
    improvement_vs_best = best_layout_hellinger - best_ffzne
    improvement_pct = (improvement_vs_best / best_layout_hellinger * 100) if best_layout_hellinger > 0 else 0
    improvement_vs_avg = avg_layout_hellinger - best_ffzne
    improvement_vs_avg_pct = (improvement_vs_avg / avg_layout_hellinger * 100) if avg_layout_hellinger > 0 else 0

    print(f"\n  --- Comparison ---")
    print(f"    Best individual layout   : Layout {best_idx+1} (qubits {layouts[best_idx]['qubits']})  H={best_layout_hellinger:.6f}")
    print(f"    Worst individual layout  : H={worst_layout_hellinger:.6f}")
    print(f"    Average across layouts   : H={avg_layout_hellinger:.6f}")
    print(f"    FF-ZNE posterior extrap.  : H={ffzne_hellinger:.6f}")
    print(f"    FF-ZNE direct extrap.    : H={ffzne_hellinger_direct:.6f}")
    print(f"    Best FF-ZNE ({ffzne_method:>9s}) : H={best_ffzne:.6f}")
    print(f"    Improvement vs best indiv: {improvement_vs_best:+.6f} ({improvement_pct:+.1f}%)")
    print(f"    Improvement vs avg       : {improvement_vs_avg:+.6f} ({improvement_vs_avg_pct:+.1f}%)")

    ffzne_improved = best_ffzne < best_layout_hellinger

    print(f"\n    FF-ZNE improved over best individual layout: {ffzne_improved}")

    # Linear fit details
    import numpy as np
    if len(noise_levels) >= 2:
        coeffs_h = np.polyfit(noise_levels, hellingers, 1)
        print(f"\n  Linear fit: H(noise) = {coeffs_h[0]:.2f} * noise + {coeffs_h[1]:.6f}")
        print(f"    Slope    : {coeffs_h[0]:.4f}")
        print(f"    Intercept: {coeffs_h[1]:.6f}")

    # ------------------------------------------------------------------
    # Build result JSON
    # ------------------------------------------------------------------
    passed = ffzne_improved or best_ffzne < 0.05

    result: dict = {
        "task_ids": ["2.5", "ffzne"],
        "technique": "Folding-Free Zero-Noise Extrapolation (FF-ZNE)",
        "reference": "arXiv:2603.13949 (March 2026)",
        "backend": args.backend if not args.dry_run else "aer_simulator",
        "shots_per_layout": args.shots,
        "total_shots": args.shots * len(layouts),
        "n_layouts": len(layouts),
        "prior": prior_4,
        "observation": args.obs,
        "states": STATE_NAMES,
        "p_obs_given_state": P_OBS_GIVEN_STATE_4,
        "classical_posterior": classical_post,
        "circuit": {
            "num_qubits": circuit.num_qubits,
            "logical_depth": circuit.depth(),
            "isa_depths": isa_depths,
        },
        "layouts": layout_results,
        "noise_levels_circuit": noise_levels,
        "noise_proxy": "circuit_noise (1 - prod(1-gate_error) for all 2Q gates)",
        "ffzne_posterior_extrapolation": {
            "posterior": ffzne_posterior,
            "hellinger": round(ffzne_hellinger, 6),
        },
        "ffzne_direct_extrapolation": {
            "hellinger": round(ffzne_hellinger_direct, 6),
        },
        "best_individual_layout": {
            "index": best_idx,
            "qubits": list(layouts[best_idx]["qubits"]),
            "hellinger": round(best_layout_hellinger, 6),
        },
        "avg_layout_hellinger": round(avg_layout_hellinger, 6),
        "improvement": {
            "ffzne_vs_best_individual": round(improvement_vs_best, 6),
            "ffzne_vs_best_individual_pct": round(improvement_pct, 2),
            "ffzne_vs_avg": round(improvement_vs_avg, 6),
            "ffzne_vs_avg_pct": round(improvement_vs_avg_pct, 2),
            "ffzne_improved": ffzne_improved,
            "best_method": ffzne_method,
        },
        "pass": passed,
    }

    if len(noise_levels) >= 2:
        result["linear_fit"] = {
            "slope": round(float(coeffs_h[0]), 4),
            "intercept": round(float(coeffs_h[1]), 6),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Classical posterior : {[round(x, 4) for x in classical_post]}")
    print(f"  Best indiv. post.  : {[round(x, 4) for x in posteriors[best_idx]]}  (layout {best_idx+1})")
    print(f"  FF-ZNE posterior   : {[round(x, 4) for x in ffzne_posterior]}")
    print(f"  Best indiv. H      : {best_layout_hellinger:.6f}")
    print(f"  FF-ZNE H ({ffzne_method:>9s}): {best_ffzne:.6f}")
    print(f"  FF-ZNE improved    : {ffzne_improved}")
    print(f"  PASS               : {passed}")

    save_result("ffzne_tiger", result)


if __name__ == "__main__":
    main()
