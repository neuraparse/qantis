"""VALIDATION-ROADMAP Tasks 1.2 + 1.6 — MTDA QUBO on D-Wave Advantage2.

Runs forward annealing and reverse annealing (warm-start) on D-Wave hardware
for the N=5 track, M=8 measurement scenario (53 QUBO variables).
Compares against the optimal Hungarian baseline.

Usage
-----
# Full hardware run (forward + reverse):
DWAVE_API_TOKEN=xxx python scripts/hardware/run_mtda_dwave.py

# Simulator-only dry run (no credentials):
python scripts/hardware/run_mtda_dwave.py --dry-run

# Custom problem size:
DWAVE_API_TOKEN=xxx python scripts/hardware/run_mtda_dwave.py \\
    --n-tracks 3 --n-meas 5 --num-reads 2000

Output
------
  output/hardware/mtda_dwave_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-mht" / "src"))

from scripts.hardware import make_qubo_instance, require_dwave_token, save_result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MTDA QUBO on D-Wave Advantage2")
    p.add_argument("--n-tracks", type=int, default=5, help="Number of tracks")
    p.add_argument("--n-meas", type=int, default=8, help="Number of measurements")
    p.add_argument("--num-reads", type=int, default=1000, help="D-Wave num_reads")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Use simulated annealing locally, skip D-Wave QPU",
    )
    p.add_argument(
        "--no-reverse", action="store_true",
        help="Skip reverse annealing (forward only)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header(args: argparse.Namespace) -> None:
    print(f"\n=== MTDA QUBO — D-Wave Advantage2 ===")
    print(f"  Tracks: {args.n_tracks}, Measurements: {args.n_meas}")
    print(f"  num_reads: {args.num_reads}")
    print(f"  Backend: {'SimulatedAnnealing (dry-run)' if args.dry_run else 'D-Wave Advantage2'}")
    print(f"  Reverse annealing: {not args.no_reverse}")


def _run_hungarian(qubo_result) -> tuple:
    from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver

    solver = HungarianSolver()
    result = solver.solve(qubo_result)
    return result


def _run_forward(qubo_result, num_reads: int, use_simulator: bool):
    from quantum_mht.solvers.annealing_solver import AnnealingSolver

    solver = AnnealingSolver(
        num_reads=num_reads,
        annealing_time_us=20.0,
        use_reverse_annealing=False,
        use_simulator=use_simulator,
    )
    return solver.solve(qubo_result), solver


def _run_reverse(qubo_result, num_reads: int, use_simulator: bool, warm_start_solution):
    from quantum_mht.solvers.annealing_solver import AnnealingSolver

    solver = AnnealingSolver(
        num_reads=num_reads,
        annealing_time_us=20.0,
        use_reverse_annealing=True,
        reverse_anneal_s_target=0.45,
        reverse_anneal_hold_us=10.0,
        reinitialize_state=True,
        use_simulator=use_simulator,
    )
    # Inject warm-start from forward solution
    if warm_start_solution is not None:
        solver._previous_solution = {
            var_idx: int(warm_start_solution[var_idx])
            for var_idx in range(qubo_result.num_variables)
        }
    return solver.solve(qubo_result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _print_header(args)

    # Build QUBO instance
    qubo = make_qubo_instance(n_tracks=args.n_tracks, n_meas=args.n_meas, seed=42)
    print(f"\n  QUBO variables: {qubo.num_variables}")
    print(f"  Penalty (lambda): {qubo.penalty:.4f}")

    # Hungarian baseline (classical optimum)
    hungarian = _run_hungarian(qubo)
    print(f"\n  Hungarian (optimal) objective: {hungarian.objective_value:.4f}")
    print(f"  Assignments: {hungarian.assignments}")

    use_sim = args.dry_run
    if not use_sim:
        require_dwave_token()

    # Forward annealing
    print(f"\n[forward] Running {'simulated' if use_sim else 'D-Wave hardware'} annealing ...")
    try:
        fwd_result, fwd_solver = _run_forward(qubo, args.num_reads, use_sim)
    except ModuleNotFoundError as exc:
        print(f"\n[error] Missing dependency: {exc}")
        print("  Install D-Wave Ocean SDK: pip install dwave-ocean-sdk")
        save_result("mtda_dwave", {"task_ids": ["1.2", "1.6"], "pass": False,
                                   "notes": f"Missing dependency: {exc}"})
        sys.exit(1)
    fwd_ratio = fwd_result.objective_value / max(abs(hungarian.objective_value), 1e-9)
    print(f"  Forward objective: {fwd_result.objective_value:.4f}")
    print(f"  Forward feasible : {fwd_result.is_feasible if hasattr(fwd_result, 'is_feasible') else 'N/A'}")
    print(f"  Approx ratio     : {fwd_ratio:.3f}")
    print(f"  Solve time       : {fwd_result.solve_time_s:.3f}s")

    result_data: dict = {
        "task_ids": ["1.2", "1.6"],
        "backend": "simulated_annealing" if use_sim else "dwave_advantage2",
        "n_tracks": args.n_tracks,
        "n_meas": args.n_meas,
        "num_variables": qubo.num_variables,
        "num_reads": args.num_reads,
        "hungarian_objective": hungarian.objective_value,
        "hungarian_assignments": hungarian.assignments,
        "forward": {
            "objective": fwd_result.objective_value,
            "approx_ratio": fwd_ratio,
            "solve_time_s": fwd_result.solve_time_s,
            "assignments": fwd_result.assignments,
        },
    }

    # Reverse annealing
    if not args.no_reverse:
        print(f"\n[reverse] Running reverse annealing with forward warm-start ...")
        rev_result = _run_reverse(
            qubo, args.num_reads, use_sim,
            fwd_solver._previous_solution
        )
        rev_ratio = rev_result.objective_value / max(abs(hungarian.objective_value), 1e-9)
        print(f"  Reverse objective: {rev_result.objective_value:.4f}")
        print(f"  Approx ratio     : {rev_ratio:.3f}")
        print(f"  Solve time       : {rev_result.solve_time_s:.3f}s")
        print(f"  Reverse <= Forward: {rev_result.objective_value <= fwd_result.objective_value}")

        result_data["reverse"] = {
            "objective": rev_result.objective_value,
            "approx_ratio": rev_ratio,
            "solve_time_s": rev_result.solve_time_s,
            "assignments": rev_result.assignments,
            "better_than_forward": rev_result.objective_value <= fwd_result.objective_value,
        }

    # Pass/fail
    fwd_pass = fwd_ratio < 2.0
    result_data["pass"] = fwd_pass
    result_data["notes"] = (
        f"Forward approx ratio {fwd_ratio:.3f} "
        f"{'< 2.0 PASS' if fwd_pass else '>= 2.0 FAIL'}"
    )

    save_result("mtda_dwave", result_data)
    print(f"\n  PASS: {fwd_pass}  (fwd_ratio={fwd_ratio:.3f}, threshold=2.0)")


if __name__ == "__main__":
    main()
