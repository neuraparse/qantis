"""Compute GNN classical baseline on the exact hardware QUBO instance.

Evaluates Hungarian (optimal) and GNN (greedy) solvers on the same
N=2, M=3, seed=42 instance used in the FPC-QAOA hardware experiments.
Reports full QUBO objectives so results are directly comparable to
hardware FPC-QAOA reported in the paper (64.1% ± 3.3% of optimal).

Run from the repo root:
    python scripts/compute_gnn_baseline.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure packages are on the path when run from repo root
repo_root = Path(__file__).resolve().parents[1]
for pkg in ["quantum-common", "quantum-mht", "quantum-pomdp"]:
    src = repo_root / "packages" / pkg / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

sys.path.insert(0, str(repo_root / "scripts"))

from hardware import make_qubo_instance
from quantum_mht.solvers.classical_solvers.hungarian_solver import HungarianSolver
from quantum_mht.solvers.classical_solvers.gnn_solver import GNNSolver


def evaluate_full_qubo(qubo_result, assignments, missed, false_alarms):
    """Evaluate the full QUBO objective for a given solution.

    For a feasible solution, penalty terms are zero (all constraints satisfied).
    Only diagonal entries for active (=1) variables contribute.
    This matches how FPC-QAOA hardware objectives are computed.
    """
    variables = qubo_result.variables
    active = {}
    for i, j in assignments:
        active[variables.var_index(i, j)] = 1
    for i in missed:
        if variables.include_missed:
            active[variables.var_index(i, -1)] = 1
    for j in false_alarms:
        if variables.include_false_alarm:
            active[variables.var_index(-1, j)] = 1
    return sum(qubo_result.Q.get((k, k), 0.0) for k in active)


def main():
    print("=" * 60)
    print("GNN Baseline vs Hungarian on Hardware QUBO Instance")
    print("N=2 tracks, M=3 measurements, seed=42")
    print("(Exact instance used in FPC-QAOA hardware experiments)")
    print("=" * 60)

    # Build exact hardware instance
    qubo = make_qubo_instance(n_tracks=2, n_meas=3, seed=42)
    variables = qubo.variables
    print(f"\nQUBO variables: {variables.num_variables}")
    print(f"  Assignment (2x3):       6")
    print(f"  Missed detections (N):  {variables.n_tracks}")
    print(f"  False alarms (M):       {variables.n_measurements}")

    # --- Hungarian (optimal) ---
    hungarian = HungarianSolver()
    h_result = hungarian.solve(qubo)
    h_obj = h_result.objective_value
    print(f"\nHungarian (optimal, O(n³)):")
    print(f"  Assignments:    {h_result.assignments}")
    print(f"  Missed:         {h_result.missed_detections}")
    print(f"  False alarms:   {h_result.false_alarms}")
    print(f"  QUBO objective: {h_obj:.4f}")

    # --- GNN (greedy, O(NM log NM)) ---
    gnn = GNNSolver()
    g_result = gnn.solve(qubo)
    # Fix: evaluate full QUBO objective (GNN solver only sums assignment costs,
    # missing false alarm and missed detection diagonal entries)
    g_obj_full = evaluate_full_qubo(
        qubo,
        g_result.assignments,
        g_result.missed_detections,
        g_result.false_alarms,
    )
    print(f"\nGNN (greedy, O(NM log NM)):")
    print(f"  Assignments:              {g_result.assignments}")
    print(f"  Missed:                   {g_result.missed_detections}")
    print(f"  False alarms:             {g_result.false_alarms}")
    print(f"  QUBO obj (solver):        {g_result.objective_value:.4f}  [assignment cost only]")
    print(f"  QUBO obj (full, correct): {g_obj_full:.4f}  [includes FA/MD terms]")

    # --- Comparison ---
    # For QUBO minimisation: lower is better. % of optimal = obj / h_obj * 100
    # Since both are negative, we use obj / h_obj * 100 directly.
    if abs(h_obj) > 1e-9:
        gnn_pct = g_obj_full / h_obj * 100.0
    else:
        gnn_pct = float("nan")

    print(f"\n{'─' * 60}")
    print(f"Hungarian optimal:           {h_obj:.4f} (100.0%)")
    print(f"GNN full objective:          {g_obj_full:.4f} ({gnn_pct:.1f}% of optimal)")
    print(f"FPC-QAOA hw (p=3, 3 runs):   mean 64.1% ± 3.3% of optimal")
    print(f"{'─' * 60}")

    # Assess whether GNN equals optimal (N=2 is trivial for greedy)
    if abs(g_obj_full - h_obj) < 0.01:
        print("\n✓ GNN = Hungarian (N=2, M=3 is trivial for greedy: correct)")
        print("  FPC-QAOA hardware (64.1%) is BELOW classical GNN (100%) baseline.")
        print("  Paper should report this gap honestly.")
    else:
        diff = g_obj_full - h_obj
        print(f"\nGNN suboptimal by {diff:.4f} ({100.0 - gnn_pct:.1f}% gap vs optimal)")

    return {
        "hungarian_obj": h_obj,
        "gnn_obj_full": g_obj_full,
        "gnn_pct_of_optimal": gnn_pct,
        "gnn_assignments": g_result.assignments,
        "gnn_false_alarms": g_result.false_alarms,
    }


if __name__ == "__main__":
    main()
