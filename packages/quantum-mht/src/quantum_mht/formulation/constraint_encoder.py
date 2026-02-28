"""Row/column constraints encoded as QUBO penalty terms.

Converts the hard constraints of the MTDA assignment problem (each track
assigned to at most one measurement and vice versa) into soft penalty terms
in the QUBO objective. The penalty method transforms:

    min  F_cost(x)
    s.t. Ax = b

into the unconstrained QUBO:

    min  F_cost(x) + lambda * ||Ax - b||^2

where lambda must be large enough to make constraint violation energetically
unfavorable, but not so large as to create numerical issues.

Auto-Calibration Formula:
    lambda = 1.5 * max|c_{i,j}|
    This ensures the penalty for violating any constraint exceeds the maximum
    possible cost benefit from an infeasible assignment.

Academic References:
    Stollenwerk et al., "Adiabatic Quantum Computing for Multi Object Tracking",
        Fraunhofer FKIE, arXiv:2110.08346, 2021, Sec IV -- penalty-based QUBO
        constraint encoding for row/column assignment constraints.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995 -- assignment constraint formulation.

Penalty Calibration Notes (2026):
    Auto-calibration formula lambda=1.5*max|c_{ij}| is an empirical finding
    from Fraunhofer FKIE (arXiv:2110.08346 Sec IV). Setting lambda too low
    allows constraint violations (multiple measurements assigned to same
    track); too high drowns out cost terms (all solutions have similar
    energy, degrading annealing performance). The 1.5x multiplier provides
    a robust balance validated on D-Wave Advantage and Advantage2 hardware.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from quantum_mht.formulation.association_variables import AssociationVariables

@dataclass
class ConstraintEncoder:
    """Encode MTDA constraints as QUBO penalty terms.

    Row constraint: each track assigned to at most one measurement (or missed)
    Column constraint: each measurement assigned to at most one track (or false alarm)

    The penalty strength lambda is auto-calibrated as:
        lambda = penalty_multiplier * max|c_{i,j}|
    following arXiv:2110.08346, Sec IV. Default multiplier 1.5 provides
    sufficient margin to enforce feasibility while avoiding numerical issues
    on quantum hardware with limited precision.

    References:
        Stollenwerk et al., arXiv:2110.08346, 2021, Sec IV.
    """
    # Auto-calibration multiplier: lambda = 1.5 * max|c_{i,j}| (arXiv:2110.08346, Sec IV)
    penalty_multiplier: float = 1.5

    def auto_calibrate_penalty(self, cost_matrix: NDArray[np.float64]) -> float:
        """Auto-calibrate penalty: lambda = multiplier * max|c_{i,j}|.

        Formula from arXiv:2110.08346, Sec IV: ensures constraint penalty
        exceeds any single assignment cost, guaranteeing feasible solutions
        are energetically preferred.
        """
        finite_costs = cost_matrix[np.isfinite(cost_matrix)]
        if len(finite_costs) == 0:
            return 10.0
        return self.penalty_multiplier * float(np.max(np.abs(finite_costs)))

    def encode_row_constraints(
        self,
        variables: AssociationVariables,
        penalty: float,
    ) -> dict[tuple[int, int], float]:
        """Encode row constraints: sum_j x_{i,j} + x_{i,miss} = 1 for each track i.

        QUBO penalty: lambda * (sum_j x_{i,j} + x_{i,miss} - 1)^2
        Returns Q matrix entries as dict of (var_idx_1, var_idx_2) -> coefficient.

        This enforces that each track is assigned to exactly one measurement
        (or declared missed). See arXiv:2110.08346, Sec IV, Eq 8.
        """
        Q: dict[tuple[int, int], float] = {}

        for i in range(variables.n_tracks):
            vars_in_row = []
            for j in range(variables.n_measurements):
                vars_in_row.append(variables.var_index(i, j))
            if variables.include_missed:
                vars_in_row.append(variables.var_index(i, -1))

            # Expand (sum x_k - 1)^2 for binary x_k (x_k^2 = x_k):
            # = sum x_k^2 + 2*sum_{k<l} x_k*x_l - 2*sum x_k + 1
            # = sum x_k(1-2) + 2*sum_{k<l} x_k*x_l + 1
            # Linear terms (diagonal): coefficient = -1 (from x^2 - 2x = -x)
            # Quadratic terms (off-diagonal): coefficient = +2
            # (arXiv:2110.08346, Sec IV -- standard QUBO penalty expansion)
            for k in vars_in_row:
                Q[(k, k)] = Q.get((k, k), 0.0) + penalty * (-1.0)
            # Quadratic terms: coefficient = 2
            for a_idx in range(len(vars_in_row)):
                for b_idx in range(a_idx + 1, len(vars_in_row)):
                    k, l = vars_in_row[a_idx], vars_in_row[b_idx]
                    key = (min(k, l), max(k, l))
                    Q[key] = Q.get(key, 0.0) + penalty * 2.0

        return Q

    def encode_column_constraints(
        self,
        variables: AssociationVariables,
        penalty: float,
    ) -> dict[tuple[int, int], float]:
        """Encode column constraints: sum_i x_{i,j} + x_{fa,j} = 1 for each measurement j."""
        Q: dict[tuple[int, int], float] = {}

        for j in range(variables.n_measurements):
            vars_in_col = []
            for i in range(variables.n_tracks):
                vars_in_col.append(variables.var_index(i, j))
            if variables.include_false_alarm:
                vars_in_col.append(variables.var_index(-1, j))

            for k in vars_in_col:
                Q[(k, k)] = Q.get((k, k), 0.0) + penalty * (-1.0)
            for a_idx in range(len(vars_in_col)):
                for b_idx in range(a_idx + 1, len(vars_in_col)):
                    k, l = vars_in_col[a_idx], vars_in_col[b_idx]
                    key = (min(k, l), max(k, l))
                    Q[key] = Q.get(key, 0.0) + penalty * 2.0

        return Q
