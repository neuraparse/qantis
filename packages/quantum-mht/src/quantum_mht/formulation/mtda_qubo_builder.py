"""Main QUBO builder orchestrator for Multi-Target Data Association (MTDA).

Constructs the complete QUBO (Quadratic Unconstrained Binary Optimization)
matrix for the MTDA problem by combining:
    1. Cost terms from log-likelihood ratios (diagonal of Q)
    2. Row constraint penalties (each track -> one measurement or missed)
    3. Column constraint penalties (each measurement -> one track or false alarm)

The final QUBO objective is:
    F(x) = F_cost(x) + lambda_row * F_row(x) + lambda_col * F_col(x)

where F_cost encodes assignment costs, and F_row/F_col encode constraint
violations as quadratic penalties.

Qubit Count Scaling Table (with missed detection + false alarm slack vars):
    Tracks  Meas   Qubits (N*M + N + M)
    ------  ----   --------------------
      3       5     15 + 3 + 5  =  23
      5       8     40 + 5 + 8  =  53
     10      15    150 + 10 + 15 = 175
     20      30    600 + 20 + 30 = 650
     50      75   3750 + 50 + 75 = 3875

Advantage2 Feasibility (Zephyr 20-way, 4400+ qubits, May 2025 GA):
    N=10, M=15 -> 175 QUBO vars -> ~500 physical qubits -> FEASIBLE
    N=20, M=30 -> 650 QUBO vars -> ~2000 physical qubits -> FEASIBLE
    N=50, M=75 -> 3875 QUBO vars -> ~12000 physical qubits -> LeapHybrid required

LeapHybrid (D-Wave cloud): Supports up to 2M variables via classical-quantum
    decomposition. Suitable for large-scale tracking beyond direct QPU capacity.

Academic References:
    Stollenwerk et al., "Adiabatic Quantum Computing for Multi Object Tracking",
        Fraunhofer FKIE, arXiv:2110.08346, 2021 -- complete MTDA-to-QUBO
        formulation including cost matrix, constraint penalties, and variable
        encoding. Table 1 provides empirical qubit scaling analysis.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995 -- underlying assignment problem.
    McCormick et al., "Bayesian Diabatic Quantum Annealing for Multi-Target
        Tracking", arXiv:2209.00615, 2022 -- multi-hypothesis QUBO enumeration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray
from quantum_mht.formulation.association_variables import AssociationVariables
from quantum_mht.formulation.constraint_encoder import ConstraintEncoder
from quantum_mht.formulation.cost_matrix import CostMatrixBuilder

@dataclass
class QUBOResult:
    """Result of QUBO construction (arXiv:2110.08346, Sec III-IV)."""
    Q: dict[tuple[int, int], float]
    num_variables: int
    variables: AssociationVariables
    penalty: float
    offset: float = 0.0

    def to_bqm(self) -> Any:
        """Convert to dimod BinaryQuadraticModel.

        Creates a BQM compatible with D-Wave Ocean SDK 9.x samplers
        (Advantage2, 4400+ qubits, Zephyr topology) and dimod-based solvers.
        """
        import dimod
        linear = {}
        quadratic = {}
        for (i, j), val in self.Q.items():
            if i == j:
                linear[i] = linear.get(i, 0.0) + val
            else:
                quadratic[(i, j)] = quadratic.get((i, j), 0.0) + val
        return dimod.BinaryQuadraticModel(linear, quadratic, self.offset, dimod.BINARY)

@dataclass
class MTDAQuboBuilder:
    """Main QUBO builder for Multi-Target Data Association.

    Objective: F(x) = F_cost + lambda_row * F_row + lambda_col * F_col

    Orchestrates the full MTDA-to-QUBO pipeline:
        1. Build cost matrix from sensor data (CostMatrixBuilder)
        2. Create binary variables with slack (AssociationVariables)
        3. Auto-calibrate penalty strength (ConstraintEncoder)
        4. Assemble final QUBO matrix Q

    References:
        Stollenwerk et al., arXiv:2110.08346, 2021 (FKIE QUBO formulation).
        Bar-Shalom & Li, "Multitarget-Multisensor Tracking", YBS 1995.
    """
    cost_builder: CostMatrixBuilder | None = None
    constraint_encoder: ConstraintEncoder | None = None
    include_missed_detection: bool = True
    include_false_alarm: bool = True
    missed_detection_cost: float = 5.0
    false_alarm_cost: float = 3.0

    def __post_init__(self) -> None:
        if self.cost_builder is None:
            self.cost_builder = CostMatrixBuilder()
        if self.constraint_encoder is None:
            self.constraint_encoder = ConstraintEncoder()

    def build_from_cost_matrix(
        self,
        cost_matrix: NDArray[np.float64],
        gate_mask: NDArray[np.bool_] | None = None,
    ) -> QUBOResult:
        """Build QUBO from a precomputed cost matrix.

        Args:
            cost_matrix: (N_tracks, N_meas) assignment costs
            gate_mask: (N_tracks, N_meas) True where assignment is feasible
        """
        n_tracks, n_meas = cost_matrix.shape
        if gate_mask is None:
            gate_mask = np.ones_like(cost_matrix, dtype=bool)

        variables = AssociationVariables(
            n_tracks=n_tracks,
            n_measurements=n_meas,
            include_missed=self.include_missed_detection,
            include_false_alarm=self.include_false_alarm,
        )

        # Auto-calibrate penalty: lambda = 1.5 * max|c_{i,j}| (arXiv:2110.08346, Sec IV)
        penalty = self.constraint_encoder.auto_calibrate_penalty(cost_matrix)

        Q: dict[tuple[int, int], float] = {}

        # Objective: cost terms c_{i,j} on QUBO diagonal (arXiv:2110.08346, Eq 3-5)
        for i in range(n_tracks):
            for j in range(n_meas):
                if gate_mask[i, j]:
                    idx = variables.var_index(i, j)
                    Q[(idx, idx)] = Q.get((idx, idx), 0.0) + cost_matrix[i, j]

        # Missed detection costs
        if self.include_missed_detection:
            for i in range(n_tracks):
                idx = variables.var_index(i, -1)
                Q[(idx, idx)] = Q.get((idx, idx), 0.0) + self.missed_detection_cost

        # False alarm costs
        if self.include_false_alarm:
            for j in range(n_meas):
                idx = variables.var_index(-1, j)
                Q[(idx, idx)] = Q.get((idx, idx), 0.0) + self.false_alarm_cost

        # Row constraints: each track -> exactly one measurement or missed (arXiv:2110.08346, Eq 8)
        row_Q = self.constraint_encoder.encode_row_constraints(variables, penalty)
        for key, val in row_Q.items():
            Q[key] = Q.get(key, 0.0) + val

        # Column constraints: each measurement -> exactly one track or false alarm (arXiv:2110.08346, Eq 9)
        col_Q = self.constraint_encoder.encode_column_constraints(variables, penalty)
        for key, val in col_Q.items():
            Q[key] = Q.get(key, 0.0) + val

        return QUBOResult(
            Q=Q,
            num_variables=variables.num_variables,
            variables=variables,
            penalty=penalty,
        )

    def build(
        self,
        predicted_states: NDArray[np.float64],
        measurements: NDArray[np.float64],
        covariances: NDArray[np.float64],
    ) -> QUBOResult:
        """Build QUBO from raw tracking data."""
        cost_matrix, gate_mask = self.cost_builder.build_with_gating_mask(
            predicted_states, measurements, covariances,
        )
        return self.build_from_cost_matrix(cost_matrix, gate_mask)
