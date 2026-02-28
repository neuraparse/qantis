"""Base solver interface for Multi-Target Data Association (MTDA).

Defines the abstract solver interface and result data structure used by all
MTDA solvers in this package. The solver taxonomy includes:

Quantum Solvers:
    - AnnealingSolver: D-Wave quantum annealing (Advantage2, Zephyr topology)
      with optional reverse annealing warm-start (Ihara 2025).
    - QAOASolver: IBM Quantum QAOA via Qiskit v2.3 SamplerV2/EstimatorV2
      (Farhi, Goldstone & Gutmann, arXiv:1411.4028, 2014).
    - FPCQAOASolver: Fixed-parameter-count QAOA with smooth schedule functions
      (Saavedra-Pino et al., arXiv:2512.21181, Dec 2025).
    - HybridSolver: D-Wave LeapHybrid classical-quantum decomposition.

Classical Solvers (baselines):
    - HungarianSolver: Optimal 2D assignment, O(n^3) (Kuhn 1955).
    - GNNSolver: Global Nearest Neighbor greedy assignment (Blackman 1986).
    - JPDASolver: Joint Probabilistic Data Association (Bar-Shalom & Li 1995).
    - ReidMHTSolver: Classical MHT with Murty's K-best (Reid 1979, Murty 1968).

Academic References:
    Stollenwerk et al., arXiv:2110.08346, 2021 -- QUBO formulation for MTDA.
    Farhi, Goldstone & Gutmann, arXiv:1411.4028, 2014 -- original QAOA.
    Cai et al., Rev. Mod. Phys. 95, 045005, 2023 -- composable error mitigation
        for quantum solvers.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

@dataclass
class SolverResult:
    """Result from solving an MTDA problem.

    Contains the decoded assignment solution plus solver metadata.
    The is_feasible property checks that no track or measurement is
    double-assigned (satisfying the row/column constraints from
    arXiv:2110.08346, Sec IV).
    """
    assignments: list[tuple[int, int]]
    missed_detections: list[int]
    false_alarms: list[int]
    objective_value: float
    solve_time_s: float
    solver_name: str
    raw_solution: NDArray[np.int_] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_feasible(self) -> bool:
        tracks_assigned = [a[0] for a in self.assignments]
        meas_assigned = [a[1] for a in self.assignments]
        return len(tracks_assigned) == len(set(tracks_assigned)) and len(meas_assigned) == len(set(meas_assigned))

class MTDASolver(abc.ABC):
    """Abstract base class for MTDA solvers.

    All solvers accept a QUBOResult and return a SolverResult.
    This interface enables fair comparison between quantum and classical
    solvers on the same QUBO instance (see solver taxonomy in module docstring).
    """
    @abc.abstractmethod
    def solve(self, qubo_result: Any) -> SolverResult: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...
