"""Binary variable encoding for Multi-Target Data Association (MTDA).

Maps the MTDA assignment problem to binary decision variables x_{i,j} suitable
for QUBO formulation. Each variable represents whether track i is associated
with measurement j. Additional slack variables handle missed detections and
false alarms.

Qubit Scaling Analysis:
    For N tracks and M measurements, the total number of binary variables is:
        N*M  (assignment variables)
      + N    (missed detection slack variables, if enabled)
      + M    (false alarm slack variables, if enabled)
      = N*M + N + M  total QUBO variables (qubits)

    Example: 10 tracks x 15 measurements = 150 + 10 + 15 = 175 qubits.
    This scales as O(N*M), making qubit count the primary bottleneck for
    near-term quantum hardware.

Academic References:
    Stollenwerk et al., "Adiabatic Quantum Computing for Multi Object Tracking",
        Fraunhofer FKIE, arXiv:2110.08346, 2021, Sec III -- binary variable
        encoding for MTDA and QUBO variable layout.
    Bar-Shalom & Li, "Multitarget-Multisensor Tracking: Principles and
        Techniques", YBS Publishing 1995 -- assignment problem formulation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

@dataclass
class AssociationVariables:
    """Binary decision variables x_{i,j} for track-measurement association.

    Encodes the MTDA assignment problem as binary variables following the
    formulation in Stollenwerk et al., arXiv:2110.08346, Sec III.

    Variables:
    - x_{i,j}: track i assigned to measurement j (i=1..N, j=1..M)
    - x_{i,0}: track i has missed detection (slack variable)
    - x_{0,j}: measurement j is a false alarm (slack variable)

    Total qubits required: N*M + N + M (see module docstring for scaling).
    """
    n_tracks: int
    n_measurements: int
    include_missed: bool = True
    include_false_alarm: bool = True
    _var_map: dict[tuple[int, int], int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        idx = 0
        # Assignment variables x_{i,j} (arXiv:2110.08346, Sec III)
        for i in range(self.n_tracks):
            for j in range(self.n_measurements):
                self._var_map[(i, j)] = idx
                idx += 1
        # Missed detection slack variables x_{i, -1} (arXiv:2110.08346, Sec III)
        if self.include_missed:
            for i in range(self.n_tracks):
                self._var_map[(i, -1)] = idx
                idx += 1
        # False alarm slack variables x_{-1, j} (arXiv:2110.08346, Sec III)
        if self.include_false_alarm:
            for j in range(self.n_measurements):
                self._var_map[(-1, j)] = idx
                idx += 1

    @property
    def num_variables(self) -> int:
        return len(self._var_map)

    def var_index(self, track: int, measurement: int) -> int:
        """Get variable index for track-measurement pair."""
        return self._var_map[(track, measurement)]

    def var_label(self, track: int, measurement: int) -> str:
        """Get human-readable variable label."""
        if track == -1:
            return f"fa_{measurement}"
        if measurement == -1:
            return f"miss_{track}"
        return f"x_{track}_{measurement}"

    def decode_solution(self, solution: NDArray[np.int_]) -> dict[str, list]:
        """Decode binary solution vector to assignments."""
        assignments = []
        missed = []
        false_alarms = []

        for (i, j), idx in self._var_map.items():
            if idx < len(solution) and solution[idx] == 1:
                if i == -1:
                    false_alarms.append(j)
                elif j == -1:
                    missed.append(i)
                else:
                    assignments.append((i, j))

        return {
            "assignments": assignments,
            "missed_detections": missed,
            "false_alarms": false_alarms,
        }
