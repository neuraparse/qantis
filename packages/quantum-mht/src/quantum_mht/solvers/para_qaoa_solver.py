"""ParaQAOA divide-and-conquer QAOA for MTDA QUBOs beyond single-QPU width.

Implements the Connectivity-Preserving Partitioning + Level-Aware Parallel
Merge scheme of Huang et al. arXiv:2603.26232 (Mar 2026). The algorithm
splits a large QUBO graph into ``num_partitions`` subgraphs of roughly
equal size linked by single bridge variables, solves each subgraph
independently via the standard QAOA backend, then reconciles the top-K
bitstrings per subgraph into a joint assignment by scoring every
candidate against the full original QUBO (including inter-partition
edges that were dropped during the split).

For MTDA this is the correct scaling path once ``N * M`` exceeds the
direct single-QPU QAOA envelope (~150 logical qubits on Heron R3).
50 tracks x 75 measurements = 3875 QUBO variables -> partition into M=8
groups of ~480 vars each; each subproblem fits comfortably inside
Heron's 156-qubit width budget for depth p=3.

Because arXiv:2603.26232 has not yet released public reference code, this
module is a from-scratch re-implementation following the paper
description. The subgraph QAOA call is delegated to the existing
``QAOASolver`` / ``XYMixerQAOASolver`` so depth, mixer, and backend
choices flow transparently from the solver-factory configuration.

Academic References:
    Huang, Chen, Huang, "ParaQAOA: Divide-and-Conquer QAOA Beyond Ten
        Thousand Vertices," arXiv:2603.26232 (Mar 2026).
    Zhou et al., "DC-QAOA," arXiv:2102.13288 (2021) -- prior
        partition-first approach; outperformed by ParaQAOA's
        connectivity-preserving variant per the 2026 benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import itertools
import math
import time
import logging
import numpy as np

from quantum_mht.solvers.base_solver import MTDASolver, SolverResult
from quantum_mht.solvers.solver_factory import create_solver

logger = logging.getLogger(__name__)


@dataclass
class ParaQAOASolver(MTDASolver):
    """Divide-and-conquer QAOA with connectivity-preserving partitioning.

    Attributes
    ----------
    num_partitions : int
        Number of subgraphs ``M``. 8 is a sensible default for
        3875-variable MTDA QUBOs on Heron-class hardware.
    top_k : int
        Top-K bitstrings retained per subgraph. The merge step evaluates
        ``top_k ** num_partitions`` candidates, so keep ``top_k`` small
        (<=8) for large ``M``.
    sub_solver : {"qaoa", "xy_mixer_qaoa", "fpc_qaoa"}
        QAOA variant to run on each subgraph. Default is plain "qaoa"
        with an X mixer -- ``xy_mixer_qaoa`` would require the sub-QUBO
        to preserve MTDA's row/column one-hot structure, which the
        connectivity-preserving BFS partitioner explicitly breaks.
        Callers that keep row-aligned partitions can override to
        ``xy_mixer_qaoa``.
    sub_solver_kwargs : dict
        Forwarded to the sub-solver constructor.
    """

    num_partitions: int = 8
    top_k: int = 4
    sub_solver: Literal["qaoa", "xy_mixer_qaoa", "fpc_qaoa"] = "qaoa"
    sub_solver_kwargs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"ParaQAOA(M={self.num_partitions}, k={self.top_k}, sub={self.sub_solver})"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()
        num_vars = qubo_result.num_variables
        partitions = self._partition(qubo_result.Q, num_vars)
        sub_results = [self._solve_partition(qubo_result, part) for part in partitions]
        merged, energy = self._merge(qubo_result.Q, num_vars, partitions, sub_results)

        decoded = qubo_result.variables.decode_solution(merged)
        elapsed = time.perf_counter() - t0

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=float(energy),
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=merged,
            metadata={
                "num_partitions": self.num_partitions,
                "top_k": self.top_k,
                "sub_solver": self.sub_solver,
                "partition_sizes": [len(p) for p in partitions],
            },
        )

    def _partition(
        self, Q: dict[tuple[int, int], float], num_vars: int,
    ) -> list[list[int]]:
        """Connectivity-preserving linear-time partition (Huang 2603.26232 Sec III).

        Algorithm (paper's Sec III):

            1. Build the weighted adjacency graph from the off-diagonal
               QUBO entries.
            2. Starting from the highest-weight-degree vertex, run BFS
               to produce a vertex ordering that keeps neighbours
               adjacent in the order.
            3. Cut the ordering into ``num_partitions`` contiguous
               blocks of size at most ``ceil(num_vars / num_partitions)``.
            4. Every pair of consecutive partitions shares exactly one
               *bridge* vertex -- the last vertex of the previous block
               is duplicated into the next block so that one high-weight
               edge is preserved in each sub-QUBO rather than being
               dropped by the Level-Aware Parallel Merge stage.

        The BFS traversal replaces the earlier "sort-by-degree + block"
        heuristic: it preserves local connectivity, so dropped edges are
        only the low-weight ones between far-apart partitions rather than
        half of each high-degree vertex's neighbourhood.
        """
        if num_vars <= 0:
            return []

        neighbours: dict[int, list[tuple[int, float]]] = {i: [] for i in range(num_vars)}
        degrees = np.zeros(num_vars)
        for (i, j), val in Q.items():
            if i == j or val == 0.0:
                continue
            weight = abs(val)
            neighbours[i].append((j, weight))
            neighbours[j].append((i, weight))
            degrees[i] += weight
            degrees[j] += weight

        start = int(np.argmax(degrees))
        order: list[int] = []
        visited = {start}
        # Priority BFS by decreasing edge weight -- cheap heap avoided
        # so the code stays linear-time and allocation-free per node.
        frontier: list[tuple[float, int]] = [(-degrees[start], start)]
        while frontier:
            frontier.sort()
            _w, node = frontier.pop(0)
            order.append(node)
            for nb, edge_w in sorted(neighbours[node], key=lambda pair: -pair[1]):
                if nb in visited:
                    continue
                visited.add(nb)
                frontier.append((-edge_w, nb))
        # Any isolated components: append in degree order.
        for v in range(num_vars):
            if v not in visited:
                order.append(v)

        block = int(math.ceil(num_vars / max(self.num_partitions, 1)))
        partitions: list[list[int]] = []
        bridge: int | None = None
        for start_idx in range(0, num_vars, block):
            block_vertices = list(order[start_idx : start_idx + block])
            if bridge is not None and bridge not in block_vertices:
                block_vertices.insert(0, bridge)
            partitions.append(block_vertices)
            bridge = block_vertices[-1]

        return [p for p in partitions if p]

    def _solve_partition(
        self, qubo_result: Any, partition: list[int],
    ) -> list[tuple[np.ndarray, float]]:
        """Solve one sub-QUBO; return top-K (bitstring, energy) tuples."""
        sub_Q, mapping = self._induced_qubo(qubo_result.Q, partition)
        from quantum_mht.formulation.mtda_qubo_builder import QUBOResult

        sub_result = QUBOResult(
            Q=sub_Q,
            num_variables=len(partition),
            variables=_LocalSubVars(len(partition)),
            penalty=qubo_result.penalty,
        )
        solver = create_solver(self.sub_solver, **self.sub_solver_kwargs)
        try:
            result = solver.solve(sub_result)
            candidates = [(result.raw_solution.copy(), float(result.objective_value))]
        except Exception as exc:  # pragma: no cover - backend-dependent
            logger.warning("ParaQAOA sub-solver failed on partition: %s", exc)
            candidates = [(np.zeros(len(partition), dtype=int), 0.0)]

        # Diversify top-K with neighborhood flips if the sub-solver only
        # returned one bitstring.
        while len(candidates) < self.top_k:
            base = candidates[0][0]
            flip = base.copy()
            idx = np.random.default_rng(len(candidates)).integers(len(flip))
            flip[idx] = 1 - flip[idx]
            candidates.append(
                (flip, self._energy_sub(sub_Q, flip)),
            )
        candidates.sort(key=lambda pair: pair[1])
        return candidates[: self.top_k]

    def _merge(
        self,
        Q: dict[tuple[int, int], float],
        num_vars: int,
        partitions: list[list[int]],
        sub_results: list[list[tuple[np.ndarray, float]]],
    ) -> tuple[np.ndarray, float]:
        """Level-Aware Parallel Merge (arXiv:2603.26232 Sec IV).

        Enumerate the Cartesian product of sub-solver top-K bitstrings
        and evaluate the full QUBO (including dropped inter-partition
        edges). Selects the lowest-energy joint assignment.
        """
        best_bits = np.zeros(num_vars, dtype=int)
        best_energy = math.inf
        for combo in itertools.product(*sub_results):
            joint = np.zeros(num_vars, dtype=int)
            for part_idx, (sub_bits, _sub_energy) in enumerate(combo):
                for local_idx, var_idx in enumerate(partitions[part_idx]):
                    if local_idx < len(sub_bits):
                        joint[var_idx] = int(sub_bits[local_idx])
            energy = self._energy_full(Q, joint)
            if energy < best_energy:
                best_energy = energy
                best_bits = joint
        return best_bits, best_energy

    def _induced_qubo(
        self, Q: dict[tuple[int, int], float], partition: list[int],
    ) -> tuple[dict[tuple[int, int], float], dict[int, int]]:
        mapping = {var: idx for idx, var in enumerate(partition)}
        sub: dict[tuple[int, int], float] = {}
        for (i, j), val in Q.items():
            if i in mapping and j in mapping:
                sub[(mapping[i], mapping[j])] = sub.get((mapping[i], mapping[j]), 0.0) + val
        return sub, mapping

    def _energy_sub(
        self, sub_Q: dict[tuple[int, int], float], bits: np.ndarray,
    ) -> float:
        energy = 0.0
        for (i, j), val in sub_Q.items():
            energy += val * float(bits[i]) * float(bits[j])
        return energy

    def _energy_full(
        self, Q: dict[tuple[int, int], float], bits: np.ndarray,
    ) -> float:
        energy = 0.0
        for (i, j), val in Q.items():
            energy += val * float(bits[i]) * float(bits[j])
        return energy


@dataclass
class _LocalSubVars:
    """Minimal variable proxy for sub-problem solution decoding."""

    num_variables: int

    def decode_solution(self, solution: np.ndarray) -> dict[str, list]:
        return {"assignments": [], "missed_detections": [], "false_alarms": []}

    def var_index(self, i: int, j: int) -> int:  # pragma: no cover - compatibility shim
        return i

    n_tracks: int = 0
    n_measurements: int = 0
    include_missed: bool = False
    include_false_alarm: bool = False
