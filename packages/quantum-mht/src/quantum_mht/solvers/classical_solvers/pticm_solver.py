"""Parallel-Tempering + Isoenergetic-Cluster-Moves (PT-ICM) baseline.

Numba-accelerated implementation of the PT-ICM recipe that Chowdhury, Aadit, Mohseni,
Camsari et al. identified as the 2025 SOTA classical competitor to
quantum annealing on hard combinatorial landscapes (Nature Communications
16, 2025, DOI 10.1038/s41467-025-64235-y). This solver is the single
most important baseline to ship alongside QANTIS's quantum-annealing
results because reviewers expect to see that a well-tuned PT-ICM does
not already match the D-Wave numbers.

Architecture:

    - **Parallel tempering**: ``num_replicas`` Metropolis chains at a
      geometric temperature ladder; pair swaps attempted each sweep with
      Metropolis acceptance.
    - **Houdayer isoenergetic cluster moves**: at each replica two
      independent spin configurations are maintained; on the
      ``q == -1`` overlap subgraph a random connected cluster is flipped
      in both copies, which is energy-preserving and hence accepted with
      probability 1 (Zhu-Ochoa-Katzgraber PRL 115, 077201, 2015).
    - **Feedback-optimized temperature placement** (Katzgraber et al.,
      cond-mat/0602085) is left as a future pass; we start with a
      geometric ladder over ``beta_range``.

Numba is optional; when unavailable the algorithm falls back to pure
NumPy (slower but functionally identical). This keeps the baseline
importable on CI machines without a Numba toolchain.

Academic References:
    Chowdhury, Aadit, Grimaldi, Mohseni, Theogarajan, Camsari et al., "Pushing the boundary of quantum
        advantage in hard combinatorial optimization with probabilistic
        computers," Nat. Commun. 16 (2025),
        DOI 10.1038/s41467-025-64235-y.
    Zhu, Ochoa, Katzgraber, "Efficient Cluster Algorithm for Spin
        Glasses in Any Space Dimension," PRL 115, 077201 (2015).
    Houdayer, "A cluster Monte Carlo algorithm for 2-dimensional spin
        glasses," Eur. Phys. J. B 22, 479 (2001).
    Katzgraber, Trebst, Huse, Troyer, "Feedback-optimized parallel
        tempering Monte Carlo," J. Stat. Mech. (2006) P03018,
        cond-mat/0602085.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import time
import logging

import numpy as np

from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import-time optional acceleration
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrapper


@dataclass
class PTICMSolver(MTDASolver):
    """Parallel Tempering + Isoenergetic Cluster Moves baseline.

    Attributes
    ----------
    num_replicas : int | None
        Temperature-ladder length. ``None`` -> ``ceil(sqrt(N) * log2(max(N, 4)))``
        which scales to the Chowdhury-Aadit-Camsari 2025 recipe.
    num_sweeps : int
        Metropolis sweeps (one full pass over variables per replica per
        copy per sweep).
    beta_range : tuple[float, float]
        ``(beta_min, beta_max)``. For dense SK-like MTDA QUBOs we start
        with T in [0.2, 3.0].
    icm_interval : int
        Attempt ICM cluster flips every ``icm_interval`` sweeps.
    swap_interval : int
        Attempt replica-exchange every ``swap_interval`` sweeps.
    seed : int
    """

    num_replicas: int | None = None
    num_sweeps: int = 10_000
    beta_range: tuple[float, float] = (0.33, 5.0)
    icm_interval: int = 1
    swap_interval: int = 1
    seed: int = 42

    @property
    def name(self) -> str:
        return "PT-ICM"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()

        num_vars = qubo_result.num_variables
        J, h = self._qubo_to_ising(qubo_result.Q, num_vars)
        betas = self._temperature_ladder(num_vars)
        rng = np.random.default_rng(self.seed)

        # Two ICM copies per temperature.
        shape = (len(betas), 2, num_vars)
        spins = rng.choice([-1, 1], size=shape).astype(np.int8)

        _pticm_run(
            spins,
            J.astype(np.float64),
            h.astype(np.float64),
            betas.astype(np.float64),
            self.num_sweeps,
            self.icm_interval,
            self.swap_interval,
            rng.integers(0, 2 ** 31 - 1),
        )

        # Best sample across all replicas + ICM copies, at the coldest T.
        best_spins = spins[0]  # coldest temperature
        best_x_index = int(np.argmin([_ising_energy(best_spins[c], J, h) for c in (0, 1)]))
        best = best_spins[best_x_index]

        solution = ((best + 1) // 2).astype(int)
        decoded = qubo_result.variables.decode_solution(solution)
        energy = float(
            sum(val * solution[i] * solution[j] for (i, j), val in qubo_result.Q.items())
        )
        elapsed = time.perf_counter() - t0

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=energy,
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata={
                "num_replicas": len(betas),
                "num_sweeps": self.num_sweeps,
                "beta_range": self.beta_range,
                "icm_interval": self.icm_interval,
                "swap_interval": self.swap_interval,
                "numba": _NUMBA_AVAILABLE,
            },
        )

    def _temperature_ladder(self, num_vars: int) -> np.ndarray:
        if self.num_replicas is not None:
            R = int(self.num_replicas)
        else:
            R = int(math.ceil(math.sqrt(max(num_vars, 4)) * math.log2(max(num_vars, 4))))
        beta_min, beta_max = self.beta_range
        return np.geomspace(beta_min, beta_max, R)

    def _qubo_to_ising(
        self, Q: dict[tuple[int, int], float], num_vars: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """QUBO -> Ising via ``x = (1 + s) / 2``.

        ``Q_ii x_i -> (Q_ii / 2) s_i``; off-diagonal
        ``Q_ij x_i x_j -> (Q_ij / 4) s_i s_j + (Q_ij / 4) (s_i + s_j)``.

        We store the coupling symmetrically so both ``_metropolis_sweep``
        (which uses ``J[i, j]`` with ``j != i``) and ``_ising_energy``
        (which sums the upper triangle only) see the correct coefficient.
        """
        J = np.zeros((num_vars, num_vars))
        h = np.zeros(num_vars)
        for (i, j), val in Q.items():
            if val == 0.0:
                continue
            if i == j:
                h[i] += 0.5 * val
                continue
            # Normalise to upper triangle but write both sides so the
            # sweep's local field sum picks up the correct coupling.
            coupling = 0.25 * val
            a, b = (i, j) if i < j else (j, i)
            J[a, b] += coupling
            J[b, a] += coupling
            h[a] += coupling
            h[b] += coupling
        return J, h


@njit(cache=True, fastmath=True)
def _ising_energy(spins: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
    n = spins.shape[0]
    energy = 0.0
    for i in range(n):
        energy += h[i] * spins[i]
        for j in range(i + 1, n):
            energy += J[i, j] * spins[i] * spins[j]
    return energy


@njit(cache=True, fastmath=True)
def _metropolis_sweep(
    spins: np.ndarray, J: np.ndarray, h: np.ndarray, beta: float, seed: int,
) -> None:
    n = spins.shape[0]
    rng_state = np.uint32(seed | 1)
    for i in range(n):
        local = h[i]
        for j in range(n):
            if i != j:
                local += J[i, j] * spins[j]
        delta_e = 2.0 * spins[i] * local
        if delta_e <= 0.0:
            spins[i] = -spins[i]
        else:
            rng_state = _lcg(rng_state)
            u = float(rng_state) / 4294967296.0
            if u < math.exp(-beta * delta_e):
                spins[i] = -spins[i]


@njit(cache=True, fastmath=True)
def _lcg(state: np.uint32) -> np.uint32:
    return np.uint32(np.int64(state) * 1664525 + 1013904223)


@njit(cache=True)
def _houdayer_flip(
    spins_a: np.ndarray, spins_b: np.ndarray, J: np.ndarray, seed: int,
) -> None:
    """Houdayer isoenergetic cluster move (Zhu-Ochoa-Katzgraber PRL 115 2015).

    Avoids Python list comprehensions inside the Numba body so the JIT
    compiler stays happy with typed-ndarray buffers. ``stack`` is a
    pre-allocated ``int64`` array, with ``stack_top`` tracking the
    logical size; ``visited`` flags the BFS frontier.
    """
    n = spins_a.shape[0]
    negative_count = 0
    for i in range(n):
        if spins_a[i] * spins_b[i] == -1:
            negative_count += 1
    if negative_count == 0:
        return

    rng_state = np.uint32(seed | 1)
    rng_state = _lcg(rng_state)
    pick = int(rng_state) % negative_count

    seed_idx = 0
    running = 0
    for i in range(n):
        if spins_a[i] * spins_b[i] == -1:
            if running == pick:
                seed_idx = i
                break
            running += 1

    visited = np.zeros(n, dtype=np.bool_)
    stack = np.empty(n, dtype=np.int64)
    stack[0] = seed_idx
    stack_top = 1
    visited[seed_idx] = True

    while stack_top > 0:
        stack_top -= 1
        v = stack[stack_top]
        for u in range(n):
            if (
                not visited[u]
                and spins_a[u] * spins_b[u] == -1
                and J[v, u] != 0.0
            ):
                visited[u] = True
                stack[stack_top] = u
                stack_top += 1
    for i in range(n):
        if visited[i]:
            spins_a[i] = -spins_a[i]
            spins_b[i] = -spins_b[i]


@njit(cache=True, fastmath=True)
def _replica_swap(
    spins: np.ndarray, J: np.ndarray, h: np.ndarray, betas: np.ndarray, seed: int,
) -> None:
    R = spins.shape[0]
    parity = seed & 1
    rng_state = np.uint32(seed | 1)
    for r in range(parity, R - 1, 2):
        for copy in range(spins.shape[1]):
            e_r = _ising_energy(spins[r, copy], J, h)
            e_s = _ising_energy(spins[r + 1, copy], J, h)
            delta = (betas[r] - betas[r + 1]) * (e_r - e_s)
            if delta <= 0.0:
                tmp = spins[r, copy].copy()
                spins[r, copy] = spins[r + 1, copy]
                spins[r + 1, copy] = tmp
                continue
            rng_state = _lcg(rng_state)
            u = float(rng_state) / 4294967296.0
            if u < math.exp(-delta):
                tmp = spins[r, copy].copy()
                spins[r, copy] = spins[r + 1, copy]
                spins[r + 1, copy] = tmp


@njit(cache=True)
def _pticm_run(
    spins: np.ndarray,
    J: np.ndarray,
    h: np.ndarray,
    betas: np.ndarray,
    num_sweeps: int,
    icm_interval: int,
    swap_interval: int,
    seed: int,
) -> None:
    for sweep in range(num_sweeps):
        for r in range(spins.shape[0]):
            for copy in range(spins.shape[1]):
                _metropolis_sweep(spins[r, copy], J, h, betas[r], seed + sweep + r * 7 + copy)
        if icm_interval > 0 and sweep % icm_interval == 0:
            for r in range(spins.shape[0]):
                _houdayer_flip(spins[r, 0], spins[r, 1], J, seed + sweep + r * 31)
        if swap_interval > 0 and sweep % swap_interval == 0:
            _replica_swap(spins, J, h, betas, seed + sweep)
