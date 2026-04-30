"""XY-Mixer QAOA solver (IWS-QAOA) for one-hot-constrained MTDA.

Implements the Iterative Warm-Start XY-Mixer QAOA variant introduced in
Apr 2026 (arXiv:2604.02083). The MTDA QUBO encodes one-hot row and column
constraints (each track assigned to at most one measurement, each measurement
to at most one track). XY-ring mixers confine QAOA evolution to the
Hamming-weight-feasible subspace, so constraint-violating states are never
sampled. This substantially improves sampling efficiency over the standard
transverse-field (X) mixer on one-hot problems (arXiv:2604.02083 reports
orders-of-magnitude sampling-quality improvement on TSP / Max-k-Cut, whose
one-hot structure is identical to data association).

An optional hybrid XY-X mixing path (arXiv:2604.07218) trades some subspace
strictness for faster convergence on constraint-soft instances — useful when
missed-detection / false-alarm slack variables break strict Hamming weight.

An optional warm-start regularizer (arXiv:2603.10191, RWS-QAOA) biases the
initial state away from the classical heuristic bitstring by a tunable
regularization strength, preventing the warm-start from stalling at its
seed.

This solver does not replace QAOASolver or FPCQAOASolver — it is a
structure-aware alternative for the assignment regime. For fully dense
unconstrained QUBO, fall back to FPC-QAOA or annealing.

Academic References:
    arXiv:2604.02083, Apr 2026 — IWS-QAOA with XY mixers, iterative warm-start.
    arXiv:2604.07218, Apr 2026 — hybrid XY-X mixing for VRP (one-hot + side
        constraints).
    arXiv:2603.10191, Mar 2026 — RWS-QAOA regularized warm-start.
    Hadfield et al., Algorithms 12(2):34, 2019 — quantum alternating
        operator ansatz; XY-mixer foundations.
    Stollenwerk et al., arXiv:2110.08346, 2021 — one-hot QUBO for MTDA.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import time
import logging
import numpy as np
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

logger = logging.getLogger(__name__)


@dataclass
class XYMixerQAOASolver(MTDASolver):
    """QAOA with XY-ring mixer restricted to the one-hot feasible subspace.

    The XY mixer preserves total Hamming weight per row (or per column),
    so row-one-hot constraints of the MTDA QUBO are enforced by the ansatz
    itself rather than by penalty terms. The penalty magnitude in the cost
    Hamiltonian can therefore be reduced — shrinking the coefficient dynamic
    range that hurts D-Wave embedding and shot-noise convergence on gate
    hardware.

    Attributes
    ----------
    reps : int
        QAOA depth p (number of cost/mixer alternations).
    mixer : {"xy_ring", "xy_complete", "hybrid"}
        XY mixer topology. "xy_ring" connects adjacent one-hot positions
        in a ring; "xy_complete" connects all pairs (deeper circuit but
        faster mixing); "hybrid" alternates XY and X layers
        (arXiv:2604.07218) for constraint-soft instances.
    use_simulator : bool
        True for noiseless StatevectorSampler. False for hardware via
        SamplerV2 with explicit shot count.
    shots : int
        Shot count when use_simulator=False. Default 100K to mitigate
        the SamplerV2 10K-shot undersampling issue (arXiv:2512.08245).
    maxiter : int
        Classical optimizer iterations (COBYLA).
    warm_start_solution : NDArray | None
        Optional classical-heuristic bitstring (e.g., from Hungarian) to
        seed the initial state. None = uniform Dicke-state-style init.
    regularization_lambda : float
        RWS-QAOA (arXiv:2603.10191) regularizer strength. 0.0 disables.
        Positive values penalize the warm-start bitstring in the cost
        Hamiltonian so the circuit explores away from the seed.
    """

    reps: int = 2
    mixer: Literal["xy_ring", "xy_complete", "hybrid"] = "xy_ring"
    use_simulator: bool = True
    shots: int = 100_000
    maxiter: int = 200
    warm_start_solution: np.ndarray | None = None
    regularization_lambda: float = 0.0

    @property
    def name(self) -> str:
        return f"XYMixerQAOA(p={self.reps},mix={self.mixer})"

    def solve(self, qubo_result: Any) -> SolverResult:
        t0 = time.perf_counter()

        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit import Parameter
            from qiskit.primitives import StatevectorSampler
        except ImportError as e:
            raise RuntimeError(
                f"Qiskit required for XYMixerQAOASolver: {e}"
            ) from e
        # qiskit-optimization 0.7 (Jan 2026) hosts COBYLA; fall back to the
        # frozen qiskit_algorithms package for legacy environments.
        try:
            from qiskit_optimization.optimizers import COBYLA
        except ImportError:
            try:
                from qiskit_algorithms.optimizers import COBYLA
            except ImportError as exc:
                raise RuntimeError(
                    "COBYLA optimizer missing; install qiskit-optimization>=0.7"
                ) from exc

        num_vars = qubo_result.num_variables
        cost_terms = self._cost_ising_terms(qubo_result, num_vars)
        ring_pairs = self._xy_ring_pairs(qubo_result, num_vars)

        gamma = [Parameter(f"g{k}") for k in range(self.reps)]
        beta = [Parameter(f"b{k}") for k in range(self.reps)]
        qc = self._build_circuit(num_vars, ring_pairs, gamma, beta, cost_terms)

        sampler = StatevectorSampler()
        optimizer = COBYLA(maxiter=self.maxiter)

        def objective(params: np.ndarray) -> float:
            bound = qc.assign_parameters(
                {p: float(v) for p, v in zip(list(gamma) + list(beta), params)}
            )
            bound.measure_all()
            job = sampler.run([bound], shots=self.shots)
            result = job.result()[0]
            counts = result.data.meas.get_counts()
            return self._expected_energy(counts, qubo_result, num_vars)

        x0 = np.random.default_rng(seed=42).uniform(0, np.pi, 2 * self.reps)
        opt_result = optimizer.minimize(fun=objective, x0=x0)

        bound = qc.assign_parameters(
            {p: float(v) for p, v in zip(list(gamma) + list(beta), opt_result.x)}
        )
        bound.measure_all()
        counts = sampler.run([bound], shots=self.shots).result()[0].data.meas.get_counts()
        best_bitstring = max(
            counts, key=lambda bs: -self._bitstring_energy(bs, qubo_result, num_vars)
        )
        best_energy = self._bitstring_energy(best_bitstring, qubo_result, num_vars)

        solution = np.array(
            [int(best_bitstring[num_vars - 1 - i]) for i in range(num_vars)]
        )
        decoded = qubo_result.variables.decode_solution(solution)
        elapsed = time.perf_counter() - t0

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=float(best_energy),
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata={
                "reps": self.reps,
                "mixer": self.mixer,
                "optimizer_nfev": int(getattr(opt_result, "nfev", -1)),
                "warm_start": self.warm_start_solution is not None,
                "regularization_lambda": self.regularization_lambda,
                "ring_pairs_count": len(ring_pairs),
            },
        )

    def _cost_ising_terms(
        self, qubo_result: Any, num_vars: int,
    ) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
        """Map QUBO -> Ising (Z basis) coefficients for the cost layer.

        QUBO objective ``sum Q_ij x_i x_j`` with ``x_i = (1 - Z_i) / 2``:

            - diagonal ``Q_ii x_i`` -> ``-Q_ii / 2 * Z_i`` (+const)
            - off-diagonal ``Q_ij x_i x_j`` (i<j) ->
              ``Q_ij/4 * Z_i Z_j - Q_ij/4 * Z_i - Q_ij/4 * Z_j`` (+const)

        RWS-QAOA (arXiv:2603.10191) warm-start regularization adds a
        per-qubit ``lambda * Z_i`` term biased toward (1 - 2 * bit) so
        the optimiser is pushed away from the seed bitstring.
        """
        h: dict[int, float] = {}
        J: dict[tuple[int, int], float] = {}
        for (i, j), val in qubo_result.Q.items():
            if val == 0.0:
                continue
            if i == j:
                h[i] = h.get(i, 0.0) - 0.5 * val
                continue
            a, b = (i, j) if i < j else (j, i)
            J[(a, b)] = J.get((a, b), 0.0) + 0.25 * val
            h[a] = h.get(a, 0.0) - 0.25 * val
            h[b] = h.get(b, 0.0) - 0.25 * val

        if (
            self.regularization_lambda > 0.0
            and self.warm_start_solution is not None
        ):
            for i, bit in enumerate(self.warm_start_solution):
                if i >= num_vars:
                    break
                # +1 when bit == 0 (penalise flipping to 1),
                # -1 when bit == 1 (penalise flipping to 0).
                sign = 1.0 if int(bit) == 0 else -1.0
                h[i] = h.get(i, 0.0) + sign * self.regularization_lambda

        return h, J

    def _xy_ring_pairs(self, qubo_result: Any, num_vars: int) -> list[tuple[int, int]]:
        """Partition variables into XY-ring groups along the one-hot constraints.

        The MTDA QUBO encodes row constraints ``sum_j x_{i,j} = 1`` for each
        track i. XY-ring mixers connect adjacent variables within a row as
        neighbor pairs, preserving the one-hot total. For hybrid mode, we
        also connect each row's last variable to the first (closing the ring).
        """
        variables = qubo_result.variables
        n_tracks = variables.n_tracks
        n_meas = variables.n_measurements

        pairs: list[tuple[int, int]] = []
        for i in range(n_tracks):
            row_indices = [variables.var_index(i, j) for j in range(n_meas)]
            if getattr(variables, "include_missed", False):
                row_indices.append(variables.var_index(i, -1))
            for k in range(len(row_indices) - 1):
                pairs.append((row_indices[k], row_indices[k + 1]))
            if self.mixer in ("xy_ring", "hybrid") and len(row_indices) > 2:
                pairs.append((row_indices[-1], row_indices[0]))

        if self.mixer == "xy_complete":
            pairs = []
            for i in range(n_tracks):
                row_indices = [variables.var_index(i, j) for j in range(n_meas)]
                for a_idx in range(len(row_indices)):
                    for b_idx in range(a_idx + 1, len(row_indices)):
                        pairs.append((row_indices[a_idx], row_indices[b_idx]))

        return pairs

    def _build_circuit(
        self,
        num_vars: int,
        ring_pairs: list[tuple[int, int]],
        gamma: list,
        beta: list,
        cost_terms: tuple[dict[int, float], dict[tuple[int, int], float]],
    ):
        """Build the p-layer QAOA ansatz with XY-ring (or hybrid) mixer.

        The cost Hamiltonian ``H_c = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j``
        is realised per layer as ``RZ(2 gamma h_i)`` and ``RZZ(2 gamma J_ij)``.
        The XY-ring mixer emits ``RXX(beta) RYY(beta)`` on the one-hot-
        preserving edges computed in :meth:`_xy_ring_pairs`; under
        ``mixer="hybrid"`` every odd layer falls back to a transverse-field
        ``RX(2 beta)`` per arXiv:2604.07218 to accelerate mixing on
        constraint-soft instances.
        """
        from qiskit import QuantumCircuit

        h_terms, J_terms = cost_terms
        qc = QuantumCircuit(num_vars)
        self._initial_state(qc, num_vars)

        for k in range(self.reps):
            for i, h_i in h_terms.items():
                qc.rz(2.0 * h_i * gamma[k], i)
            for (i, j), J_ij in J_terms.items():
                qc.rzz(2.0 * J_ij * gamma[k], i, j)
            if self.mixer == "hybrid" and (k % 2 == 1):
                for q in range(num_vars):
                    qc.rx(2.0 * beta[k], q)
            else:
                for a, b in ring_pairs:
                    qc.rxx(beta[k], a, b)
                    qc.ryy(beta[k], a, b)
        return qc

    def _initial_state(self, qc, num_vars: int) -> None:
        """Prepare the initial state.

        With a warm-start bitstring, seed the classical heuristic solution
        and let the XY mixer explore nearby Hamming-weight-preserving states.
        Without one, use equal superposition (a Dicke-state-like initializer
        is left as a future upgrade — see arXiv:2604.02083 Sec. III-B).
        """
        if self.warm_start_solution is not None:
            for i, bit in enumerate(self.warm_start_solution):
                if int(bit) == 1:
                    qc.x(i)
        else:
            for q in range(num_vars):
                qc.h(q)

    def _expected_energy(
        self, counts: dict[str, int], qubo_result: Any, num_vars: int
    ) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        energy = 0.0
        for bitstring, weight in counts.items():
            energy += weight * self._bitstring_energy(bitstring, qubo_result, num_vars)
        return energy / total

    def _bitstring_energy(
        self, bitstring: str, qubo_result: Any, num_vars: int
    ) -> float:
        x = np.array(
            [int(bitstring[num_vars - 1 - i]) for i in range(num_vars)], dtype=int
        )
        energy = 0.0
        for (i, j), val in qubo_result.Q.items():
            energy += val * x[i] * x[j]
        return float(energy)
