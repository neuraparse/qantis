"""Scaling curve analysis for quantum POMDP.

Academic References:
    arXiv:2507.18606 - "Hybrid quantum-classical algorithm for near-optimal
    planning in POMDPs" (Jul 2025).
    -- Quantum advantage scaling: the belief update speedup
       O(P(e)^{-1/2}) vs classical O(P(e)^{-1}) translates to
       overall planning speedup that grows with problem size. The
       crossover point (where quantum becomes faster) depends on
       circuit compilation overhead, hardware gate fidelity, and
       the observation probability distribution.

    Brassard, Hoyer, Mosca, Tapp, "Quantum Amplitude Amplification
    and Estimation", Contemporary Math 305:53-74 (2002).
    -- The quadratic speedup bound that governs the quantum scaling
       curve. For Grover-style AA: classical O(1/P(e)) queries vs
       quantum O(1/sqrt(P(e))) queries per belief update.

    Shende, Bullock, Markov, "Synthesis of Quantum Logic Circuits",
    IEEE Trans. CAD 25(6) (2006).
    -- Circuit depth scaling: O(2^n) for arbitrary n-qubit state
       preparation, bounding the quantum circuit compilation cost.

    Qiskit v2.3 (Jan 2026): Circuit resource estimates (qubits,
    depth, gate count, CX count) determined via Qiskit transpilation
    to IBM Heron R3 native gate set.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class ScalingAnalysis:
    """Analyze scaling behavior of quantum vs classical POMDP.

    Tracks quantum vs classical wall-clock times across problem sizes
    to empirically measure the speedup predicted by arXiv:2507.18606
    and Brassard et al. (2002).
    """
    problem_sizes: list[int] = field(default_factory=list)
    quantum_times: list[float] = field(default_factory=list)
    classical_times: list[float] = field(default_factory=list)
    quantum_qubits: list[int] = field(default_factory=list)
    circuit_depths: list[int] = field(default_factory=list)

    def add_data_point(self, size: int, q_time: float, c_time: float, qubits: int = 0, depth: int = 0) -> None:
        self.problem_sizes.append(size)
        self.quantum_times.append(q_time)
        self.classical_times.append(c_time)
        self.quantum_qubits.append(qubits)
        self.circuit_depths.append(depth)

    @property
    def speedups(self) -> list[float]:
        return [c / q if q > 0 else 0.0 for c, q in zip(self.classical_times, self.quantum_times)]

    def fit_scaling(self, times: list[float]) -> dict[str, float]:
        """Fit polynomial scaling to log-log data."""
        if len(self.problem_sizes) < 2:
            return {"exponent": 0.0, "coefficient": 0.0}
        log_n = np.log(self.problem_sizes)
        log_t = np.log(np.maximum(times, 1e-10))
        coeffs = np.polyfit(log_n, log_t, 1)
        return {"exponent": float(coeffs[0]), "coefficient": float(np.exp(coeffs[1]))}

    @property
    def quantum_scaling(self) -> dict[str, float]:
        return self.fit_scaling(self.quantum_times)

    @property
    def classical_scaling(self) -> dict[str, float]:
        return self.fit_scaling(self.classical_times)

    def crossover_estimate(self) -> int | None:
        """Estimate problem size where quantum becomes faster.

        The crossover point depends on quantum circuit compilation
        overhead (Shende et al. 2006), hardware gate fidelity, and
        the observation probability distribution (arXiv:2507.18606).
        Below this size, classical methods are faster despite worse
        asymptotic scaling.
        """
        for i, (q, c) in enumerate(zip(self.quantum_times, self.classical_times)):
            if q < c:
                return self.problem_sizes[i]
        return None
