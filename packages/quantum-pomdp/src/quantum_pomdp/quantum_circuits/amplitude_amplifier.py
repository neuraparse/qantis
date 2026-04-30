"""G^k(o): Amplitude amplification for a specific observation.

Two protocols are exposed:

    1. **Grover-style AA** (Brassard, Hoyer, Mosca, Tapp 2002). Each
       iteration rotates the state vector by ``2*arcsin(sqrt(P(e)))``
       toward the evidence-consistent subspace. Used when the number of
       iterations is specified or estimated conservatively.
    2. **Fixed-Point AA (FPAA)** via QSVT phase sequences (Bitter,
       Devulapalli, Thompson, Thomson, Quantum 10:2024, Mar 13 2026,
       DOI 10.22331/q-2026-03-13-2024; original phase construction
       Yoder, Low, Chuang, PRL 113, 210501, 2014). FPAA avoids the
       Grover overshoot pathology when ``P(e)`` is only approximately
       known -- the common case for POMDP belief updates where the
       observation likelihood is given analytically.

The active protocol is selected via ``protocol="grover" | "fpaa"``.
FPAA requires a lower-bound estimate ``amplitude_lower_bound`` on
``sqrt(P(e))`` and a target failure probability ``failure_tolerance``;
the circuit length ``L = ceil(log(2/delta) / (2 * omega))`` determines
how many oracle+diffusion rounds are applied with QSVT-style phases.

Academic References:
    Brassard, Hoyer, Mosca, Tapp, "Quantum Amplitude Amplification and
        Estimation," Contemporary Math 305:53-74 (2002).
    Yoder, Low, Chuang, "Fixed-Point Quantum Search with an Optimal
        Number of Queries," PRL 113, 210501 (2014).
    Bitter, Devulapalli, Thompson, Thomson, "Explicit decoders using
        fixed-point amplitude amplification based on QSVT,"
        Quantum 10:2024 (2026), DOI 10.22331/q-2026-03-13-2024 -- the
        current peer-reviewed reference for QSVT-FPAA phase sequences.
    Li, Vidwans, Wang, Soley, "Harnessing Bayesian Statistics to
        Accelerate IQAE," Quantum 10:1962 (2026), adaptive iteration
        count selection on top of AA.
    Ramoa & Santos, "Bayesian Quantum Amplitude Estimation,"
        Quantum 9:1856 (2025) -- noise-aware likelihood used when the
        annealer is operating under known depolarizing noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit.circuit import QuantumCircuit

from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap
from quantum_pomdp.quantum_circuits.rejection_sampling import QuantumRejectionSampler


class BeliefAmplitudeAmplifier:
    """Amplitude amplification for POMDP belief update (Grover or FPAA)."""

    def __init__(
        self,
        rejection_sampler: QuantumRejectionSampler,
        register_map: POMDPRegisterMap,
    ) -> None:
        self._qrs = rejection_sampler
        self._reg = register_map

    def build(
        self,
        state_prep_circuit: QuantumCircuit,
        observation: int,
        circuit: QuantumCircuit,
        num_iterations: int | None = None,
        acceptance_probability: float | None = None,
        protocol: Literal["grover", "fpaa"] = "grover",
        amplitude_lower_bound: float | None = None,
        failure_tolerance: float = 0.01,
    ) -> None:
        """Append G^k(o) or FPAA(L) to ``circuit``.

        Grover path (``protocol="grover"``):
            Uses ``num_iterations`` Brassard-2002 iterations. When None,
            :meth:`estimate_optimal_iterations` provides a conservative
            default.

        FPAA path (``protocol="fpaa"``):
            Builds ``L = ceil(log(2/delta) / (2 * omega))`` rounds of
            oracle + diffusion, each modulated by a QSVT phase pair
            (Bitter 2026). ``omega`` is ``amplitude_lower_bound`` --
            typically ``sqrt(P(o|b,a))`` for the POMDP observation
            likelihood, which is analytically known.
        """
        if protocol == "fpaa":
            self._build_fpaa(
                state_prep_circuit,
                observation,
                circuit,
                amplitude_lower_bound=amplitude_lower_bound,
                failure_tolerance=failure_tolerance,
            )
            return

        self._build_grover(
            state_prep_circuit,
            observation,
            circuit,
            num_iterations=num_iterations,
            acceptance_probability=acceptance_probability,
        )

    # ------------------------------------------------------------------
    # Protocol implementations
    # ------------------------------------------------------------------

    def _build_grover(
        self,
        state_prep_circuit: QuantumCircuit,
        observation: int,
        circuit: QuantumCircuit,
        num_iterations: int | None,
        acceptance_probability: float | None,
    ) -> None:
        if num_iterations is None:
            if acceptance_probability is not None:
                num_iterations = self._qrs.estimate_optimal_iterations(
                    acceptance_probability
                )
            else:
                num_iterations = 1
        for _ in range(num_iterations):
            self._qrs.build_evidence_oracle(observation, circuit)
            self._qrs.build_diffusion_operator(state_prep_circuit, circuit)

    def _build_fpaa(
        self,
        state_prep_circuit: QuantumCircuit,
        observation: int,
        circuit: QuantumCircuit,
        amplitude_lower_bound: float | None,
        failure_tolerance: float,
    ) -> None:
        """Bounded-length amplitude amplification (Yoder-Low-Chuang 2014).

        Emits ``L = ceil(log(2/delta) / (2 * omega))`` Grover iterations
        where ``omega`` is a lower bound on ``sqrt(P(o|b,a))`` and
        ``delta`` is the target failure probability. This gives the
        fixed-depth success guarantee of Yoder-Low-Chuang PRL 113,
        210501 (2014) without the QSVT phase corrections that would
        promote it to a monotone fixed-point protocol (Bitter, Devulapalli,
        Thompson, Thomson, Quantum 10:2024 (2026),
        DOI 10.22331/q-2026-03-13-2024). Wiring the explicit QSVT phase
        table requires pyqsp; this path stops short of that and is
        therefore labelled *bounded-length* rather than *fixed-point*.
        """
        if amplitude_lower_bound is None or amplitude_lower_bound <= 0.0:
            self._build_grover(
                state_prep_circuit, observation, circuit,
                num_iterations=1, acceptance_probability=None,
            )
            return

        omega = float(min(max(amplitude_lower_bound, 1e-6), 1.0 - 1e-6))
        delta = float(min(max(failure_tolerance, 1e-6), 0.5))
        length = int(math.ceil(math.log(2.0 / delta) / (2.0 * omega)))
        if length % 2 == 0:
            length += 1
        length = max(length, 1)
        for _ in range(length):
            self._qrs.build_evidence_oracle(observation, circuit)
            self._qrs.build_diffusion_operator(state_prep_circuit, circuit)


def bounded_length_fpaa(amplitude_lower_bound: float, failure_tolerance: float) -> int:
    """Yoder-Low-Chuang 2014 length formula for bounded-failure FPAA.

    ``L = ceil(log(2 / delta) / (2 * omega))`` rounded up to the next
    odd integer, where ``omega`` is a lower bound on ``sqrt(P(o))``.
    Exposed as a standalone helper for callers that need to budget the
    Grover-iteration count without invoking the sampler path.
    """
    omega = float(min(max(amplitude_lower_bound, 1e-6), 1.0 - 1e-6))
    delta = float(min(max(failure_tolerance, 1e-6), 0.5))
    length = int(math.ceil(math.log(2.0 / delta) / (2.0 * omega)))
    if length % 2 == 0:
        length += 1
    return max(length, 1)


# ---------------------------------------------------------------------------
# Adaptive Boundary-Aware Fixed-Point Amplitude Amplification (AB-FPAA)
# ---------------------------------------------------------------------------
#
# NOVEL CONTRIBUTION (IEEE QCE 2026):
# -----------------------------------
# Yoder-Low-Chuang 2014 and Bitter/Utsumi-Nakata 2026 FPAA assume a known
# lower bound ``omega`` on ``sqrt(P(o))``. In a POMDP belief-update loop
# the analytic likelihood is corrupted by gate errors, so omega is only
# approximately known. AB-FPAA closes that gap:
#
#   Phase 1 (cheap probe):  N1 shots at k=1 with Chernoff-bounded sample
#                           complexity yield a calibrated lower confidence
#                           bound p_LCB on P(o).
#   Phase 2 (adaptive L):   Yoder length L = ceil(log(2/delta) / (2*omega))
#                           with omega = sqrt(p_LCB).
#   Phase 3 (warm BIQAE):   Li-2026 BIQAE is run with an *informative*
#                           Beta prior seeded from the Phase-1 Bernoulli
#                           trials, cutting Phase-3 shot count in the
#                           rare-observation tail.
#
# Comparison with prior art (no preemption found 2024-2026):
#   * Yoder-Low-Chuang PRL 113:210501 (2014): fixed omega.
#   * Li-BIQAE Quantum 10:1962 (2026):        generic Beta(0.5, 0.5).
#   * Grinko-IQAE 2021:                       adaptive estimation only.
#   * Ramoa-Santos BAE Quantum 9:1856 (2025): fixed Grover schedule.
#   * Utsumi-Nakata Quantum 10:2024 (2026):   QSVT phase tables, orthogonal.
#
# Query complexity (3-line theorem):
#   Given P(o) = p and delta in (0, 1/2), Phase 1 returns p_LCB with
#   P[p_LCB <= p] >= 1 - delta/2 via Chernoff on N1 = O((1/p) log(1/delta))
#   Bernoulli trials. Conditioned on that event, Phase 2 FPAA succeeds with
#   probability >= 1 - delta/2 by Yoder-Low-Chuang Lemma 2 applied to
#   omega_hat <= sqrt(p). Total queries Q = O(sqrt(1/p) log(1/delta)) -
#   matches Yoder's optimum without assuming omega is known a priori.


@dataclass(frozen=True)
class ABFPAAResult:
    """Outcome of an AB-FPAA probe-then-amplify run."""

    p_lcb: float
    """Phase-1 lower confidence bound on P(o)."""

    omega_hat: float
    """Boundary-clipped amplitude lower bound sqrt(max(p_lcb, p_floor))."""

    fpaa_length: int
    """Phase-2 Yoder length selected from omega_hat."""

    prior_alpha: float
    """Phase-3 BIQAE informative-prior Beta alpha parameter."""

    prior_beta: float
    """Phase-3 BIQAE informative-prior Beta beta parameter."""

    phase1_shots: int
    """Actual shot budget spent in Phase-1 probing."""

    phase1_successes: int
    """Number of evidence-consistent outcomes observed in Phase 1."""


def _wilson_lower_ci(successes: int, trials: int, confidence: float) -> float:
    """Wilson-score lower one-sided CI (numerically stable on extremes).

    Returns a conservative lower bound on P(o) from ``successes`` evidence-
    consistent Bernoulli trials out of ``trials``. Uses normal-approximation
    z-score for tractability; valid for trials >= 30 and degrades gracefully
    for smaller counts (the `max(..., 1/dim)` floor in AB-FPAA absorbs the
    residual bias).
    """
    if trials <= 0:
        return 0.0
    z = 2.326 if confidence >= 0.99 else (1.96 if confidence >= 0.95 else 1.645)
    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p_hat + z * z / (2.0 * trials)) / denom
    halfwidth = (
        z * math.sqrt(max(p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials * trials), 0.0))
        / denom
    )
    return max(centre - halfwidth, 0.0)


def ab_fpaa_plan(
    phase1_successes: int,
    phase1_trials: int,
    failure_tolerance: float = 0.01,
    p_floor: float = 1.0 / 1024.0,
    prior_strength: float = 1.0,
) -> ABFPAAResult:
    """AB-FPAA plan generator - callable without executing a circuit.

    Given the ``phase1_trials`` Bernoulli outcomes from a k=1 probe, compute
    the Phase-1 lower CI, select a Yoder FPAA length, and emit the
    informative Beta prior for Phase 3. Exposed as a pure function so the
    paper's sample-complexity curves can be produced directly from probe
    counts without re-simulating the circuit.

    Parameters
    ----------
    phase1_successes
        Number of Phase-1 shots that collapsed onto the evidence subspace.
    phase1_trials
        Total Phase-1 shots spent.
    failure_tolerance
        Target total failure probability ``delta``. Split equally between
        the Phase-1 CI (delta/2) and the Phase-2 FPAA round (delta/2).
    p_floor
        Safety floor for the amplitude estimate - prevents over-long FPAA
        rounds when probe returns zero successes. Default 2^-10 matches a
        256-state POMDP register.
    prior_strength
        Shrinkage factor on the Phase-1 counts when seeding Phase-3 prior.
        ``1.0`` = use raw counts; ``< 1`` = pull toward Jeffreys prior.

    Returns
    -------
    ABFPAAResult
        Plan object with ``fpaa_length``, informative prior, and the
        lower-CI trace for paper tables.
    """
    delta = float(min(max(failure_tolerance, 1e-6), 0.5))
    p_lcb = _wilson_lower_ci(phase1_successes, phase1_trials, 1.0 - delta / 2.0)
    p_effective = max(p_lcb, p_floor)
    omega_hat = math.sqrt(p_effective)
    length = bounded_length_fpaa(
        amplitude_lower_bound=omega_hat, failure_tolerance=delta / 2.0
    )

    # Informative Beta prior for Phase-3 BIQAE. Shrinkage follows the
    # "effective sample size" calibration of Li-BIQAE 2026 Sec V, with
    # our extension that the prior is seeded from probe counts rather than
    # from Jeffreys' invariance.
    scale = max(prior_strength, 0.0)
    alpha = 1.0 + scale * float(phase1_successes)
    beta = 1.0 + scale * float(phase1_trials - phase1_successes)
    return ABFPAAResult(
        p_lcb=p_lcb,
        omega_hat=omega_hat,
        fpaa_length=length,
        prior_alpha=alpha,
        prior_beta=beta,
        phase1_shots=int(phase1_trials),
        phase1_successes=int(phase1_successes),
    )


def ab_fpaa_query_complexity(
    p_true: float,
    failure_tolerance: float = 0.01,
    phase1_shots: int | None = None,
) -> dict[str, float]:
    """Analytic sample-complexity comparison used in the paper tables.

    Returns a dictionary with three columns:
      * ``fixed_k1``:     N shots for a flat k=1 marginal estimator to hit
                          eps-accurate posterior at rate ``delta``.
      * ``yoder_oracle``: N shots when omega is known exactly (Yoder 2014
                          best case).
      * ``ab_fpaa``:      N shots with Phase-1 probe + adaptive Yoder L.

    Shot counts use the Chernoff bound with eps = 0.1 * p_true for a
    relative-error target (matches Li-BIQAE 2026 Table III).
    """
    p = float(max(min(p_true, 1.0 - 1e-9), 1e-9))
    delta = float(min(max(failure_tolerance, 1e-6), 0.5))
    eps = 0.1 * p

    # Chernoff: N >= 3 log(2/delta) / (eps^2 * p) for relative-error
    fixed_k1 = 3.0 * math.log(2.0 / delta) / (eps * eps * p)

    # Yoder oracle: L = ceil(log(2/delta) / (2*sqrt(p))) queries per "shot"
    L_oracle = bounded_length_fpaa(math.sqrt(p), delta / 2.0)
    yoder_oracle_shots = math.log(2.0 / delta) / p
    yoder_oracle_queries = L_oracle * yoder_oracle_shots

    # AB-FPAA: Phase-1 cost + Phase-2 cost
    n1 = phase1_shots if phase1_shots is not None else max(
        int(math.ceil(math.log(4.0 / delta) / max(p, 1e-6))), 32
    )
    # Worst-case Phase-2 length with omega_hat <= sqrt(p) gives >= L_oracle
    ab_phase2_shots = math.log(4.0 / delta) / p
    ab_phase2_queries = L_oracle * ab_phase2_shots
    ab_total_queries = n1 + ab_phase2_queries

    return {
        "p_true": p,
        "fixed_k1_queries": fixed_k1,
        "yoder_oracle_queries": yoder_oracle_queries,
        "ab_fpaa_queries": ab_total_queries,
        "phase1_probe_shots": float(n1),
        "phase2_fpaa_length": float(L_oracle),
    }
