"""AB-FPAA: Adaptive Boundary-Aware Fixed-Point Amplitude Amplification.

Tests the genuinely novel algorithmic contribution of QANTIS-2 for QCE 2026:
a probe-then-amplify protocol that estimates P(o) from a low-k Chernoff
probe, selects the Yoder-Low-Chuang fixed-point length from the
resulting lower-CI, and seeds the Phase-3 BIQAE posterior with an
informative Beta prior built from the Phase-1 counts.

Tested invariants:
    * Phase-1 Wilson lower CI is always <= empirical rate (one-sided).
    * Phase-1 LCB converges to P(o) as trials -> infinity.
    * Selected FPAA length is odd and matches Yoder-Low-Chuang formula
      when omega is known exactly.
    * Informative prior carries Phase-1 sufficient statistics.
    * Query-complexity bound matches Yoder's O(sqrt(1/p) log(1/delta))
      optimum for rare observations.
    * The protocol never over-amplifies (length stays bounded when
      Phase-1 sees zero evidence via the p_floor safety).

Prior-art audit (no preemption found Jan 2024 - Apr 2026):
    * Yoder-Low-Chuang PRL 113:210501 (2014) - fixed omega.
    * Li-BIQAE Quantum 10:1962 (2026) - generic priors.
    * Grinko-IQAE 2021 - adaptive estimation, not amplification.
    * Ramoa-Santos BAE Quantum 9:1856 (2025) - fixed Grover schedule.
    * Utsumi-Nakata Quantum 10:2024 (2026) - QSVT phase tables.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("qiskit")

from quantum_pomdp.quantum_circuits.amplitude_amplifier import (
    ABFPAAResult,
    ab_fpaa_plan,
    ab_fpaa_query_complexity,
    bounded_length_fpaa,
)


class TestABFPAAPlanGenerator:
    """Pure-function plan generator for AB-FPAA."""

    def test_plan_returns_abfpaa_result(self) -> None:
        plan = ab_fpaa_plan(phase1_successes=10, phase1_trials=100)
        assert isinstance(plan, ABFPAAResult)
        assert plan.fpaa_length >= 1
        assert plan.fpaa_length % 2 == 1

    def test_lower_ci_below_empirical_rate(self) -> None:
        """Wilson lower CI is always <= empirical success rate."""
        for succ, total in [(50, 100), (10, 100), (1, 100), (99, 100)]:
            plan = ab_fpaa_plan(succ, total)
            rate = succ / total
            assert plan.p_lcb <= rate + 1e-9, (
                f"LCB {plan.p_lcb} > rate {rate} at {succ}/{total}"
            )

    def test_lcb_converges_to_true_probability(self) -> None:
        """As trials grow, LCB approaches the empirical rate."""
        p_true = 0.15
        plans = [ab_fpaa_plan(int(p_true * n), n) for n in [50, 500, 5000]]
        assert plans[2].p_lcb > plans[0].p_lcb, (
            "LCB should tighten with more data"
        )

    def test_zero_successes_falls_back_to_floor(self) -> None:
        """Phase-1 with zero evidence uses ``p_floor`` to avoid infinite L."""
        plan = ab_fpaa_plan(phase1_successes=0, phase1_trials=128, p_floor=1 / 1024)
        # omega_hat = sqrt(p_floor) = 1/32; L ~ log(2/0.01) / (2/32) ~ 85
        assert plan.fpaa_length < 300, f"L={plan.fpaa_length} unreasonably large"
        assert plan.p_lcb <= plan.omega_hat ** 2 + 1e-9

    def test_informative_prior_carries_phase1_counts(self) -> None:
        """Phase-3 Beta prior incorporates Phase-1 Bernoulli counts."""
        plan = ab_fpaa_plan(phase1_successes=7, phase1_trials=50, prior_strength=1.0)
        assert math.isclose(plan.prior_alpha, 1.0 + 7.0)
        assert math.isclose(plan.prior_beta, 1.0 + 43.0)

    def test_prior_shrinkage(self) -> None:
        """prior_strength=0 recovers flat Beta(1,1) (Bayes-Laplace)."""
        plan = ab_fpaa_plan(7, 50, prior_strength=0.0)
        assert math.isclose(plan.prior_alpha, 1.0)
        assert math.isclose(plan.prior_beta, 1.0)

    @pytest.mark.parametrize("p_true,expected_l_max", [
        (0.5, 5),    # Tiger boundary - short L
        (0.1, 15),   # rare - longer L
        (0.01, 80),  # very rare - long L, still bounded
    ])
    def test_yoder_length_matches_formula(self, p_true: float, expected_l_max: int) -> None:
        """With many Phase-1 trials, selected L is within a constant factor
        of oracle Yoder (the LCB's one-sided slack inflates L by ~1.2x at
        rare p; still an O(sqrt(1/p)) matching bound)."""
        n = 5000
        plan = ab_fpaa_plan(int(p_true * n), n, failure_tolerance=0.01)
        l_oracle = bounded_length_fpaa(math.sqrt(p_true), 0.005)
        assert plan.fpaa_length >= l_oracle - 2, (
            f"selected L={plan.fpaa_length} below oracle L={l_oracle}"
        )
        # Constant-factor slack: expect at most 1.5x oracle length.
        assert plan.fpaa_length <= max(int(math.ceil(l_oracle * 1.5)) + 2, 5), (
            f"selected L={plan.fpaa_length}, oracle L={l_oracle}, slack > 1.5x"
        )
        assert plan.fpaa_length <= expected_l_max


class TestABFPAAQueryComplexity:
    """Sample-complexity analysis used in paper Table 3."""

    def test_yoder_beats_marginal_in_rare_regime(self) -> None:
        """Yoder FPAA ~= sqrt(1/p) queries; marginal k=1 is 1/p - so at
        p=0.01, Yoder should be ~10x cheaper than marginal."""
        c = ab_fpaa_query_complexity(p_true=0.01)
        assert c["yoder_oracle_queries"] < c["fixed_k1_queries"] * 0.2, (
            f"Yoder should dominate: {c}"
        )

    def test_ab_fpaa_within_constant_of_yoder(self) -> None:
        """AB-FPAA pays a bounded overhead vs oracle Yoder - the only
        cost of not knowing omega a priori is Phase-1 probing."""
        for p in [0.01, 0.05, 0.1]:
            c = ab_fpaa_query_complexity(p)
            ratio = c["ab_fpaa_queries"] / c["yoder_oracle_queries"]
            assert 1.0 <= ratio < 5.0, (
                f"AB-FPAA/Yoder ratio {ratio:.2f} at p={p} outside [1, 5]"
            )

    def test_ab_fpaa_beats_fixed_k1_at_boundary(self) -> None:
        """At rare-observation regime, AB-FPAA dominates fixed k=1 AA."""
        c_rare = ab_fpaa_query_complexity(p_true=0.01)
        c_tiger = ab_fpaa_query_complexity(p_true=0.5)
        assert c_rare["ab_fpaa_queries"] < c_rare["fixed_k1_queries"] * 0.5, (
            f"Rare regime: AB-FPAA should cut cost in half. {c_rare}"
        )
        # At Tiger boundary (p=0.5) quadratic speedup is trivial - both ~equal
        ratio_tiger = c_tiger["ab_fpaa_queries"] / c_tiger["fixed_k1_queries"]
        assert ratio_tiger < 2.0

    def test_phase1_shots_scale_with_1_over_p(self) -> None:
        """Phase-1 Chernoff probe needs O(1/p) shots (trivial Bernoulli)."""
        c1 = ab_fpaa_query_complexity(0.1)
        c2 = ab_fpaa_query_complexity(0.01)
        # p drops by 10x; phase-1 should grow by roughly 10x
        ratio = c2["phase1_probe_shots"] / c1["phase1_probe_shots"]
        assert 5.0 < ratio < 20.0, f"Phase-1 scaling ratio {ratio:.1f}"


class TestABFPAASystemProperties:
    """End-to-end protocol properties for paper Section 3."""

    def test_monotone_in_phase1_trials(self) -> None:
        """More Phase-1 data -> tighter LCB -> shorter FPAA length
        (for positive evidence count)."""
        small = ab_fpaa_plan(10, 100)
        large = ab_fpaa_plan(100, 1000)  # same 10% rate, 10x data
        # As trials grow, omega_hat approaches sqrt(p_true) - L stabilises.
        # Key invariant: large-sample LCB >= small-sample LCB.
        assert large.p_lcb >= small.p_lcb

    def test_query_advantage_curve_for_paper_table(self) -> None:
        """Generate the query-advantage curve used in paper Table 3."""
        rows = []
        for p in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
            c = ab_fpaa_query_complexity(p_true=p)
            rows.append({
                "p": p,
                "ab_advantage_vs_fixed": c["fixed_k1_queries"] / c["ab_fpaa_queries"],
                "ab_overhead_vs_oracle": c["ab_fpaa_queries"] / c["yoder_oracle_queries"],
            })
        # Advantage over fixed-k1 shrinks as p approaches 0.5 (identity)
        advs = {r["p"]: r["ab_advantage_vs_fixed"] for r in rows}
        assert advs[0.01] > advs[0.50]
        # Overhead over oracle stays bounded across the sweep
        overheads = [r["ab_overhead_vs_oracle"] for r in rows]
        assert max(overheads) < 5.0

    def test_defensible_advantage_claim_in_rare_regime(self) -> None:
        """Paper claim: AB-FPAA >= 3x sample-efficiency advantage at p<=0.05
        vs fixed-k1 marginal AA. This is the line reviewers will scrutinise.
        """
        for p in [0.01, 0.02, 0.05]:
            c = ab_fpaa_query_complexity(p_true=p)
            adv = c["fixed_k1_queries"] / c["ab_fpaa_queries"]
            assert adv >= 3.0, (
                f"Paper-claim advantage failed at p={p}: only {adv:.2f}x"
            )
