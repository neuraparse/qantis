"""Ronnow-Shaydulin TTS(99%) scaling protocol.

Implements the 2024-2026 canonical benchmarking methodology for quantum
optimization claims:

    1. ``TTS(p_s)`` per instance: time to reach the ground state with
       probability ``p_s`` (default 0.99). Formula:

           TTS(p_s) = T_run * log(1 - p_s) / log(1 - p_hit)

       where ``T_run`` is single-run wall-time and ``p_hit`` is the
       empirical success probability across ``runs_per_instance`` runs.
    2. Instance ensemble: >= 50 seeded instances per problem size
       (Shaydulin minimum).
    3. Scaling fit: linear regression of log10(median TTS) vs size with
       bootstrap 95% CI on the slope (10_000 resamples).
    4. Pairwise significance: Wilcoxon signed-rank on per-instance TTS.
    5. Embedding-aware effective TTS (arXiv:2504.13376 Mar 2026):

           TTS_eff = TTS / (1 - chain_break_fraction)**chain_length_max.

The protocol sits alongside the existing ``BenchmarkRunner`` -- the
runner is generic, this class is specifically for optimization-TTS
claims that reviewers insist on.

Academic References:
    Shaydulin, Safro, Larson et al., "Evidence of scaling advantage for
        the QAOA on a classically intractable problem," Science Advances
        10, eadm6761 (2024), DOI 10.1126/sciadv.adm6761.
    Ronnow et al., "Defining and Detecting Quantum Speedup," Science
        345, 420 (2014), DOI 10.1126/science.1252319 -- foundational
        TTS framework; methodology remains canonical.
    Pelofske, Bartschi, Eidenbenz, "Short-depth QAOA circuits and
        quantum annealing on higher-order Ising models," npj Quantum
        Information 10, 30 (2024), DOI 10.1038/s41534-024-00825-w --
        D-Wave advantage protocol with embedding accounting.
    Lubinski et al. (QED-C), "Application-Oriented Performance
        Benchmarks for Quantum Computing" (2024-2026 suite v2).
    Minor-embedding chain-length correlation, arXiv:2504.13376
        (Mar 2026) -- effective-TTS penalty formula.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstanceRun:
    """Result of one solver invocation on one instance.

    Attributes
    ----------
    instance_id : str
    size : int
    hit : bool
        True if the returned objective matches the reference optimum
        within ``energy_tolerance``.
    energy : float
    ref_energy : float
    wall_time_s : float
    chain_length_max : int
    chain_break_fraction : float
    """

    instance_id: str
    size: int
    hit: bool
    energy: float
    ref_energy: float
    wall_time_s: float
    chain_length_max: int = 1
    chain_break_fraction: float = 0.0


@dataclass(frozen=True)
class InstanceSummary:
    """Aggregate of ``runs_per_instance`` runs for a single instance."""

    instance_id: str
    size: int
    p_hit: float
    median_wall_time_s: float
    tts_s: float
    tts_eff_s: float
    chain_length_max: int
    chain_break_fraction_mean: float


class Solver(Protocol):
    """Minimal solver callable used by :class:`BenchmarkProtocol`."""

    def __call__(self, instance: Any, seed: int) -> InstanceRun: ...


@dataclass
class BenchmarkProtocol:
    """TTS(p_s) scaling protocol with bootstrap confidence intervals.

    Attributes
    ----------
    target_success : float
        Default 0.99 per Ronnow 2014.
    sizes : Sequence[int]
        Problem sizes to sweep. 3+ sizes required for scaling fit.
    instances_per_size : int
        Shaydulin 2024 minimum is 50; 20 is the sanity threshold.
    runs_per_instance : int
        100 is the Shaydulin 2024 recommendation for ``p_hit`` estimation.
    energy_tolerance : float
    bootstrap_samples : int
    """

    target_success: float = 0.99
    sizes: Sequence[int] = (10, 20, 50)
    instances_per_size: int = 50
    runs_per_instance: int = 100
    energy_tolerance: float = 1e-6
    bootstrap_samples: int = 10_000
    seed: int = 42

    def __post_init__(self) -> None:
        if len(self.sizes) < 3:
            logger.warning(
                "BenchmarkProtocol: at least 3 sizes are required for scaling fit; "
                "received %d. CI will be undefined.",
                len(self.sizes),
            )
        if self.instances_per_size < 20:
            logger.warning(
                "BenchmarkProtocol: instances_per_size=%d is below the Shaydulin 2024 "
                "recommendation (>=50). Results will be noisy.",
                self.instances_per_size,
            )

    # ------------------------------------------------------------------
    # Per-instance TTS
    # ------------------------------------------------------------------

    def tts(self, p_hit: float, wall_time_s: float) -> float:
        """Ronnow 2014 time-to-solution formula."""
        p_hit = float(min(max(p_hit, 1e-6), 1.0 - 1e-6))
        return wall_time_s * math.log1p(-self.target_success) / math.log1p(-p_hit)

    def effective_tts(
        self,
        tts_s: float,
        chain_length_max: int,
        chain_break_fraction: float,
    ) -> float:
        """Embedding-penalized TTS (arXiv:2504.13376 Mar 2026)."""
        penalty = max((1.0 - float(chain_break_fraction)), 1e-9) ** max(chain_length_max, 1)
        return tts_s / penalty

    def summarize_instance(self, runs: Sequence[InstanceRun]) -> InstanceSummary:
        if not runs:
            raise ValueError("summarize_instance requires at least one run")
        size = runs[0].size
        instance_id = runs[0].instance_id
        p_hit = float(np.mean([1.0 if r.hit else 0.0 for r in runs]))
        wall_times = np.array([r.wall_time_s for r in runs])
        median_wall = float(np.median(wall_times))
        tts_s = self.tts(p_hit, median_wall)
        cl_max = max(r.chain_length_max for r in runs)
        cbf = float(np.mean([r.chain_break_fraction for r in runs]))
        tts_eff = self.effective_tts(tts_s, cl_max, cbf)
        return InstanceSummary(
            instance_id=instance_id,
            size=size,
            p_hit=p_hit,
            median_wall_time_s=median_wall,
            tts_s=tts_s,
            tts_eff_s=tts_eff,
            chain_length_max=cl_max,
            chain_break_fraction_mean=cbf,
        )

    # ------------------------------------------------------------------
    # Scaling + significance
    # ------------------------------------------------------------------

    def scaling_fit(
        self,
        size_to_median_tts: dict[int, float],
    ) -> dict[str, float | tuple[float, float]]:
        """Linear fit of log10(median TTS) vs size with bootstrap CI."""
        if len(size_to_median_tts) < 3:
            return {"slope": float("nan"), "intercept": float("nan"),
                    "r2": float("nan"), "ci95": (float("nan"), float("nan"))}

        sizes = np.array(sorted(size_to_median_tts))
        values = np.array([size_to_median_tts[s] for s in sizes])
        logs = np.log10(values)
        fit = stats.linregress(sizes, logs)

        rng = np.random.default_rng(self.seed)
        slopes: list[float] = []
        for _ in range(self.bootstrap_samples):
            idx = rng.choice(len(sizes), size=len(sizes), replace=True)
            if len(set(sizes[idx])) < 2:
                continue
            slopes.append(float(stats.linregress(sizes[idx], logs[idx]).slope))
        if slopes:
            ci = (float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975)))
        else:
            ci = (float("nan"), float("nan"))

        return {
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r2": float(fit.rvalue ** 2),
            "ci95": ci,
        }

    def pairwise_significance(
        self, tts_a: Sequence[float], tts_b: Sequence[float],
    ) -> dict[str, float | bool]:
        """Wilcoxon signed-rank test on paired per-instance TTS."""
        if len(tts_a) != len(tts_b):
            raise ValueError("pairwise_significance requires equal-length inputs")
        if len(tts_a) < 6:
            return {"wilcoxon_stat": float("nan"), "p_value": 1.0,
                    "significant_p01": False}
        stat, p = stats.wilcoxon(np.asarray(tts_a), np.asarray(tts_b))
        return {
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "significant_p01": bool(p < 0.01),
        }

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    def run(
        self,
        solver: Solver,
        instances_by_size: dict[int, Iterable[Any]],
    ) -> dict[str, Any]:
        """Run ``solver`` across all instances, aggregate per size.

        ``instances_by_size`` maps each declared size to an iterable of
        opaque instance objects that ``solver`` understands. ``solver``
        must return an :class:`InstanceRun` per call.
        """
        report: dict[str, Any] = {
            "protocol_version": "qantis-bench-2026.1",
            "target_success": self.target_success,
            "per_size": {},
        }
        for size in self.sizes:
            summaries: list[InstanceSummary] = []
            for instance in instances_by_size.get(size, []):
                runs: list[InstanceRun] = []
                for repeat in range(self.runs_per_instance):
                    seed = self.seed + repeat + 1000 * size
                    runs.append(solver(instance, seed))
                summaries.append(self.summarize_instance(runs))
            if not summaries:
                continue
            tts_values = [s.tts_s for s in summaries]
            tts_eff_values = [s.tts_eff_s for s in summaries]
            report["per_size"][size] = {
                "median_tts_s": float(np.median(tts_values)),
                "q25_tts_s": float(np.quantile(tts_values, 0.25)),
                "q75_tts_s": float(np.quantile(tts_values, 0.75)),
                "median_tts_eff_s": float(np.median(tts_eff_values)),
                "median_chain_length_max": int(np.median([s.chain_length_max for s in summaries])),
                "median_chain_break_fraction": float(
                    np.median([s.chain_break_fraction_mean for s in summaries])
                ),
                "instance_count": len(summaries),
            }

        medians = {size: entry["median_tts_s"]
                   for size, entry in report["per_size"].items()}
        report["scaling"] = self.scaling_fit(medians)
        return report
