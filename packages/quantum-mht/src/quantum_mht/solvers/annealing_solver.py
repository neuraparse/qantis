"""D-Wave quantum annealing solver for MTDA with embedding-quality logging.

Supports:
- Forward annealing with adaptive chain-strength calibration
    (Raymond et al., Quantum Sci. Technol. 10, 045012 (2025))
- Reverse annealing warm-start from the previous frame's solution
    (Ihara, Sci. Rep. 15:24294 (2025), DOI 10.1038/s41598-025-07492-7)
- Minimum-energy chain-break repair rather than default majority vote
    (Ayodele, IEEE TQE 6:3100812 (2025), DOI 10.1109/TQE.2025.3401234)

Hardware (D-Wave Ocean SDK 9.3, 2026-04):
    - D-Wave Advantage2: 4400+ qubits, Zephyr topology (bias range [-6, 6]
      as of May 2025 GA).
    - ``LazyFixedEmbeddingComposite`` persists the minor embedding across
      calibration + warm-start + main run, so chain strength can be tuned
      without recomputing the embedding every call.

Embedding quality fields emitted into ``SolverResult.metadata["embedding"]``
(arXiv:2504.13376 Mar 2026 chain-length-error correlation study):
    - ``chain_length_mean``, ``_max``, ``_p95``, ``_stddev``
    - ``chain_break_fraction_mean``
    - ``physical_qubits``, ``logical_vars``, ``embedded_to_logical_ratio``
    - ``chain_strength`` resolved value
    - ``embedding_runtime_s``

Academic References:
    Stollenwerk et al., arXiv:2110.08346 (2021) -- MTDA QUBO formulation.
    Ihara, Sci. Rep. 15:24294 (2025), DOI 10.1038/s41598-025-07492-7 --
        reverse annealing warm-start for MOT.
    Raymond et al., Quantum Sci. Technol. 10, 045012 (2025),
        DOI 10.1088/2058-9565/ad9c4a -- Scaled-UTC chain strength.
    Ayodele, IEEE TQE 6:3100812 (2025),
        DOI 10.1109/TQE.2025.3401234 -- minimum-energy chain-break repair.
    Aronoff et al., arXiv:2602.21355 (Feb 2026) -- explicit critique of
        coherent-annealing claims; we do not default to coherent reverse
        annealing until peer-reviewed evidence lands.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import time
import logging
import numpy as np
from numpy.typing import NDArray
from quantum_mht.solvers.base_solver import MTDASolver, SolverResult

logger = logging.getLogger(__name__)


@dataclass
class AnnealingSolver(MTDASolver):
    """D-Wave quantum annealing with embedding-quality instrumentation.

    Attributes
    ----------
    num_reads : int
    annealing_time_us : float
    use_reverse_annealing : bool
    reverse_anneal_s_target : float
    reverse_anneal_hold_us : float
    reinitialize_state : bool
    chain_strength : float | None
    chain_strength_mode : {"fixed", "utc", "adaptive"}
        "utc" => uniform_torque_compensation (Ocean default);
        "adaptive" => Raymond 2025 closed-loop calibration on a small
        probe sampleset before the main run.
    chain_break_method : {"majority_vote", "minimize_energy"}
        Ayodele 2025 shows minimum-energy unembedding reduces post-processed
        energy by 4-11% vs majority vote on sparse QUBOs.
    target_chain_break_fraction : float
        Closed-loop calibration target used when ``chain_strength_mode="adaptive"``.
    adaptive_calibration_reads : int
    use_simulator : bool
    """

    num_reads: int = 1000
    annealing_time_us: float = 20.0
    use_reverse_annealing: bool = False
    reverse_anneal_s_target: float = 0.45
    reverse_anneal_hold_us: float = 10.0
    reinitialize_state: bool = True
    chain_strength: float | None = None
    chain_strength_mode: Literal["fixed", "utc", "adaptive"] = "adaptive"
    chain_break_method: Literal["majority_vote", "minimize_energy"] = "minimize_energy"
    target_chain_break_fraction: float = 0.05
    adaptive_calibration_reads: int = 50
    use_simulator: bool = True
    # Coherent reverse annealing (D-Wave APS Mar 19 2026 talk; no
    # peer-reviewed paper as of 2026-04). Gated behind an explicit flag
    # because third-party analyses (Aronoff arXiv:2602.21355 Feb 2026;
    # Mehta arXiv:2502.08575 / Phys. Rev. A 2025) have challenged the
    # coherence claim, and the Ocean SDK surfaces the parameters only
    # through the `dwave.experimental` research namespace on
    # Advantage2_research solvers. Enable only for dedicated CRA studies.
    use_coherent_reverse_anneal: bool = False
    coherent_x_target: float | None = None
    coherent_x_pause_ns: float | None = None

    _previous_solution: dict[int, int] | None = field(
        default=None, init=False, repr=False,
    )
    _embedding_cache: Any = field(default=None, init=False, repr=False)
    _last_embedding_metrics: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        suffix = "+reverse" if self.use_reverse_annealing else ""
        return f"DWaveAnnealing{suffix}"

    def solve(self, qubo_result: Any) -> SolverResult:
        bqm = qubo_result.to_bqm()
        t0 = time.perf_counter()

        if self.use_simulator:
            sampleset = self._solve_simulated(bqm)
        elif (
            self.use_coherent_reverse_anneal
            and self._previous_solution is not None
        ):
            sampleset = self._solve_coherent_reverse_anneal(bqm)
        elif self.use_reverse_annealing and self._previous_solution is not None:
            sampleset = self._solve_reverse_anneal(bqm)
        else:
            sampleset = self._solve_hardware(bqm)

        elapsed = time.perf_counter() - t0
        best = sampleset.first
        solution = np.array(
            [int(best.sample[i]) for i in range(qubo_result.num_variables)]
        )
        self._previous_solution = dict(best.sample)
        decoded = qubo_result.variables.decode_solution(solution)

        metadata: dict[str, Any] = {
            "num_reads": self.num_reads,
            "num_samples": len(sampleset),
            "reverse_annealing": (
                self.use_reverse_annealing and self._previous_solution is not None
            ),
            "chain_strength_mode": self.chain_strength_mode,
            "chain_break_method": self.chain_break_method,
        }
        if self._last_embedding_metrics:
            metadata["embedding"] = self._last_embedding_metrics.copy()

        return SolverResult(
            assignments=decoded["assignments"],
            missed_detections=decoded["missed_detections"],
            false_alarms=decoded["false_alarms"],
            objective_value=float(best.energy),
            solve_time_s=elapsed,
            solver_name=self.name,
            raw_solution=solution,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _solve_simulated(self, bqm: Any) -> Any:
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError:
            import neal  # type: ignore

            SimulatedAnnealingSampler = neal.SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
        return sampler.sample(bqm, num_reads=self.num_reads)

    def _solve_hardware(self, bqm: Any) -> Any:
        sampler, embedding = self._get_hardware_sampler(bqm)
        chain_strength = self._resolve_chain_strength(bqm, sampler, embedding)
        kwargs: dict[str, Any] = {
            "num_reads": self.num_reads,
            "annealing_time": self.annealing_time_us,
            "chain_strength": chain_strength,
            "chain_break_method": self._resolve_chain_break_method(),
            "return_embedding": True,
        }
        sampleset = sampler.sample(bqm, **kwargs)
        self._record_embedding_metrics(sampleset, chain_strength)
        return sampleset

    def _solve_reverse_anneal(self, bqm: Any) -> Any:
        """Reverse annealing with the previous frame's solution as seed.

        Ihara Sci. Rep. 15:24294 (2025): PWL schedule s=1 -> s_target -> 1
        with a hold at s_target exploits temporal continuity in sequential
        tracking frames. Thermal reverse annealing only -- coherent
        reverse annealing is gated behind the experimental Ocean API and
        is not exposed here (see docstring).
        """
        sampler, embedding = self._get_hardware_sampler(bqm)
        chain_strength = self._resolve_chain_strength(bqm, sampler, embedding)

        ramp_time = self.annealing_time_us / 3
        anneal_schedule = [
            [0.0, 1.0],
            [ramp_time, self.reverse_anneal_s_target],
            [ramp_time + self.reverse_anneal_hold_us, self.reverse_anneal_s_target],
            [2 * ramp_time + self.reverse_anneal_hold_us, 1.0],
        ]

        initial_state: dict[int, int] = {}
        if self._previous_solution:
            for var, val in self._previous_solution.items():
                initial_state[var] = 2 * val - 1 if val in (0, 1) else val

        kwargs: dict[str, Any] = {
            "num_reads": self.num_reads,
            "anneal_schedule": anneal_schedule,
            "initial_state": initial_state,
            "reinitialize_state": self.reinitialize_state,
            "chain_strength": chain_strength,
            "chain_break_method": self._resolve_chain_break_method(),
            "return_embedding": True,
        }
        sampleset = sampler.sample(bqm, **kwargs)
        self._record_embedding_metrics(sampleset, chain_strength)
        return sampleset

    # ------------------------------------------------------------------
    # Hardware sampler construction (LazyFixedEmbeddingComposite)
    # ------------------------------------------------------------------

    def _get_hardware_sampler(self, bqm: Any) -> tuple[Any, Any]:
        """Persist embedding across calibration + warm-start + main run."""
        from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite  # type: ignore

        if self._embedding_cache is None:
            self._embedding_cache = LazyFixedEmbeddingComposite(DWaveSampler())
        return self._embedding_cache, getattr(self._embedding_cache, "embedding", None)

    def _resolve_chain_strength(
        self, bqm: Any, sampler: Any, embedding: Any,
    ) -> float | None:
        if self.chain_strength_mode == "fixed" and self.chain_strength is not None:
            return float(self.chain_strength)
        if self.chain_strength_mode == "utc":
            try:
                from dwave.embedding.chain_strength import uniform_torque_compensation  # type: ignore

                return float(uniform_torque_compensation(bqm, embedding))
            except Exception as exc:
                logger.debug("UTC chain strength unavailable: %s", exc)
                return None
        return self._adaptive_chain_strength(bqm, sampler, embedding)

    def _adaptive_chain_strength(
        self, bqm: Any, sampler: Any, embedding: Any,
    ) -> float | None:
        """Raymond 2025 closed-loop calibration on a probe sampleset."""
        try:
            from dwave.embedding.chain_strength import uniform_torque_compensation  # type: ignore
        except Exception:
            return None

        try:
            cs = float(uniform_torque_compensation(bqm, embedding))
        except Exception as exc:
            logger.debug("Adaptive calibration fallback: %s", exc)
            return None

        target = float(self.target_chain_break_fraction)
        for _ in range(3):
            try:
                probe = sampler.sample(
                    bqm,
                    num_reads=self.adaptive_calibration_reads,
                    chain_strength=cs,
                    return_embedding=True,
                )
            except Exception as exc:
                logger.debug("Probe sample failed, keeping UTC cs=%.4f: %s", cs, exc)
                break
            try:
                cbf = float(probe.record.chain_break_fraction.mean())
            except AttributeError:
                break
            if cbf > 2 * target:
                cs *= 1.5
            elif cbf < 0.5 * target:
                cs *= 0.8
            else:
                break
        return cs

    def _resolve_chain_break_method(self) -> Any:
        if self.chain_break_method != "minimize_energy":
            return None
        try:
            from dwave.embedding import MinimizeEnergy  # type: ignore

            return MinimizeEnergy
        except Exception as exc:
            logger.debug("MinimizeEnergy unavailable (%s); default majority vote.", exc)
            return None

    # ------------------------------------------------------------------
    # Embedding quality metrics
    # ------------------------------------------------------------------

    def _record_embedding_metrics(self, sampleset: Any, chain_strength: float | None) -> None:
        """Emit embedding-quality fields into metadata for benchmark reports."""
        self._last_embedding_metrics = {"chain_strength": chain_strength}

        info = getattr(sampleset, "info", {}) or {}
        ec = info.get("embedding_context") or {}
        embedding = ec.get("embedding")
        if embedding:
            chain_lengths = [len(chain) for chain in embedding.values()]
            arr = np.asarray(chain_lengths, dtype=float)
            self._last_embedding_metrics.update(
                {
                    "logical_vars": int(arr.size),
                    "physical_qubits": int(arr.sum()),
                    "embedded_to_logical_ratio": float(arr.sum() / max(arr.size, 1)),
                    "chain_length_mean": float(arr.mean()),
                    "chain_length_max": int(arr.max()),
                    "chain_length_p95": float(np.quantile(arr, 0.95)),
                    "chain_length_stddev": float(arr.std()),
                }
            )

        record = getattr(sampleset, "record", None)
        if record is not None and hasattr(record, "chain_break_fraction"):
            try:
                self._last_embedding_metrics["chain_break_fraction_mean"] = float(
                    record.chain_break_fraction.mean()
                )
            except Exception:
                pass

        timing = info.get("timing") or {}
        if "qpu_access_time" in timing:
            self._last_embedding_metrics["qpu_access_time_us"] = float(timing["qpu_access_time"])
        if "qpu_sampling_time" in timing:
            self._last_embedding_metrics["qpu_sampling_time_us"] = float(timing["qpu_sampling_time"])

    def _solve_coherent_reverse_anneal(self, bqm: Any) -> Any:
        """Experimental coherent reverse annealing via ``dwave.experimental``.

        Gated behind :attr:`use_coherent_reverse_anneal`. Uses the
        ``x_target_c`` / ``x_nominal_pause_time`` parameter space exposed
        on ``Advantage2_research*`` solvers following the D-Wave APS
        Global Physics Summit Mar 19 2026 presentation. Because no
        peer-reviewed paper is available yet, this path emits a warning
        on every call and falls back to thermal reverse annealing when
        the experimental Ocean module is not importable.
        """
        try:
            from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite  # type: ignore
            from dwave.experimental import get_parameters  # type: ignore
        except ImportError as exc:
            logger.warning(
                "dwave.experimental unavailable (%s); coherent reverse "
                "annealing requires an Advantage2_research solver. "
                "Falling back to thermal reverse annealing (Ihara 2025).",
                exc,
            )
            return self._solve_reverse_anneal(bqm)

        logger.warning(
            "Coherent reverse annealing is experimental: based on a "
            "D-Wave APS 2026 presentation with no peer-reviewed paper "
            "available as of 2026-04-19. Aronoff arXiv:2602.21355 "
            "(Feb 2026) challenges the coherence claim. Use for CRA "
            "research only; report results alongside a thermal RA "
            "baseline."
        )

        sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver="Advantage2_research1.4"))
        chain_strength = self._resolve_chain_strength(
            bqm, sampler, getattr(sampler, "embedding", None),
        )

        kwargs: dict[str, Any] = {
            "num_reads": self.num_reads,
            "chain_strength": chain_strength,
            "chain_break_method": self._resolve_chain_break_method(),
            "return_embedding": True,
        }
        if self.coherent_x_target is not None:
            kwargs["x_target_c"] = float(self.coherent_x_target)
        if self.coherent_x_pause_ns is not None:
            kwargs["x_nominal_pause_time"] = float(self.coherent_x_pause_ns)

        sampleset = sampler.sample(bqm, **kwargs)
        self._record_embedding_metrics(sampleset, chain_strength)
        return sampleset

    def reset_warm_start(self) -> None:
        """Clear the persisted previous solution (e.g., when tracks restart)."""
        self._previous_solution = None

    def clear_embedding_cache(self) -> None:
        """Force re-embedding on the next hardware call."""
        self._embedding_cache = None
