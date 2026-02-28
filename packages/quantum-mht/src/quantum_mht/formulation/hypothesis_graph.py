"""MHT hypothesis tree management.

Implements the hypothesis tree data structure central to Multiple Hypothesis
Tracking (MHT). Each hypothesis represents a complete, globally consistent
assignment of measurements to tracks across one or more scans.

The hypothesis tree grows combinatorially with each scan; pruning strategies
(score-based thresholding and N-best retention) are essential to keep the
tree tractable. Murty's algorithm enables efficient enumeration of the
K-best hypotheses from the assignment matrix.

Academic References:
    Reid, "An Algorithm for Tracking Multiple Targets", IEEE Trans. Automatic
        Control AC-24(6):843-854, 1979 -- original MHT hypothesis tree
        formulation with deferred decision logic.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 8 -- comprehensive treatment of MHT hypothesis
        management, pruning, and merging strategies.
    Murty, "An Algorithm for Ranking all the Assignments in Increasing Order
        of Cost", Operations Research 16(3):682-687, 1968 -- K-best
        assignment enumeration used to generate ranked hypotheses.
    McCormick et al., "Bayesian Diabatic Quantum Annealing for Multi-Target
        Tracking", arXiv:2209.00615, 2022 -- quantum-native multi-hypothesis
        enumeration via diabatic annealing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class Hypothesis:
    """Single MHT hypothesis (track-measurement mapping).

    Represents one complete assignment hypothesis in the MHT tree
    (Reid 1979). Each hypothesis contains a globally consistent set of
    track-measurement assignments, missed detections, and false alarms,
    scored by cumulative log-likelihood.
    """
    hypothesis_id: int
    assignments: list[tuple[int, int]]  # (track_idx, meas_idx)
    missed_detections: list[int]  # track indices
    false_alarms: list[int]  # measurement indices
    log_likelihood: float = 0.0
    parent_id: int | None = None

    @property
    def score(self) -> float:
        return self.log_likelihood

@dataclass
class HypothesisGraph:
    """Manages MHT hypothesis tree across scans.

    Implements hypothesis generation, scoring, and pruning as described in
    Blackman & Popoli 1999, Ch 8. Pruning uses both score-threshold gating
    and N-best hypothesis retention (Murty 1968) to maintain tractability.

    References:
        Reid 1979 (IEEE TAES) -- hypothesis tree structure.
        Blackman & Popoli 1999, Ch 8 -- pruning and management.
        Murty 1968 (Operations Research) -- K-best enumeration.
    """
    hypotheses: list[Hypothesis] = field(default_factory=list)
    max_hypotheses: int = 100
    pruning_threshold: float = -50.0
    _next_id: int = field(default=0, init=False)

    def add_hypothesis(self, assignments: list[tuple[int, int]], missed: list[int], false_alarms: list[int], log_likelihood: float, parent_id: int | None = None) -> Hypothesis:
        h = Hypothesis(
            hypothesis_id=self._next_id,
            assignments=assignments,
            missed_detections=missed,
            false_alarms=false_alarms,
            log_likelihood=log_likelihood,
            parent_id=parent_id,
        )
        self._next_id += 1
        self.hypotheses.append(h)
        return h

    def prune(self) -> int:
        """Prune low-scoring hypotheses.

        Two-stage pruning (Blackman & Popoli 1999, Ch 8.3):
        1. Remove hypotheses below score threshold (pruning_threshold).
        2. Retain only the top max_hypotheses by score (N-best, Murty 1968).
        """
        before = len(self.hypotheses)
        self.hypotheses = [h for h in self.hypotheses if h.log_likelihood > self.pruning_threshold]
        self.hypotheses.sort(key=lambda h: h.log_likelihood, reverse=True)
        if len(self.hypotheses) > self.max_hypotheses:
            self.hypotheses = self.hypotheses[:self.max_hypotheses]
        return before - len(self.hypotheses)

    @property
    def best_hypothesis(self) -> Hypothesis | None:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.log_likelihood)

    def n_best(self, n: int) -> list[Hypothesis]:
        """Return the N-best hypotheses by score (Murty 1968)."""
        return sorted(self.hypotheses, key=lambda h: h.log_likelihood, reverse=True)[:n]
