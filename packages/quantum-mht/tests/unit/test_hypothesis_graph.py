"""Tests for quantum_mht.formulation.hypothesis_graph module.

Validates the Hypothesis and HypothesisGraph classes that implement the
MHT hypothesis tree management (Reid 1979, Blackman & Popoli 1999 Ch 8).
"""

import numpy as np
import pytest

from quantum_mht.formulation.hypothesis_graph import Hypothesis, HypothesisGraph


class TestHypothesis:
    """Test the Hypothesis data class."""

    def test_creation_with_all_fields(self) -> None:
        """Hypothesis should store assignments, missed_detections, false_alarms."""
        h = Hypothesis(
            hypothesis_id=0,
            assignments=[(0, 1), (1, 0)],
            missed_detections=[2],
            false_alarms=[3],
            log_likelihood=-5.0,
            parent_id=None,
        )
        assert h.hypothesis_id == 0
        assert h.assignments == [(0, 1), (1, 0)]
        assert h.missed_detections == [2]
        assert h.false_alarms == [3]
        assert h.log_likelihood == -5.0
        assert h.parent_id is None

    def test_creation_minimal(self) -> None:
        """Hypothesis should work with minimal required fields."""
        h = Hypothesis(
            hypothesis_id=1,
            assignments=[],
            missed_detections=[],
            false_alarms=[],
        )
        assert h.hypothesis_id == 1
        assert h.log_likelihood == 0.0

    def test_score_property_equals_log_likelihood(self) -> None:
        """The score property should return log_likelihood."""
        h = Hypothesis(
            hypothesis_id=0,
            assignments=[(0, 0)],
            missed_detections=[],
            false_alarms=[],
            log_likelihood=-3.5,
        )
        assert h.score == h.log_likelihood
        assert h.score == -3.5

    def test_score_with_positive_log_likelihood(self) -> None:
        """Score should handle positive log_likelihood values."""
        h = Hypothesis(
            hypothesis_id=0,
            assignments=[(0, 0)],
            missed_detections=[],
            false_alarms=[],
            log_likelihood=2.0,
        )
        assert h.score == 2.0


class TestHypothesisGraphCreation:
    """Test HypothesisGraph instantiation and defaults."""

    def test_defaults(self) -> None:
        """HypothesisGraph should initialize with sensible defaults."""
        graph = HypothesisGraph()
        assert graph.hypotheses == []
        assert graph.max_hypotheses == 100
        assert graph.pruning_threshold == -50.0

    def test_custom_parameters(self) -> None:
        """Should accept custom max_hypotheses and pruning_threshold."""
        graph = HypothesisGraph(max_hypotheses=50, pruning_threshold=-20.0)
        assert graph.max_hypotheses == 50
        assert graph.pruning_threshold == -20.0


class TestAddHypothesis:
    """Test adding hypotheses to the graph."""

    def test_returns_hypothesis_with_auto_id(self) -> None:
        """add_hypothesis should return a Hypothesis with auto-incremented ID."""
        graph = HypothesisGraph()
        h0 = graph.add_hypothesis(
            assignments=[(0, 0)],
            missed=[],
            false_alarms=[],
            log_likelihood=-1.0,
        )
        assert h0.hypothesis_id == 0

        h1 = graph.add_hypothesis(
            assignments=[(1, 1)],
            missed=[],
            false_alarms=[],
            log_likelihood=-2.0,
        )
        assert h1.hypothesis_id == 1

    def test_hypothesis_added_to_list(self) -> None:
        """Added hypotheses should appear in the hypotheses list."""
        graph = HypothesisGraph()
        graph.add_hypothesis(
            assignments=[(0, 0)],
            missed=[],
            false_alarms=[],
            log_likelihood=-1.0,
        )
        assert len(graph.hypotheses) == 1

    def test_hypothesis_stores_parent_id(self) -> None:
        """add_hypothesis should preserve the parent_id parameter."""
        graph = HypothesisGraph()
        h0 = graph.add_hypothesis(
            assignments=[(0, 0)],
            missed=[],
            false_alarms=[],
            log_likelihood=-1.0,
            parent_id=None,
        )
        h1 = graph.add_hypothesis(
            assignments=[(0, 0), (1, 1)],
            missed=[],
            false_alarms=[],
            log_likelihood=-0.5,
            parent_id=h0.hypothesis_id,
        )
        assert h1.parent_id == 0


class TestBestHypothesis:
    """Test best_hypothesis property."""

    def test_returns_highest_scoring(self) -> None:
        """best_hypothesis should return the hypothesis with highest log_likelihood."""
        graph = HypothesisGraph()
        graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-10.0)
        graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-1.0)
        graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-5.0)

        best = graph.best_hypothesis
        assert best is not None
        assert best.log_likelihood == -1.0

    def test_empty_graph_returns_none(self) -> None:
        """best_hypothesis on empty graph should return None."""
        graph = HypothesisGraph()
        assert graph.best_hypothesis is None


class TestNBest:
    """Test n_best method for top-N hypothesis retrieval."""

    def test_returns_top_n(self) -> None:
        """n_best(n) should return the top-N hypotheses by score, descending."""
        graph = HypothesisGraph()
        scores = [-10.0, -1.0, -5.0, -3.0, -7.0]
        for s in scores:
            graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=s)

        top3 = graph.n_best(3)
        assert len(top3) == 3
        assert top3[0].log_likelihood == -1.0
        assert top3[1].log_likelihood == -3.0
        assert top3[2].log_likelihood == -5.0

    def test_n_larger_than_available(self) -> None:
        """n_best with n > len(hypotheses) should return all hypotheses."""
        graph = HypothesisGraph()
        graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-2.0)
        graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-1.0)

        result = graph.n_best(10)
        assert len(result) == 2

    def test_n_best_empty_graph(self) -> None:
        """n_best on empty graph should return empty list."""
        graph = HypothesisGraph()
        assert graph.n_best(5) == []


class TestPrune:
    """Test hypothesis pruning."""

    def test_removes_below_threshold(self) -> None:
        """Prune should remove hypotheses below pruning_threshold."""
        graph = HypothesisGraph(pruning_threshold=-50.0)
        graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-100.0)
        graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-10.0)
        graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-60.0)

        removed = graph.prune()
        # -100.0 and -60.0 are <= -50.0, so removed
        assert removed == 2
        assert len(graph.hypotheses) == 1
        assert graph.hypotheses[0].log_likelihood == -10.0

    def test_prune_returns_count_of_removed(self) -> None:
        """prune() should return the number of hypotheses removed."""
        graph = HypothesisGraph(pruning_threshold=-50.0)
        for _ in range(5):
            graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-100.0)
        for _ in range(3):
            graph.add_hypothesis(assignments=[], missed=[], false_alarms=[], log_likelihood=-10.0)

        removed = graph.prune()
        assert removed == 5

    def test_max_hypotheses_cap(self) -> None:
        """Pruning should cap hypotheses at max_hypotheses even if scores are above threshold."""
        graph = HypothesisGraph(max_hypotheses=100, pruning_threshold=-1000.0)

        # Add 150 hypotheses, all above threshold
        for i in range(150):
            graph.add_hypothesis(
                assignments=[],
                missed=[],
                false_alarms=[],
                log_likelihood=-float(i),  # scores: 0, -1, -2, ..., -149
            )
        assert len(graph.hypotheses) == 150

        removed = graph.prune()
        assert len(graph.hypotheses) == 100
        assert removed == 50
        # Should keep the 100 best (highest scores: 0 through -99)
        best_remaining = max(h.log_likelihood for h in graph.hypotheses)
        worst_remaining = min(h.log_likelihood for h in graph.hypotheses)
        assert best_remaining == 0.0
        assert worst_remaining == -99.0

    def test_prune_no_removal_needed(self) -> None:
        """Prune with all hypotheses above threshold and below cap should remove 0."""
        graph = HypothesisGraph(max_hypotheses=100, pruning_threshold=-50.0)
        for i in range(5):
            graph.add_hypothesis(
                assignments=[],
                missed=[],
                false_alarms=[],
                log_likelihood=-float(i),
            )
        removed = graph.prune()
        assert removed == 0
        assert len(graph.hypotheses) == 5
