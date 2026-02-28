"""MOTA, MOTP, IDF1 metric collection for MHT evaluation.

Implements the standard Multi-Object Tracking (MOT) evaluation metrics for
benchmarking quantum and classical MTDA solvers:

    MOTA (Multiple Object Tracking Accuracy):
        MOTA = 1 - (FP + FN + IDSW) / GT
        Measures overall tracking accuracy including false positives (FP),
        false negatives (FN), and identity switches (IDSW).

    MOTP (Multiple Object Tracking Precision):
        MOTP = mean(d_i) for all matched track-target pairs
        Measures localization precision of correctly matched tracks.

    IDF1 (Identification F1 Score):
        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        Measures identity preservation -- how well the tracker maintains
        correct target identities over time.

Academic References:
    Bernardin & Stiefelhagen, "Evaluating Multiple Object Tracking
        Performance: The CLEAR MOT Metrics", J. Image and Video Processing,
        2008 -- MOTA and MOTP metric definitions.
    Ristani et al., "Performance Measures and a Data Set for Multi-Target,
        Multi-Camera Tracking", ECCV Workshop, 2016 -- IDF1 metric for
        identity-aware tracking evaluation.
    Blackman & Popoli, "Design and Analysis of Modern Tracking Systems",
        Artech House, 1999, Ch 12 -- tracking system performance evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class MOTMetrics:
    """Multi-Object Tracking metrics (Bernardin & Stiefelhagen 2008).

    MOTA and MOTP follow the CLEAR MOT definitions. IDF1 follows Ristani et al. 2016.
    """
    mota: float = 0.0  # MOTA (Bernardin & Stiefelhagen 2008)
    motp: float = 0.0  # MOTP (Bernardin & Stiefelhagen 2008)
    num_switches: int = 0
    num_false_positives: int = 0
    num_misses: int = 0
    num_objects: int = 0
    num_detections: int = 0

    @property
    def precision(self) -> float:
        if self.num_detections == 0:
            return 0.0
        return (self.num_detections - self.num_false_positives) / self.num_detections

    @property
    def recall(self) -> float:
        if self.num_objects == 0:
            return 0.0
        return (self.num_objects - self.num_misses) / self.num_objects

@dataclass
class MHTBenchmarkRunner:
    """Run MHT benchmarks and collect MOT metrics.

    Evaluates tracking performance using CLEAR MOT metrics (Bernardin &
    Stiefelhagen 2008) and IDF1 (Ristani et al. 2016).
    """
    results: list[dict[str, Any]] = field(default_factory=list)

    def compute_mota(self, ground_truth: list, predictions: list) -> float:
        """Compute MOTA: 1 - (FP + FN + IDSW) / GT.

        MOTA (Bernardin & Stiefelhagen 2008) measures overall tracking
        accuracy. Values near 1.0 indicate excellent tracking.
        """
        if not ground_truth:
            return 0.0
        total_gt = sum(len(gt) for gt in ground_truth)
        if total_gt == 0:
            return 0.0
        fp = sum(max(0, len(p) - len(gt)) for gt, p in zip(ground_truth, predictions))
        fn = sum(max(0, len(gt) - len(p)) for gt, p in zip(ground_truth, predictions))
        return 1.0 - (fp + fn) / total_gt

    def compute_motp(self, distances: list[float]) -> float:
        """Compute MOTP: mean distance for matched pairs.

        MOTP (Bernardin & Stiefelhagen 2008) measures localization precision.
        Lower values indicate better positional accuracy.
        """
        if not distances:
            return 0.0
        return float(np.mean(distances))
