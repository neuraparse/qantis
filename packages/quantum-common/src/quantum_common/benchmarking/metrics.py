"""Benchmark metrics and comparison results.

2026 Academic References — Quantum Benchmark Metrics
======================================================
- **Quantum Volume (QV)**: Hardware-agnostic benchmark measuring the
  largest random circuit of equal width and depth that a quantum computer
  can successfully implement. Defined by:
  Cross et al., "Validating Quantum Computers Using Randomized Model
  Circuits", PRA 100, 032328 (2019). DOI: 10.1103/PhysRevA.100.032328

- **CLOPS (Circuit Layer Operations Per Second)**: Throughput metric
  measuring how many QV-equivalent circuit layers a system can execute
  per second, including compilation, queuing, and data transfer overhead.
  Defined by IBM Quantum: Wack et al., "Quality, Speed, and Scale: Three
  Key Attributes to Measure the Performance of Near-Term Quantum
  Computers", arXiv:2110.14108 (2021).

- **MetricType.FIDELITY**: Typically measured via state tomography or
  cross-entropy benchmarking (XEB). See Google Quantum AI's supremacy
  experiment methodology.

- **Speedup calculation**: The ``ComparisonResult.speedup`` property
  computes wall-clock speedup (classical_time / quantum_time). Note that
  fair comparison must account for compilation overhead, queue wait times,
  and shot count (per arXiv:2512.08245).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import numpy as np

class MetricType(Enum):
    """Standard quantum benchmark metric categories.

    TIME: Wall-clock execution time (seconds)
    ACCURACY: Solution quality (0.0-1.0 scale)
    FIDELITY: Quantum state fidelity, e.g., Quantum Volume (Cross et al. 2019)
    COST: Resource cost (e.g., USD per job, total shots)
    QUBITS: Number of qubits used
    DEPTH: Circuit depth after transpilation
    CUSTOM: User-defined metrics (e.g., CLOPS — Wack et al. 2021)
    """
    TIME = auto()
    ACCURACY = auto()
    FIDELITY = auto()
    COST = auto()
    QUBITS = auto()
    DEPTH = auto()
    CUSTOM = auto()

@dataclass
class BenchmarkMetric:
    """Single benchmark measurement."""
    name: str
    value: float
    metric_type: MetricType
    unit: str = ""
    lower_is_better: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"BenchmarkMetric({self.name}={self.value:.4f} {self.unit})"

@dataclass
class ComparisonResult:
    """Comparison between quantum and classical approaches."""
    problem_size: int
    quantum_metrics: list[BenchmarkMetric]
    classical_metrics: list[BenchmarkMetric]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def speedup(self) -> float | None:
        """Calculate quantum speedup for time metrics."""
        q_time = next((m.value for m in self.quantum_metrics if m.metric_type == MetricType.TIME), None)
        c_time = next((m.value for m in self.classical_metrics if m.metric_type == MetricType.TIME), None)
        if q_time and c_time and q_time > 0:
            return c_time / q_time
        return None

    @property
    def accuracy_improvement(self) -> float | None:
        """Calculate accuracy improvement."""
        q_acc = next((m.value for m in self.quantum_metrics if m.metric_type == MetricType.ACCURACY), None)
        c_acc = next((m.value for m in self.classical_metrics if m.metric_type == MetricType.ACCURACY), None)
        if q_acc is not None and c_acc is not None and c_acc > 0:
            return (q_acc - c_acc) / c_acc
        return None
