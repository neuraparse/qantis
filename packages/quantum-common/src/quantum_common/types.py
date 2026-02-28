"""Shared type aliases and enumerations for the quantum autonomous systems project.

2026 Academic References — Quantum Frameworks
==============================================
This module defines backend type enumerations aligned with the following
quantum computing frameworks and hardware generations (as of Jan 2026):

- **Qiskit v2.3** (Jan 2026): Introduces SamplerV2, EstimatorV2 primitives,
  PauliProductMeasurement, gridsynth_rz() synthesis, and targets IBM Heron R3
  processors (156 qubits). See: https://docs.quantum.ibm.com/api/qiskit

- **D-Wave Ocean SDK 9.x**: Supports Advantage2 system with 4400+ qubits on
  Zephyr topology. Namespace migration from ``neal`` to ``dwave.samplers``.
  See: D-Wave Ocean Documentation, https://docs.ocean.dwavesys.com

- **PennyLane v0.44** (Jan 2026): QRAM templates (BBQRAM, SelectOnlyQRAM,
  HybridQRAM), MultiplexerStatePreparation for amplitude encoding.
  See: https://docs.pennylane.ai/en/stable/

- **Azure Quantum SDK 2.3**: IonQ Aria-2 (25 algorithmic qubits),
  Quantinuum H2 (56 qubits). See: https://learn.microsoft.com/azure/quantum

- **Mitiq v0.44+**: ``scaled_circuits`` renamed to ``construct_circuits``,
  Virtual Distillation and PEA workflow support.
  See: https://mitiq.readthedocs.io
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Core type aliases
StateVector: TypeAlias = NDArray[np.complex128]
ProbabilityDistribution: TypeAlias = NDArray[np.float64]
CostMatrix: TypeAlias = NDArray[np.float64]
QubitIndex: TypeAlias = int
ShotCount: TypeAlias = int
BitstringCounts: TypeAlias = dict[str, int]


class QuantumParadigm(Enum):
    """Supported quantum computing paradigms."""

    GATE_BASED = auto()
    ANNEALING = auto()
    HYBRID = auto()


class BackendType(Enum):
    """Supported backend providers.

    Each member maps to a specific quantum framework and hardware generation:
      - IBM_QUANTUM: Qiskit v2.3 / IBM Heron R3 (156q) via Qiskit Runtime
      - AZURE_QUANTUM: Azure Quantum SDK 2.3 / IonQ Aria-2, Quantinuum H2
      - DWAVE: D-Wave Ocean SDK 9.x / Advantage2 (4400+ qubits, Zephyr)
      - PENNYLANE: PennyLane v0.44 / hardware-agnostic device abstraction
      - LOCAL_AER: Qiskit Aer v0.17 statevector/QASM simulator
      - LOCAL_PENNYLANE: PennyLane v0.44 default.qubit local simulator
      - LOCAL_DWAVE_SIM: ``dwave.samplers.SimulatedAnnealingSampler``
        (migrated from ``neal`` in Ocean SDK 9.x)
    """

    IBM_QUANTUM = "ibm_quantum"
    AZURE_QUANTUM = "azure_quantum"
    DWAVE = "dwave"
    PENNYLANE = "pennylane"
    LOCAL_AER = "local_aer"
    LOCAL_PENNYLANE = "local_pennylane"
    LOCAL_DWAVE_SIM = "local_dwave_sim"


class JobStatus(Enum):
    """Unified job status across all backends."""

    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
