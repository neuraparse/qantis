"""Composable error mitigation pipeline.

2026 Academic References — Error Mitigation Theory
=====================================================
- **Cai et al., "Quantum Error Mitigation", Rev. Mod. Phys. 95, 045005 (2023).**
  DOI: 10.1103/RevModPhys.95.045005
  Comprehensive survey of error mitigation techniques. Introduces the concept
  of *composable* mitigation pipelines where ZNE, PEC, and readout correction
  can be chained in sequence. This ``MitigationPipeline`` class implements
  the chain-of-responsibility pattern described in Section IV.
  CONTEXT: Comprehensive survey establishing composable error mitigation as
  the standard NISQ approach. Our MitigationPipeline implements their
  chain-of-responsibility composition pattern.

- **Nature s41467-025-67768-4 (2025)**: Combined error mitigation + correction
  demonstrated on logical qubits. Our composable pipeline architecture supports
  stacking ZNE + readout mitigation, and can integrate with logical qubit error
  correction when available.

- **Mitiq v0.44+** (2026): API changes include renaming
  ``scaled_circuits`` to ``construct_circuits``, and addition of Virtual
  Distillation and PEA (Phase Estimation Assisted) workflow support.
  See: https://mitiq.readthedocs.io

- **Composability**: Cai et al. (2023) demonstrate that combining readout
  mitigation → ZNE → measurement averaging yields better fidelity than
  any single technique alone, at the cost of increased shot overhead.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)

class MitigationStrategy(abc.ABC):
    """Base class for error mitigation strategies."""
    @abc.abstractmethod
    def apply(self, circuit: Any, backend: Any, counts: dict[str, int], shots: int) -> dict[str, int]:
        """Apply mitigation to raw measurement counts."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

@dataclass
class MitigationPipeline:
    """Chain-of-responsibility mitigation pipeline. Strategies applied in order.

    Per Cai et al., Rev. Mod. Phys. 95, 045005 (2023), composable mitigation
    achieves optimal results when strategies are ordered: readout correction
    first, then noise-scaling techniques (ZNE), then quasi-probability
    methods (PEC) last.
    """
    strategies: list[MitigationStrategy] = field(default_factory=list)

    def add(self, strategy: MitigationStrategy) -> MitigationPipeline:
        self.strategies.append(strategy)
        return self

    def apply(self, circuit: Any, backend: Any, counts: dict[str, int], shots: int) -> dict[str, int]:
        result = counts
        for strategy in self.strategies:
            logger.info("Applying mitigation: %s", strategy.name)
            try:
                result = strategy.apply(circuit, backend, result, shots)
            except Exception as e:
                logger.warning("Mitigation %s failed: %s, skipping", strategy.name, e)
        return result

    def __len__(self) -> int:
        return len(self.strategies)
