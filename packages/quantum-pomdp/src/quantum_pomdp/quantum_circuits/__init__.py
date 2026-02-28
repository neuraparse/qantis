"""Quantum circuit construction for POMDP belief state estimation.

Requires ``qiskit>=2.0`` (install via ``pip install quantum-common[ibm]``).
Imports are deferred so the rest of the package works without Qiskit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantum_pomdp.quantum_circuits.belief_update import (
        BeliefUpdateCircuitConfig,
        QuantumBeliefUpdateCircuit,
    )
    from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap

__all__ = [
    "BeliefUpdateCircuitConfig",
    "POMDPRegisterMap",
    "QuantumBeliefUpdateCircuit",
]


def __getattr__(name: str):  # noqa: N807
    """Lazy import to avoid hard qiskit dependency at package level."""
    if name == "BeliefUpdateCircuitConfig":
        from quantum_pomdp.quantum_circuits.belief_update import BeliefUpdateCircuitConfig
        return BeliefUpdateCircuitConfig
    if name == "QuantumBeliefUpdateCircuit":
        from quantum_pomdp.quantum_circuits.belief_update import QuantumBeliefUpdateCircuit
        return QuantumBeliefUpdateCircuit
    if name == "POMDPRegisterMap":
        from quantum_pomdp.quantum_circuits.register_map import POMDPRegisterMap
        return POMDPRegisterMap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
