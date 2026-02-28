"""POMDP pipeline for hybrid quantum-classical execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantum_pomdp.pipeline.belief_updater import QuantumBeliefUpdater
    from quantum_pomdp.pipeline.hybrid_pipeline import HybridPOMDPPipeline

__all__ = [
    "HybridPOMDPPipeline",
    "QuantumBeliefUpdater",
]


def __getattr__(name: str):  # noqa: N807
    if name == "HybridPOMDPPipeline":
        from quantum_pomdp.pipeline.hybrid_pipeline import HybridPOMDPPipeline
        return HybridPOMDPPipeline
    if name == "QuantumBeliefUpdater":
        from quantum_pomdp.pipeline.belief_updater import QuantumBeliefUpdater
        return QuantumBeliefUpdater
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
