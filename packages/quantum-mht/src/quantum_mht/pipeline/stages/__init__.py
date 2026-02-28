"""Pipeline stages for MHT tracking.

Stages implement the predict -> gate -> QUBO -> solve -> update -> manage
pipeline (Stollenwerk et al., arXiv:2110.08346, 2021). Each stage is a
thin orchestrator that delegates to the corresponding tracking/formulation
module.
"""
from quantum_mht.pipeline.stages.prediction_stage import PredictionStage
from quantum_mht.pipeline.stages.gating_stage import GatingStage
from quantum_mht.pipeline.stages.association_stage import AssociationStage
from quantum_mht.pipeline.stages.update_stage import UpdateStage
from quantum_mht.pipeline.stages.management_stage import ManagementStage

__all__ = [
    "PredictionStage",
    "GatingStage",
    "AssociationStage",
    "UpdateStage",
    "ManagementStage",
]
