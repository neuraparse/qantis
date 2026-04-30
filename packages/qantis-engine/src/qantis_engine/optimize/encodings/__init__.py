"""Feasible-subspace encodings for constraint-native optimization.

Each encoder exposes:
    - ``num_qubits()``         — qubit budget given the problem size
    - ``feasible_states()``    — enumerator for very small instances (testing)
    - ``project(bitstring)``   — closest feasible bitstring (for repair)
    - ``initial_state()``      — Dicke-like superposition over feasible set
"""
from __future__ import annotations

from qantis_engine.optimize.encodings.colored_permutation import ColoredPermutationEncoding
from qantis_engine.optimize.encodings.dicke_subspace import DickeSubspaceEncoding
from qantis_engine.optimize.encodings.hamming_weight import HammingWeightEncoding
from qantis_engine.optimize.encodings.permutation_oracle import PermutationOracleEncoding

__all__ = [
    "HammingWeightEncoding",
    "PermutationOracleEncoding",
    "ColoredPermutationEncoding",
    "DickeSubspaceEncoding",
]
