"""Seed management for reproducible experiments."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import numpy as np

@dataclass
class SeedManager:
    """Manage random seeds for reproducible quantum experiments."""
    base_seed: int = 42
    _counter: int = field(default=0, init=False)

    def get_seed(self, label: str = "") -> int:
        """Get a deterministic seed based on base_seed and label."""
        if label:
            hash_input = f"{self.base_seed}:{label}".encode()
            return int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
        self._counter += 1
        return self.base_seed + self._counter

    def get_rng(self, label: str = "") -> np.random.Generator:
        """Get a numpy random generator with deterministic seed."""
        return np.random.default_rng(self.get_seed(label))

    def reset(self) -> None:
        """Reset the counter."""
        self._counter = 0
