"""Zero-Noise Extrapolation (ZNE) via Mitiq >= 0.44.

Three extrapolation modes are exposed:

    - **Global folding ZNE** (default) -- Temme et al. PRL 119, 180509
      (2017); Li & Benjamin, PRX 7, 021050 (2017). Richardson / Linear /
      Polynomial extrapolation of expectation values under scaled noise.
    - **Layerwise Richardson Extrapolation (LRE)** -- Russo, Mari,
      LaRose et al., PRX Quantum 5, 040313 (2024). Scales noise per
      layer rather than globally. Preferred when circuit depth > 40
      layers or noise is spatially heterogeneous (Heron R3 regime).
    - **Non-Clifford ZNE** -- Ezzell, Pokharel, Lidar,
      Quantum 10, 2003 (2026), DOI 10.22331/q-2026-02-10-2003. Custom
      ``scale_method`` that inserts identity circuits around continuous
      non-Clifford rotations (R_x, R_y, R_z) to amplify coherent
      over-rotation noise linearly. Advertised here; implementation
      lives in ``NonCliffordScaling`` (experimental).

API compatibility: this module targets Mitiq >= 0.44 (Feb 2026). The
``scaled_circuits`` helper was renamed ``construct_circuits`` in that
release; the names of extrapolation factories (``RichardsonFactory``,
``LinearFactory``, ``PolyFactory``) are unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import logging
from quantum_common.mitigation.pipeline import MitigationStrategy

logger = logging.getLogger(__name__)


@dataclass
class ZNEStrategy(MitigationStrategy):
    """Global-folding Zero-Noise Extrapolation (Mitiq wrapper).

    Attributes
    ----------
    scale_factors : list[float]
        Noise amplification factors. Defaults follow Mitiq best-practice
        (1.0, 2.0, 3.0) for Richardson extrapolation.
    factory_type : {"Richardson", "Linear", "Poly"}
        Extrapolation inference method.
    """

    scale_factors: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    factory_type: Literal["Richardson", "Linear", "Poly"] = "Richardson"

    @property
    def name(self) -> str:
        return f"ZNE({self.factory_type})"

    def apply(
        self,
        circuit: Any,
        backend: Any,
        counts: dict[str, int],
        shots: int,
    ) -> dict[str, int]:
        """ZNE is an expectation-value procedure, not a counts-rewriter.

        Prefer :meth:`execute_with_zne` or :meth:`construct_circuits`
        from callers that know how to evaluate expectation values.
        """
        raise NotImplementedError(
            "ZNEStrategy.apply() is not a counts transform. Use "
            "execute_with_zne() or construct_circuits()."
        )

    def construct_circuits(self, circuit: Any) -> list[Any]:
        """Return the list of noise-scaled folded circuits.

        Uses Mitiq's ``zne.scaling.fold_gates_at_random`` primitive which
        has a stable call signature across 0.38 -> 0.46: ``fold(circuit,
        scale_factor)`` returns a single folded circuit. The top-level
        ``zne.construct_circuits`` helper was renamed several times;
        going through the scaling primitive directly avoids churn.
        """
        from mitiq import zne  # type: ignore

        folder = zne.scaling.fold_gates_at_random
        return [folder(circuit, scale_factor=s) for s in self.scale_factors]

    def execute_with_zne(self, circuit: Any, executor: Callable[[Any], float]) -> float:
        """Full ZNE evaluation; ``executor`` returns a scalar expectation value."""
        from mitiq import zne  # type: ignore

        factory = self._factory(zne)
        return zne.execute_with_zne(
            circuit,
            executor,
            scale_noise=zne.scaling.fold_gates_at_random,
            factory=factory,
        )

    def _factory(self, zne_module):
        inference = zne_module.inference
        factory_map = {
            "Richardson": inference.RichardsonFactory,
            "Linear": inference.LinearFactory,
            "Poly": inference.PolyFactory,
        }
        cls = factory_map[self.factory_type]
        if self.factory_type == "Poly":
            return cls(scale_factors=self.scale_factors, order=2)
        return cls(scale_factors=self.scale_factors)


def non_clifford_scale_method(
    circuit: Any,
    scale_factor: float,
    rotation_gate_names: tuple[str, ...] = ("rx", "ry", "rz"),
) -> Any:
    """Ezzell-Pokharel-Lidar non-Clifford ZNE scaler (Quantum 10:2003, 2026).

    Amplifies noise on continuous rotation gates without perturbing the
    encoded angle: for each occurrence of ``R_alpha(theta)`` the scaler
    inserts identity pairs ``R_alpha(phi) R_alpha(-phi)`` with
    ``phi = (scale_factor - 1) * theta / 2``. The logical rotation
    remains ``theta`` while the coherent-over-rotation noise scales
    linearly with ``scale_factor``. Validated on 127-qubit Eagle in
    Ezzell et al. Quantum 10, 2003 (2026), DOI
    10.22331/q-2026-02-10-2003.

    Suitable as the ``scale_noise`` argument in Mitiq 0.44+'s
    ``execute_with_zne`` or ``construct_circuits``. Falls back to the
    input circuit unchanged when the circuit framework does not expose
    the expected Qiskit-like API.
    """
    if scale_factor <= 1.0:
        return circuit

    try:
        from qiskit.circuit import QuantumCircuit  # type: ignore
    except ImportError:
        return circuit

    if not isinstance(circuit, QuantumCircuit):
        return circuit

    scaled = circuit.copy_empty_like()
    phi_ratio = (scale_factor - 1.0) / 2.0
    for instr in circuit.data:
        op = instr.operation
        qubits = instr.qubits
        cbits = instr.clbits
        name = op.name.lower()
        scaled.append(op, qubits, cbits)
        if name in rotation_gate_names and op.params:
            theta = float(op.params[0])
            phi = phi_ratio * theta
            if abs(phi) < 1e-12:
                continue
            forward = op.__class__(phi)
            backward = op.__class__(-phi)
            scaled.append(forward, qubits, cbits)
            scaled.append(backward, qubits, cbits)
    return scaled


@dataclass
class LREStrategy(MitigationStrategy):
    """Layerwise Richardson Extrapolation (Russo, Mari, LaRose et al. 2024).

    Scales noise per circuit layer rather than folding the whole circuit.
    Prefer for:

        - circuit depth > 40 layers,
        - spatially heterogeneous noise (Heron R2/R3 fleet),
        - QAOA with p >= 3 where global folds overcount coherent errors.

    Reference
    ---------
    Russo, Mari, LaRose et al., PRX Quantum 5, 040313 (2024),
    DOI 10.1103/PRXQuantum.5.040313.
    """

    degree: int = 2
    fold_multiplier: int = 3
    num_chunks: int | None = None

    @property
    def name(self) -> str:
        return f"LRE(deg={self.degree}, m={self.fold_multiplier})"

    def apply(
        self,
        circuit: Any,
        backend: Any,
        counts: dict[str, int],
        shots: int,
    ) -> dict[str, int]:
        raise NotImplementedError(
            "LREStrategy.apply() is not a counts transform; use execute_with_lre."
        )

    def execute_with_lre(self, circuit: Any, executor: Callable[[Any], float]) -> float:
        from mitiq.lre import execute_with_lre  # type: ignore

        return execute_with_lre(
            circuit,
            executor,
            degree=self.degree,
            fold_multiplier=self.fold_multiplier,
            num_chunks=self.num_chunks,
        )
