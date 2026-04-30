"""QED-C Application-Oriented Benchmark adapter.

Lightweight bridge between QANTIS's benchmark harness and the QED-C /
MITRE "Application-Oriented Performance Benchmarks for Quantum
Computing" suite (Lubinski et al., 2024-2026; GitHub:
SRI-International/QC-App-Oriented-Benchmarks). Rather than vendoring the
upstream suite source, this module defines a common instance schema,
runner hooks, and a new MHT-assignment benchmark stub that can be
submitted upstream once its calibration data is collected.

Usage:
    >>> adapter = QEDCAdapter(suite_root="/path/to/qedc/checkout")
    >>> for instance in adapter.iter_benchmarks(["amplitude_estimation", "maxcut"]):
    ...     run_id = harness.submit(instance)

The adapter runs in three modes:

    1. **"in-process"** -- ``suite_root`` points to a checked-out QED-C
       repo; we import their benchmark modules and iterate the included
       problem instances.
    2. **"metadata-only"** -- no suite checkout; we expose a small
       hardcoded instance catalogue that matches QED-C's JSON schema so
       smoke-tests and CI can still exercise the QANTIS reporting path.
    3. **"mht-extension"** -- emits a new MHT / multi-hypothesis
       tracking benchmark family that QANTIS will submit upstream once
       the seed instances are finalised.

Academic references:
    Lubinski, Johri, Varosy, Coleman, Zhao, Necaise, Baldwin, Mayer,
        Proctor, "Application-Oriented Performance Benchmarks for
        Quantum Computing," IEEE Transactions on Quantum Engineering 4,
        3100332 (2023).
    QED-C / MITRE GitHub:
        https://github.com/SRI-International/QC-App-Oriented-Benchmarks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal
import importlib.util
import logging

logger = logging.getLogger(__name__)


@dataclass
class QEDCInstance:
    """Problem instance in QED-C / MITRE schema."""

    suite: str
    name: str
    size: int
    parameters: dict[str, Any] = field(default_factory=dict)
    qubit_count: int | None = None
    description: str | None = None


@dataclass
class QEDCAdapter:
    """Adapter between the QED-C benchmark suite and QANTIS harness.

    Attributes
    ----------
    suite_root : str | None
        Filesystem path to a checked-out ``QC-App-Oriented-Benchmarks``
        repo. When ``None`` the adapter runs in ``metadata-only`` mode.
    mode : {"in-process", "metadata-only", "mht-extension"}
    """

    suite_root: str | None = None
    mode: Literal["in-process", "metadata-only", "mht-extension"] = "metadata-only"

    def __post_init__(self) -> None:
        if self.suite_root is not None:
            path = Path(self.suite_root)
            if path.exists():
                self.mode = "in-process"
            else:
                logger.warning(
                    "QED-C suite_root %s not found; falling back to metadata-only mode",
                    path,
                )

    def iter_benchmarks(
        self, families: Iterable[str] | None = None,
    ) -> Iterator[QEDCInstance]:
        if self.mode == "in-process":
            yield from self._iter_in_process(families)
        else:
            yield from self._iter_metadata(families)

    def iter_mht_extension(self) -> Iterator[QEDCInstance]:
        """Emit the MHT-assignment benchmark family stub.

        QANTIS intends to upstream this family as a pull request to the
        QED-C suite once seed instances are finalised; until then the
        adapter ships a small representative set that the benchmark
        harness can use locally.
        """
        for n_tracks, n_meas, clutter_density in [
            (3, 5, 0.1),
            (5, 8, 0.2),
            (10, 15, 0.3),
            (20, 30, 0.3),
            (50, 75, 0.4),
        ]:
            yield QEDCInstance(
                suite="qantis_mht",
                name=f"mtda_{n_tracks}x{n_meas}_clutter{clutter_density:.1f}",
                size=n_tracks * n_meas + n_tracks + n_meas,
                parameters={
                    "n_tracks": n_tracks,
                    "n_measurements": n_meas,
                    "clutter_density": clutter_density,
                    "include_missed": True,
                    "include_false_alarm": True,
                },
                qubit_count=n_tracks * n_meas + n_tracks + n_meas,
                description=(
                    "Multi-Target Data Association QUBO instance with "
                    "Stollenwerk 2021 encoding (rows/cols one-hot plus "
                    "slack variables). Reviewer ammunition: include "
                    "Hungarian + PT-ICM + Gurobi classical baselines in "
                    "all reports per Shaydulin Sci Adv 2024."
                ),
            )

    # ------------------------------------------------------------------
    # Internal iteration paths
    # ------------------------------------------------------------------

    def _iter_in_process(
        self, families: Iterable[str] | None,
    ) -> Iterator[QEDCInstance]:
        if self.suite_root is None:
            return
        root = Path(self.suite_root)
        targets = list(families) if families is not None else [
            "amplitude_estimation",
            "maxcut",
            "vqe",
            "hamiltonian_simulation",
        ]
        for family in targets:
            family_dir = root / family
            if not family_dir.exists():
                logger.debug("QED-C family directory missing: %s", family_dir)
                continue
            spec = importlib.util.spec_from_file_location(
                f"qedc_{family}", family_dir / "__init__.py",
            )
            if spec is None or spec.loader is None:
                logger.debug("QED-C family %s has no __init__.py", family)
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # pragma: no cover - upstream-dependent
                logger.warning("Failed to import QED-C %s: %s", family, exc)
                continue
            iterator = getattr(module, "iter_instances", None)
            if iterator is None:
                logger.debug("QED-C %s has no iter_instances(); skipping", family)
                continue
            for raw in iterator():
                yield QEDCInstance(
                    suite=family,
                    name=str(raw.get("name", "unnamed")),
                    size=int(raw.get("size", 0)),
                    parameters=dict(raw.get("parameters", {})),
                    qubit_count=raw.get("qubit_count"),
                    description=raw.get("description"),
                )

    def _iter_metadata(
        self, families: Iterable[str] | None,
    ) -> Iterator[QEDCInstance]:
        catalogue: list[QEDCInstance] = [
            QEDCInstance(
                suite="amplitude_estimation",
                name="monte_carlo_n8",
                size=8,
                parameters={"precision_bits": 4},
                qubit_count=8,
            ),
            QEDCInstance(
                suite="maxcut",
                name="3-regular_n12",
                size=12,
                parameters={"graph_type": "3-regular", "reps": 2},
                qubit_count=12,
            ),
            QEDCInstance(
                suite="vqe",
                name="h2_sto3g",
                size=4,
                parameters={"molecule": "H2", "basis": "sto-3g"},
                qubit_count=4,
            ),
            QEDCInstance(
                suite="hamiltonian_simulation",
                name="tfim_n10",
                size=10,
                parameters={"model": "tfim", "trotter_steps": 4},
                qubit_count=10,
            ),
        ]
        if families is None:
            yield from catalogue
            return
        allowed = set(families)
        for instance in catalogue:
            if instance.suite in allowed:
                yield instance
