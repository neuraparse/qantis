"""Experiment metadata tracking and session management."""

from __future__ import annotations

import json
import platform
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentSession:
    """Tracks metadata for a single experiment run."""

    experiment_name: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    status: str = "running"

    def __post_init__(self) -> None:
        self.environment = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        }

    def add_note(self, note: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self.notes.append(f"[{ts}] {note}")

    def add_artifact(self, path: str | Path) -> None:
        self.artifacts.append(str(path))

    def finish(self, status: str = "completed") -> None:
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = status

    def save(self, output_dir: Path | str = "experiments") -> Path:
        """Persist session metadata to JSON."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.experiment_name}_{self.session_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path
