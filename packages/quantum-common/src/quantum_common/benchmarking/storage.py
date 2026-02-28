"""Benchmark result persistence (JSON + SQLite)."""
from __future__ import annotations
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class JSONStorage:
    """Store benchmark results as JSON files."""
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: list[dict[str, Any]], experiment_name: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump({"experiment": experiment_name, "timestamp": timestamp, "results": results}, f, indent=2, default=str)
        logger.info("Saved results to %s", filepath)
        return filepath

    def load_latest(self, experiment_name: str) -> dict[str, Any] | None:
        pattern = f"{experiment_name}_*.json"
        files = sorted(self.output_dir.glob(pattern))
        if not files:
            return None
        with open(files[-1]) as f:
            return json.load(f)


class SQLiteStorage:
    """Store benchmark results in SQLite database."""
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    case_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment ON benchmark_results(experiment_name)")

    def save_metric(self, experiment_name: str, case_name: str, metric_name: str, value: float, unit: str = "", metadata: dict | None = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO benchmark_results (experiment_name, case_name, metric_name, metric_value, metric_unit, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (experiment_name, case_name, metric_name, value, unit, json.dumps(metadata or {})),
            )

    def query(self, experiment_name: str, metric_name: str | None = None) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if metric_name:
                rows = conn.execute(
                    "SELECT * FROM benchmark_results WHERE experiment_name = ? AND metric_name = ? ORDER BY created_at",
                    (experiment_name, metric_name),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM benchmark_results WHERE experiment_name = ? ORDER BY created_at",
                    (experiment_name,),
                ).fetchall()
            return [dict(r) for r in rows]
