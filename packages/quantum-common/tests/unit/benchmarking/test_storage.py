"""Tests for quantum_common.benchmarking.storage module — JSON and SQLite storage."""

import json
from pathlib import Path

import pytest

from quantum_common.benchmarking.storage import JSONStorage, SQLiteStorage


class TestJSONStorage:
    """Verify JSONStorage save and load operations."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        storage = JSONStorage(output_dir=tmp_path / "results")
        results = [{"case": "test1", "time": 1.5}]
        filepath = storage.save(results, experiment_name="exp1")

        assert filepath.exists()
        assert filepath.suffix == ".json"

    def test_save_file_content(self, tmp_path: Path) -> None:
        storage = JSONStorage(output_dir=tmp_path / "results")
        results = [{"case": "test1", "metric": 0.95}]
        filepath = storage.save(results, experiment_name="my_exp")

        data = json.loads(filepath.read_text())
        assert data["experiment"] == "my_exp"
        assert data["results"] == results
        assert "timestamp" in data

    def test_save_creates_output_directory(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "deep" / "nested" / "results"
        storage = JSONStorage(output_dir=nested_dir)
        storage.save([{"data": 1}], experiment_name="test")

        assert nested_dir.exists()

    def test_load_latest_returns_saved_data(self, tmp_path: Path) -> None:
        storage = JSONStorage(output_dir=tmp_path / "results")
        original_results = [{"case": "test", "value": 42}]
        storage.save(original_results, experiment_name="exp_latest")

        loaded = storage.load_latest("exp_latest")
        assert loaded is not None
        assert loaded["experiment"] == "exp_latest"
        assert loaded["results"] == original_results

    def test_load_latest_returns_none_when_no_files(self, tmp_path: Path) -> None:
        storage = JSONStorage(output_dir=tmp_path / "empty_results")
        loaded = storage.load_latest("nonexistent")
        assert loaded is None

    def test_load_latest_returns_most_recent(self, tmp_path: Path) -> None:
        """When multiple files exist, load_latest returns the last one sorted."""
        storage = JSONStorage(output_dir=tmp_path / "results")

        storage.save([{"run": 1}], experiment_name="multi")
        # Save a second time to create two files
        import time
        time.sleep(0.01)  # Ensure different timestamp
        storage.save([{"run": 2}], experiment_name="multi")

        loaded = storage.load_latest("multi")
        assert loaded is not None
        assert loaded["results"] == [{"run": 2}]

    def test_save_multiple_experiments(self, tmp_path: Path) -> None:
        storage = JSONStorage(output_dir=tmp_path / "results")
        storage.save([{"data": "a"}], experiment_name="exp_a")
        storage.save([{"data": "b"}], experiment_name="exp_b")

        loaded_a = storage.load_latest("exp_a")
        loaded_b = storage.load_latest("exp_b")
        assert loaded_a is not None
        assert loaded_b is not None
        assert loaded_a["experiment"] == "exp_a"
        assert loaded_b["experiment"] == "exp_b"


class TestSQLiteStorage:
    """Verify SQLiteStorage save_metric and query operations."""

    def test_save_metric_and_query(self, tmp_path: Path) -> None:
        db_path = tmp_path / "benchmark.db"
        storage = SQLiteStorage(db_path=db_path)

        storage.save_metric(
            experiment_name="exp1",
            case_name="case1",
            metric_name="execution_time",
            value=1.234,
            unit="seconds",
        )

        results = storage.query("exp1")
        assert len(results) == 1
        assert results[0]["metric_name"] == "execution_time"
        assert results[0]["metric_value"] == pytest.approx(1.234)
        assert results[0]["metric_unit"] == "seconds"

    def test_save_metric_with_metadata(self, tmp_path: Path) -> None:
        db_path = tmp_path / "benchmark.db"
        storage = SQLiteStorage(db_path=db_path)

        metadata = {"backend": "ibm_heron", "shots": 4096}
        storage.save_metric(
            experiment_name="exp1",
            case_name="case1",
            metric_name="fidelity",
            value=0.99,
            metadata=metadata,
        )

        results = storage.query("exp1")
        assert len(results) == 1
        stored_metadata = json.loads(results[0]["metadata"])
        assert stored_metadata["backend"] == "ibm_heron"
        assert stored_metadata["shots"] == 4096

    def test_query_filters_by_experiment(self, tmp_path: Path) -> None:
        db_path = tmp_path / "benchmark.db"
        storage = SQLiteStorage(db_path=db_path)

        storage.save_metric("exp1", "case1", "time", 1.0)
        storage.save_metric("exp2", "case1", "time", 2.0)

        results_exp1 = storage.query("exp1")
        results_exp2 = storage.query("exp2")
        assert len(results_exp1) == 1
        assert len(results_exp2) == 1
        assert results_exp1[0]["metric_value"] == pytest.approx(1.0)
        assert results_exp2[0]["metric_value"] == pytest.approx(2.0)

    def test_query_filters_by_metric_name(self, tmp_path: Path) -> None:
        db_path = tmp_path / "benchmark.db"
        storage = SQLiteStorage(db_path=db_path)

        storage.save_metric("exp1", "case1", "time", 1.0)
        storage.save_metric("exp1", "case1", "accuracy", 0.95)
        storage.save_metric("exp1", "case1", "fidelity", 0.99)

        time_results = storage.query("exp1", metric_name="time")
        assert len(time_results) == 1
        assert time_results[0]["metric_name"] == "time"

    def test_query_returns_empty_for_nonexistent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "benchmark.db"
        storage = SQLiteStorage(db_path=db_path)

        results = storage.query("nonexistent")
        assert results == []

    def test_multiple_metrics_same_experiment(self, tmp_path: Path) -> None:
        db_path = tmp_path / "benchmark.db"
        storage = SQLiteStorage(db_path=db_path)

        storage.save_metric("exp1", "case1", "time", 1.0)
        storage.save_metric("exp1", "case2", "time", 2.0)
        storage.save_metric("exp1", "case1", "accuracy", 0.9)

        all_results = storage.query("exp1")
        assert len(all_results) == 3

    def test_db_file_created(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        SQLiteStorage(db_path=db_path)
        assert db_path.exists()

    def test_db_nested_directory_created(self, tmp_path: Path) -> None:
        db_path = tmp_path / "deep" / "nested" / "test.db"
        SQLiteStorage(db_path=db_path)
        assert db_path.parent.exists()
