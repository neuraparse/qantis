"""Tests for quantum_common.logging module — ExperimentSession and setup_logging."""

import json
from pathlib import Path

import pytest

from quantum_common.logging.experiment import ExperimentSession


class TestExperimentSessionCreation:
    """Verify ExperimentSession construction and defaults."""

    def test_creation_with_name(self) -> None:
        session = ExperimentSession(experiment_name="test_exp")
        assert session.experiment_name == "test_exp"

    def test_session_id_generated(self) -> None:
        session = ExperimentSession(experiment_name="test")
        assert isinstance(session.session_id, str)
        assert len(session.session_id) == 12

    def test_started_at_is_set(self) -> None:
        session = ExperimentSession(experiment_name="test")
        assert session.started_at is not None
        assert isinstance(session.started_at, str)
        # Should be ISO format
        assert "T" in session.started_at

    def test_ended_at_initially_none(self) -> None:
        session = ExperimentSession(experiment_name="test")
        assert session.ended_at is None

    def test_default_status_is_running(self) -> None:
        session = ExperimentSession(experiment_name="test")
        assert session.status == "running"

    def test_environment_populated(self) -> None:
        session = ExperimentSession(experiment_name="test")
        assert "python_version" in session.environment
        assert "platform" in session.environment
        assert "hostname" in session.environment

    def test_notes_initially_empty(self) -> None:
        session = ExperimentSession(experiment_name="test")
        assert session.notes == []

    def test_artifacts_initially_empty(self) -> None:
        session = ExperimentSession(experiment_name="test")
        assert session.artifacts == []

    def test_unique_session_ids(self) -> None:
        session1 = ExperimentSession(experiment_name="a")
        session2 = ExperimentSession(experiment_name="b")
        assert session1.session_id != session2.session_id


class TestExperimentSessionAddNote:
    """Verify add_note appends timestamped notes."""

    def test_add_note_appends(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.add_note("Started experiment")
        assert len(session.notes) == 1

    def test_add_note_contains_text(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.add_note("First observation")
        assert "First observation" in session.notes[0]

    def test_add_note_has_timestamp(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.add_note("Timestamped note")
        # Note format is "[<ISO timestamp>] <text>"
        assert session.notes[0].startswith("[")
        assert "]" in session.notes[0]

    def test_add_multiple_notes(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.add_note("Note 1")
        session.add_note("Note 2")
        session.add_note("Note 3")
        assert len(session.notes) == 3
        assert "Note 1" in session.notes[0]
        assert "Note 3" in session.notes[2]


class TestExperimentSessionAddArtifact:
    """Verify add_artifact appends paths."""

    def test_add_artifact_string(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.add_artifact("/path/to/results.csv")
        assert len(session.artifacts) == 1
        assert session.artifacts[0] == "/path/to/results.csv"

    def test_add_artifact_path_object(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.add_artifact(Path("/path/to/model.pkl"))
        assert len(session.artifacts) == 1
        assert session.artifacts[0] == str(Path("/path/to/model.pkl"))

    def test_add_multiple_artifacts(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.add_artifact("/artifact1.json")
        session.add_artifact("/artifact2.json")
        assert len(session.artifacts) == 2


class TestExperimentSessionFinish:
    """Verify finish sets ended_at and status."""

    def test_finish_sets_ended_at(self) -> None:
        session = ExperimentSession(experiment_name="test")
        assert session.ended_at is None
        session.finish()
        assert session.ended_at is not None
        assert isinstance(session.ended_at, str)

    def test_finish_default_status(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.finish()
        assert session.status == "completed"

    def test_finish_custom_status(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.finish(status="failed")
        assert session.status == "failed"

    def test_finish_cancelled_status(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.finish(status="cancelled")
        assert session.status == "cancelled"

    def test_ended_at_is_iso_format(self) -> None:
        session = ExperimentSession(experiment_name="test")
        session.finish()
        assert "T" in session.ended_at


class TestExperimentSessionSave:
    """Verify save persists session as JSON file."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        session = ExperimentSession(experiment_name="save_test")
        filepath = session.save(output_dir=tmp_path)
        assert filepath.exists()
        assert filepath.suffix == ".json"

    def test_save_file_contains_valid_json(self, tmp_path: Path) -> None:
        session = ExperimentSession(experiment_name="json_test")
        session.add_note("Test note")
        filepath = session.save(output_dir=tmp_path)

        data = json.loads(filepath.read_text())
        assert data["experiment_name"] == "json_test"
        assert len(data["notes"]) == 1

    def test_save_filename_includes_name_and_id(self, tmp_path: Path) -> None:
        session = ExperimentSession(experiment_name="my_exp")
        filepath = session.save(output_dir=tmp_path)
        assert "my_exp" in filepath.name
        assert session.session_id in filepath.name

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        session = ExperimentSession(experiment_name="test")
        output_dir = tmp_path / "experiments" / "run1"
        filepath = session.save(output_dir=output_dir)
        assert output_dir.exists()
        assert filepath.exists()

    def test_save_includes_all_fields(self, tmp_path: Path) -> None:
        session = ExperimentSession(
            experiment_name="full_test",
            parameters={"lr": 0.01},
        )
        session.add_note("Started")
        session.add_artifact("/data/result.csv")
        session.finish()

        filepath = session.save(output_dir=tmp_path)
        data = json.loads(filepath.read_text())

        assert "experiment_name" in data
        assert "session_id" in data
        assert "started_at" in data
        assert "ended_at" in data
        assert "parameters" in data
        assert "environment" in data
        assert "notes" in data
        assert "artifacts" in data
        assert "status" in data
        assert data["status"] == "completed"
        assert data["parameters"]["lr"] == 0.01


class TestSetupLogging:
    """Verify setup_logging function doesn't crash."""

    def test_setup_logging_default_args(self) -> None:
        from quantum_common.logging import setup_logging

        # Should not raise any exceptions
        setup_logging()

    def test_setup_logging_debug_level(self) -> None:
        from quantum_common.logging import setup_logging

        setup_logging(level="DEBUG")

    def test_setup_logging_json_format(self) -> None:
        from quantum_common.logging import setup_logging

        setup_logging(log_format="json")

    def test_setup_logging_with_log_file(self, tmp_path: Path) -> None:
        from quantum_common.logging import setup_logging

        log_file = str(tmp_path / "test.log")
        setup_logging(log_file=log_file)
