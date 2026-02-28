"""Tests for configuration loading."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from quantum_common.config.loader import load_experiment_config
from quantum_common.exceptions import ConfigurationError


class TestLoadExperimentConfig:
    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        config_data = {
            "name": "test_experiment",
            "backend": {"backend_type": "local_aer", "use_simulator": True},
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_experiment_config(config_file)
        assert config.name == "test_experiment"
        assert config.backend.use_simulator is True

    def test_missing_file_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_experiment_config("nonexistent.yaml")

    def test_overrides(self, tmp_path: Path) -> None:
        config_data = {
            "name": "base",
            "backend": {"backend_type": "local_aer"},
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        config = load_experiment_config(
            config_file, overrides={"name": "overridden"}
        )
        assert config.name == "overridden"

    def test_env_overrides(self, tmp_path: Path) -> None:
        config_data = {
            "name": "env_test",
            "backend": {"backend_type": "local_aer", "use_simulator": False},
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch.dict("os.environ", {"QA_BACKEND__USE_SIMULATOR": "true"}):
            config = load_experiment_config(config_file)
            assert config.backend.use_simulator is True
