"""YAML/TOML configuration loading with environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from quantum_common.config.schema import Credentials, ExperimentConfig
from quantum_common.exceptions import ConfigurationError


def load_experiment_config(
    config_path: Path | str,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML.

    Args:
        config_path: Path to YAML configuration file.
        overrides: Optional dict of key-value overrides.

    Returns:
        Validated ExperimentConfig instance.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        _deep_merge(raw, overrides)

    _apply_env_overrides(raw)

    try:
        return ExperimentConfig(**raw)
    except Exception as e:
        raise ConfigurationError(f"Invalid configuration in {path}: {e}") from e


def load_credentials() -> Credentials:
    """Load credentials from environment variables / .env file."""
    return Credentials()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Apply QA_* environment variables as config overrides.

    Convention: QA_BACKEND__USE_SIMULATOR=true -> config["backend"]["use_simulator"] = True
    Double underscore (__) separates nested keys.
    """
    prefix = "QA_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix) :].lower().split("__")
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            if value.lower() in ("true", "false"):
                target[parts[-1]] = value.lower() == "true"
            elif value.isdigit():
                target[parts[-1]] = int(value)
            else:
                target[parts[-1]] = value
