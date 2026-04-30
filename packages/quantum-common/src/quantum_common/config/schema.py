"""Pydantic models for type-safe configuration validation.

2026 Academic References — Configuration Patterns
====================================================
- **Pydantic v2.10 Settings pattern**: Uses ``pydantic-settings`` for
  environment-variable-backed configuration with type validation.
  ``BaseSettings`` supports ``.env`` file loading, ``SecretStr`` for
  credential masking, and ``Field(alias=...)`` for environment variable
  name mapping. See: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

- **Credentials management**: Quantum API tokens (IBM Quantum, D-Wave,
  Azure Quantum) are loaded from environment variables via ``BaseSettings``
  and never serialized to config files. ``SecretStr`` ensures tokens are
  masked in logs and repr() output.

- **Framework version alignment**: Configuration fields map to parameters
  for Qiskit v2.3, D-Wave Ocean SDK 9.x, PennyLane v0.44, and Azure
  Quantum SDK 2.3. See ``BackendConfig`` for per-framework options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings


class BackendConfig(BaseModel):
    """Configuration for a single quantum backend."""

    backend_type: str
    enabled: bool = True
    constructor_kwargs: dict[str, Any] = Field(default_factory=dict)

    # IBM-specific
    channel: str | None = None
    instance: str | None = None
    backend_name: str | None = None

    # D-Wave-specific
    use_hybrid: bool = False
    num_reads: int = 1000
    annealing_time_us: int = 20

    # Common
    use_simulator: bool = False
    max_shots: int = 100_000
    optimization_level: int = 1


class MitigationConfig(BaseModel):
    """Configuration for error mitigation strategies."""

    enabled: bool = True
    strategies: list[str] = Field(default_factory=lambda: ["zne"])
    zne_scale_factors: list[float] = [1.0, 2.0, 3.0]
    zne_factory: Literal["linear", "richardson", "exponential"] = "richardson"
    readout_calibration_shots: int = 1000


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""

    output_dir: Path = Path("benchmark_results")
    use_sqlite: bool = False
    num_trials: int = 5
    base_seed: int = 42
    problem_sizes: list[int] = [5, 10, 15, 20]
    shots_per_trial: int = 1024


class LoggingConfig(BaseModel):
    """Structured logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "console"
    log_file: Path | None = None
    include_timestamps: bool = True


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str
    description: str = ""
    backend: BackendConfig
    mitigation: MitigationConfig = Field(default_factory=MitigationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    case_parameters: dict[str, Any] = Field(default_factory=dict)


class Credentials(BaseSettings):
    """Secrets loaded from environment variables. Never stored in config files.

    Uses Pydantic v2.10 ``BaseSettings`` with ``.env`` file support and
    ``SecretStr`` for safe credential handling. Environment variable aliases
    follow the naming conventions of each quantum framework:
    - IBM_QUANTUM_TOKEN: Qiskit Runtime v0.36+ authentication
    - DWAVE_API_TOKEN: D-Wave Ocean SDK 9.x Leap API access
    - AZURE_QUANTUM_*: Azure Quantum SDK 2.3 workspace configuration
    """

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Allow user-specific .env additions (per-collaborator tokens such as
        # IBM_QUANTUM_TOKEN_ALICE / IBM_QUANTUM_INSTANCE_BOB) without
        # failing validation. Only the declared fields below are promoted
        # to attributes; anything else is ignored.
        "extra": "ignore",
    }

    ibm_quantum_token: SecretStr | None = Field(
        default=None, alias="IBM_QUANTUM_TOKEN"
    )
    dwave_api_token: SecretStr | None = Field(
        default=None, alias="DWAVE_API_TOKEN"
    )
    azure_subscription_id: str | None = Field(
        default=None, alias="AZURE_QUANTUM_SUBSCRIPTION_ID"
    )
    azure_resource_group: str | None = Field(
        default=None, alias="AZURE_QUANTUM_RESOURCE_GROUP"
    )
    azure_workspace: str | None = Field(
        default=None, alias="AZURE_QUANTUM_WORKSPACE"
    )
