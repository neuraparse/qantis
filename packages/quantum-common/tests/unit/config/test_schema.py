"""Tests for configuration schema validation."""

from quantum_common.config.schema import (
    BackendConfig,
    BenchmarkConfig,
    ExperimentConfig,
    LoggingConfig,
    MitigationConfig,
)


class TestBackendConfig:
    def test_default_values(self) -> None:
        config = BackendConfig(backend_type="local_aer")
        assert config.backend_type == "local_aer"
        assert config.enabled is True
        assert config.use_simulator is False
        assert config.optimization_level == 1

    def test_ibm_config(self) -> None:
        config = BackendConfig(
            backend_type="ibm_quantum",
            channel="ibm_quantum",
            instance="ibm-q/open/main",
            backend_name="ibm_brisbane",
        )
        assert config.channel == "ibm_quantum"
        assert config.backend_name == "ibm_brisbane"

    def test_dwave_config(self) -> None:
        config = BackendConfig(
            backend_type="dwave",
            use_hybrid=True,
            num_reads=2000,
            annealing_time_us=50,
        )
        assert config.use_hybrid is True
        assert config.num_reads == 2000
        assert config.annealing_time_us == 50


class TestExperimentConfig:
    def test_minimal_config(self) -> None:
        config = ExperimentConfig(
            name="test",
            backend=BackendConfig(backend_type="local_aer"),
        )
        assert config.name == "test"
        assert config.mitigation.enabled is True
        assert config.benchmark.num_trials == 5
        assert config.logging.level == "INFO"

    def test_full_config(self) -> None:
        config = ExperimentConfig(
            name="full_test",
            description="A comprehensive test",
            backend=BackendConfig(backend_type="ibm_quantum", use_simulator=True),
            mitigation=MitigationConfig(
                enabled=True,
                strategies=["zne", "readout"],
                zne_scale_factors=[1.0, 1.5, 2.0],
            ),
            benchmark=BenchmarkConfig(
                num_trials=10,
                base_seed=123,
                problem_sizes=[4, 8, 16],
            ),
            logging=LoggingConfig(level="DEBUG", format="json"),
            case_parameters={"horizon": 20, "grid_size": 4},
        )
        assert config.description == "A comprehensive test"
        assert len(config.mitigation.strategies) == 2
        assert config.case_parameters["horizon"] == 20


class TestMitigationConfig:
    def test_defaults(self) -> None:
        config = MitigationConfig()
        assert config.enabled is True
        assert "zne" in config.strategies
        assert config.zne_factory == "richardson"

    def test_custom_scales(self) -> None:
        config = MitigationConfig(zne_scale_factors=[1.0, 3.0, 5.0])
        assert len(config.zne_scale_factors) == 3
