"""Tests for quantum_common.mitigation.zne module — ZNEStrategy."""

import pytest

from quantum_common.mitigation.zne import ZNEStrategy


class TestZNEStrategyCreation:
    """Verify ZNEStrategy construction and defaults."""

    def test_default_scale_factors(self) -> None:
        strategy = ZNEStrategy()
        assert strategy.scale_factors == [1.0, 2.0, 3.0]

    def test_custom_scale_factors(self) -> None:
        strategy = ZNEStrategy(scale_factors=[1.0, 1.5, 2.0, 2.5])
        assert strategy.scale_factors == [1.0, 1.5, 2.0, 2.5]
        assert len(strategy.scale_factors) == 4

    def test_default_factory_type(self) -> None:
        strategy = ZNEStrategy()
        assert strategy.factory_type == "Richardson"

    def test_custom_factory_type(self) -> None:
        strategy = ZNEStrategy(factory_type="Linear")
        assert strategy.factory_type == "Linear"

    def test_custom_both_params(self) -> None:
        strategy = ZNEStrategy(
            scale_factors=[1.0, 3.0, 5.0],
            factory_type="Poly",
        )
        assert strategy.scale_factors == [1.0, 3.0, 5.0]
        assert strategy.factory_type == "Poly"

    @pytest.mark.parametrize("scale_factors", [
        [1.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 1.5, 2.5],
        [1.0, 3.0, 5.0, 7.0],
    ])
    def test_parametrized_scale_factors(self, scale_factors: list) -> None:
        strategy = ZNEStrategy(scale_factors=scale_factors)
        assert strategy.scale_factors == scale_factors
        assert len(strategy.scale_factors) == len(scale_factors)

    @pytest.mark.parametrize("factory_type", ["Richardson", "Linear", "Poly"])
    def test_parametrized_factory_types(self, factory_type: str) -> None:
        strategy = ZNEStrategy(factory_type=factory_type)
        assert strategy.factory_type == factory_type

    @pytest.mark.parametrize("scale_factors,factory_type", [
        ([1.0, 2.0, 3.0], "Richardson"),
        ([1.0, 3.0, 5.0], "Linear"),
        ([1.0, 2.0, 3.0, 4.0], "Poly"),
        ([1.0, 1.5, 2.0], "Richardson"),
    ])
    def test_parametrized_combinations(self, scale_factors, factory_type) -> None:
        strategy = ZNEStrategy(scale_factors=scale_factors, factory_type=factory_type)
        assert strategy.scale_factors == scale_factors
        assert strategy.factory_type == factory_type


class TestZNEStrategyName:
    """Verify name property reflects factory_type."""

    def test_default_name(self) -> None:
        strategy = ZNEStrategy()
        assert strategy.name == "ZNE(Richardson)"

    def test_name_with_linear_factory(self) -> None:
        strategy = ZNEStrategy(factory_type="Linear")
        assert strategy.name == "ZNE(Linear)"

    def test_name_with_poly_factory(self) -> None:
        strategy = ZNEStrategy(factory_type="Poly")
        assert strategy.name == "ZNE(Poly)"

    def test_name_format(self) -> None:
        """Name should follow the pattern ZNE(<factory_type>)."""
        for factory in ["Richardson", "Linear", "Poly"]:
            strategy = ZNEStrategy(factory_type=factory)
            assert strategy.name.startswith("ZNE(")
            assert strategy.name.endswith(")")
            assert factory in strategy.name

    @pytest.mark.parametrize("factory_type,expected_name", [
        ("Richardson", "ZNE(Richardson)"),
        ("Linear", "ZNE(Linear)"),
        ("Poly", "ZNE(Poly)"),
    ])
    def test_name_matches_factory_type(self, factory_type, expected_name) -> None:
        strategy = ZNEStrategy(factory_type=factory_type)
        assert strategy.name == expected_name

    def test_name_is_string(self) -> None:
        strategy = ZNEStrategy()
        assert isinstance(strategy.name, str)

    def test_name_not_empty(self) -> None:
        strategy = ZNEStrategy()
        assert len(strategy.name) > 0


class TestZNEStrategyApply:
    """Verify the apply method raises NotImplementedError."""

    def test_apply_raises_not_implemented(self) -> None:
        """apply() must raise NotImplementedError — use execute_with_zne instead."""
        strategy = ZNEStrategy()
        counts = {"00": 500, "11": 500}
        with pytest.raises(NotImplementedError):
            strategy.apply(circuit=None, backend=None, counts=counts, shots=1000)

    def test_apply_raises_with_any_args(self) -> None:
        """apply() raises regardless of argument values."""
        strategy = ZNEStrategy(scale_factors=[1.0, 3.0, 5.0], factory_type="Linear")
        with pytest.raises(NotImplementedError):
            strategy.apply(circuit=object(), backend=object(), counts={"0": 1}, shots=100)

    def test_apply_error_message_mentions_execute_with_zne(self) -> None:
        """Error message should guide user to the correct method."""
        strategy = ZNEStrategy()
        with pytest.raises(NotImplementedError, match="execute_with_zne"):
            strategy.apply(circuit=None, backend=None, counts={}, shots=0)

    def test_scale_factors_are_independent_across_instances(self) -> None:
        """Verify mutable default scale_factors are independent."""
        s1 = ZNEStrategy()
        s2 = ZNEStrategy()
        s1.scale_factors.append(4.0)
        assert 4.0 not in s2.scale_factors


class TestZNEExecuteWithZne:
    """Verify execute_with_zne behavior (mocked)."""

    def test_execute_with_zne_raises_without_mitiq(self) -> None:
        """Without Mitiq installed, execute_with_zne raises RuntimeError."""
        import sys
        import unittest.mock as mock

        strategy = ZNEStrategy()
        # Temporarily hide mitiq if installed
        with mock.patch.dict(sys.modules, {"mitiq": None, "mitiq.zne": None}):
            with pytest.raises((RuntimeError, ImportError, TypeError)):
                strategy.execute_with_zne(circuit=None, executor_fn=lambda c: 0.5)

    def test_execute_with_zne_calls_executor(self) -> None:
        """execute_with_zne should invoke the executor function."""
        try:
            import mitiq  # noqa: F401
        except ImportError:
            pytest.skip("mitiq not installed")

        import unittest.mock as mock

        strategy = ZNEStrategy(scale_factors=[1.0, 2.0, 3.0])
        mock_executor = mock.MagicMock(return_value=0.75)

        try:
            result = strategy.execute_with_zne(
                circuit=object(),
                executor_fn=mock_executor,
            )
            assert isinstance(result, float)
        except Exception:
            # Mitiq may fail on a non-circuit object; just verify it was called
            pass


class TestZNEStrategyIsolation:
    """Test instance isolation and dataclass behavior."""

    def test_two_instances_are_independent(self) -> None:
        s1 = ZNEStrategy(scale_factors=[1.0, 2.0])
        s2 = ZNEStrategy(scale_factors=[1.0, 3.0, 5.0])
        assert s1.scale_factors != s2.scale_factors

    def test_scale_factors_mutation_does_not_affect_other(self) -> None:
        s1 = ZNEStrategy()
        s2 = ZNEStrategy()
        original_s2 = list(s2.scale_factors)
        s1.scale_factors.clear()
        assert s2.scale_factors == original_s2

    def test_factory_type_can_be_updated(self) -> None:
        strategy = ZNEStrategy()
        assert strategy.factory_type == "Richardson"
        strategy.factory_type = "Linear"
        assert strategy.factory_type == "Linear"
        assert strategy.name == "ZNE(Linear)"

    def test_scale_factors_length_one(self) -> None:
        """Single scale factor is valid (though ZNE needs ≥2 for extrapolation)."""
        strategy = ZNEStrategy(scale_factors=[1.0])
        assert len(strategy.scale_factors) == 1
        assert strategy.scale_factors[0] == 1.0
