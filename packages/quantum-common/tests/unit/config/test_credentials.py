"""Tests for quantum_common.config.credentials module — credential validation."""

import os

import pytest

from quantum_common.config.credentials import validate_credentials
from quantum_common.config.schema import Credentials
from quantum_common.exceptions import ConfigurationError


class TestValidateCredentials:
    """Verify validate_credentials returns correct structure."""

    def test_returns_dict_with_expected_keys(self) -> None:
        result = validate_credentials()
        assert isinstance(result, dict)
        assert "ibm_quantum" in result
        assert "dwave" in result
        assert "azure_quantum" in result

    def test_all_keys_are_boolean(self) -> None:
        result = validate_credentials()
        for key, value in result.items():
            assert isinstance(value, bool), f"Expected bool for '{key}', got {type(value)}"

    def test_no_env_vars_all_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no credential env vars set, all values should be False."""
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_SUBSCRIPTION_ID", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_RESOURCE_GROUP", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_WORKSPACE", raising=False)

        creds = Credentials(_env_file=None)
        result = validate_credentials(creds)

        assert result["ibm_quantum"] is False
        assert result["dwave"] is False
        assert result["azure_quantum"] is False

    def test_ibm_token_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When IBM token is provided via env var, ibm_quantum should be True."""
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "fake-ibm-token")
        monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_SUBSCRIPTION_ID", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_RESOURCE_GROUP", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_WORKSPACE", raising=False)

        creds = Credentials(_env_file=None)
        result = validate_credentials(creds)

        assert result["ibm_quantum"] is True
        assert result["dwave"] is False
        assert result["azure_quantum"] is False

    def test_dwave_token_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When D-Wave token is provided via env var, dwave should be True."""
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        monkeypatch.setenv("DWAVE_API_TOKEN", "fake-dwave-token")
        monkeypatch.delenv("AZURE_QUANTUM_SUBSCRIPTION_ID", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_RESOURCE_GROUP", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_WORKSPACE", raising=False)

        creds = Credentials(_env_file=None)
        result = validate_credentials(creds)

        assert result["ibm_quantum"] is False
        assert result["dwave"] is True

    def test_azure_requires_all_three_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Azure quantum requires subscription_id, resource_group, and workspace."""
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
        # Only subscription_id set - should be False
        monkeypatch.setenv("AZURE_QUANTUM_SUBSCRIPTION_ID", "sub-123")
        monkeypatch.delenv("AZURE_QUANTUM_RESOURCE_GROUP", raising=False)
        monkeypatch.delenv("AZURE_QUANTUM_WORKSPACE", raising=False)

        creds = Credentials(_env_file=None)
        result = validate_credentials(creds)
        assert result["azure_quantum"] is False

    def test_azure_all_fields_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When all Azure fields are provided, azure_quantum should be True."""
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
        monkeypatch.setenv("AZURE_QUANTUM_SUBSCRIPTION_ID", "sub-123")
        monkeypatch.setenv("AZURE_QUANTUM_RESOURCE_GROUP", "rg-quantum")
        monkeypatch.setenv("AZURE_QUANTUM_WORKSPACE", "ws-quantum")

        creds = Credentials(_env_file=None)
        result = validate_credentials(creds)
        assert result["azure_quantum"] is True

    def test_all_credentials_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When all credentials are provided, all values should be True."""
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "ibm-token")
        monkeypatch.setenv("DWAVE_API_TOKEN", "dwave-token")
        monkeypatch.setenv("AZURE_QUANTUM_SUBSCRIPTION_ID", "sub-123")
        monkeypatch.setenv("AZURE_QUANTUM_RESOURCE_GROUP", "rg-quantum")
        monkeypatch.setenv("AZURE_QUANTUM_WORKSPACE", "ws-quantum")

        creds = Credentials(_env_file=None)
        result = validate_credentials(creds)
        assert result["ibm_quantum"] is True
        assert result["dwave"] is True
        assert result["azure_quantum"] is True
