"""Secure credential management for quantum backend authentication."""

from __future__ import annotations

import logging

from quantum_common.config.schema import Credentials
from quantum_common.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def get_ibm_token(credentials: Credentials | None = None) -> str:
    """Retrieve IBM Quantum API token from credentials."""
    creds = credentials or Credentials()
    if creds.ibm_quantum_token is None:
        raise ConfigurationError(
            "IBM Quantum token not found. Set IBM_QUANTUM_TOKEN environment variable."
        )
    return creds.ibm_quantum_token.get_secret_value()


def get_dwave_token(credentials: Credentials | None = None) -> str:
    """Retrieve D-Wave API token from credentials."""
    creds = credentials or Credentials()
    if creds.dwave_api_token is None:
        raise ConfigurationError(
            "D-Wave API token not found. Set DWAVE_API_TOKEN environment variable."
        )
    return creds.dwave_api_token.get_secret_value()


def validate_credentials(credentials: Credentials | None = None) -> dict[str, bool]:
    """Check which backend credentials are available."""
    creds = credentials or Credentials()
    return {
        "ibm_quantum": creds.ibm_quantum_token is not None,
        "dwave": creds.dwave_api_token is not None,
        "azure_quantum": all([
            creds.azure_subscription_id,
            creds.azure_resource_group,
            creds.azure_workspace,
        ]),
    }
