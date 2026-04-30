"""One-shot IBM Quantum account setup from .env file.

Reads the token / channel / instance triple from the repo ``.env`` and
writes them via ``QiskitRuntimeService.save_account`` so every hardware
script can do ``QiskitRuntimeService()`` with no arguments.

Supports multiple profiles (e.g. US-East pay, EU-DE pay, free Open plan)
selectable via ``--profile``. Each profile maps to a suffixed env-var
group:

    --profile us    ->  IBM_QUANTUM_TOKEN_US   / IBM_QUANTUM_INSTANCE_US
    --profile eu    ->  IBM_QUANTUM_TOKEN_EU   / IBM_QUANTUM_INSTANCE_EU
    --profile open  ->  IBM_QUANTUM_TOKEN_OPEN (instance auto-resolved)
    (no flag)       ->  legacy IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE

Per-operator aliases (``ibm_quantum_token_bayram``,
``ibm_quantum_instance_bayram``) also resolved as a last-chance
fallback so the user's historical .env keys keep working.

Usage:
    python scripts/hardware/_setup_ibm_account.py             # default/us
    python scripts/hardware/_setup_ibm_account.py --profile eu
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_PROFILE_FIELDS = {
    "default": ("IBM_QUANTUM_TOKEN", "IBM_QUANTUM_CHANNEL", "IBM_QUANTUM_INSTANCE"),
    "us": ("IBM_QUANTUM_TOKEN_US", "IBM_QUANTUM_CHANNEL_US", "IBM_QUANTUM_INSTANCE_US"),
    "eu": ("IBM_QUANTUM_TOKEN_EU", "IBM_QUANTUM_CHANNEL_EU", "IBM_QUANTUM_INSTANCE_EU"),
    "open": ("IBM_QUANTUM_TOKEN_OPEN", "IBM_QUANTUM_CHANNEL_OPEN", "IBM_QUANTUM_INSTANCE_OPEN"),
}
_USER_ALIASES = {
    "bayram": (
        "ibm_quantum_token_bayram",
        "ibm_quantum_channel",
        "ibm_quantum_instance_bayram",
    ),
    "serhat": (
        "ibm_quantum_token_serhat",
        "ibm_quantum_channel_serhat",
        "ibm_quantum_instance_serhat",
    ),
    "ozgur": (
        "ibm_quantum_token_ozgur",
        "ibm_quantum_channel_ozgur",
        "ibm_quantum_instance_ozgur",
    ),
}


def _pick(env: dict, *keys: str) -> str | None:
    for k in keys:
        v = env.get(k)
        if v:
            return v
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IBM saved-account setup")
    parser.add_argument(
        "--profile",
        choices=list(_PROFILE_FIELDS.keys()) + list(_USER_ALIASES.keys()),
        default="default",
        help="Select which env-var group to load (default: legacy)",
    )
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv missing; `uv pip install python-dotenv`")
        return 2

    repo = Path(__file__).resolve().parents[2]
    env_path = repo / ".env"
    if not env_path.exists():
        print(f"[setup] .env not found at {env_path}")
        return 2
    load_dotenv(env_path)
    env = os.environ

    if args.profile in _PROFILE_FIELDS:
        t_key, c_key, i_key = _PROFILE_FIELDS[args.profile]
    else:
        t_key, c_key, i_key = _USER_ALIASES[args.profile]

    # Primary source (profile-specific) with legacy fallback so the old
    # single-profile .env keeps working.
    token = _pick(env, t_key, "IBM_QUANTUM_TOKEN", "ibm_quantum_token_bayram")
    channel = (
        _pick(env, c_key, "IBM_QUANTUM_CHANNEL", "ibm_quantum_channel")
        or "ibm_quantum_platform"
    )
    instance = _pick(env, i_key, "IBM_QUANTUM_INSTANCE", "ibm_quantum_instance")

    if not token or not instance:
        print(
            f"[setup] profile '{args.profile}' missing token or instance in .env; "
            f"looked for {t_key} / {i_key} (and fallbacks)."
        )
        return 2

    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel=channel,
        token=token,
        instance=instance,
        overwrite=True,
        set_as_default=True,
    )
    # save_account is unreliable on qiskit-ibm-runtime 0.46 (silent
    # persist failures observed 2026-04-20). Best practice going
    # forward: callers should also export QISKIT_IBM_{TOKEN,CHANNEL,
    # INSTANCE} at invocation time; env vars take precedence and
    # bypass the stale ~/.qiskit/qiskit-ibm.json file.
    print(
        f"[setup] saved account for profile={args.profile} "
        f"channel={channel} instance=<redacted>"
    )
    print(
        "[setup] recommended invocation pattern:\n"
        f"  export $(grep -E '^IBM_QUANTUM_(TOKEN|CHANNEL|INSTANCE)_{args.profile.upper()}=' .env \\\n"
        "    | sed 's/_" + args.profile.upper() + "=/=/; s/^IBM_QUANTUM_/QISKIT_IBM_/')"
    )

    # Verify connectivity without echoing credentials.
    try:
        svc = QiskitRuntimeService()
        lines = []
        for b in svc.backends():
            try:
                st = b.status()
                lines.append(
                    f"{b.name:30s} qubits={b.num_qubits} queue={st.pending_jobs} "
                    f"op={st.operational}"
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"{b.name:30s} (status error: {exc})")
        print(f"[setup] connected. {len(lines)} backend(s):")
        for line in lines:
            print(f"  - {line}")
    except Exception as exc:  # noqa: BLE001
        print(f"[setup] verification failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
