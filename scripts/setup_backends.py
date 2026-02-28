#!/usr/bin/env python3
"""Verify and setup quantum backend connections.

Usage:
    uv run python scripts/setup_backends.py
    uv run python scripts/setup_backends.py --backend ibm_quantum
    uv run python scripts/setup_backends.py --backend dwave
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and verify quantum backends")
    parser.add_argument("--backend", type=str, default=None, help="Specific backend to test")
    parser.add_argument("--list", action="store_true", help="List all registered backends")
    args = parser.parse_args()

    from quantum_common.backends.factory import create_backend, list_available_backends
    from quantum_common.types import BackendType

    if args.list:
        print("Checking available backends...")
        available = list_available_backends()
        print(f"\nAvailable backends ({len(available)}):")
        for bt in available:
            print(f"  - {bt.value}")
        if not available:
            print("  (none available - install required packages)")
        return

    backends_to_test = []
    if args.backend:
        try:
            bt = BackendType(args.backend)
            backends_to_test.append(bt)
        except ValueError:
            print(f"Unknown backend: {args.backend}")
            print(f"Available: {[bt.value for bt in BackendType]}")
            sys.exit(1)
    else:
        backends_to_test = list(BackendType)

    print("Quantum Backend Setup Verification")
    print("=" * 50)

    for bt in backends_to_test:
        print(f"\n[{bt.value}]")
        try:
            backend = create_backend(bt)
            available = backend.is_available()
            info = backend.status_info()
            status = "AVAILABLE" if available else "NOT AVAILABLE"
            print(f"  Status: {status}")
            print(f"  Name: {backend.name}")
            print(f"  Paradigm: {backend.paradigm.name}")
            print(f"  Max qubits: {backend.max_qubits}")
            for key, val in info.items():
                if key not in ("backend", "available"):
                    print(f"  {key}: {val}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 50)
    print("Setup verification complete.")


if __name__ == "__main__":
    main()
