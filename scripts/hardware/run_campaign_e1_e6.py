"""QANTIS-2 IBM Heron campaign orchestrator (E1-E6).

Chains the six experiments from
``qantis-paper/notes/ibm-campaign-plan-2026-04-19.md`` into a single
entry point. Each experiment is invoked as a subprocess so a crash in
one does not kill the rest; artifacts land in
``output/hardware/<campaign>_<timestamp>.json`` as usual.

Usage
-----
End-to-end dry-run on CPU:

    python scripts/hardware/run_campaign_e1_e6.py --dry-run

Full Pittsburgh run, 4096 shots per experiment:

    IBM_QUANTUM_TOKEN=xxx python scripts/hardware/run_campaign_e1_e6.py \
        --backend ibm_pittsburgh --shots 4096 --skip e5

Use ``--only`` or ``--skip`` to select experiments.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

from scripts.hardware import save_result  # type: ignore


EXPERIMENTS: dict[str, list[str]] = {
    "e1": ["run_tts_ensemble_heron.py"],
    "e2": ["run_nonclifford_zne_tiger.py"],
    "e3": ["run_bounded_fpaa_boundary.py"],
    "e4": ["run_lre_4state_tiger.py"],
    "e5": ["run_iceberg_tiger_heron.py"],
    "e6": ["run_tn_baseline_tiger.py"],
    # E7: QCE 2026 scaling-evidence experiment across |S| in {2, 4, 8}.
    # Delivers the Ronnow-Shaydulin bootstrap-CI slope that elevates the
    # paper's scaling claim from "trend" to "evidence".
    "e7": ["run_tts_scaling_s248.py"],
}


def _build_cmd(
    script: Path, backend: str, shots: int, dry_run: bool,
) -> list[str]:
    cmd = [sys.executable, str(script)]
    if script.name not in {"run_tn_baseline_tiger.py"}:
        cmd += ["--backend", backend, "--shots", str(shots)]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _run_one(
    key: str,
    script_name: str,
    backend: str,
    shots: int,
    dry_run: bool,
) -> dict:
    script = _repo / "scripts" / "hardware" / script_name
    cmd = _build_cmd(script, backend, shots, dry_run)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    return {
        "experiment": key,
        "script": script_name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": proc.stdout.strip().splitlines()[-20:],
        "stderr_tail": proc.stderr.strip().splitlines()[-20:],
    }


def _select(
    only: Iterable[str] | None, skip: Iterable[str] | None,
) -> list[tuple[str, str]]:
    keys = list(EXPERIMENTS.keys())
    if only:
        keys = [k for k in keys if k in set(only)]
    if skip:
        keys = [k for k in keys if k not in set(skip)]
    return [(k, EXPERIMENTS[k][0]) for k in keys]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="ibm_pittsburgh")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--skip", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = _select(args.only, args.skip)
    if not selected:
        print("[campaign] no experiments selected; nothing to do.")
        return

    results = []
    for key, script in selected:
        print(f"[campaign] running {key} ({script}) ...")
        results.append(_run_one(key, script, args.backend, args.shots, args.dry_run))

    save_result(
        "campaign_e1_e6_summary",
        {
            "campaign": "qantis2_ibm_heron_campaign",
            "backend": args.backend,
            "shots": args.shots,
            "dry_run": args.dry_run,
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "experiments": results,
        },
    )


if __name__ == "__main__":
    main()
