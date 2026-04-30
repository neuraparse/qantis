"""Recover and classify all IBM Pittsburgh jobs from 2026-04-18/19 campaign.

The Python wrappers hung on ``SamplerV2.run().result()`` blocking call
for several experiments (E2, E3-v2, E4). The QPU jobs themselves
completed successfully in the IBM archive; this script pulls the
counts from each job, clusters them by submit timestamp, and writes a
consolidated recovery artifact.

Usage:
    python scripts/hardware/_recover_jobs.py

Output:
    output/hardware/recovered_jobs_2026-04-19.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))

from scripts.hardware import databin_get_counts, save_result


def main() -> int:
    from qiskit_ibm_runtime import QiskitRuntimeService

    svc = QiskitRuntimeService()
    start = datetime.now(tz=timezone.utc) - timedelta(hours=48)
    jobs = svc.jobs(limit=200, created_after=start)
    print(f"[recover] found {len(jobs)} jobs in last 48h")

    rows = []
    for j in jobs:
        record: dict = {
            "job_id": j.job_id(),
            "backend": j.backend().name,
            "status": str(j.status()),
            "creation_date": str(j.creation_date),
        }
        try:
            meta = j.metrics() or {}
            record["qpu_seconds"] = (
                meta.get("usage", {}).get("quantum_seconds", 0)
                if isinstance(meta, dict)
                else 0
            )
        except Exception:
            record["qpu_seconds"] = None

        if str(j.status()) != "DONE":
            rows.append(record)
            continue

        try:
            result = j.result()
            per_pub = []
            for pub_idx in range(len(result)):
                pub = result[pub_idx]
                try:
                    counts = databin_get_counts(pub)
                except Exception as exc:
                    counts = {"error": f"{type(exc).__name__}: {exc}"}
                shots = sum(counts.values()) if all(
                    isinstance(v, int) for v in counts.values()
                ) else 0
                per_pub.append({
                    "pub_index": pub_idx,
                    "shots": shots,
                    "unique_bitstrings": len(counts),
                    "top_5_bitstrings": sorted(
                        counts.items(), key=lambda kv: -kv[1]
                    )[:5] if isinstance(counts.get(list(counts.keys())[0] if counts else "", None), int) else None,
                })
            record["pubs"] = per_pub
        except Exception as exc:
            record["recovery_error"] = f"{type(exc).__name__}: {exc}"

        rows.append(record)
        print(
            f"[recover] {j.job_id():<26s} "
            f"pubs={len(record.get('pubs', []))} "
            f"qpu={record.get('qpu_seconds', '?'):.1f}s"
            if isinstance(record.get("qpu_seconds"), (int, float))
            else f"[recover] {j.job_id()}"
        )

    payload = {
        "campaign": "heron_r3_recovery_2026-04-19",
        "note": (
            "Recovery of all QPU jobs from the 2026-04-19 campaign after "
            "several Python wrappers hung on SamplerV2.result() polling."
        ),
        "rows": rows,
    }
    save_result("recovered_jobs", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
