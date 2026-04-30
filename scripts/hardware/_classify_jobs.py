"""Classify recovered Pittsburgh jobs into experiments by circuit structure.

Looks at the full counts from each job, the classical-register size, and
the dominant bitstring to infer whether it belongs to E3 (1-qubit
Grover amplification), E7 (Tiger belief update 4/8/... bits), E5
(Iceberg 4-qubit encoded), E2/E4 (non-Clifford ZNE / LRE variants).

Produces a per-experiment table rather than re-running anything.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))

from scripts.hardware import databin_get_counts, save_result


def classify(num_clbits: int, counts: dict[str, int]) -> str:
    if num_clbits == 1:
        return "E3_FPAA_1q_grover"
    if num_clbits == 2:
        return "E7_tiger_Sis2_or_E2_minimal"
    if num_clbits == 3:
        return "E7_tiger_Sis4"
    if num_clbits == 4:
        # E5 Iceberg encoded (4 physical) OR E7 |S|=8 (3 state + 1 obs)
        return "E5_iceberg_or_E7_Sis8"
    if num_clbits == 5:
        return "E4_lre_4state"
    return f"unknown_{num_clbits}bit"


def hellinger(p: dict[str, int], q: dict[str, int]) -> float:
    keys = set(p) | set(q)
    sp = sum(p.values()) or 1
    sq = sum(q.values()) or 1
    total = 0.0
    for k in keys:
        total += (math.sqrt(p.get(k, 0) / sp) - math.sqrt(q.get(k, 0) / sq)) ** 2
    return math.sqrt(0.5 * total)


def main() -> int:
    from qiskit_ibm_runtime import QiskitRuntimeService

    svc = QiskitRuntimeService()
    start = datetime.now(tz=timezone.utc) - timedelta(hours=48)
    jobs = svc.jobs(limit=200, created_after=start)

    records = []
    for j in jobs:
        if str(j.status()) != "DONE":
            continue
        try:
            res = j.result()
            pub = res[0]
            counts = databin_get_counts(pub)
            data = getattr(pub, "data", pub)
            num_clbits = next(
                (
                    getattr(getattr(data, name, None), "num_bits", None)
                    for name in dir(data)
                    if not name.startswith("_")
                    and hasattr(getattr(data, name, None), "num_bits")
                ),
                None,
            ) or len(next(iter(counts.keys())).replace(" ", ""))
            shots = sum(counts.values())
            top_bs, top_n = max(counts.items(), key=lambda kv: kv[1])
            records.append({
                "job_id": j.job_id(),
                "creation": str(j.creation_date),
                "num_clbits": num_clbits,
                "shots": shots,
                "unique": len(counts),
                "top_bs": top_bs,
                "top_frac": top_n / max(shots, 1),
                "classification": classify(num_clbits, counts),
                "full_counts": counts,
            })
        except Exception as exc:
            records.append({
                "job_id": j.job_id(),
                "error": f"{type(exc).__name__}: {exc}",
            })

    print(f"{'job_id':<26s} {'created':<20s} {'bits':>4s} {'shots':>6s} "
          f"{'uniq':>5s} {'top_bs':>8s} {'top%':>6s}  experiment")
    for r in sorted(records, key=lambda x: x.get("creation", "")):
        if "error" in r:
            print(f"{r['job_id']:<26s} ERROR  {r['error']}")
            continue
        print(
            f"{r['job_id']:<26s} {r['creation'][:19]:<20s} {r['num_clbits']:>4d} "
            f"{r['shots']:>6d} {r['unique']:>5d} {r['top_bs']:>8s} "
            f"{100*r['top_frac']:>5.1f}%  {r['classification']}"
        )

    save_result(
        "classified_jobs",
        {
            "campaign": "heron_r3_campaign_2026-04-19_classified",
            "records": records,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
