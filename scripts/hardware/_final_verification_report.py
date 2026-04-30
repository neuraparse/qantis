"""Final verification: reconstruct every Pittsburgh experiment from
the 30 jobs submitted on 2026-04-18/19.

Pulls results directly from IBM's archive (no re-submission) and matches
each job to its experiment based on classical-register width + submit
order. Writes a single consolidated JSON + pretty-print summary.
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


def theoretical_amplified(p: float, k: int) -> float:
    theta = math.asin(math.sqrt(max(min(p, 1 - 1e-9), 1e-9)))
    return math.sin((2 * k + 1) * theta) ** 2


def hellinger(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    sa = sum(a.values()) or 1
    sb = sum(b.values()) or 1
    return math.sqrt(
        0.5 * sum(
            (math.sqrt(a.get(k, 0) / sa) - math.sqrt(b.get(k, 0) / sb)) ** 2
            for k in keys
        )
    )


def main() -> int:
    from qiskit_ibm_runtime import QiskitRuntimeService

    svc = QiskitRuntimeService()
    start = datetime.now(tz=timezone.utc) - timedelta(hours=48)
    jobs = svc.jobs(limit=200, created_after=start)

    # Pull counts for all done jobs, bucket by (num_clbits, creation date)
    buckets: dict[int, list[dict]] = {}
    total_qpu = 0.0
    for j in sorted(jobs, key=lambda x: x.creation_date):
        if str(j.status()) != "DONE":
            continue
        try:
            res = j.result()
            pub = res[0]
            counts = databin_get_counts(pub)
            shots = sum(counts.values())
            # Classical register width from a sample bitstring
            sample_bs = next(iter(counts))
            num_bits = len(sample_bs.replace(" ", ""))
        except Exception as exc:
            print(f"[skip] {j.job_id()}: {exc}")
            continue
        try:
            m = j.metrics() or {}
            qpu_s = m.get("usage", {}).get("quantum_seconds", 0)
        except Exception:
            qpu_s = 0
        total_qpu += float(qpu_s)
        buckets.setdefault(num_bits, []).append({
            "job_id": j.job_id(),
            "creation": str(j.creation_date),
            "shots": shots,
            "num_bits": num_bits,
            "counts": counts,
            "top": max(counts.items(), key=lambda kv: kv[1]),
            "qpu_s": qpu_s,
        })

    # ---- E3 recovery: all 1-qubit jobs are E3 (Grover amplification) ----
    e3_jobs = sorted(buckets.get(1, []), key=lambda r: r["creation"])
    print("\n=== E3 FPAA boundary (1-qubit Grover) ===")
    print(f"{'created':<20s} {'job_id':<26s} {'p1_obs':>8s} {'uniq':>4s} {'shots':>6s}")
    e3_report = []
    # Theoretical plan from re-submit: --p-obs 0.05 0.95 --ks 1 3 →
    # [(0.05,1), (0.05,3), (0.95,1), (0.95,3)] -- 4 expected circuits.
    # First-run (earlier) used different params; we report all.
    for rec in e3_jobs:
        p1 = rec["counts"].get("1", 0) / max(rec["shots"], 1)
        print(f"{rec['creation'][:19]:<20s} {rec['job_id']:<26s} "
              f"{p1:>8.4f} {len(rec['counts']):>4d} {rec['shots']:>6d}")
        e3_report.append({"job_id": rec["job_id"], "creation": rec["creation"],
                           "p1_measured": p1, "shots": rec["shots"]})

    # Best-fit against theoretical (p_obs, k) plan for the v2 resubmit
    plan_v2 = [(0.05, 1), (0.05, 3), (0.95, 1), (0.95, 3)]
    if len(e3_jobs) >= 4:
        print("\n-- E3 v2 (resubmit) recovery (last 4 in creation order) --")
        print(f"{'p_obs':>6s} {'k':>2s}  {'p1_measured':>12s} {'p1_theory':>11s} "
              f"{'err_pct':>8s} {'amplif':>7s}")
        for rec, (p_obs, k) in zip(e3_jobs[-4:], plan_v2):
            p1 = rec["counts"].get("1", 0) / max(rec["shots"], 1)
            th = theoretical_amplified(p_obs, k)
            err = abs(p1 - th) / max(th, 1e-6) * 100
            ampl = p1 / max(p_obs, 1e-9)
            print(f"{p_obs:>6.2f} {k:>2d}  {p1:>12.4f} {th:>11.4f} "
                  f"{err:>7.2f}% {ampl:>7.2f}x")

    # ---- E7 Tiger Heron R3 scaling (2-bit |S|=2, 3-bit |S|=4, 4-bit |S|=8) ----
    print("\n=== E7 Tiger |S|={2,4,8} Heron R3 (recovered) ===")
    for bits, size in ((2, 2), (3, 4), (4, 8)):
        bucket = [b for b in buckets.get(bits, []) if b["creation"] < "2026-04-19 10:55:00"]
        if not bucket:
            continue
        for b in bucket:
            # Post-select on MSB = 0 (observation=0)
            kept = {k[1:]: v for k, v in b["counts"].items() if k[0] == "0"}
            tot = sum(kept.values())
            post = {k: v/tot for k, v in kept.items()} if tot else {}
            print(f"  |S|={size:<2d} {b['job_id']:<26s} {b['creation'][:19]} "
                  f"top={b['top'][0]} ({100*b['top'][1]/b['shots']:.1f}%) "
                  f"post_sel_kept={tot}/{b['shots']}")

    # ---- E5 Iceberg (4-bit 4096-shot jobs at 10:58+) ----
    print("\n=== E5 Iceberg (from saved JSON) ===")
    e5_artifact = sorted(Path("output/hardware").glob("iceberg_tiger_heron_2026-04-19*.json"))[-1]
    with open(e5_artifact) as fh:
        e5 = json.load(fh)
    print(f"  artifact: {e5_artifact.name}")
    print(f"  encoded acceptance: {e5['encoded']['acceptance_ratio']:.4f}")
    print(f"  encoded mode_energy: {e5['encoded']['mode_energy']} (ground)")
    print(f"  unencoded mode_energy: {e5['unencoded']['mode_energy']} (excited)")

    # ---- E6 TN baseline (CPU, from saved JSON) ----
    print("\n=== E6 classical TN baseline (from saved JSON) ===")
    e6_artifact = sorted(Path("output/hardware").glob("tn_baseline_tiger_2026-04-19*.json"))[-1]
    with open(e6_artifact) as fh:
        e6 = json.load(fh)
    by_n: dict[int, list[float]] = {}
    for row in e6["rows"]:
        if row.get("relative_gap") is None or not isinstance(row.get("relative_gap"), float):
            continue
        if math.isnan(row["relative_gap"]):
            continue
        by_n.setdefault(row["n"], []).append(row["relative_gap"])
    print(f"  artifact: {e6_artifact.name}")
    for n in sorted(by_n):
        g = by_n[n]
        print(f"  n={n} seeds={len(g)} mean_gap={sum(g)/len(g):.4f} max_gap={max(g):.4f}")

    # Overall budget
    print(f"\n=== Campaign budget ===")
    print(f"  Total QPU seconds (all 30 jobs): {total_qpu:.1f}")
    print(f"  Jobs per bit-width:")
    for bits in sorted(buckets):
        print(f"    {bits}-bit: {len(buckets[bits])} jobs")

    save_result("final_verification", {
        "campaign": "heron_r3_final_verification_2026-04-19",
        "total_qpu_seconds": total_qpu,
        "e3_fpaa_resubmit": e3_report,
        "buckets_by_bits": {
            str(bits): [
                {k: v for k, v in rec.items() if k != "counts"}
                for rec in recs
            ]
            for bits, recs in buckets.items()
        },
        "e5_iceberg": e5,
        "e6_tn_baseline_summary": {
            str(n): {"seeds": len(g), "mean_gap": sum(g)/len(g), "max_gap": max(g)}
            for n, g in by_n.items()
        },
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
