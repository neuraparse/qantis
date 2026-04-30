"""Compact every paper-integrated hardware run into one CSV for inspection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "output" / "hardware"
CSV_PATH = REPO / "qantis-paper" / "hardware_results_summary.csv"

PAPER_JOBS = {
    # FPAA bounded-length sweep headlines
    "bounded_fpaa_boundary_2026-04-20T17-23-03.json": ("E3 Kingston boundary sweep", "ibm_kingston"),
    "bounded_fpaa_boundary_2026-04-20T18-35-40.json": ("E3 Fez cross-val", "ibm_fez"),
    "bounded_fpaa_boundary_2026-04-20T18-36-35.json": ("E3 Kingston ultra-rare 1e-4 to 1e-2", "ibm_kingston"),
    "bounded_fpaa_boundary_2026-04-20T18-50-05.json": ("E3 Fez ultra-deep k=78", "ibm_fez"),
    "bounded_fpaa_boundary_2026-04-20T18-51-28.json": ("E3 Kingston ultra-deep 10k x k=78", "ibm_kingston"),
    "bounded_fpaa_boundary_2026-04-20T19-02-23.json": ("E3 Kingston mega 100k x k=248", "ibm_kingston"),
    "bounded_fpaa_boundary_2026-04-20T19-10-28.json": ("E3 Marrakesh cross-val", "ibm_marrakesh"),
    "bounded_fpaa_boundary_2026-04-20T19-23-47.json": ("E3 Kingston 971832x peak @ k=700", "ibm_kingston"),
    # Non-Clifford ZNE
    "nonclifford_zne_tiger_2026-04-20T17-23-17.json": ("E2 non-Clifford ZNE 4-state", "ibm_kingston"),
    # Batched LRE
    "lre_batched_4state_tiger_2026-04-20T18-15-05.json": ("E4 batched LRE 4-state Tiger", "ibm_fez"),
}


def extract_rows(path: Path, label: str, backend: str) -> list[dict]:
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return [{"label": label, "error": f"{type(e).__name__}: {e}"}]
    rows = []
    for r in d.get("rows", []):
        rows.append({
            "campaign": label,
            "backend": d.get("backend", backend),
            "job_id": d.get("job_id", ""),
            "shots": d.get("shots", ""),
            "p_obs_or_a": r.get("p_obs", r.get("prior_label", "")),
            "k_or_extra": r.get("k", ""),
            "measured_p1_or_base": r.get("measured_p1", r.get("base_expectation", "")),
            "theory_or_lre_or_aer": r.get("expected_p1", r.get("lre_expectation", "")),
            "err_pct": None,
            "amplification": r.get("amplification", ""),
            "hellinger": r.get("hellinger_vs_ideal", ""),
        })
        # fill error percent where possible
        if rows[-1]["measured_p1_or_base"] and rows[-1]["theory_or_lre_or_aer"]:
            m = rows[-1]["measured_p1_or_base"]
            t = rows[-1]["theory_or_lre_or_aer"]
            try:
                rows[-1]["err_pct"] = f"{100*abs(float(m)-float(t))/max(float(t), 1e-12):.2f}"
            except Exception:
                pass
    # Also handle nonclifford regimes nested structure
    if d.get("regimes"):
        for reg in d["regimes"]:
            for method, pm in reg.get("per_method", {}).items():
                for rr in pm.get("rows", []):
                    rows.append({
                        "campaign": label + f" / {reg['regime']} / {method}",
                        "backend": d.get("backend", backend),
                        "job_id": d.get("job_id", ""),
                        "shots": d.get("shots", ""),
                        "p_obs_or_a": f"scale={rr['scale']}",
                        "k_or_extra": "",
                        "measured_p1_or_base": rr.get("hellinger", ""),
                        "theory_or_lre_or_aer": pm.get("zero_noise_hellinger", ""),
                        "err_pct": "",
                        "amplification": "",
                        "hellinger": rr.get("hellinger", ""),
                    })
    return rows


def main():
    all_rows = []
    for fname, (label, bkend) in PAPER_JOBS.items():
        path = OUT_DIR / fname
        if not path.exists():
            print(f"[miss] {fname}")
            continue
        all_rows.extend(extract_rows(path, label, bkend))

    cols = ["campaign", "backend", "job_id", "shots", "p_obs_or_a", "k_or_extra",
            "measured_p1_or_base", "theory_or_lre_or_aer", "err_pct",
            "amplification", "hellinger"]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[saved] {CSV_PATH}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
