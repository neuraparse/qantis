from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize BIQAE baseline/calibrated runs")
    p.add_argument("files", nargs="+", help="BIQAE result JSON files")
    return p.parse_args()


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    args = _parse_args()
    rows: list[dict] = []
    for path_str in args.files:
        path = Path(path_str)
        data = _load(path)
        summary = data.get("paper_summary_hardware") or data.get("paper_summary_simulator") or {}
        prior = summary.get("prior", {})
        phase1_shots = summary.get("phase1_shots", 0)
        total_shots = summary.get("total_shots_used")
        if isinstance(prior, dict):
            phase1_shots = prior.get("phase1_shots", phase1_shots)
            total_shots = prior.get("total_shots", total_shots)
        phase2_shots = summary.get("phase2_shots")
        if phase2_shots is None:
            if total_shots is not None:
                phase2_shots = max(total_shots - phase1_shots, 0)
            else:
                phase2_shots = summary.get("total_shots_used")
        row = {
            "file": path.name,
            "backend": summary.get("backend", data.get("backend", "")),
            "amplitude": summary.get("amplitude", data.get("a_true", "")),
            "calibrated": summary.get("calibrated", data.get("calibrated", False)),
            "estimate": summary.get("estimate"),
            "error": summary.get("absolute_error"),
            "ci_width": summary.get("ci_width"),
            "ci_contains": summary.get("ci_contains_true"),
            "iterations": summary.get("iterations"),
            "phase1_shots": phase1_shots,
            "phase2_shots": phase2_shots,
            "total_shots": total_shots,
            "prior": prior,
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["amplitude"], r["calibrated"]))

    print("| amplitude | mode | backend | estimate | abs error | CI width | CI contains | iterations | phase-1 | phase-2 | total shots | prior | file |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        prior = row["prior"]
        if isinstance(prior, dict):
            if prior.get("kind") == "baseline":
                prior_label = f"baseline(mu={prior.get('prior_mean')}, sigma={prior.get('prior_std')})"
            else:
                prior_label = (
                    f"{prior.get('regime')} Beta({prior.get('prior_alpha')}, {prior.get('prior_beta')})"
                    if prior else ""
                )
        else:
            prior_label = str(prior)
        print(
            f"| {row['amplitude']} | "
            f"{'calibrated' if row['calibrated'] else 'baseline'} | "
            f"{row['backend']} | "
            f"{row['estimate']} | "
            f"{row['error']} | "
            f"{row['ci_width']} | "
            f"{row['ci_contains']} | "
            f"{row['iterations']} | "
            f"{row['phase1_shots']} | "
            f"{row['phase2_shots']} | "
            f"{row['total_shots']} | "
            f"{prior_label} | "
            f"{row['file']} |"
        )


if __name__ == "__main__":
    main()
