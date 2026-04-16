"""Prepare simulator-first outputs that answer the advisor's QCE26 comments.

The script intentionally focuses on three questions:
1. Can we show reproducible sequential belief-update behavior locally?
2. Do we have operational modern classical POMDP baselines?
3. What is the resource and sample-efficiency pathway beyond the 2-state Tiger?
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "quantum-pomdp" / "src"))

from quantum_pomdp.analysis.advisor_experiments import (  # noqa: E402
    corridor_tiger_4state_simulator_report,
    scenario_resource_pathway_report,
    tiger_classical_baseline_report,
)

OUTPUT_DIR = ROOT / "output" / "paper"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw"
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_hardware_dry_run(label: str, args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, *args]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    text = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    matches = re.findall(r"^\[saved\] (.+\.json)$", text, re.MULTILINE)
    if not matches:
        raise RuntimeError(f"Could not find saved JSON path for {label}.\n{text}")
    json_path = Path(matches[-1])
    result = json.loads(json_path.read_text(encoding="utf-8"))
    raw_copy = RAW_OUTPUT_DIR / f"{label}_{datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%f')}.json"
    shutil.copyfile(json_path, raw_copy)
    return {
        "label": label,
        "command": cmd,
        "stdout_tail": text.strip().splitlines()[-20:],
        "json_path": str(raw_copy),
        "result": result,
    }


def _summarize_sequential_result(entry: dict[str, Any]) -> dict[str, Any]:
    result = entry["result"]
    summary = result.get("summary", {})
    return {
        "label": entry["label"],
        "json_path": entry["json_path"],
        "backend": result.get("backend"),
        "n_steps": result.get("n_steps"),
        "observation_sequence": result.get("observation_sequence"),
        "max_hellinger": summary.get("tiger_max_hw_hellinger"),
        "trajectory_ok": summary.get("belief_trajectory_correct_direction"),
        "aa_method": summary.get("aa_method", "none"),
        "aa_steps_used": summary.get("aa_steps_used", 0),
        "aa_steps_total": summary.get("aa_steps_total", result.get("n_steps")),
        "min_amplification": summary.get("min_amplification"),
        "max_amplification": summary.get("max_amplification"),
        "mean_amplification": summary.get("mean_amplification"),
        "pass": result.get("pass"),
    }


def _write_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# QCE26 Advisor Response Pack")
    lines.append("")
    lines.append("## Sequential Simulator Runs")
    lines.append("")
    for item in report["sequential_runs"]:
        lines.append(
            f"- `{item['label']}`: steps={item['n_steps']}, max Hellinger={item['max_hellinger']}, "
            f"AA={item['aa_method']}, pass={item['pass']}, source={item['json_path']}"
        )
        if item["mean_amplification"] is not None:
            lines.append(
                f"  mean amplification={item['mean_amplification']}x "
                f"(min={item['min_amplification']}x, max={item['max_amplification']}x)"
            )
    lines.append("")
    lines.append("## Classical Tiger Baselines")
    lines.append("")
    for item in report["classical_baselines"]:
        actions = ", ".join(
            f"{solver}={action}"
            for solver, action in item["recommended_action_names"].items()
        )
        lines.append(f"- `{item['belief_label']}` belief {item['belief']}: {actions}")
    lines.append("")
    lines.append("## Corridor Tiger (4-state) Simulator")
    lines.append("")
    ct = report["corridor_tiger_4state"]
    lines.append(
        f"- prior={ct['prior']}, obs={ct['observation']}, Hellinger={ct['hellinger_distance']}, "
        f"logical depth={ct['logical_depth']}, qubits={ct['num_qubits']}"
    )
    lines.append("")
    lines.append("## Resource / Sample-Efficiency Pathway")
    lines.append("")
    for item in report["resource_pathway"]:
        lines.append(
            f"- `{item['label']}`: |S|={item['num_states']}, total qubits={item['total_circuit_qubits']}, "
            f"P(e)={item['representative_evidence_probability']:.5f}, "
            f"classical≈{item['classical_queries_per_accept']:.2f}, "
            f"quantum-scale≈{item['quantum_query_scale_only']:.2f}, "
            f"gain≈{item['asymptotic_sample_efficiency_gain']:.2f}x"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The larger-scenario entries are query-scaling context only; they do not claim end-to-end runtime advantage."
    )
    lines.append(
        "- MTDA remains outside this report because the advisor feedback targeted the validated inference core and its baselines."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sequential_jobs = [
        (
            "t4_baseline",
            [
                "scripts/hardware/run_end_to_end_pomdp_ibm.py",
                "--dry-run",
                "--shots", "8192",
                "--biqae-shots", "300",
                "--n-steps", "4",
                "--obs-sequence", "0", "0", "0", "0",
            ],
        ),
        (
            "t8_baseline",
            [
                "scripts/hardware/run_end_to_end_pomdp_ibm.py",
                "--dry-run",
                "--shots", "8192",
                "--biqae-shots", "300",
                "--n-steps", "8",
                "--obs-sequence", "0", "0", "0", "0", "1", "1", "0", "0",
            ],
        ),
        (
            "t8_fpaa",
            [
                "scripts/hardware/run_end_to_end_pomdp_ibm.py",
                "--dry-run",
                "--fpaa",
                "--shots", "8192",
                "--biqae-shots", "300",
                "--n-steps", "8",
                "--obs-sequence", "0", "0", "0", "0", "1", "1", "0", "0",
            ],
        ),
        (
            "t12_fpaa",
            [
                "scripts/hardware/run_end_to_end_pomdp_ibm.py",
                "--dry-run",
                "--fpaa",
                "--shots", "8192",
                "--biqae-shots", "300",
                "--n-steps", "12",
                "--obs-sequence", "0", "0", "0", "0", "1", "1", "1", "0", "0", "0", "1", "0",
            ],
        ),
    ]

    sequential_runs = [
        _summarize_sequential_result(_run_hardware_dry_run(label, args))
        for label, args in sequential_jobs
    ]
    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "sequential_runs": sequential_runs,
        "classical_baselines": tiger_classical_baseline_report(),
        "corridor_tiger_4state": corridor_tiger_4state_simulator_report(shots=8192),
        "resource_pathway": scenario_resource_pathway_report(),
        "notes": {
            "sample_efficiency_only": (
                "Larger-scenario results are reported as evidence-conditioning "
                "query scaling, not as end-to-end runtime advantage."
            ),
            "advisor_focus": (
                "This pack responds to concerns about Tiger being small, the lack "
                "of runtime claims, and the need for modern classical baselines."
            ),
        },
    }

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    json_path = OUTPUT_DIR / f"qce26_advisor_pack_{timestamp}.json"
    md_path = OUTPUT_DIR / f"qce26_advisor_pack_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)

    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
