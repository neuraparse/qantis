"""Create a submission-freeze snapshot for the QCE26 paper."""

from __future__ import annotations

import shutil
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "qantis-paper"
OUTPUT_DIR = ROOT / "output" / "paper"

FREEZE_DATE = date.today().isoformat()
FREEZE_DIR = OUTPUT_DIR / f"qce26_submission_freeze_{FREEZE_DATE}"


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def _copy(src: Path, rel_dst: str) -> None:
    dst = FREEZE_DIR / rel_dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    pdfinfo = _run(["pdfinfo", str(PAPER_DIR / "main-qce.pdf")])
    pages = "unknown"
    for line in pdfinfo.splitlines():
        if line.startswith("Pages:"):
            pages = line.split(":", 1)[1].strip()
            break

    key_files: list[tuple[Path, str]] = [
        (PAPER_DIR / "main-qce.pdf", "paper/main-qce.pdf"),
        (PAPER_DIR / "main-qce.tex", "paper/main-qce.tex"),
        (PAPER_DIR / "metadata.tex", "paper/metadata.tex"),
        (PAPER_DIR / "sections/03-multistep-grover.tex", "paper/sections/03-multistep-grover.tex"),
        (PAPER_DIR / "sections/04-biqae-calibration.tex", "paper/sections/04-biqae-calibration.tex"),
        (PAPER_DIR / "sections/06-nisq-feasibility.tex", "paper/sections/06-nisq-feasibility.tex"),
        (PAPER_DIR / "sections/07-conclusion.tex", "paper/sections/07-conclusion.tex"),
        (OUTPUT_DIR / "qce26_finalization_2026-04-11.md", "summaries/qce26_finalization_2026-04-11.md"),
        (OUTPUT_DIR / "qce26_ibm_r3_complete_2026-04-11.md", "summaries/qce26_ibm_r3_complete_2026-04-11.md"),
        (OUTPUT_DIR / "biqae_pittsburgh_pairs_2026-04-11.md", "summaries/biqae_pittsburgh_pairs_2026-04-11.md"),
        (OUTPUT_DIR / "pittsburgh_t8_fpaa_2026-04-11.md", "summaries/pittsburgh_t8_fpaa_2026-04-11.md"),
        (OUTPUT_DIR / "pittsburgh_t12_fpaa_2026-04-11.md", "summaries/pittsburgh_t12_fpaa_2026-04-11.md"),
        (
            ROOT / "output/hardware/biqae_ibm_2026-04-11T02-44-57.json",
            "hardware/biqae_ibm_2026-04-11T02-44-57.json",
        ),
        (
            ROOT / "output/hardware/biqae_ibm_2026-04-11T02-45-42.json",
            "hardware/biqae_ibm_2026-04-11T02-45-42.json",
        ),
        (
            ROOT / "output/hardware/biqae_ibm_2026-04-11T02-46-27.json",
            "hardware/biqae_ibm_2026-04-11T02-46-27.json",
        ),
        (
            ROOT / "output/hardware/biqae_ibm_2026-04-11T02-47-08.json",
            "hardware/biqae_ibm_2026-04-11T02-47-08.json",
        ),
        (
            ROOT / "output/hardware/tiger_4state_multiprior_ibm_2026-04-11T02-47-58.json",
            "hardware/tiger_4state_multiprior_ibm_2026-04-11T02-47-58.json",
        ),
        (
            ROOT / "output/hardware/tiger_4state_ibm_2026-04-11T02-48-20.json",
            "hardware/tiger_4state_ibm_2026-04-11T02-48-20.json",
        ),
        (
            ROOT / "output/hardware/tiger_4state_ibm_2026-04-11T02-48-39.json",
            "hardware/tiger_4state_ibm_2026-04-11T02-48-39.json",
        ),
        (
            ROOT / "output/hardware/e2e_pomdp_fpaa_ibm_2026-04-11T03-41-40.json",
            "hardware/e2e_pomdp_fpaa_ibm_2026-04-11T03-41-40.json",
        ),
        (
            ROOT / "output/hardware/e2e_pomdp_fpaa_ibm_2026-04-11T04-21-17.json",
            "hardware/e2e_pomdp_fpaa_ibm_2026-04-11T04-21-17.json",
        ),
    ]

    copied: list[str] = []
    for src, rel_dst in key_files:
        if src.exists():
            _copy(src, rel_dst)
            copied.append(rel_dst)

    manifest = FREEZE_DIR / "FREEZE_MANIFEST.md"
    manifest.write_text(
        "\n".join(
            [
                "# QCE26 Submission Freeze",
                "",
                f"- Freeze date: `{FREEZE_DATE}`",
                f"- Git commit: `{commit}`",
                f"- Paper PDF pages: `{pages}`",
                "- Paper status: `QECS / RESP final submission freeze`",
                "",
                "Included materials:",
                *[f"- `{item}`" for item in copied],
                "",
                "Primary evidence frozen in this snapshot:",
                "- Same-backend Pittsburgh `T=12` FPAA trajectory",
                "- Same-backend Pittsburgh `T=8` FPAA trajectory",
                "- Mixed-backend Boston->Pittsburgh `T=12` continuation summary",
                "- Same-backend Pittsburgh BIQAE pairs at `a=0.01` and `a=0.95`",
                "- Optimized Pittsburgh 4-state six-case sweep",
                "- Optimized Pittsburgh boundary reruns",
                "",
                "Notes:",
                "- This freeze intentionally keeps MTDA at architectural-context level only.",
                "- This freeze is intended to support the final QCE26 paper submission state.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(FREEZE_DIR)


if __name__ == "__main__":
    main()
