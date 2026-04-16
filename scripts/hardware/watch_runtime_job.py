"""Poll an IBM Runtime job until it reaches a terminal state."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "packages" / "quantum-common" / "src"))
sys.path.insert(0, str(_repo / "packages" / "quantum-pomdp" / "src"))

from scripts.hardware import make_ibm_runtime_service, save_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch an IBM Runtime job.")
    parser.add_argument("--job-id", required=True, help="IBM Runtime job id")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path for plain-text status log. Defaults to output/hardware/job_watch_<job>.log",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_file = Path(args.log_file) if args.log_file else (
        _repo / "output" / "hardware" / f"job_watch_{args.job_id}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    service = make_ibm_runtime_service()
    terminal_states = {"DONE", "ERROR", "CANCELLED"}

    while True:
        job = service.job(args.job_id)
        status = str(job.status())
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {status}\n")

        print(f"{timestamp} {status}")

        if status in terminal_states:
            summary = {
                "job_id": args.job_id,
                "status": status,
                "timestamp": timestamp,
                "backend": str(job.backend()),
            }
            save_result(f"runtime_job_status_{args.job_id}", summary)
            return

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
