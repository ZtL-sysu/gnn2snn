"""R7/v2: detached-session launcher for gpu_queue_driver_v2.py.

Same rationale as gpu_queue_launcher_v1.py: a NO-TRIGGER Windows scheduled
task owns the process tree (Task Scheduler service), so the E5 falsifier GPU
job survives the automation session boundary.  stdout/stderr are redirected
to a log because a scheduled task has no attached console.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RR = "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix"
RUNROOT = ROOT / RR
LOG = RUNROOT / "logs" / "gpu_queue_driver_v2.launcher.log"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    env = os.environ.copy()
    env.setdefault("DCGS_REPO", str(Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"))
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-X", "utf8", str(CODE / "gpu_queue_driver_v2.py")]
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== LAUNCHER_V2_START {utc()} pid={os.getpid()} ===\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f,
                            stderr=subprocess.STDOUT).returncode
        f.write(f"=== LAUNCHER_V2_END {utc()} rc={rc} ===\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
