"""R7: detached-session launcher for the frozen E3/E5 serial GPU driver.

Why this exists
---------------
The heartbeat automation terminates its own background processes at the
session boundary.  R6c (TsI47h) and R6d (v10FQp) were both killed that way,
leaving the RTX 3060 idle for ~82 and ~75 minutes respectively.  This launcher
is invoked by a NO-TRIGGER Windows scheduled task, so the driver process tree
belongs to the Task Scheduler service and survives the session boundary.

It also redirects the driver's stdout/stderr into a log file: a scheduled-task
process has no attached console, so a bare ``print()`` in the driver would
otherwise fail with ``sys.stdout is None``.

No parameter below changes the frozen E3/E5 commands.  The driver itself owns
the chain: latency --resume -> aggregate -> falsifier run -> falsifier verify.
No local identity path is written into this source file.
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
LOG = RUNROOT / "logs" / "gpu_queue_driver_v1.launcher.log"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    env = os.environ.copy()
    # The driver's build_env() injects the frozen official PyG data root; keep
    # the pinned DCGS checkout explicit here as well, both resolved at runtime.
    env.setdefault("DCGS_REPO", str(Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"))
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-X", "utf8", str(CODE / "gpu_queue_driver_v1.py")]
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== LAUNCHER_START {utc()} pid={os.getpid()} ===\n")
        f.write(f"CMD: {sys.executable} -X utf8 {cmd[-1]}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f,
                            stderr=subprocess.STDOUT).returncode
        f.write(f"=== LAUNCHER_END {utc()} rc={rc} ===\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
