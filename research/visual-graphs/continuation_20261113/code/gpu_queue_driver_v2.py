"""R7/v2: session-independent serial driver for the repaired E5 falsifier.

After the v1 driver completed step 01 (37/37 latency configs, rc=0) and step
02 (aggregate, rc=0), steps 03/03b/04 failed: e5_falsifier_v1.py crashed on
an unsliced per-element tolerance tensor.  This driver runs the MINIMAL-FIX
v2 script with exactly the same GPU parameters (--device cuda --batches 40),
then its verify step.  It does NOT re-run latency or aggregate: those two
reports already exist and are immutable published evidence.

It is launched by a NO-TRIGGER Windows scheduled task via
gpu_queue_launcher_v2.py, so the process tree belongs to the Task Scheduler
service and survives the automation session boundary (root cause of the three
previous interruptions).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RR = "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix"
RUNROOT = ROOT / RR
LOGDIR = RUNROOT / "logs" / "gpu_queue_driver_v2"
DRIVER_LOG = RUNROOT / "logs" / "gpu_queue_driver_v2.driver.log"
STATUS = RUNROOT / "logs" / "gpu_queue_driver_v2.status.jsonl"
GATE_REPORT = RUNROOT / "results" / "e5-falsifier-v1" / "e5_gate_bound_v1.json"

DEFAULT_DATA_ROOT = (
    "C:/code/phd-snn/auto-paper/20261110-152407/"
    "Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/"
    "f-research-v1-run-20261110-172308-superpixel-graph-vision/data/pyg_bench"
)

PY = sys.executable

PLAN = [
    ("01_e5_falsifier_v2_run",
     ["e5_falsifier_v2.py", "run", "--device", "cuda", "--batches", "40"]),
    ("02_e5_falsifier_v2_verify",
     ["e5_falsifier_v2.py", "verify"]),
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def scrub(text: str) -> str:
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME") or ""
    for raw, tag in ((home, "<HOME>"), (str(Path(tempfile.gettempdir())), "<TEMP>"),
                     (PY, "<PYTHON>")):
        if raw:
            text = text.replace(raw, tag)
            text = text.replace(raw.replace("\\", "/"), tag)
    return text


def emit(text: str) -> None:
    line = f"{utc()} {scrub(text)}\n"
    with DRIVER_LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def command(argv: list[str]) -> list[str]:
    return [PY, "-X", "utf8", str(CODE / argv[0])] + argv[1:]


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GNN2SNN_DATA_ROOT"] = env.get("GNN2SNN_DATA_ROOT") or DEFAULT_DATA_ROOT
    env.setdefault("DCGS_REPO", str(Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"))
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def main() -> None:
    env = build_env()
    if "--dry-print" in sys.argv:
        print(json.dumps({
            "dry_print": True, "root": str(ROOT),
            "data_root": env["GNN2SNN_DATA_ROOT"], "dcgs_repo": env["DCGS_REPO"],
            "plan": [{"name": n, "cmd": command(a)} for n, a in PLAN],
            "gate_report_exists": GATE_REPORT.is_file(),
        }, indent=2))
        return

    LOGDIR.mkdir(parents=True, exist_ok=True)
    emit(f"DRIVER_V2_START pid={os.getpid()} python={PY} steps={len(PLAN)}")
    for name, argv in PLAN:
        cmd = command(argv)
        log = LOGDIR / f"{name}.log"
        emit(f"STEP_START {name} cmd={' '.join(cmd[2:])}")
        t0 = time.monotonic()
        with log.open("a", encoding="utf-8") as lf:
            lf.write(f"\n=== {name} START {utc()} ===\nCMD: {scrub(' '.join(cmd))}\n")
            lf.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf,
                                  stderr=subprocess.STDOUT)
        rc = proc.returncode
        rec = {"step": name, "rc": rc,
               "elapsed_seconds": round(time.monotonic() - t0, 1),
               "start_utc": utc(),
               "log": f"{RR}/logs/gpu_queue_driver_v2/{name}.log",
               "cmd": [scrub(x) for x in cmd[2:]]}
        with STATUS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        emit(f"STEP_END {name} rc={rc} elapsed_sec={rec['elapsed_seconds']}")
    emit("DRIVER_V2_END " + json.dumps({"gate_report": GATE_REPORT.is_file()},
                                       sort_keys=True))


if __name__ == "__main__":
    main()
