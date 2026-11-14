"""R6d: single serial GPU queue driver for the frozen E3/E5 evidence chain.

Purpose
-------
One background process owns the local RTX 3060 and consumes the GPU_QUEUE.md
head-of-line items in strict order, so the card is not idle between short
heartbeats.  At most ONE `e3_e5_v1.py latency` process exists at any instant
(post-2026-11-14T00:04Z concurrency rule).

Chain (all commands are the frozen, unchanged ones):
  01 e3_e5_v1.py latency --device cuda --batches 40 --warmup 1 --repeats 3 --resume
  02 e3_e5_v1.py aggregate                                      (CPU)
  03 e5_falsifier_v1.py run --device cuda --batches 40
  04 e5_falsifier_v1.py verify                                  (CPU)

Failure handling: every step keeps its own raw log and rc; a failed step does
NOT stop the driver.  If step 01 did not produce e5_latency_v1.json, a single
identical `--resume` retry is inserted (incremental JSONL skips finished
configurations) and `aggregate` is re-run afterwards.  The same applies to the
falsifier report.  No parameter is shortened, no repeat is dropped, no seed is
selected, no output directory is deleted.

Env: GNN2SNN_DATA_ROOT (official PyG root, no local username in the default),
     DCGS_REPO (pinned checkout; defaults to <tempdir>/dcgs-5e0e4c73).
Both are resolved at runtime so this source file contains no private path.
"""
from __future__ import annotations

import argparse
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
LOGDIR = RUNROOT / "logs" / "gpu_queue_driver_v1"
DRIVER_LOG = RUNROOT / "logs" / "gpu_queue_driver_v1.driver.log"
STATUS = RUNROOT / "logs" / "gpu_queue_driver_v1.status.jsonl"
E3E5_OUT = RUNROOT / "results" / "e3-e5-v1"
FALS_OUT = RUNROOT / "results" / "e5-falsifier-v1"
LATENCY_REPORT = E3E5_OUT / "e5_latency_v1.json"
CRITERION_REPORT = E3E5_OUT / "e3_e5_criterion_v1.json"
GATE_REPORT = FALS_OUT / "e5_gate_bound_v1.json"

DEFAULT_DATA_ROOT = (
    "C:/code/phd-snn/auto-paper/20261110-152407/"
    "Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/"
    "f-research-v1-run-20261110-172308-superpixel-graph-vision/data/pyg_bench"
)

PY = sys.executable

LATENCY = ["e3_e5_v1.py", "latency", "--device", "cuda", "--batches", "40",
           "--warmup", "1", "--repeats", "3", "--resume"]
AGGREGATE = ["e3_e5_v1.py", "aggregate"]
FALS_RUN = ["e5_falsifier_v1.py", "run", "--device", "cuda", "--batches", "40"]
FALS_VERIFY = ["e5_falsifier_v1.py", "verify"]

PLAN = [
    ("01_e5_latency_resume", LATENCY),
    ("02_e3e5_aggregate", AGGREGATE),
    ("03_e5_falsifier_run", FALS_RUN),
    ("04_e5_falsifier_verify", FALS_VERIFY),
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def scrub(text: str) -> str:
    """Remove local identity/home/temp paths before anything is written to an
    uploaded log.  Public evidence keeps the command arguments verbatim."""
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-print", action="store_true",
                    help="print the serial plan and resolved paths, run nothing")
    args = ap.parse_args()
    env = build_env()
    if args.dry_print:
        print(json.dumps({
            "dry_print": True, "root": str(ROOT),
            "data_root": env["GNN2SNN_DATA_ROOT"],
            "dcgs_repo": env["DCGS_REPO"],
            "python": PY,
            "plan": [{"name": n, "cmd": command(a)} for n, a in PLAN],
            "latency_report_exists": LATENCY_REPORT.is_file(),
            "criterion_report_exists": CRITERION_REPORT.is_file(),
            "gate_report_exists": GATE_REPORT.is_file(),
        }, indent=2))
        return

    LOGDIR.mkdir(parents=True, exist_ok=True)
    emit(f"DRIVER_START pid={os.getpid()} python={PY} steps={len(PLAN)}")
    plan = list(PLAN)
    summary = []
    i = 0
    while i < len(plan):
        name, argv = plan[i]
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
        elapsed = round(time.monotonic() - t0, 1)
        rec = {"step": name, "rc": rc, "elapsed_seconds": elapsed,
               "start_utc": utc(), "log": f"{RR}/logs/gpu_queue_driver_v1/{name}.log",
               "cmd": [scrub(x) for x in cmd[2:]]}
        with STATUS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        emit(f"STEP_END {name} rc={rc} elapsed_sec={elapsed} log={log.name}")
        summary.append(rec)

        # Keep the queue moving: a failed head-of-line item never stalls the round.
        if name == "01_e5_latency_resume" and not LATENCY_REPORT.is_file():
            plan.insert(i + 1, ("01b_e5_latency_resume_retry", LATENCY))
            emit("QUEUE_INSERT 01b_e5_latency_resume_retry (identical --resume params)")
        if name == "01b_e5_latency_resume_retry" and LATENCY_REPORT.is_file():
            plan.insert(i + 1, ("02b_e3e5_aggregate_retry", AGGREGATE))
            emit("QUEUE_INSERT 02b_e3e5_aggregate_retry")
        if name == "03_e5_falsifier_run" and not GATE_REPORT.is_file():
            plan.insert(i + 1, ("03b_e5_falsifier_run_retry", FALS_RUN))
            emit("QUEUE_INSERT 03b_e5_falsifier_run_retry")
        if name == "03b_e5_falsifier_run_retry" and GATE_REPORT.is_file():
            plan.insert(i + 1, ("04b_e5_falsifier_verify_retry", FALS_VERIFY))
            emit("QUEUE_INSERT 04b_e5_falsifier_verify_retry")
        i += 1

    emit("DRIVER_END " + json.dumps({
        "steps": len(summary),
        "failed": [s["step"] for s in summary if s["rc"] != 0],
        "latency_report": LATENCY_REPORT.is_file(),
        "criterion_report": CRITERION_REPORT.is_file(),
        "gate_report": GATE_REPORT.is_file(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
