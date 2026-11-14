"""R7c/v3: session-independent serial GPU queue driver with an appendable queue.

Why this file is NEW (manifest discipline): e5_falsifier_v1.py is bound by the
published R6c manifest, and e5_falsifier_v2.py / gpu_queue_driver_v2.py /
gpu_queue_launcher_v2.py are bound by the R7b2 manifest.  Their bytes are never
edited.  v2 completed all 40 GPU batches (527.8 s) and then crashed while
SERIALISING the report ('Acc' object has no attribute 'items'); v3 fixes that
and has passed a full CPU run -> report -> verify smoke test.

The driver is launched by a NO-TRIGGER Windows scheduled task via
gpu_queue_launcher_v3.py, so its process tree belongs to the Task Scheduler
service and survives the automation session boundary (the root cause of the
three earlier interruptions, repaired in R7).

Queue semantics: steps live in logs/gpu_queue_v3.json as a list of
{"name": ..., "argv": [...], "retry_argv": [...] | null}.  The queue file is
RE-READ between steps, so a wake-up may append the next real GPU job while an
earlier step is still running; the driver then consumes it serially and the
3060 never idles.  Already-executed step names are tracked in
gpu_queue_driver_v3.status.jsonl and are never re-run.  A failed step keeps its
raw log and rc and does NOT stall the remaining queue; when a step declares
retry_argv, exactly one identical-parameter retry is inserted.

No scientific parameter is changed here: --device cuda --batches 40, frozen
thresholds, all 5000 official validation graphs, 1 warmup where applicable,
and the pinned DCGS commit are untouched by this driver.
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
LOGDIR = RUNROOT / "logs" / "gpu_queue_driver_v3"
DRIVER_LOG = RUNROOT / "logs" / "gpu_queue_driver_v3.driver.log"
STATUS = RUNROOT / "logs" / "gpu_queue_driver_v3.status.jsonl"
QUEUE = RUNROOT / "logs" / "gpu_queue_v3.json"
GATE_REPORT = RUNROOT / "results" / "e5-falsifier-v1" / "e5_gate_bound_v1.json"

DEFAULT_DATA_ROOT = (
    "C:/code/phd-snn/auto-paper/20261110-152407/"
    "Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/"
    "f-research-v1-run-20261110-172308-superpixel-graph-vision/data/pyg_bench"
)

PY = sys.executable

DEFAULT_PLAN = [
    {"name": "01_e5_falsifier_v3_run",
     "argv": ["e5_falsifier_v3.py", "run", "--device", "cuda", "--batches", "40"],
     "retry_argv": ["e5_falsifier_v3.py", "run", "--device", "cuda", "--batches", "40"]},
    {"name": "02_e5_falsifier_v3_verify",
     "argv": ["e5_falsifier_v3.py", "verify"],
     "retry_argv": None},
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


def load_queue() -> list[dict]:
    if not QUEUE.is_file():
        return list(DEFAULT_PLAN)
    raw = json.loads(QUEUE.read_text(encoding="utf-8"))
    steps = raw["steps"] if isinstance(raw, dict) else raw
    for s in steps:
        require_str = isinstance(s.get("name"), str) and isinstance(s.get("argv"), list)
        if not require_str:
            raise RuntimeError("malformed queue step: need name and argv")
    return steps


def done_names() -> set[str]:
    if not STATUS.is_file():
        return set()
    names = set()
    for line in STATUS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                names.add(json.loads(line)["step"])
            except (KeyError, ValueError):
                pass
    return names


def run_step(name: str, argv: list[str], env: dict[str, str]) -> int:
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
           "log": f"{RR}/logs/gpu_queue_driver_v3/{name}.log",
           "cmd": [scrub(x) for x in cmd[2:]]}
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    emit(f"STEP_END {name} rc={rc} elapsed_sec={rec['elapsed_seconds']}")
    return rc


def main() -> None:
    env = build_env()
    if "--dry-print" in sys.argv:
        print(json.dumps({
            "dry_print": True, "root": str(ROOT),
            "data_root": env["GNN2SNN_DATA_ROOT"], "dcgs_repo": env["DCGS_REPO"],
            "queue_file": str(QUEUE), "plan": [
                {"name": s["name"], "cmd": command(s["argv"])} for s in load_queue()],
            "gate_report_exists": GATE_REPORT.is_file(),
        }, indent=2))
        return

    LOGDIR.mkdir(parents=True, exist_ok=True)
    emit(f"DRIVER_V3_START pid={os.getpid()} python={PY}")
    executed = done_names()
    # Re-read the queue after every step: a wake-up can append the next real
    # GPU job while an earlier one is still running, so the 3060 never idles.
    idle_polls = 0
    while True:
        steps = load_queue()
        pending = [s for s in steps if s["name"] not in executed]
        if not pending:
            idle_polls += 1
            if idle_polls > 120:  # 120 * 15 s = 30 min grace for queue appends
                emit("DRIVER_V3_IDLE_END no_new_steps_within_grace_period")
                break
            time.sleep(15)
            continue
        idle_polls = 0
        step = pending[0]
        name = step["name"]
        rc = run_step(name, step["argv"], env)
        executed.add(name)
        if rc != 0 and step.get("retry_argv"):
            retry_name = name + "_retry"
            if retry_name not in executed:
                emit(f"STEP_RETRY {name} -> {retry_name} (identical parameters)")
                rc2 = run_step(retry_name, step["retry_argv"], env)
                executed.add(retry_name)
                if rc2 == 0:
                    emit(f"STEP_RETRY_OK {retry_name}")
    emit("DRIVER_V3_END " + json.dumps({
        "gate_report": GATE_REPORT.is_file(),
        "steps_executed": len(executed),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
