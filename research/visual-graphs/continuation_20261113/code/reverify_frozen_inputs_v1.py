"""Wake-up frozen-input gate for the R4b sync (read-only, no GPU, no data pass).

Re-verifies every hash the user pinned, plus CURRENT_SCOPE and the pinned DCGS
commit, and asserts the canonical f-research-v1 pipeline state has not drifted.
Writes a public-safe JSON report (relative names and hashes only, no local
absolute paths) so the gate itself is auditable by other agents.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = STUDY.parents[2]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
SOURCE = ROOT / "research" / "visual-graphs" / "takeover_20261112"
DATA = Path(r"C:\code\phd-snn\auto-paper\20261110-152407"
            r"\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs"
            r"\f-research-v1-run-20261110-172308-superpixel-graph-vision\data\pyg_bench")
AUTHOR = Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"

EXPECTED_SCOPE = "e4439c00264a841e3ec7f08f197c17ba0398645bbe304fe5825acdd27f785ba9"
EXPECTED_DCGS = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
PASSED_PREFIX = ("P0", "P1", "P1A", "P1B", "P2", "P3", "P3A")


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    calib = load("r4b_calib", CODE / "calibrate_multiseed_v1.py")
    errors: list[str] = []
    checked: dict[str, str] = {}

    # 1. Frozen protocol / calibration / teacher / five official processed files.
    try:
        frozen = calib.verify_frozen_inputs(RUNROOT, DATA, SOURCE)
        for abs_path, digest in frozen.items():
            checked[Path(abs_path).name] = digest
    except Exception as exc:  # noqa: BLE001 - gate must report, not crash silently
        errors.append(f"frozen-input block: {exc}")

    # 2. CURRENT_SCOPE.md (user-pinned scope document).
    scope = STUDY / "CURRENT_SCOPE.md"
    scope_hash = sha256(scope)
    checked["CURRENT_SCOPE.md"] = scope_hash
    if scope_hash != EXPECTED_SCOPE:
        errors.append("CURRENT_SCOPE.md hash mismatch")

    # 3. Pinned DCGS checkout commit, resolved at runtime (no home path stored).
    dcgs_head = ""
    try:
        dcgs_head = subprocess.check_output(
            ["git", "-C", str(AUTHOR), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"DCGS rev-parse failed: {exc}")
    checked["dcgs_pinned_commit"] = dcgs_head
    if dcgs_head != EXPECTED_DCGS:
        errors.append("DCGS commit mismatch")

    # 4. Canonical pipeline state: contract v4, P0-P3A passed, P4 running only.
    state = json.loads((RUNROOT / "pipeline-state.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: s["status"] for s in state["stages"]}
    if state.get("contract_version") != "v4":
        errors.append("pipeline contract version is not v4")
    for sid in PASSED_PREFIX:
        if by_id.get(sid) != "passed":
            errors.append(f"{sid} is {by_id.get(sid)!r}, expected passed")
    if by_id.get("P4") != "running":
        errors.append(f"P4 is {by_id.get('P4')!r}, expected running")
    if any(s["status"] == "passed" for s in state["stages"] if s["id"] not in PASSED_PREFIX):
        errors.append("a stage after P4 is marked passed")

    report = {
        "check": "frozen_input_gate_r4b_v1",
        "status": "PASS" if not errors else "FAIL",
        "contract_version": state.get("contract_version"),
        "P4_status": by_id.get("P4"),
        "frozen_items_verified": len(checked),
        "hashes": checked,
        "errors": errors,
        "note": "hash-only gate; no scientific result and no remote-sync claim",
    }
    out = RUNROOT / "logs" / "frozen-input-gate-r4b-v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "hashes"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
