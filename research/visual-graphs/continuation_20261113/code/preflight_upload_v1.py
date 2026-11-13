"""Pre-upload content check for an explicit R4b artifact set (read-only).

Reads the checkpoint SHA-256 manifest plus an explicit list of collaboration
documents, and verifies every file exists, is below the 50 MiB upload ceiling,
and carries no credentials / private-home identity paths. NPZ per-sample
evidence is opened with allow_pickle=False and checked for numeric-only arrays.

Writes a public-safe JSON report: relative paths and booleans only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

MAX_BYTES = 50 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".json", ".csv", ".md", ".log", ".txt", ".yaml", ".yml"}
FORBIDDEN_TEXT = [
    re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),
    re.compile(r"/User[s]/", re.I),
    re.compile(r"AppData[\\/]Local[\\/]Temp", re.I),
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"(api[_-]?key|access[_-]?token|bearer\s+[A-Za-z0-9._-]{20,})\s*[:=]", re.I),
]
# Derive the local account name at runtime so the checker itself carries no
# hard-coded machine identity. On a machine without USERNAME, fall back to the
# home directory name; if both are absent, no username pattern is applied.
_LOCAL_ACCOUNT = os.environ.get("USERNAME") or Path.home().name
LOCAL_USER = re.compile(re.escape(_LOCAL_ACCOUNT), re.I) if _LOCAL_ACCOUNT else None
DRIVE_PATH = re.compile(r"[A-Za-z]:[\\/]")
DISALLOWED_SUFFIXES = {".pt", ".pth", ".ckpt", ".zip", ".tar", ".gz", ".7z", ".rar"}
EXPECTED_NPZ_KEYS = {"indices", "labels", "teacher_logits", "logits"}


def check_text(path: Path, is_public_report: bool, problems: list[str]) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in FORBIDDEN_TEXT:
        if pattern.search(text):
            problems.append(f"forbidden pattern {pattern.pattern!r}")
    if LOCAL_USER is not None and LOCAL_USER.search(text):
        problems.append("local username present")
    if is_public_report and DRIVE_PATH.search(text):
        problems.append("absolute drive path in a public report")


def check_npz(path: Path, problems: list[str]) -> dict:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        kinds = {k: z[k].dtype.kind for k in z.files}
        shapes = {k: list(z[k].shape) for k in z.files}
    if not EXPECTED_NPZ_KEYS.issubset(keys):
        problems.append(f"missing NPZ keys: {sorted(EXPECTED_NPZ_KEYS - keys)}")
    if any(k in "OUSV" for k in kinds.values()):
        problems.append("non-numeric (object/string/void) array in NPZ")
    return {"keys": sorted(keys), "dtype_kinds": kinds, "shapes": shapes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rel_paths = [a["path"] for a in doc["artifacts"]] + list(args.extra)

    problems: dict[str, list[str]] = {}
    total = 0
    largest = 0
    npz_count = 0
    npz_example = None
    for rel in rel_paths:
        p = repo / rel
        issues: list[str] = []
        if not p.is_file():
            issues.append("missing file")
        else:
            size = p.stat().st_size
            total += size
            largest = max(largest, size)
            if size > MAX_BYTES:
                issues.append(f"exceeds 50 MiB ({size} bytes)")
            if p.suffix.lower() in DISALLOWED_SUFFIXES:
                issues.append(f"disallowed binary type {p.suffix}")
            if p.suffix.lower() in TEXT_SUFFIXES:
                check_text(p, ".public.json" in p.name or p.name.endswith("public.json"), issues)
            if p.suffix.lower() == ".npz":
                npz_count += 1
                info = check_npz(p, issues)
                if npz_example is None:
                    npz_example = info
        if issues:
            problems[rel] = issues

    report = {
        "check": "upload_content_preflight_r4b_v1",
        "status": "PASS" if not problems else "FAIL",
        "files_checked": len(rel_paths),
        "npz_checked": npz_count,
        "npz_schema_example": npz_example,
        "total_bytes": total,
        "largest_single_file_bytes": largest,
        "max_allowed_bytes": MAX_BYTES,
        "problems": problems,
        "note": "content/format check only; no scientific or remote-sync claim",
    }
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "npz_schema_example"}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
