"""Independent deterministic reproducibility rerun of E5 mechanism evidence.

WHY A WRAPPER, NOT AN EDIT: e5_falsifier_v3.py is bound by the published
R7b-sync-supplement SHA-256 manifest and MUST NOT be edited.  This module
imports it read-only and changes ONLY its output directory, then calls the
unmodified run() / verify().  Every scientific parameter is therefore
byte-identical to the published v3 code: 40 batches, all 5000 official
validation graphs in fixed sequential order, batch128, T in {1,2,4,8}, the
same frozen centered thresholds, the same gate-bound / degree / sharpness
subset definitions, the same raw-vs-decoded and fresh-vs-stale falsifiers,
the same fidelity threshold 1e-4, the same protocol and DCGS hashes.

This is a DETERMINISM CHECK, not a new scientific claim:
  * the canonical results/e5-falsifier-v1/ report is NEVER overwritten;
  * the rerun writes to results/e5-falsifier-repro-v1/;
  * compare() requires the numeric content of both reports (excluding
    wall-clock/creation timestamps) and the fixed-column METRICS.csv bytes
    to be identical.  A mismatch is reported as a real FAIL, never hidden.

No energy, hardware-latency, accuracy, novelty or method-advantage claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
CANON_OUT = RUNROOT / "results" / "e5-falsifier-v1"
REPRO_OUT = RUNROOT / "results" / "e5-falsifier-repro-v1"
REPORT_NAME = "e5_gate_bound_v1.json"
METRICS_NAME = "falsifier_METRICS.csv"
VOLATILE_KEYS = {"total_job_seconds", "created_at_utc", "created_at",
                 "elapsed_seconds", "timestamp", "utc"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items()
                if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def compare() -> None:
    a, b = CANON_OUT / REPORT_NAME, REPRO_OUT / REPORT_NAME
    ca = CANON_OUT / METRICS_NAME
    cb = REPRO_OUT / METRICS_NAME
    for p in (a, b, ca, cb):
        if not p.is_file():
            raise RuntimeError(f"missing artifact: {p}")
    ra = strip_volatile(json.loads(a.read_text(encoding="utf-8")))
    rb = strip_volatile(json.loads(b.read_text(encoding="utf-8")))
    differing = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
    csv_equal = ca.read_bytes() == cb.read_bytes()
    result = {
        "check": "e5_falsifier_repro_compare_v1",
        "status": "PASS" if (not differing and csv_equal) else "FAIL",
        "canonical_report_sha256": sha256(a),
        "rerun_report_sha256": sha256(b),
        "canonical_csv_sha256": sha256(ca),
        "rerun_csv_sha256": sha256(cb),
        "numeric_report_equal_excluding_timestamps": not differing,
        "differing_top_level_keys": differing,
        "fixed_column_metrics_csv_byte_identical": csv_equal,
        "note": ("determinism/reproducibility check only; NOT new evidence, "
                 "NOT a scientific acceptance, no energy/latency/accuracy claim"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


def main() -> None:
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e5_falsifier_v3 as V3

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("precheck")
    sub.add_parser("verify")
    r = sub.add_parser("run")
    r.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    r.add_argument("--batches", type=int, default=40)
    sub.add_parser("compare")
    a = p.parse_args()

    if a.cmd == "compare":
        compare()
        return

    # The ONLY mutation: redirect output away from the published canonical
    # report.  Nothing else in the bound module is changed.
    V3.OUT = REPRO_OUT
    if a.cmd == "run":
        V3.run(a)
    elif a.cmd == "verify":
        V3.verify()
    else:
        V3.precheck()


if __name__ == "__main__":
    main()
