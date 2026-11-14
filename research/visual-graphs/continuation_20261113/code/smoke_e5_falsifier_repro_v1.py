"""CPU 1-batch smoke for the read-only E5 reproducibility wrapper.

Writes ONLY into logs/e5_falsifier_repro_smoke_v1/, never into the canonical
results/e5-falsifier-v1/ or the planned results/e5-falsifier-repro-v1/ report
paths.  Verifies that redirecting the imported v3 module's OUT works and that
the canonical published report bytes are unchanged.
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
CANON = RUNROOT / "results" / "e5-falsifier-v1" / "e5_gate_bound_v1.json"
SMOKE = RUNROOT / "logs" / "e5_falsifier_repro_smoke_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    if not CANON.is_file():
        raise RuntimeError("canonical E5 report missing")
    before = sha256(CANON)
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e5_falsifier_v3 as V3

    V3.OUT = SMOKE
    V3.run(argparse.Namespace(device="cpu", batches=1))
    after = sha256(CANON)
    rp = SMOKE / "e5_gate_bound_v1.json"
    if not rp.is_file():
        raise RuntimeError("redirected OUT did not receive the smoke report")
    d = json.loads(rp.read_text(encoding="utf-8"))
    if d["fidelity_check"]["frozen_vs_instrumented_max_abs_diff"] > 1e-4:
        raise RuntimeError("smoke fidelity check failed")
    if before != after:
        raise RuntimeError("canonical report was touched by the smoke")
    print(json.dumps({
        "smoke_status": "PASS",
        "wrapper_out_redirect": True,
        "canonical_report_untouched": True,
        "canonical_sha256": before,
        "smoke_report": str(rp.relative_to(ROOT)),
        "fidelity_max_abs_diff": d["fidelity_check"]["frozen_vs_instrumented_max_abs_diff"],
    }, indent=2))


if __name__ == "__main__":
    main()
