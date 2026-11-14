"""R7b/v3 CPU end-to-end smoke test (temporary output, NOT evidence).

v2 completed all 40 GPU batches and only then failed while serialising the
report.  A unit test of dump_acc() alone cannot prove the whole `run` ->
report -> `verify` path.  This smoke executes that full path on ONE batch on
CPU with OUT redirected to a throwaway directory, so it cannot create the
canonical results/e5-falsifier-v1/e5_gate_bound_v1.json that the GPU job
refuses to overwrite.

Typical use (requires GNN2SNN_DATA_ROOT and DCGS_REPO in the environment):
  python -X utf8 smoke_e5_falsifier_v3.py --device cpu --batches 1

This is a plumbing test.  It produces no scientific evidence, no metrics for
the paper and no published artifact.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import types
from pathlib import Path

CODE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location(
        "e5_falsifier_v3_smoke", str(CODE / "e5_falsifier_v3.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--batches", type=int, default=1)
    args = ap.parse_args()

    mod = load()
    real_out = mod.OUT
    assert not (real_out / "e5_gate_bound_v1.json").exists(), (
        "canonical report already exists; smoke must not touch it")
    smoke = Path(tempfile.mkdtemp(prefix="e5f3-smoke-")) / "out"
    mod.OUT = smoke
    print(json.dumps({"smoke": True, "device": args.device,
                      "batches": args.batches, "temp_out": str(smoke)}), flush=True)

    mod.run(types.SimpleNamespace(device=args.device, batches=args.batches))
    mod.verify()

    report = json.loads((smoke / "e5_gate_bound_v1.json").read_text(encoding="utf-8"))
    assert set(report["gate_bound"]) == {str(T) for T in mod.TIMESTEPS}
    for T in mod.TIMESTEPS:
        assert set(report["gate_bound"][str(T)]) == {
            "all", "degree_low", "degree_high", "sharpness_low", "sharpness_high"}
    with (smoke / "falsifier_METRICS.csv").open(encoding="utf-8", newline="") as f:
        rows = list(__import__("csv").DictReader(f))
    assert len(rows) == len(mod.TIMESTEPS) * 2 * 5 * 3, len(rows)
    assert not (real_out / "e5_gate_bound_v1.json").exists(), (
        "smoke must not create the canonical report")
    print(json.dumps({"smoke_status": "PASS",
                      "fidelity_max_abs_diff":
                          report["fidelity_check"]["frozen_vs_instrumented_max_abs_diff"],
                      "metric_rows": len(rows),
                      "canonical_report_untouched": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
