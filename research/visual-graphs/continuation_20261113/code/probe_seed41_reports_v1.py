"""Read-only inspection of the two seed-41 E1 reports (structure for aggregation)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r"C:\code\gnn2snn-wb-resume")
PATHS = [
    ROOT / "research" / "visual-graphs" / "continuation_20261113" / "results" / "e1-seed41-affine-v1" / "report.json",
    ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix" / "results" / "e1-seed41-controls-v1" / "report.json",
]
for p in PATHS:
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d.get("rows", [])
    print(json.dumps({
        "path": str(p.relative_to(ROOT)),
        "seed": d.get("seed"), "graphs": d.get("graphs"),
        "teacher_accuracy": d.get("teacher_accuracy"),
        "n_rows": len(rows),
        "row0_keys": sorted(rows[0].keys()) if rows else [],
        "modes": sorted({r.get("mode") for r in rows}),
        "timesteps": sorted({r.get("T") for r in rows}),
        "accuracy_columns": [k for k in rows[0].keys() if k in
                             ("accuracy", "teacher_agreement", "logit_mse",
                              "teacher_accuracy", "teacher_to_output_kl")] if rows else [],
    }, indent=2))
