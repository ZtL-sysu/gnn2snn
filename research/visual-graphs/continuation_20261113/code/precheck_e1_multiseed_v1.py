"""Read-only precheck for the multiseed E1 evaluator: no GPU, no 5000-graph pass."""
from __future__ import annotations
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = Path(r"C:\code\gnn2snn-wb-resume")
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
SOURCE = ROOT / "research" / "visual-graphs" / "takeover_20261112"
DATA = Path(r"C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\data\pyg_bench")
AUTHOR = Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"

sys.path.insert(0, str(CODE))
sys.path.insert(0, str(SOURCE))
os_root = __import__("os")
os_root.environ.setdefault("GNN2SNN_ROOT", str(ROOT))
os_root.environ.setdefault("GNN2SNN_DATA_ROOT", str(DATA))
os_root.environ.setdefault("DCGS_REPO", str(AUTHOR))

spec = importlib.util.spec_from_file_location("e1ms", CODE / "eval_e1_multiseed_v1.py")
e1 = importlib.util.module_from_spec(spec)
sys.modules["e1ms"] = e1
spec.loader.exec_module(e1)

cal_helper = importlib.import_module("gated_affine_cal_v1")
affine_helper = importlib.import_module("gated_affine_base_v1")
MTH_diff, MTH_diff_line, provenance = affine_helper.require_author_sources(AUTHOR)

report = {"precheck": True, "author_sources": sorted(provenance["source_files"].keys())}
for seed in (42, 43):
    td = RUNROOT / "clean-restore-v1" / "research" / "visual-graphs" / "takeover_20261112" / "full_teacher_runs" / f"gated_cifar10_seed{seed}_full_v2"
    caldir = RUNROOT / "results" / f"e1-calibration-seed{seed}-v1"
    affine_cal = caldir / f"affine_cal_seed{seed}_v1.json"
    six_cal = caldir / f"gcal_seed{seed}_v1.json"
    metadata = json.loads((td / "metadata.json").read_text(encoding="utf-8"))
    done = json.loads((td / "done.json").read_text(encoding="utf-8"))
    ck = e1.base.sha256(td / "best.pt")
    adoc = e1.verify_affine_calibration(affine_cal, td, seed, ck, metadata, done, cal_helper)
    sdoc = e1.verify_six_calibration(six_cal, seed, ck, metadata, done)
    shapes = {}
    for mode in e1.MODES:
        th = e1.thresholds_for(mode, adoc, sdoc, cal_helper, torch.device("cpu"), torch.float32)
        shapes[mode] = [int(t.numel()) for t in th]
        assert all(t.numel() == 70 for t in th), f"threshold width at {mode}"
        assert all(bool(torch.isfinite(t).all()) and bool((t > 0).all()) for t in th), f"threshold positivity at {mode}"
    report[f"seed{seed}"] = {
        "affine_sites": len(adoc["calibration"]["sites"]),
        "six_sites": len(sdoc["calibration"]["channel_thresholds"]),
        "threshold_vector_counts": {m: len(v) for m, v in shapes.items()},
        "threshold_widths_ok": all(all(w == 70 for w in v) for v in shapes.values()),
        "expected_teacher_accuracy": done["best_val_accuracy"],
        "affine_cal_sha256": e1.base.sha256(affine_cal),
        "six_cal_sha256": e1.base.sha256(six_cal),
    }
report["status"] = "PASS"
print(json.dumps(report, indent=2))
