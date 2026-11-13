"""Prove the multiseed E1 threshold/forward mapping is bitwise unchanged vs seed41.

Pure CPU, synthetic arrays only; no official dataset, no GPU job, no model
forward. Compares, on the SAME frozen seed-41 calibration documents:

  * eval_e1_multiseed_v1.thresholds_for  (new, per-seed parameterised)
  * gated_affine_cal_v1.profile_thresholds            (seed41 affine evaluator)
  * eval_controls_validation_v2.control_thresholds   (seed41 control evaluator)

All six implemented rows x T in {1,2,4,8} threshold TENSORS must be equal.
"""
from __future__ import annotations
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = Path(r"C:\code\gnn2snn-wb-resume")
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
TAKEOVER = RUNROOT / "clean-restore-v1" / "research" / "visual-graphs" / "takeover_20261112"
SOURCE = ROOT / "research" / "visual-graphs" / "takeover_20261112"
DATA = Path(r"C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\data\pyg_bench")
AUTHOR = Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"

sys.path.insert(0, str(CODE))
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("GNN2SNN_ROOT", str(ROOT))
os.environ.setdefault("GNN2SNN_DATA_ROOT", str(DATA))
os.environ.setdefault("DCGS_REPO", str(AUTHOR))

spec = importlib.util.spec_from_file_location("e1ms", CODE / "eval_e1_multiseed_v1.py")
e1 = importlib.util.module_from_spec(spec)
sys.modules["e1ms"] = e1
spec.loader.exec_module(e1)

import eval_controls_validation_v2 as ref_controls  # noqa: E402
import gated_affine_cal_v1 as cal_helper  # noqa: E402

affine_doc = json.loads((TAKEOVER / "affine_cal1.json").read_text(encoding="utf-8"))
six_doc = json.loads((TAKEOVER / "full_teacher_runs" / "gated_cifar10_seed41_full_v1" / "gcal_v1.json").read_text(encoding="utf-8"))
cal_helper.validate_statistics(affine_doc["calibration"])

dev, dtype = torch.device("cpu"), torch.float32
report = {"test": "flow_equivalence_v1", "rows": {}, "all_equal": True}
for mode in e1.MODES:
    new = e1.thresholds_for(mode, affine_doc, six_doc, cal_helper, dev, dtype)
    if mode in e1.AFFINE_MODES:
        old = [t.to(device=dev, dtype=dtype)
               for t in cal_helper.profile_thresholds(affine_doc, mode, 1)]
    else:
        old = ref_controls.control_thresholds(mode, affine_doc, six_doc, cal_helper, dev, dtype)
    assert len(new) == len(old), f"vector count differs at {mode}"
    equal = all(torch.equal(a, b) for a, b in zip(new, old))
    report["rows"][mode] = {"vectors": len(new),
                            "widths": [int(t.numel()) for t in new],
                            "bitwise_equal_to_seed41_reference": equal}
    report["all_equal"] = report["all_equal"] and equal

# Also verify the per-seed calibrations keep the identical site/threshold schema.
for seed in (42, 43):
    d = RUNROOT / "results" / f"e1-calibration-seed{seed}-v1"
    a = json.loads((d / f"affine_cal_seed{seed}_v1.json").read_text(encoding="utf-8"))
    s = json.loads((d / f"gcal_seed{seed}_v1.json").read_text(encoding="utf-8"))
    cal_helper.validate_statistics(a["calibration"])
    report[f"seed{seed}_schema"] = {
        "affine_sites_equal": set(a["calibration"]["channel_thresholds"]) == set(cal_helper.SITES),
        "six_sites_equal": set(s["calibration"]["channel_thresholds"]) == set(ref_controls.SIX_SITES),
        "affine_function": a["algorithm_binding"]["calibration_function"],
        "six_function": s["algorithm_binding"]["calibration_function"],
    }
    report["all_equal"] = report["all_equal"] and all(
        (report[f"seed{seed}_schema"][k] if isinstance(v, bool) else True)
        for k, v in report[f"seed{seed}_schema"].items() if isinstance(v, bool))

report["status"] = "PASS" if report["all_equal"] else "FAIL"
print(json.dumps(report, indent=2))
if not report["all_equal"]:
    raise SystemExit(1)
