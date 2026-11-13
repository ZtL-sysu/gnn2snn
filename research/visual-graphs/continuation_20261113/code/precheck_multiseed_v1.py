"""Read-only precheck for the multiseed calibration driver (no GPU, no 45000-graph pass)."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
ROOT = Path(r"C:\code\gnn2snn-wb-resume")
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
SOURCE = ROOT / "research" / "visual-graphs" / "takeover_20261112"
DATA = Path(r"C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\data\pyg_bench")
TEACHERS = {
    42: RUNROOT / "clean-restore-v1" / "research" / "visual-graphs" / "takeover_20261112" / "full_teacher_runs" / "gated_cifar10_seed42_full_v2",
    43: RUNROOT / "clean-restore-v1" / "research" / "visual-graphs" / "takeover_20261112" / "full_teacher_runs" / "gated_cifar10_seed43_full_v2",
}

spec = importlib.util.spec_from_file_location("calib", CODE / "calibrate_multiseed_v1.py")
calib = importlib.util.module_from_spec(spec)
sys.modules["calib"] = calib
spec.loader.exec_module(calib)

sys.path.insert(0, str(SOURCE))
six_helper = importlib.import_module("gated_frozen_gate_calibrate")
aff_helper = importlib.import_module("gated_affine_cal_v1")

report = {"precheck": True}
fh = calib.verify_frozen_inputs(RUNROOT, DATA, SOURCE)
report["frozen_inputs_verified"] = len(fh)
report["helper_hashes"] = {
    "six_site": calib.sha256(Path(six_helper.__file__).resolve()),
    "affine": calib.sha256(Path(aff_helper.__file__).resolve()),
}
report["six_site_sites"] = list(six_helper.SITE_NAMES)
report["affine_sites"] = list(aff_helper.SITES)
report["calibrate_loader_signatures"] = {
    "six": str(__import__("inspect").signature(six_helper.calibrate_loader)),
    "affine": str(__import__("inspect").signature(aff_helper.calibrate_loader)),
}
for seed, td in TEACHERS.items():
    t = calib.verify_teacher(td, seed, SOURCE, DATA, six_helper)
    bound = calib.build_bound_inputs(td, SOURCE, DATA, t["metadata"], fh)
    report[f"seed{seed}"] = {
        "checkpoint_sha256": t["checkpoint_sha256"],
        "frozen_val_accuracy": t["done"]["best_val_accuracy"],
        "source_hashes_verified": len(t["source_hashes"]),
        "bound_inputs": len(bound),
        "metadata_seed": t["metadata"]["seed"],
    }
report["status"] = "PASS"
print(json.dumps(report, indent=2))
