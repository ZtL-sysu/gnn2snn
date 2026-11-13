"""Train-only shared-threshold controls for affine-base profiles.

This is a new immutable driver. It reuses the v1 calibration/profile/helper
implementations but does not change them. No labels, validation or test data
are used. The dense profile is diagnostic, not an efficiency/energy result.
"""
import os
_EXPECTED_CUBLAS = ":4096:8"
_cublas_now = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _cublas_now not in (None, _EXPECTED_CUBLAS):
    raise SystemExit("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before importing Torch; got %r" % _cublas_now)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _EXPECTED_CUBLAS

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import gated_affine_cal_v1 as C
import gated_affine_profile_v1 as P
import gated_affine_base_v1 as A

HERE = Path(__file__).resolve().parent
CAL_PATH = HERE / "affine_cal1.json"
PROTOCOL = HERE / "affine-base-fulltrain-protocol-v1.md"
CLARIFICATION = HERE / "affine-screen-decision-clarification-v1.md"
AUDIT_JSON = HERE / "affine-six-profile-audit-v1.json"
AUDIT_MD = HERE / "affine-six-profile-audit-v1.md"
AUDIT_SCRIPT = HERE / "audit_affine_profiles_v1.py"
MODES = ("first_analog", "centered")
SCALES = (1, 8)
PREFIXES = (1, 2, 4)
PROFILE_MODES = ("generic", "first_analog", "centered")
DEPENDENCIES = (
    "gated_affine_shared_profile_v1.py", "check_gated_affine_shared_profile_v1.py",
    "gated_affine_profile_v1.py", "check_gated_affine_profile_v1.py",
    "gated_affine_base_v1.py", "check_gated_affine_base_v1.py",
    "gated_affine_cal_v1.py", "check_gated_affine_cal_v1.py",
    "activation_histogram.py", "gated_frozen_gate_calibrate.py",
    "gated_teacher_model.py", "affine-base-fulltrain-protocol-v1.md",
    "affine-screen-decision-clarification-v1.md", "audit_affine_profiles_v1.py",
    "affine-six-profile-audit-v1.json", "affine-six-profile-audit-v1.md",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile_result_path(mode, scale):
    return HERE / f"affine_{mode}_s{scale}_tr1.json"


def shared_thresholds(calibration_doc, scale):
    if scale not in SCALES:
        raise ValueError("scale must be 1 or 8")
    raw = C.profile_thresholds(calibration_doc, "first_analog", scale)
    residual = C.profile_thresholds(calibration_doc, "centered", scale)
    if len(raw) != 6 or len(residual) != 6:
        raise RuntimeError("expected six layer-1..3 raw and residual threshold vectors")
    return merge_threshold_families(raw, residual)


def merge_threshold_families(raw, residual):
    """Elementwise max of corresponding raw/residual per-channel thresholds."""
    if len(raw) != 6 or len(residual) != 6:
        raise ValueError("expected six layer-1..3 raw and residual threshold vectors")
    merged = []
    for index, (a, b) in enumerate(zip(raw, residual)):
        if a.shape != b.shape or a.ndim != 1 or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise RuntimeError("invalid corresponding threshold vectors at site %d" % index)
        theta = torch.maximum(a, b)
        if not bool(torch.isfinite(theta).all()) or bool((theta <= 0).any()):
            raise RuntimeError("shared threshold must be finite and positive")
        merged.append(theta)
    return merged


def prior_audit_gate(author_repo):
    if not all(p.is_file() for p in (AUDIT_JSON, AUDIT_MD, AUDIT_SCRIPT, CLARIFICATION, PROTOCOL)):
        raise RuntimeError("missing six-profile audit, clarification, audit script, or frozen protocol")
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    controls = audit.get("conditional_controls", {})
    if (audit.get("status") != "complete" or
            controls.get("overall_shared_threshold_controls_trigger") is not True or
            not any(controls.get(str(s), {}).get("shared_threshold_controls_trigger") is True for s in SCALES)):
        raise RuntimeError("shared-threshold control is not authorized by completed six-profile audit")
    records = {(x.get("mode"), x.get("scale")): x for x in audit.get("profiles", [])}
    expected = {(m, s) for m in PROFILE_MODES for s in SCALES}
    if set(records) != expected:
        raise RuntimeError("audit does not bind exactly the six source profiles")
    previous_inputs = {}
    for mode, scale in expected:
        path = profile_result_path(mode, scale)
        if not path.is_file() or sha(path) != records[(mode, scale)].get("profile_sha256"):
            raise RuntimeError("one of the six frozen profile JSON hashes changed: %s" % path.name)
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_start = prior.get("input_hashes_start", {})
        if not prior_start or prior.get("input_hashes_end") != prior_start:
            raise RuntimeError("prior profile has missing/mismatched input binding")
        for source_path, digest in prior_start.items():
            if source_path in previous_inputs and previous_inputs[source_path] != digest:
                raise RuntimeError("conflicting prior profile input bindings")
            previous_inputs[source_path] = digest
    C.require_unchanged(previous_inputs)
    # Read and bind the exact clarification/protocol text that predeclared this screen.
    clarify_text = CLARIFICATION.read_text(encoding="utf-8")
    if "ALL T1/T2/T4" not in clarify_text or "all three prefixes" not in clarify_text.lower():
        raise RuntimeError("conditional-control clarification lacks the all-prefix rule")
    return audit, records


def shared_threshold_profile_loader(model, loader, thresholds, mode, unit_relu, unit_line, device,
                                    expected=45000, width=70, batch_size=128):
    if mode not in MODES:
        raise ValueError("shared controls support only first_analog/centered")
    if len(thresholds) != 6:
        raise ValueError("shared controls require six corresponding vectors")
    return P.profile_loader(model, loader, thresholds, mode, unit_relu, unit_line,
                            device, expected=expected, width=width, batch_size=batch_size)


def start_hashes(cal_doc, profile_docs, audit, author):
    bound = dict(cal_doc["input_hashes_start"])
    paths = [HERE/name for name in DEPENDENCIES]
    paths += [CAL_PATH, AUDIT_JSON, AUDIT_MD, AUDIT_SCRIPT, CLARIFICATION, PROTOCOL]
    paths += [profile_result_path(m, s) for m in PROFILE_MODES for s in SCALES]
    for item in author["source_files"].values():
        paths.append(Path(author["author_repo"]) / item["path"])
    for path in paths:
        path = path.resolve()
        bound[str(path)] = sha(path)
    # Do not accept a modified source set between calibration and this run.
    for path, digest in cal_doc["input_hashes_start"].items():
        if bound.get(path) != digest:
            raise RuntimeError("calibration-bound input changed before shared profile: %s" % path)
    return dict(sorted(bound.items()))


def require_unchanged(start):
    end = {path: sha(path) for path in start}
    if end != start:
        raise RuntimeError("bound source/data/checkpoint/profile/audit changed during shared profile")
    return end


def run(output, calibration, mode, scale, author_repo, device):
    output = C.output_path(output)
    if mode not in MODES or scale not in SCALES:
        raise ValueError("unsupported mode/scale")
    if Path(calibration).resolve() != CAL_PATH.resolve():
        raise ValueError("only the frozen affine_cal1.json is allowed")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    dev = torch.device(device)
    model, teacher = C.load_teacher(dev)
    cal_doc, calhash = C.validate_calibration_document(calibration, teacher)
    audit, audit_records = prior_audit_gate(author_repo)
    relu, line, author = A.require_author_sources(author_repo)
    thresholds = shared_thresholds(cal_doc, scale)
    bound = start_hashes(cal_doc, audit_records, audit, author)
    data = GNNBenchmarkDataset(root=str(C.DATA_ROOT), name="CIFAR10", split="train")
    if len(data) != 45000:
        raise RuntimeError("full 45k training split required")
    loader = DataLoader(C.LabelFreeGraphView(data), batch_size=128, shuffle=False, num_workers=0)
    timer = time.perf_counter()
    result = shared_threshold_profile_loader(model, loader, thresholds, mode, relu, line, dev)
    end = require_unchanged(bound)
    A.validate_source_unchanged(author_repo, author)
    # Confirm the conditional gate and six result hashes still hold at completion.
    _, end_records = prior_audit_gate(author_repo)
    if end_records != audit_records:
        raise RuntimeError("six-profile audit changed during shared profile")
    result.update({
        "status": "complete", "dataset": "CIFAR10", "seed": 41, "mode": mode,
        "threshold_scale": scale, "threshold_rule": "per-channel max(raw_abs, branch_cumsum) × fixed scale",
        "scope": "all_45000_official_train_graphs_label_free_fixed_order_shared_threshold_control",
        "executed_T": 4, "prefix_time_grid": list(PREFIXES), "teacher": teacher,
        "threshold_vectors": [v.tolist() for v in thresholds],
        "calibration": str(CAL_PATH.resolve()), "calibration_sha256": calhash,
        "prior_profile_audit": str(AUDIT_JSON.resolve()), "prior_profile_audit_sha256": sha(AUDIT_JSON),
        "prior_profile_audit_md_sha256": sha(AUDIT_MD),
        "prior_profile_hashes": {f"{m}_s{s}": audit_records[(m, s)]["profile_sha256"]
                                 for m in PROFILE_MODES for s in SCALES},
        "author_source": author, "input_hashes_start": bound, "input_hashes_end": end,
        "elapsed_seconds": time.perf_counter()-timer,
        "runtime": {"device": str(dev), "torch": torch.__version__, "python": sys.version,
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "labels_passed": False, "validation_or_test_opened": False},
        "qualification": "four conditional shared-threshold controls; dense reference; teacher fidelity/event/prospective ADD diagnostics only; no accuracy/energy/latency/novelty claim"})
    C.write_exclusive(output, result)
    print(json.dumps({"status": "complete", "mode": mode, "scale": scale,
                      "output": str(output)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--scale", type=int, choices=SCALES, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    run(args.output, args.calibration, args.mode, args.scale, args.author_repo, args.device)


if __name__ == "__main__":
    main()
