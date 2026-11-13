"""CPU-only provenance and synthetic parity checks for dcgs_fixed_seed_eval_v2."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
RUNROOT = HERE.parent
DEFAULT_AUTHOR_REPO = Path(r"C:\Users\xhf\AppData\Local\Temp\dcgs-5e0e4c73")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expect_runtime_error(fn, fragment):
    try:
        fn()
    except RuntimeError as exc:
        if fragment not in str(exc):
            raise AssertionError("unexpected validation error: %s" % exc)
        return str(exc)
    raise AssertionError("expected RuntimeError containing %r" % fragment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", type=Path, default=DEFAULT_AUTHOR_REPO)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    v1 = load_module("dcgs_fixed_forward_reference_v1", HERE / "dcgs_fixed_residual_gcn_event_forward_v1.py")
    v2 = load_module("dcgs_fixed_seed_eval_under_test_v2", HERE / "dcgs_fixed_seed_eval_v2.py")

    MTH_diff, commit, provenance = v2.require_author_source(args.author_repo)
    checked = []
    records = [
        ("MNIST", 42, HERE / "full_teacher_runs" / "MNIST_seed42_full", HERE / "rep" / "M42" / "cal.json"),
        ("MNIST", 43, HERE / "full_teacher_runs" / "MNIST_seed43_full", HERE / "rep" / "M43" / "cal.json"),
        ("CIFAR10", 42, HERE / "full_teacher_runs" / "CIFAR10_seed42_full", HERE / "rep" / "C42" / "cal.json"),
    ]
    for dataset, seed, teacher_run, calibration in records:
        done, meta, cal, checkpoint, ck_sha, _ = v2.validate_inputs(
            dataset, seed, teacher_run, calibration, provenance)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(state.get("epoch", -1)) != int(cal["teacher_epoch"]):
            raise AssertionError("calibration teacher epoch does not match checkpoint state")
        checked.append({"dataset": dataset, "seed": seed, "status": done["status"],
                        "checkpoint_sha256": ck_sha, "teacher_epoch": cal["teacher_epoch"]})

    mn42 = records[0]
    expect_runtime_error(
        lambda: v2.validate_inputs("MNIST", 43, mn42[2], mn42[3], provenance),
        "teacher dataset/seed mismatch")
    expect_runtime_error(
        lambda: v2.validate_inputs("MNIST", 42, RUNROOT / "outside", mn42[3], provenance),
        "teacher run must be inside")

    # Invoke the exact saved synthetic source-oracle exercise through both
    # files; it covers all locked T values, no dataset loader, and no GPU.
    new_result = v2.synthetic_acceptance(MTH_diff, v2.load_oracle())
    ref_result = v1.synthetic_acceptance(MTH_diff, v1.load_oracle())
    if json.dumps(new_result, sort_keys=True) != json.dumps(ref_result, sort_keys=True):
        raise AssertionError("v2 synthetic outputs differ from immutable v1 forward")
    for t in (1, 2, 4, 8):
        row = new_result["results"]["T%d" % t]
        if row["warmup_calls"] != 6 or row["active_unit_calls"] != 6 * t:
            raise AssertionError("unexpected warmup/active call accounting at T=%d" % t)
    print(json.dumps({"device": "cpu", "author_commit": commit,
                      "real_completed_inputs_validated_without_dataset_load": checked,
                      "negative_checks": ["seed mismatch rejected", "teacher path escape rejected"],
                      "synthetic_forward_matches_immutable_v1": True,
                      "T_values": [1, 2, 4, 8], "warmup_calls_per_T": 6,
                      "active_calls_per_T": {"T%d" % t: 6 * t for t in (1, 2, 4, 8)},
                      "official_dataset_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
