"""CPU-only checks for v3 reuse, full-test reference validation, and provenance guards."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch_geometric.data import Batch, Data

HERE = Path(__file__).resolve().parent
RUNROOT = HERE.parent
DEFAULT_AUTHOR_REPO = Path(r"C:\Users\xhf\AppData\Local\Temp\dcgs-5e0e4c73")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expect_error(fn, fragment):
    try:
        fn()
    except Exception as exc:
        if fragment not in str(exc):
            raise AssertionError("unexpected error: %s" % exc) from exc
        return
    raise AssertionError("expected error containing %r" % fragment)


def make_reference(directory, n=10000):
    rng = np.random.RandomState(20261113)
    labels = rng.randint(0, 10, size=n).astype(np.int64)
    logits = rng.normal(size=(n, 10)).astype(np.float32)
    path = directory / "teacher_full_test_predictions.npz"
    np.savez_compressed(path, index=np.arange(n, dtype=np.int64), labels=labels,
                        logits=logits, predictions=logits.argmax(1))
    sidecar = path.with_name("teacher_full_test.json")
    sidecar.write_text(json.dumps({"dataset": "CIFAR10", "seed": 41, "graphs": n,
                                   "split": "full official test, historically exposed",
                                   "checkpoint_sha256": "checkpoint-lock",
                                   "prediction_artifact_sha256": _sha256(path)}),
                       encoding="utf-8")
    return path, labels, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", type=Path, default=DEFAULT_AUTHOR_REPO)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    v1 = load_module("dcgs_fixed_eval_v1_forward_ref", HERE / "dcgs_fixed_residual_gcn_event_forward_v1.py")
    v2 = load_module("dcgs_fixed_eval_v2_ref", HERE / "dcgs_fixed_seed_eval_v2.py")
    v3 = load_module("dcgs_fixed_eval_v3_under_test", HERE / "dcgs_fixed_seed_eval_v3.py")

    MTH_diff, commit, provenance = v2.require_author_source(args.author_repo)
    if commit != v2.PINNED_COMMIT:
        raise AssertionError("unexpected pinned author source")

    # Reuse must preserve the locked fixed-forward synthetic outputs across T.
    got = v3.V2.synthetic_acceptance(MTH_diff, v3.V2.load_oracle())
    ref = v1.synthetic_acceptance(MTH_diff, v1.load_oracle())
    if json.dumps(got, sort_keys=True) != json.dumps(ref, sort_keys=True):
        raise AssertionError("v3 imported forward differs from v1 synthetic reference")
    for T in (1, 2, 4, 8):
        row = got["results"]["T%d" % T]
        if row["warmup_calls"] != 6 or row["active_unit_calls"] != 6 * T:
            raise AssertionError("unexpected call accounting at T=%d" % T)

    from full_teacher_gcn import BenchmarkGCN
    from fold_teacher_bn import fold_model
    torch.manual_seed(41)
    graph = Data(x=torch.tensor([[0.2, 0.4], [0.8, -0.1], [0.3, 0.7]]),
                 pos=torch.tensor([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2]]),
                 edge_index=torch.tensor([[0, 1, 1, 2, 2, 0],
                                          [1, 0, 2, 1, 0, 2]], dtype=torch.long),
                 y=torch.tensor([2]))
    batch = Batch.from_data_list([graph])
    stripped = v3._strip_targets(batch)
    if "y" in stripped or int(stripped.num_graphs) != 1:
        raise AssertionError("conversion batch retained labels or lost graph count")
    folded = fold_model(BenchmarkGCN(input_dim=4, width=8, layers=4,
                                     num_classes=3).eval()).eval()
    theta = [[0.23] * 8] * 4 + [[0.25] * 4, [0.27] * 2]
    for T in (1, 2, 4, 8):
        with torch.no_grad():
            with_y = v2.dcgs_event_forward(folded, batch, theta, T, MTH_diff)[0]
            without_y = v2.dcgs_event_forward(folded, stripped, theta, T, MTH_diff)[0]
        if not torch.allclose(with_y, without_y, atol=1e-7, rtol=0):
            raise AssertionError("stripping y changed event forward at T=%d" % T)

    # Real completed seed-41 run and calibration validate without loading data.
    teacher_run = HERE / "full_teacher_runs" / "CIFAR10_seed41_full"
    calibration = teacher_run / "calibration_hist_v1.json"
    _, meta, _, _, checkpoint_sha, _ = v3._bound_inputs(
        "CIFAR10", 41, teacher_run, calibration, args.author_repo, provenance)
    if meta.get("dataset") != "CIFAR10":
        raise AssertionError("wrong validated teacher metadata")
    real_reference = v3._read_reference(
        teacher_run / "teacher_full_test_predictions.npz", "CIFAR10", 41, 10000,
        checkpoint_sha)
    if not np.array_equal(real_reference["indices"], np.arange(10000)):
        raise AssertionError("real seed-41 test reference ordering mismatch")
    expect_error(lambda: v2.validate_inputs("CIFAR10", 42, teacher_run, calibration, provenance),
                 "teacher dataset/seed mismatch")
    expect_error(lambda: v2.validate_inputs("CIFAR10", 41, RUNROOT / "outside", calibration, provenance),
                 "teacher run must be inside")

    # Synthetic 10k archive exercises full coverage/order, labels, identity, and hash guards.
    with tempfile.TemporaryDirectory(prefix=".g3-", dir=str(HERE)) as temp:
        td = Path(temp)
        path, labels, logits = make_reference(td)
        valid = v3._read_reference(path, "CIFAR10", 41, 10000, "checkpoint-lock")
        if len(valid["indices"]) != 10000 or not np.array_equal(valid["labels"], labels):
            raise AssertionError("synthetic full-test reference was not accepted intact")
        if v3._compare_reference(np.arange(10000), labels, logits, valid) != 0.0:
            raise AssertionError("identical synthetic teacher predictions did not match")
        bad_logits = logits.copy(); bad_logits[12, 2] += 0.01
        expect_error(lambda: v3._compare_reference(np.arange(10000), labels, bad_logits, valid),
                     "teacher logits differ")
        bad_labels = labels.copy(); bad_labels[12] = (bad_labels[12] + 1) % 10
        expect_error(lambda: v3._compare_reference(np.arange(10000), bad_labels, logits, valid),
                     "labels differ")
        bad_indices = np.arange(10000); bad_indices[[0, 1]] = bad_indices[[1, 0]]
        expect_error(lambda: v3._compare_reference(bad_indices, labels, logits, valid),
                     "indices differ")
        expect_error(lambda: v3._read_reference(path, "CIFAR10", 42, 10000, "checkpoint-lock"),
                     "reference dataset/seed mismatch")
        expect_error(lambda: v3._read_reference(RUNROOT / "outside.npz", "CIFAR10", 41,
                                                10000, "checkpoint-lock"),
                     "reference predictions must be inside")
        sidecar = path.with_name("teacher_full_test.json")
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        sidecar_data["prediction_artifact_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(sidecar_data), encoding="utf-8")
        expect_error(lambda: v3._read_reference(path, "CIFAR10", 41, 10000,
                                                "checkpoint-lock"),
                     "reference prediction archive hash mismatch")

    print(json.dumps({"device": "cpu", "official_dataset_loaded": False,
                      "full_test_synthetic_rows": 10000,
                      "synthetic_t_grid_matches_v1": True,
                      "T_values": [1, 2, 4, 8], "warmup_calls_per_batch": 6,
                      "active_calls_per_batch": {"T%d" % T: 6 * T for T in (1, 2, 4, 8)},
                      "real_completed_input_validated": {"dataset": "CIFAR10", "seed": 41,
                                                          "checkpoint_sha256": checkpoint_sha},
                      "real_seed41_test_reference_validated": {
                          "rows": len(real_reference["indices"]),
                          "archive_sha256": real_reference["archive_sha256"]},
                      "negative_checks": ["teacher seed mismatch", "teacher path escape",
                                          "reference wrong seed", "reference path escape",
                                          "reference archive hash mismatch", "reference label mismatch",
                                          "reference index order mismatch",
                                          "reference teacher-logit mismatch"]}, indent=2))


if __name__ == "__main__":
    main()
