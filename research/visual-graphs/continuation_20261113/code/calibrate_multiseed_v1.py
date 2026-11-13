"""Frozen full-train label-free two-pass calibration for CIFAR10 GatedGCN teachers.

Seeds 42/43 only. This driver does NOT redefine the calibration algorithm. It
imports the immutable project helpers and calls the SAME functions used for the
seed-41 reference:

  * six ReLU sites  -> gated_frozen_gate_calibrate.calibrate_loader / reference_activations
  * 14 affine sites -> gated_affine_cal_v1.calibrate_loader / capture_inputs / validate_statistics

Only the teacher checkpoint, seed label, and PyG data root are parameterised.
Two ordered passes, 512 bins, quantile 0.999, all 45000 official TRAIN graphs,
label-free, batch 128, fixed sequential order, deterministic CUDA: unchanged.

The validation split, test split and any scoring label are never opened here.
Generated calibration JSONs keep the seed-41 schema so that the existing E1
evaluator/consumers can bind them by hash without changing their threshold
mapping. Original seed-41 artifacts are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch_geometric.datasets import GNNBenchmarkDataset  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
STUDY_ROOT = CODE_DIR.parent

# Frozen inputs. Any mismatch aborts before any GPU work.
FROZEN = {
    "PROTOCOL-v3.md": "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543",
    "PROTOCOL-E1-head-addendum-v1.md": "8bde5832d556c8c7b04ee630f2390703c33a26fa750f196f4dc1c94da61291f6",
    "affine_cal1.json": "1618e3b46f75646ab051229b0d954815b5e28bf1729cbbf2042e4d596766a511",
    "gcal_v1.json": "b28485ddfa8ef259ca048f3272662fedc1e4f519b0324cfa514ccf82b2cf8a9e",
    "seed41_best.pt": "d26df76d724872edc91ea11055d114935ea173282a8ef049d1b85cc1f5598b40",
}
PROCESSED_EXPECTED = {
    "pre_filter.pt": "fe7c8f11dba2284d2e9fcdc06d2333ee1a9f381570e210c4188fe6290f392192",
    "pre_transform.pt": "fe7c8f11dba2284d2e9fcdc06d2333ee1a9f381570e210c4188fe6290f392192",
    "test_data.pt": "04d71270557759c939b783d33cf69303834388d287a92b17792d9da24f3725d9",
    "train_data.pt": "b0755438ef38d1fe3f69d75ef61acc06d9334c2f35e490d4816886db163ee060",
    "val_data.pt": "e5c407f4fa7f3530e04101a7832cf66687df7224a0dfbe792efb8ce693373dce",
}
SIX_SITES = ("gated_0", "gated_1", "gated_2", "gated_3", "mlp_0", "mlp_1")
EXPECTED_TRAIN, BATCH = 45000, 128

AFFINE_DEPS = (
    "gated_affine_cal_v1.py", "check_gated_affine_cal_v1.py",
    "activation_histogram.py", "gated_teacher_model.py",
    "gated_frozen_gate_calibrate.py", "affine-base-fulltrain-protocol-v1.md",
)
SIX_DEPS = (
    "gated_frozen_gate_calibrate.py", "gated_frozen_gate_validate.py",
    "activation_histogram.py", "gated_teacher_model.py",
    "gated_frozen_gate_prototype.py", "gated_teacher_train.py",
    "gated_teacher_train_protocol_v1.md", "gated-frozen-gate-prototype-protocol.md",
    "gated-frozen-gate-calibration-validation-protocol.md",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def verify_frozen_inputs(run_root: Path, data_root: Path, source_root: Path) -> dict:
    rr = run_root / "clean-restore-v1" / "research" / "visual-graphs" / "takeover_20261112"
    paths = {
        "PROTOCOL-v3.md": STUDY_ROOT / "PROTOCOL-v3.md",
        "PROTOCOL-E1-head-addendum-v1.md": STUDY_ROOT / "PROTOCOL-E1-head-addendum-v1.md",
        "affine_cal1.json": rr / "affine_cal1.json",
        "gcal_v1.json": rr / "full_teacher_runs" / "gated_cifar10_seed41_full_v1" / "gcal_v1.json",
        "seed41_best.pt": rr / "full_teacher_runs" / "gated_cifar10_seed41_full_v1" / "best.pt",
    }
    out = {}
    for name, expected in FROZEN.items():
        p = paths[name]
        require(p.is_file(), f"frozen input missing: {p}")
        actual = sha256(p)
        require(actual == expected, f"frozen input hash mismatch: {name}")
        out[str(p)] = actual
    processed = data_root / "CIFAR10" / "processed"
    for name, expected in PROCESSED_EXPECTED.items():
        p = processed / name
        require(p.is_file(), f"processed data missing: {p}")
        require(sha256(p) == expected, f"processed data hash mismatch: {name}")
        out[str(p)] = expected
    return out


def verify_teacher(teacher_dir: Path, seed: int, source_root: Path, data_root: Path,
                   six_helper) -> dict:
    metadata = json.loads((teacher_dir / "metadata.json").read_text(encoding="utf-8"))
    done = json.loads((teacher_dir / "done.json").read_text(encoding="utf-8"))
    checkpoint = teacher_dir / "best.pt"
    require(metadata.get("dataset") == "CIFAR10", "teacher dataset is not CIFAR10")
    require(metadata.get("scope") == "all_official_pyg_train_and_val", "teacher scope mismatch")
    require(metadata.get("official_counts") == {"train": 45000, "val": 5000}, "official counts mismatch")
    require(int(metadata.get("seed", -1)) == seed, f"teacher seed is not {seed}")
    require(done.get("status") == "finished", "teacher run is not terminal")
    require(done.get("official_counts") == metadata["official_counts"], "done counts mismatch")
    ck_hash = sha256(checkpoint)
    require(done.get("best_checkpoint_sha256") == ck_hash, "checkpoint hash differs from done marker")
    processed = data_root / "CIFAR10" / "processed"
    current = {p.name: sha256(p) for p in sorted(processed.glob("*.pt")) if p.is_file()}
    require(current == metadata["provenance"]["processed_data_sha256"], "data differ from teacher metadata")
    source_hashes = {}
    for recorded, expected in metadata["provenance"]["source_sha256"].items():
        p = six_helper.resolve_teacher_source(source_root, recorded)
        require(sha256(p) == expected, f"teacher source hash mismatch: {recorded}")
        source_hashes[str(p)] = expected
    return {"metadata": metadata, "done": done, "checkpoint": checkpoint,
            "checkpoint_sha256": ck_hash, "source_hashes": source_hashes,
            "data_hashes": current}


def build_bound_inputs(teacher_dir: Path, source_root: Path, data_root: Path,
                       metadata: dict, seed41_hashes: dict) -> dict:
    bound: dict[str, str] = {}
    bound[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    for name in ("PROTOCOL-v3.md", "PROTOCOL-E1-head-addendum-v1.md"):
        p = STUDY_ROOT / name
        bound[str(p)] = sha256(p)
    for name in set(AFFINE_DEPS) | set(SIX_DEPS):
        p = source_root / name
        if p.is_file():
            bound[str(p)] = sha256(p)
    for name in ("metadata.json", "done.json", "best.pt"):
        p = teacher_dir / name
        bound[str(p)] = sha256(p)
    for p in sorted((data_root / "CIFAR10" / "processed").glob("*.pt")):
        bound[str(p)] = sha256(p)
    for key, digest in seed41_hashes.items():
        bound[key] = digest
    return bound


def validate_six(cal: dict) -> None:
    require(cal.get("passes") == 2, "six-site calibration is not two-pass")
    require(cal.get("pass1_graphs") == EXPECTED_TRAIN and cal.get("pass2_graphs") == EXPECTED_TRAIN,
            "six-site calibration did not cover 45000 train graphs twice")
    require(cal.get("bins") == 512 and abs(float(cal.get("quantile", 0)) - 0.999) < 1e-12,
            "six-site bins/quantile changed")
    require(set(cal.get("sites", {})) == set(SIX_SITES), "six-site site set changed")
    for site in SIX_SITES:
        vals = cal["sites"][site]["per_channel_thresholds"]
        require(bool(vals) and all(float(v) > 0 and torch.isfinite(torch.tensor(float(v))) for v in vals),
                f"invalid threshold at {site}")
        require(cal["channel_thresholds"][site] == vals, f"threshold alias mismatch at {site}")


def write_exclusive(path: Path, doc: dict) -> None:
    require(not path.exists(), f"refusing existing calibration output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gnn2snn-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(42, 43), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()

    source_root = args.gnn2snn_root.resolve() / "research" / "visual-graphs" / "takeover_20261112"
    require(source_root.is_dir(), f"source root missing: {source_root}")
    data_root = args.data_root.resolve()
    teacher_dir = args.teacher_dir.resolve()
    out_dir = args.output_dir.resolve()
    run_root = args.run_root.resolve()
    require(not out_dir.exists(), f"refusing existing output directory: {out_dir}")

    sys.path.insert(0, str(source_root))
    six_helper = importlib.import_module("gated_frozen_gate_calibrate")
    aff_helper = importlib.import_module("gated_affine_cal_v1")

    # Freeze helper identities in the report (algorithm must be the seed-41 code).
    helper_hashes = {
        "six_site_helper": sha256(Path(six_helper.__file__).resolve()),
        "affine_helper": sha256(Path(aff_helper.__file__).resolve()),
        "activation_histogram": sha256(source_root / "activation_histogram.py"),
        "teacher_model": sha256(source_root / "gated_teacher_model.py"),
    }
    require(helper_hashes["six_site_helper"] == "90cfd78e4c34fe1fc25cd022b9710f684a104413ccdba4732f386f330dda3fbe",
            "six-site helper changed from frozen E0/E1 evidence")
    require(helper_hashes["affine_helper"] == "0fb4f431b142dca0829e8ca57f2a12199f80af89a839863d5ad0001eb4ac1b39",
            "affine helper changed from frozen E0/E1 evidence")

    frozen_hashes = verify_frozen_inputs(run_root, data_root, source_root)
    teacher = verify_teacher(teacher_dir, args.seed, source_root, data_root, six_helper)
    bound = build_bound_inputs(teacher_dir, source_root, data_root, teacher["metadata"], frozen_hashes)
    bound_start = dict(bound)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable")
        torch.cuda.reset_peak_memory_stats(dev)

    state = torch.load(teacher["checkpoint"], map_location=dev, weights_only=False)
    require(state.get("metadata") == teacher["metadata"], "checkpoint metadata differs from run metadata")
    require(int(state.get("epoch", -1)) >= 1, "checkpoint lacks a valid teacher epoch")
    model = six_helper.build_teacher(state, dev).eval()
    require(len(model.layers) == 4, "expected frozen four-layer GatedGCN teacher")

    train = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="train")
    require(len(train) == EXPECTED_TRAIN, f"unexpected official train graph count: {len(train)}")
    loader = DataLoader(six_helper.LabelFreeGraphView(train), batch_size=BATCH,
                        shuffle=False, num_workers=0)

    t0 = time.perf_counter()
    print(json.dumps({"stage": "six_site_start", "seed": args.seed, "graphs": EXPECTED_TRAIN}), flush=True)
    six_cal = six_helper.calibrate_loader(model, loader, expected_graphs=EXPECTED_TRAIN,
                                          bins=512, quantile=0.999, device=dev)
    validate_six(six_cal)
    print(json.dumps({"stage": "six_site_done", "elapsed_sec": time.perf_counter() - t0}), flush=True)

    t1 = time.perf_counter()
    print(json.dumps({"stage": "affine_start", "seed": args.seed, "graphs": EXPECTED_TRAIN}), flush=True)
    aff_cal = aff_helper.calibrate_loader(model, loader, dev, expected=EXPECTED_TRAIN,
                                          width=70, batch_size=BATCH)
    aff_helper.validate_statistics(aff_cal)
    print(json.dumps({"stage": "affine_done", "elapsed_sec": time.perf_counter() - t1}), flush=True)

    # Re-hash every bound input; nothing may have changed during calibration.
    bound_end = {p: sha256(Path(p)) for p in bound_start}
    require(bound_start == bound_end, "bound input changed during calibration")

    runtime = {"python": sys.version, "torch": torch.__version__, "device": str(dev),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "batch_size": BATCH, "labels_passed": False,
               "validation_or_test_opened": False}
    teacher_info = {"run": str(teacher_dir), "checkpoint_sha256": teacher["checkpoint_sha256"],
                    "metadata": teacher["metadata"], "done": teacher["done"],
                    "epoch": int(state["epoch"])}

    aff_doc = {
        "status": "complete", "dataset": "CIFAR10", "seed": args.seed,
        "scope": "all_45000_official_train_graphs_label_free_two_pass",
        "teacher": teacher_info, "calibration": aff_cal,
        "input_hashes_start": bound_start, "input_hashes_end": bound_end,
        "elapsed_seconds": t1 - t0 + (time.perf_counter() - t1),
        "quantity_definition": {"raw": "abs teacher layer input",
                                "branch_cumsum": "sum earlier post-BN ReLU branches; not subtraction"},
        "algorithm_binding": {"driver": "calibrate_multiseed_v1.py",
                              "calibration_function": "gated_affine_cal_v1.calibrate_loader",
                              "helper_sha256": helper_hashes,
                              "statement": "same immutable function and frozen parameters as seed-41 affine_cal1.json; only teacher/seed/data root differ"},
        "runtime": runtime,
    }
    six_doc = {
        "status": "complete", "dataset": "CIFAR10",
        "scope": "all_45000_official_train_graphs_label_free_two_pass_fixed_order",
        "teacher_run": str(teacher_dir), "teacher_epoch": int(state["epoch"]),
        "teacher_metadata": teacher["metadata"], "teacher_done": teacher["done"],
        "checkpoint": {"path": str(teacher["checkpoint"]), "sha256": teacher["checkpoint_sha256"]},
        "checkpoint_sha256": teacher["checkpoint_sha256"],
        "seed": args.seed,
        "data": {"root": str(data_root), "train_graphs": len(train),
                 "processed_files_sha256": teacher["data_hashes"]},
        "code_sha256": helper_hashes,
        "calibration": {"full_train_graph_count": EXPECTED_TRAIN, **six_cal},
        "input_hashes_start": bound_start, "input_hashes_end": bound_end,
        "algorithm_binding": {"driver": "calibrate_multiseed_v1.py",
                              "calibration_function": "gated_frozen_gate_calibrate.calibrate_loader",
                              "statement": "same immutable function and frozen parameters as seed-41 gcal_v1.json; only teacher/seed/data root differ"},
        "runtime": runtime,
    }

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=False)
    aff_path = out_dir / f"affine_cal_seed{args.seed}_v1.json"
    six_path = out_dir / f"gcal_seed{args.seed}_v1.json"
    write_exclusive(aff_path, aff_doc)
    write_exclusive(six_path, six_doc)
    summary = {"status": "complete", "seed": args.seed, "graphs": EXPECTED_TRAIN,
               "affine_calibration": str(aff_path), "affine_sha256": sha256(aff_path),
               "six_site_calibration": str(six_path), "six_site_sha256": sha256(six_path),
               "checkpoint_sha256": teacher["checkpoint_sha256"],
               "affine_sites": len(aff_cal["sites"]), "six_sites": len(six_cal["sites"]),
               "elapsed_sec": time.perf_counter() - t0}
    with (out_dir / "calibration_summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
