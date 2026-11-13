"""CPU-only checks for gated_teacher_seed_train_v2; no official data or GPU."""
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import gated_teacher_seed_train_v2 as trainer
import gated_frozen_gate_calibrate as calibrator

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_cli_smoke():
    output_before = {seed: trainer.run_dir_for(seed).exists() for seed in trainer.TEACHER_SEEDS}
    proc = subprocess.run([sys.executable, str(HERE / "gated_teacher_seed_train_v2.py"),
                           "--seed", "42", "--smoke"], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    require(payload["status"] == "synthetic_smoke_passed", "synthetic CLI did not complete")
    require(payload["scope"] == "synthetic_cpu_only" and payload["dataset_loaded"] is False,
            "smoke payload does not certify synthetic-only data scope")
    require(payload["device"] == "cpu" and payload["seed"] == 42,
            "smoke did not run on CPU with requested teacher seed")
    require(payload["student_seed"] == 41, "student-seed reference changed")
    original_dataset = trainer.GNNBenchmarkDataset
    def reject_official_dataset(*_args, **_kwargs):
        raise AssertionError("synthetic smoke attempted official dataset access")
    trainer.GNNBenchmarkDataset = reject_official_dataset
    try:
        in_process = trainer.synthetic_smoke(43)
    finally:
        trainer.GNNBenchmarkDataset = original_dataset
    require(in_process["dataset_loaded"] is False, "in-process smoke reported dataset access")
    output_after = {seed: trainer.run_dir_for(seed).exists() for seed in trainer.TEACHER_SEEDS}
    require(output_after == output_before, "smoke changed a full-run output directory")
    trainer_source = (HERE / "gated_teacher_seed_train_v2.py").read_text(encoding="utf-8-sig")
    smoke_source = ast.parse(trainer_source)
    synthetic_node = next(n for n in smoke_source.body if isinstance(n, ast.FunctionDef)
                          and n.name == "synthetic_smoke")
    synthetic_text = ast.get_source_segment(trainer_source, synthetic_node)
    require("GNNBenchmarkDataset" not in synthetic_text and "DATA_ROOT" not in synthetic_text,
            "synthetic smoke function references official dataset access")
    return payload


def check_initialization():
    same_a = trainer.initial_state_digest(42)
    same_b = trainer.initial_state_digest(42)
    different = trainer.initial_state_digest(43)
    require(same_a == same_b, "same-seed initial model state is not deterministic")
    require(same_a != different, "different teacher seeds produced identical initial state")
    return {"seed42_sha256": same_a, "seed43_sha256": different}


def check_manifest_and_protocol():
    manifest = trainer.source_manifest()
    expected = set(trainer.SOURCE_PATHS)
    require(set(manifest["files"]) == expected, "source manifest file set mismatch")
    for relative, digest in manifest["files"].items():
        require(digest == source_hash(HERE / relative), "source hash mismatch: " + relative)
    require(set(manifest["files"]) >= {
        "gated_teacher_seed_train_v2.py", "gated-teacher-seed-protocol-v2.md",
        "gated_teacher_train.py", "gated_teacher_model.py",
        "gated_teacher_train_protocol_v1.md", "public_baselines/gated_net.py",
        "public_baselines/gated_layer.py", "public_baselines/benchmark_mlp_readout.py",
        "public_baselines/gated_cifar.json",
    }, "manifest omits required v1/public provenance files")
    # Exercise the unchanged calibration validator against a synthetic completed
    # teacher record so relative-path resolution and the legacy alias are tested.
    with tempfile.TemporaryDirectory(prefix=".gs-", dir=HERE) as temp:
        fixture = Path(temp)
        checkpoint = fixture / "best.pt"
        checkpoint.write_bytes(b"synthetic-checkpoint-only")
        metadata = {"dataset": "CIFAR10", "scope": "all_official_pyg_train_and_val",
                    "official_counts": {"train": 45000, "val": 5000},
                    "provenance": {"source_sha256": manifest["files"]}}
        done = {"status": "finished", "official_counts": metadata["official_counts"],
                "best_checkpoint_sha256": source_hash(checkpoint)}
        (fixture / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (fixture / "done.json").write_text(json.dumps(done), encoding="utf-8")
        calibrator.validate_teacher_run(fixture)
    require(trainer.EPOCH_CAP == 200 and trainer.TIME_CAP_HOURS == 12.0,
            "full-training caps drifted from protocol")
    require(trainer.BATCH_SIZE == 128 and trainer.LEARNING_RATE == 0.001,
            "optimizer/batch settings drifted from protocol")
    tree = ast.parse((HERE / "gated_teacher_seed_train_v2.py").read_text(encoding="utf-8-sig"))
    full_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_full")
    loaded_splits = []
    for node in ast.walk(full_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GNNBenchmarkDataset":
            loaded_splits.extend(k.value.value for k in node.keywords
                                 if k.arg == "split" and isinstance(k.value, ast.Constant))
    require(set(loaded_splits) == {"train", "val"}, "full trainer must open only train and validation splits")
    require(trainer.run_dir_for(42).name == "gated_cifar10_seed42_full_v2"
            and trainer.run_dir_for(43).name == "gated_cifar10_seed43_full_v2",
            "seed-specific output paths mismatch")
    with tempfile.TemporaryDirectory(prefix=".gs-", dir=HERE) as temp:
        existing = Path(temp) / "occupied"
        existing.mkdir()
        try:
            trainer.refuse_existing_output(existing)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output directory was not refused")
    return manifest


def main():
    smoke = check_cli_smoke()
    init = check_initialization()
    manifest = check_manifest_and_protocol()
    print(json.dumps({"status": "checks_passed", "smoke_status": smoke["status"],
                      "initialization": init, "source_manifest_sha256": manifest["manifest_sha256"],
                      "official_dataset_loaded": False, "full_training_run": False}, sort_keys=True))


if __name__ == "__main__":
    main()
