"""Predeclared complete validation: DCGS hybrid versus causal IF controls."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from gated_frozen_gate_calibrate import DATA_ROOT, OUT_ROOT, RUN_NAME, SITE_NAMES, build_teacher, sha256
from gated_frozen_gate_validate import validate_calibration
from dcgs_fixed_residual_gcn_event_forward_v1 import require_author_source
from gated_differential_hybrid_v1 import differential_hybrid_forward
from gated_causal_control_v2 import causal_hybrid_forward

HERE = Path(__file__).resolve().parent
T_VALUES = (4, 8)


@torch.no_grad()
def evaluate(args):
    expected = {'val': 5000, 'test': 10000}[args.split]
    output = args.output.resolve()
    output.relative_to(HERE)
    archive = output.with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError("refuse existing output/archive")
    unit, commit, author_provenance = require_author_source(args.author_repo)
    run = args.teacher_run.resolve()
    run.relative_to(OUT_ROOT.resolve())
    calibration_path = args.calibration.resolve()
    calibration_path.relative_to(HERE)
    calibration, checkpoint, checkpoint_hash = validate_calibration(calibration_path, run)
    if Path(calibration["teacher_run"]).resolve() != run.resolve():
        raise RuntimeError("calibration teacher mismatch")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["metadata"] != calibration["teacher_metadata"] or int(state["metadata"].get("seed", -1)) != args.teacher_seed:
        raise RuntimeError("teacher metadata mismatch")
    model = build_teacher(state, device).eval()
    thresholds = [calibration["calibration"]["channel_thresholds"][name] for name in SITE_NAMES]
    dataset = GNNBenchmarkDataset(root=str(DATA_ROOT), name="CIFAR10", split=args.split)
    if len(dataset) != expected:
        raise RuntimeError("full official validation count mismatch")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels, teacher_parts = [], []
    keys = [f"dc_T{T}" for T in T_VALUES]
    keys += [f"{init}_{readout}_T{T}" for T in T_VALUES
             for init in ("zero", "half") for readout in ("mean", "terminal")]
    predictions = {key: [] for key in keys}
    events = {key: 0 for key in keys}
    graphs = 0
    started = time.monotonic()
    for batch in loader:
        labels.append(batch.y.view(-1).long().numpy())
        del batch.y
        batch = batch.to(device)
        teacher_parts.append(model(batch).cpu().numpy())
        for T in T_VALUES:
            logits, diag = differential_hybrid_forward(model, batch, thresholds, T, unit)
            predictions[f"dc_T{T}"].append(logits.cpu().numpy())
            events[f"dc_T{T}"] += sum(site["nonzero_scalar_events"] for site in diag["sites"])
            for name, fraction in (("zero", 0.), ("half", .5)):
                mean_logits, control = causal_hybrid_forward(model, batch, thresholds, T, init_fraction=fraction)
                for readout, values in (("mean", mean_logits), ("terminal", control["tick_logits"][-1])):
                    key = f"{name}_{readout}_T{T}"
                    predictions[key].append(values.cpu().numpy())
                    events[key] += control["total_spikes"]
        graphs += int(batch.num_graphs)
        if graphs % 1280 == 0 or graphs == expected:
            print(json.dumps({"graphs": graphs, "elapsed_sec": time.monotonic() - started}), flush=True)
    y = np.concatenate(labels)
    teacher = np.concatenate(teacher_parts)
    if graphs != expected or len(y) != expected:
        raise RuntimeError("incomplete validation")
    teacher_test_reference = None
    if args.split == "test":
        teacher_test_reference = run / "gtest_v1_pred.npz"
        with np.load(teacher_test_reference, allow_pickle=False) as reference:
            if not np.array_equal(reference["index"], np.arange(expected)):
                raise RuntimeError("teacher test archive index mismatch")
            if not np.array_equal(reference["labels"], y):
                raise RuntimeError("teacher test archive labels mismatch")
            if not np.array_equal(reference["logits"].argmax(1), teacher.argmax(1)):
                raise RuntimeError("teacher test archive predictions mismatch")
    arrays = {key: np.concatenate(parts) for key, parts in predictions.items()}
    rows = []
    for key, logits in arrays.items():
        if logits.shape != (expected, 10) or not np.isfinite(logits).all():
            raise RuntimeError(f"invalid logits: {key}")
        rows.append({"method": key, "accuracy": float(np.mean(logits.argmax(1) == y)),
                     "teacher_agreement": float(np.mean(logits.argmax(1) == teacher.argmax(1))),
                     "logit_mse": float(np.mean((logits - teacher) ** 2)),
                     "nonzero_scalar_events": events[key],
                     "event_type": "signed graded eight levels" if key.startswith("dc") else "binary positive IF"})
    source_names = ("gated_diff_seed_eval_v3.py", "gated_diff_eval_v2.py", "gated_diff_fullval_v1.py", "gated_differential_hybrid_v1.py",
                    "gated_causal_control_v2.py", "gated_causal_hybrid.py",
                    "gated_teacher_model.py", "dcgs_fixed_residual_gcn_event_forward_v1.py",
                    "gated-differential-hybrid-design-v1.md")
    report = {"status": "complete", "dataset": "CIFAR10", "split": "all official " + args.split,
              "graphs": graphs, "prior_validation_exposure": True, "test_loaded": args.split == "test",
              "prior_test_exposure": True,
              "teacher_test_archive_sha256": sha256(teacher_test_reference) if teacher_test_reference else None,
              "teacher_accuracy": float(np.mean(teacher.argmax(1) == y)),
              "checkpoint_sha256": checkpoint_hash, "calibration_sha256": sha256(calibration_path),
              "teacher_epoch": int(state["epoch"]), "teacher_seed": args.teacher_seed, "source_commit": commit,
              "author_source_provenance": author_provenance,
              "source_sha256": {name: sha256(HERE / name) for name in source_names},
              "processed_data_sha256": calibration["data"]["processed_files_sha256"],
              "runtime": {"torch": torch.__version__, "python": sys.version, "device": str(device)},
              "elapsed_sec": time.monotonic() - started, "rows": rows,
              "scope": "project nonlinear-current differential encoding; analog edges and dense affine computations each tick; source signed graded node units; no teacher hidden-state oracle",
              "time": "same-tick sequential traversal of six activation sites; six source zero latch calls plus six*T active DC calls; no hardware-latency or energy claim"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as stream:
        np.savez_compressed(stream, indices=np.arange(expected), labels=y, teacher_logits=teacher,
                            report_json=np.array(json.dumps(report)), **arrays)
    report["predictions_sha256"] = sha256(archive)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"teacher_accuracy": report["teacher_accuracy"], "rows": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val',), default='val')
    parser.add_argument('--teacher-run', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--teacher-seed', required=True, type=int, choices=(42,43))
    parser.add_argument("--author-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("positive batch size required")
    evaluate(args)
