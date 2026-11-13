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
    output = args.output.resolve()
    output.relative_to(HERE)
    archive = output.with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError("refuse existing output/archive")
    unit, commit, author_provenance = require_author_source(args.author_repo)
    run = OUT_ROOT / RUN_NAME
    calibration_path = run / "gcal_v1.json"
    calibration, checkpoint, checkpoint_hash = validate_calibration(calibration_path, run)
    if Path(calibration["teacher_run"]).resolve() != run.resolve():
        raise RuntimeError("calibration teacher mismatch")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["metadata"] != calibration["teacher_metadata"]:
        raise RuntimeError("teacher metadata mismatch")
    model = build_teacher(state, device).eval()
    thresholds = [calibration["calibration"]["channel_thresholds"][name] for name in SITE_NAMES]
    dataset = GNNBenchmarkDataset(root=str(DATA_ROOT), name="CIFAR10", split="val")
    if len(dataset) != 5000:
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
        if graphs % 1280 == 0 or graphs == 5000:
            print(json.dumps({"graphs": graphs, "elapsed_sec": time.monotonic() - started}), flush=True)
    y = np.concatenate(labels)
    teacher = np.concatenate(teacher_parts)
    if graphs != 5000 or len(y) != 5000:
        raise RuntimeError("incomplete validation")
    arrays = {key: np.concatenate(parts) for key, parts in predictions.items()}
    rows = []
    for key, logits in arrays.items():
        if logits.shape != (5000, 10) or not np.isfinite(logits).all():
            raise RuntimeError(f"invalid logits: {key}")
        rows.append({"method": key, "accuracy": float(np.mean(logits.argmax(1) == y)),
                     "teacher_agreement": float(np.mean(logits.argmax(1) == teacher.argmax(1))),
                     "logit_mse": float(np.mean((logits - teacher) ** 2)),
                     "nonzero_scalar_events": events[key],
                     "event_type": "signed graded eight levels" if key.startswith("dc") else "binary positive IF"})
    source_names = ("gated_diff_fullval_v1.py", "gated_differential_hybrid_v1.py",
                    "gated_causal_control_v2.py", "gated_causal_hybrid.py",
                    "gated_teacher_model.py", "dcgs_fixed_residual_gcn_event_forward_v1.py",
                    "gated-differential-hybrid-design-v1.md")
    report = {"status": "complete", "dataset": "CIFAR10", "split": "all official validation",
              "graphs": graphs, "prior_validation_exposure": True, "test_loaded": False,
              "teacher_accuracy": float(np.mean(teacher.argmax(1) == y)),
              "checkpoint_sha256": checkpoint_hash, "calibration_sha256": sha256(calibration_path),
              "teacher_epoch": int(state["epoch"]), "source_commit": commit,
              "author_source_provenance": author_provenance,
              "source_sha256": {name: sha256(HERE / name) for name in source_names},
              "processed_data_sha256": calibration["data"]["processed_files_sha256"],
              "runtime": {"torch": torch.__version__, "python": sys.version, "device": str(device)},
              "elapsed_sec": time.monotonic() - started, "rows": rows,
              "scope": "project nonlinear-current differential encoding; analog edges and dense affine computations each tick; source signed graded node units; no teacher hidden-state oracle",
              "time": "same-tick sequential traversal of six activation sites; six source zero latch calls plus six*T active DC calls; no hardware-latency or energy claim"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as stream:
        np.savez_compressed(stream, indices=np.arange(5000), labels=y, teacher_logits=teacher,
                            report_json=np.array(json.dumps(report)), **arrays)
    report["predictions_sha256"] = sha256(archive)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"teacher_accuracy": report["teacher_accuracy"], "rows": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("positive batch size required")
    evaluate(args)
