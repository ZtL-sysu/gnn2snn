"""Full official-validation evaluation for calibrated frozen-gate GatedTeacher."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from gated_frozen_gate_calibrate import (
    DATA_ROOT, EXPECTED_VAL, OUT_ROOT, RUN_NAME, SITE_NAMES, build_teacher,
    processed_file_hashes, sha256, validate_teacher_run,
)
from gated_frozen_gate_prototype import extract_teacher_gates, frozen_gate_forward

TIMESTEPS = (8, 16, 32)


def validate_calibration(calibration_path, teacher_run):
    calibration_path = Path(calibration_path)
    if not calibration_path.is_file():
        raise RuntimeError(f"Missing complete threshold calibration: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (calibration.get("status") != "complete" or calibration.get("dataset") != "CIFAR10"
            or calibration.get("scope") != "all_45000_official_train_graphs_label_free_two_pass_fixed_order"):
        raise RuntimeError("Calibration is incomplete or not the frozen all-train CIFAR10 calibration")
    run, metadata, _done, checkpoint, checkpoint_hash = validate_teacher_run(teacher_run)
    if calibration.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Calibration teacher checkpoint hash mismatch")
    if Path(calibration["teacher_run"]).resolve() != run.resolve():
        raise RuntimeError("Calibration points to a different teacher run")
    for filename, expected in calibration.get("code_sha256", {}).items():
        path = Path(__file__).resolve().parent / filename
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Calibrated source hash mismatch: {filename}")
    recorded_train_count = calibration.get("calibration", {}).get("pass2_graphs")
    if recorded_train_count != 45000:
        raise RuntimeError(f"Calibration did not cover all 45,000 train graphs: {recorded_train_count}")
    if set(calibration.get("calibration", {}).get("channel_thresholds", {})) != set(SITE_NAMES):
        raise RuntimeError("Calibration does not contain all six channel-threshold sites")
    current_data_hashes = processed_file_hashes(DATA_ROOT / "CIFAR10" / "processed")
    recorded_data_hashes = calibration.get("data", {}).get("processed_files_sha256", {})
    teacher_data_hashes = metadata.get("provenance", {}).get("processed_data_sha256", {})
    if not recorded_data_hashes or current_data_hashes != recorded_data_hashes or current_data_hashes != teacher_data_hashes:
        raise RuntimeError("Processed CIFAR10 data hashes differ from teacher/calibration records")
    return calibration, checkpoint, checkpoint_hash


def _metric_row(name, logits, teacher_logits, labels, spikes=None, T=None):
    prediction = logits.argmax(axis=1)
    teacher_prediction = teacher_logits.argmax(axis=1)
    return {
        "method": name, "T_per_site": T,
        "accuracy": float(np.mean(prediction == labels)),
        "teacher_accuracy": float(np.mean(teacher_prediction == labels)),
        "teacher_agreement": float(np.mean(prediction == teacher_prediction)),
        "logit_mse_vs_teacher": float(np.mean((logits - teacher_logits) ** 2)),
        "logit_mae_vs_teacher": float(np.mean(np.abs(logits - teacher_logits))),
        "total_spikes": None if spikes is None else int(sum(spikes)),
        "site_spikes": None if spikes is None else [int(value) for value in spikes],
    }


@torch.no_grad()
def evaluate_loader(model, loader, calibration, device, expected_graphs=EXPECTED_VAL):
    theta = [calibration["calibration"]["channel_thresholds"][name] for name in SITE_NAMES]
    teacher_parts, oracle_parts, labels_parts = [], [], []
    staged_parts = {t: [] for t in TIMESTEPS}
    spike_totals = {t: [0] * len(SITE_NAMES) for t in TIMESTEPS}
    exact_max_abs_error = 0.0
    graphs = 0
    for batch in loader:
        batch = batch.to(device)
        teacher_logits = model(batch)
        controller = extract_teacher_gates(model, batch)
        oracle_logits, _ = frozen_gate_forward(model, batch, controller)
        err = float((oracle_logits - teacher_logits).abs().max().item())
        exact_max_abs_error = max(exact_max_abs_error, err)
        if not torch.allclose(oracle_logits, teacher_logits, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"Exact oracle replay mismatch: max abs error {err}")
        teacher_parts.append(teacher_logits.cpu().numpy())
        oracle_parts.append(oracle_logits.cpu().numpy())
        # The validation labels are read only here, after model inputs are used.
        labels_parts.append(batch.y.view(-1).long().cpu().numpy())
        graphs += int(batch.num_graphs)
        for t in TIMESTEPS:
            logits, diagnostics = frozen_gate_forward(model, batch, controller, theta, T=t)
            staged_parts[t].append(logits.cpu().numpy())
            for site, record in enumerate(diagnostics["sites"]):
                spike_totals[t][site] += record["spikes"]
    if graphs != expected_graphs:
        raise RuntimeError(f"Validation graph count mismatch: {graphs} != {expected_graphs}")
    teacher = np.concatenate(teacher_parts)
    oracle = np.concatenate(oracle_parts)
    labels = np.concatenate(labels_parts)
    staged = {t: np.concatenate(parts) for t, parts in staged_parts.items()}
    return {"teacher_logits": teacher, "exact_oracle_logits": oracle,
            "staged_logits": staged, "labels": labels, "graphs": graphs,
            "exact_oracle_max_abs_logit_error": exact_max_abs_error,
            "site_spikes": spike_totals}


def run_validation(teacher_run, calibration_path, summary_path, predictions_path,
                   batch_size=128, device="cpu"):
    summary_path, predictions_path = Path(summary_path), Path(predictions_path)
    if summary_path.resolve() == predictions_path.resolve():
        raise ValueError("Summary and predictions must use distinct output paths")
    if summary_path.exists() or predictions_path.exists():
        raise FileExistsError("Refusing to overwrite validation outputs")
    calibration, checkpoint, checkpoint_hash = validate_calibration(calibration_path, teacher_run)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if dev.type == "cpu":
        torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location=dev, weights_only=False)
    model = build_teacher(state, dev).eval()
    validation = GNNBenchmarkDataset(root=str(DATA_ROOT), name="CIFAR10", split="val")
    if len(validation) != EXPECTED_VAL:
        raise RuntimeError(f"Unexpected official validation count: {len(validation)}")
    loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    start = time.monotonic()
    values = evaluate_loader(model, loader, calibration, dev)
    teacher = values["teacher_logits"]
    oracle = values["exact_oracle_logits"]
    labels = values["labels"]
    staged = values["staged_logits"]
    rows = [_metric_row("exact_teacher_conditioned_oracle", oracle, teacher, labels)]
    for t in TIMESTEPS:
        rows.append(_metric_row(f"staged_frozen_gate_T{t}", staged[t], teacher, labels,
                                values["site_spikes"][t], T=t))

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("xb") as stream:
        np.savez_compressed(
            stream, indices=np.arange(EXPECTED_VAL), labels=labels,
            teacher_logits=teacher, exact_oracle_logits=oracle,
            **{f"staged_T{t}_logits": staged[t] for t in TIMESTEPS},
        )
    result = {
        "status": "complete", "dataset": "CIFAR10", "split": "all_official_validation_graphs",
        "scope": "full_validation_exposed_to_teacher_checkpoint_selection",
        "graphs": values["graphs"], "teacher_epoch": int(state["epoch"]),
        "teacher_accuracy": float(np.mean(teacher.argmax(axis=1) == labels)),
        "checkpoint_sha256": checkpoint_hash,
        "calibration_sha256": sha256(calibration_path),
        "script_sha256": sha256(__file__),
        "prototype_sha256": sha256(Path(__file__).resolve().parent / "gated_frozen_gate_prototype.py"),
        "predictions_path": str(predictions_path.resolve()),
        "predictions_sha256": sha256(predictions_path),
        "threshold_policy": "per_channel_two_pass_full_train_99.9th_percentile_512_bins",
        "timesteps": list(TIMESTEPS),
        "exact_oracle_max_abs_logit_error": values["exact_oracle_max_abs_logit_error"],
        "interpretation": {
            "controller": "each input requires a complete ANN gated-convolution forward to generate gates; this is a teacher-oracle hybrid",
            "timing": "six staged activation windows are serialized numerical stages; T per site is not end-to-end T latency",
            "energy": "dense ANN gate computation and graph operations are not an event-driven or energy measurement",
            "validation": "all validation labels are used only for final metrics; this split was used for checkpoint selection and is not independent test evidence",
        },
        "device": str(dev), "elapsed_sec": time.monotonic() - start, "rows": rows,
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, default=OUT_ROOT / RUN_NAME)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    result = run_validation(args.teacher_run, args.calibration, args.summary,
                            args.predictions, args.batch_size, args.device)
    print(json.dumps({"status": result["status"], "graphs": result["graphs"],
                      "summary": str(args.summary), "predictions": str(args.predictions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
