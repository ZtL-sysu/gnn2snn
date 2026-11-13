"""Unified full-validation evaluator for the frozen affine-base profiles.

This uses the immutable project reference modes at threshold scale 1. Its
operation ledger contains observed calls/events only; it is not a compute,
latency, or energy benchmark.
"""
from __future__ import annotations

# Configure CUDA before importing Torch; conflicting caller configuration is an error.
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be unset or :4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

EXPECTED_CALIBRATION_SHA256 = "1618e3b46f75646ab051229b0d954815b5e28bf1729cbbf2042e4d596766a511"
EXPECTED_TRAIN, EXPECTED_VAL, BATCH_SIZE = 45000, 5000, 128
TIMESTEPS = (1, 2, 4, 8)
MODES = ("centered", "first_analog", "generic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LabelFreeDataset(Dataset):
    """Keep scoring labels out of every model/converter batch."""
    def __init__(self, source): self.source = source
    def __len__(self): return len(self.source)
    def __getitem__(self, index):
        item = self.source[index]
        return Data(**{key: getattr(item, key) for key in ("x", "edge_index", "edge_attr", "pos")
                       if getattr(item, key, None) is not None})


def remap_bound_input(recorded: str, source_root: Path, data_root: Path,
                      teacher_dir: Path) -> Path:
    """Map calibration's recorded absolute inputs to configured immutable inputs."""
    normalized = recorded.replace("/", "\\")
    parts = [p for p in normalized.split("\\") if p]
    lower = [p.lower() for p in parts]
    if "pyg_bench" in lower:
        i = lower.index("pyg_bench")
        candidate = data_root.joinpath(*parts[i + 1:])
    elif "full_teacher_runs" in lower:
        i = lower.index("full_teacher_runs")
        tail = parts[i + 2:]  # skip full_teacher_runs/<recorded run name>
        candidate = teacher_dir.joinpath(*tail)
    elif "public_baselines" in lower:
        i = lower.index("public_baselines")
        candidate = source_root.joinpath(*parts[i:])
    else:
        candidate = source_root / parts[-1]
    return candidate.resolve()


def metric_row(logits: np.ndarray, teacher: np.ndarray, labels: np.ndarray,
               kl_eps: float = 1e-12) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.ndim != 2 or teacher.shape != logits.shape or logits.shape[0] != labels.size:
        raise ValueError("logits/teacher/labels shape mismatch")
    if not np.isfinite(logits).all() or not np.isfinite(teacher).all():
        raise ValueError("non-finite logits")
    pred, teacher_pred = logits.argmax(1), teacher.argmax(1)
    p_logits = teacher - teacher.max(1, keepdims=True)
    q_logits = logits - logits.max(1, keepdims=True)
    p = np.exp(p_logits); p /= p.sum(1, keepdims=True)
    logp = np.log(np.maximum(p, kl_eps))
    logq = q_logits - np.log(np.exp(q_logits).sum(1, keepdims=True))
    return {"accuracy": float(np.mean(pred == labels)),
            "teacher_accuracy": float(np.mean(teacher_pred == labels)),
            "teacher_agreement": float(np.mean(pred == teacher_pred)),
            "logit_mse": float(np.mean((logits - teacher) ** 2)),
            "teacher_to_output_kl": float(np.mean(np.sum(p * (logp - logq), axis=1)))}


def verify_calibration(doc: dict, calibration_path: Path, teacher_dir: Path,
                       source_root: Path, data_root: Path, teacher_metadata: dict,
                       teacher_done: dict, checkpoint_hash: str, cal_helper):
    if sha256(calibration_path) != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError("affine_cal1.json hash mismatch")
    if (doc.get("status") != "complete" or doc.get("dataset") != "CIFAR10" or
            doc.get("seed") != 41 or doc.get("scope") != "all_45000_official_train_graphs_label_free_two_pass"):
        raise RuntimeError("wrong/incomplete affine calibration scope")
    teacher_doc = doc.get("teacher", {})
    if (teacher_doc.get("checkpoint_sha256") != checkpoint_hash or
            teacher_doc.get("metadata") != teacher_metadata or teacher_doc.get("done") != teacher_done):
        raise RuntimeError("calibration does not bind the selected seed-41 teacher")
    if int(teacher_doc.get("epoch", -1)) < 1:
        raise RuntimeError("calibration lacks a valid teacher epoch")
    if not doc.get("input_hashes_start") or doc.get("input_hashes_start") != doc.get("input_hashes_end"):
        raise RuntimeError("calibration input hash ledger is incomplete or changed")
    mapped = {}
    for original, expected in doc["input_hashes_start"].items():
        path = remap_bound_input(original, source_root, data_root, teacher_dir)
        if not path.is_file(): raise FileNotFoundError(f"Calibration-bound input is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Calibration-bound input hash mismatch: {path}")
        mapped[str(path)] = actual
    cal_helper.validate_statistics(doc["calibration"])
    if set(doc["calibration"]["channel_thresholds"]) != set(cal_helper.SITES):
        raise RuntimeError("Calibration threshold sites do not match helper")
    return mapped


def processed_hashes(data_root: Path) -> dict:
    paths = sorted((data_root / "CIFAR10" / "processed").glob("*.pt"))
    if not paths: raise FileNotFoundError(f"No processed CIFAR10 data under {data_root}")
    return {p.name: sha256(p) for p in paths}


@torch.no_grad()
def evaluate(args):
    source_root = args.gnn2snn_root.resolve() / "research" / "visual-graphs" / "takeover_20261112"
    if not source_root.is_dir(): raise FileNotFoundError(source_root)
    teacher_dir, data_root = args.teacher_dir.resolve(), args.data_root.resolve()
    calibration_path = args.calibration.resolve()
    out_dir, allowed_root = args.output_dir.resolve(), args.newrun_root.resolve()
    try: out_dir.relative_to(allowed_root)
    except ValueError as exc: raise ValueError("output directory must stay under --newrun-root") from exc
    if out_dir.exists(): raise FileExistsError(f"Refusing existing result directory: {out_dir}")

    sys.path.insert(0, str(source_root))
    cal_helper = importlib.import_module("gated_affine_cal_v1")
    affine_helper = importlib.import_module("gated_affine_base_v1")
    teacher_helper = importlib.import_module("gated_frozen_gate_calibrate")
    MTH_diff, MTH_diff_line, author_provenance = affine_helper.require_author_sources(args.author_repo)

    metadata_path, done_path, checkpoint = (teacher_dir / "metadata.json", teacher_dir / "done.json", teacher_dir / "best.pt")
    for path in (metadata_path, done_path, checkpoint, calibration_path):
        if not path.is_file(): raise FileNotFoundError(path)
    metadata, done = json.loads(metadata_path.read_text()), json.loads(done_path.read_text())
    checkpoint_hash = sha256(checkpoint)
    if (metadata.get("dataset") != "CIFAR10" or metadata.get("scope") != "all_official_pyg_train_and_val" or
            metadata.get("official_counts") != {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL} or
            int(metadata.get("seed", -1)) != 41):
        raise RuntimeError("Expected full CIFAR10 seed-41 teacher")
    if done.get("status") != "finished" or done.get("official_counts") != metadata["official_counts"]:
        raise RuntimeError("Teacher run is not complete")
    if done.get("best_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Teacher checkpoint hash differs from terminal marker")
    doc = json.loads(calibration_path.read_text(encoding="utf-8"))
    mapped_calibration_inputs = verify_calibration(
        doc, calibration_path, teacher_dir, source_root, data_root, metadata, done,
        checkpoint_hash, cal_helper)
    start_data = processed_hashes(data_root)
    teacher_data_hashes = metadata.get("provenance", {}).get("processed_data_sha256", {})
    if start_data != teacher_data_hashes:
        raise RuntimeError("Processed data differ from teacher metadata")
    # Include teacher source files as individually hashed inputs.
    teacher_source_paths = {}
    for recorded, expected in metadata.get("provenance", {}).get("source_sha256", {}).items():
        path = teacher_helper.resolve_teacher_source(source_root, recorded)
        if sha256(path) != expected:
            raise RuntimeError(f"Teacher source hash mismatch: {recorded}")
        teacher_source_paths[str(path.resolve())] = path.resolve()
    source_paths = {"evaluator": Path(__file__).resolve(),
                    "frozen_protocol":Path(__file__).resolve().parents[1]/'PROTOCOL-v3.md',
                    "head_matching_addendum":Path(__file__).resolve().parents[1]/'PROTOCOL-E1-head-addendum-v1.md',
                    "canonical_baseline":Path(__file__).resolve().parents[1]/'inputs/canonical-baseline-val.npz',
                    "affine_helper": Path(affine_helper.__file__).resolve(),
                    "calibration_helper": Path(cal_helper.__file__).resolve(),
                    "teacher_helper": Path(teacher_helper.__file__).resolve(),
                    "author_relu_source": args.author_repo.resolve() / author_provenance["source_files"]["relu_neuron"]["path"],
                    "author_line_source": args.author_repo.resolve() / author_provenance["source_files"]["line_neuron"]["path"]}
    source_paths.update({f"calibration_bound:{Path(path).name}": Path(path) for path in mapped_calibration_inputs
                         if Path(path).suffix.lower() in (".py", ".md", ".json")})
    source_paths.update({f"teacher_source:{Path(path).name}": path for path in teacher_source_paths.values()})
    source_start = {name: sha256(path) for name, path in source_paths.items()}
    checkpoint_start, calibration_start = checkpoint_hash, sha256(calibration_path)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    torch.manual_seed(41)
    np.random.seed(41)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("metadata") != metadata: raise RuntimeError("Checkpoint metadata differs from run metadata")
    model = teacher_helper.build_teacher(state, device).eval()
    dataset = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    if len(dataset) != EXPECTED_VAL: raise RuntimeError(f"Unexpected validation count: {len(dataset)}")
    labels_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model_loader = DataLoader(LabelFreeDataset(dataset), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    label_iter = iter(labels_loader)
    method_keys = [(mode, t) for mode in MODES for t in TIMESTEPS]
    prediction_parts = {key: [] for key in method_keys}
    events = {key: {} for key in method_keys}
    ledger = {key: {"source_warmup_calls": 0, "source_active_unit_calls": 0,
                    "analog_head_calls": 0, "graph_layer_evaluations": 0} for key in method_keys}
    label_parts, teacher_parts, indices = [], [], []
    graphs, start_time = 0, time.monotonic()
    resource_probe=None
    for batch_no, batch in enumerate(model_loader):
        label_batch = next(label_iter)
        n = int(batch.num_graphs)
        y = label_batch.y.view(-1).long().cpu().numpy()
        label_parts.append(y)
        indices.append(np.arange(graphs, graphs + n, dtype=np.int64))
        batch = batch.to(device)
        teacher_logits = model(batch)
        teacher_parts.append(teacher_logits.detach().cpu().numpy())
        for mode, timestep in method_keys:
            thresholds = cal_helper.profile_thresholds(doc, mode, 1)
            result = affine_helper.affine_mode_forward(model, batch, thresholds, timestep,
                                                       mode, MTH_diff, MTH_diff_line)
            prediction_parts[(mode, timestep)].append(result["logits"].detach().cpu().numpy())
            key = (mode, timestep)
            for name, stat in zip(result["site_names"], result["site_stats"]):
                events[key][name] = events[key].get(name, 0) + int(stat["nonzero_scalar_events"])
            ledger[key]["source_warmup_calls"] += int(result["source_warmup_calls"])
            ledger[key]["source_active_unit_calls"] += int(result["active_unit_calls"])
            ledger[key]["analog_head_calls"] += int(result["head_calls"])
            ledger[key]["graph_layer_evaluations"] += len(model.layers) * int(timestep)
        graphs += n
        if batch_no==0:
            if device.type=='cuda':torch.cuda.synchronize(device)
            resource_probe={'first_batch_graphs':n,'elapsed_sec':time.monotonic()-start_time,
                'peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None,
                'scope':'resource feasibility only; not a latency benchmark'}
            print(json.dumps({'resource_probe':resource_probe}),flush=True)
            if resource_probe['elapsed_sec']*40>80*3600:
                raise RuntimeError('Projected fixed validation grid exceeds declared E1 compute ceiling')
        if batch_no % 10 == 9 or graphs == EXPECTED_VAL:
            print(json.dumps({"graphs": graphs, "elapsed_sec": time.monotonic()-start_time}), flush=True)
    labels = np.concatenate(label_parts)
    teacher = np.concatenate(teacher_parts)
    indices_all = np.concatenate(indices)
    if graphs != EXPECTED_VAL or not np.array_equal(indices_all, np.arange(EXPECTED_VAL)):
        raise RuntimeError("Incomplete or noncanonical validation traversal")
    if float(np.mean(teacher.argmax(1) == labels)) != 0.7066:
        raise RuntimeError("Seed-41 teacher validation accuracy does not match its frozen reference")
    with np.load(source_paths['canonical_baseline'],allow_pickle=False) as reference:
        if not np.array_equal(indices_all,reference['indices']) or not np.array_equal(labels,reference['labels']):
            raise RuntimeError('Canonical teacher reference index/label mismatch')
        if not np.array_equal(teacher,reference['teacher_logits']):
            raise RuntimeError('Canonical teacher logits differ under fixed deterministic setup')
    predictions = {key: np.concatenate(parts) for key, parts in prediction_parts.items()}
    rows = []
    for mode, timestep in method_keys:
        row = {"mode": mode, "T": timestep, **metric_row(predictions[(mode, timestep)], teacher, labels),
               "per_site_nonzero_scalar_events": events[(mode, timestep)]}
        rows.append(row)

    source_end = {name: sha256(path) for name, path in source_paths.items()}
    data_end = processed_hashes(data_root)
    calibration_end, checkpoint_end = sha256(calibration_path), sha256(checkpoint)
    if source_start != source_end or start_data != data_end or calibration_start != calibration_end or checkpoint_start != checkpoint_end:
        raise RuntimeError("Source, calibration, model, or data changed during evaluation")
    if author_provenance != affine_helper.validate_source_unchanged(args.author_repo, author_provenance):
        raise RuntimeError("Pinned author sources changed during evaluation")

    report = {"status": "complete", "dataset": "CIFAR10", "split": "all official validation",
              "graphs": graphs, "seed": 41, "threshold_scale": 1,
              "historical_validation_exposure":True,"historical_test_exposure":True,
              "test_inference_performed":False,
              "modes": list(MODES), "timesteps": list(TIMESTEPS),
              "calibration_sha256_start": calibration_start, "calibration_sha256_end": calibration_end,
              "checkpoint_sha256_start": checkpoint_start, "checkpoint_sha256_end": checkpoint_end,
              "data_sha256_start": start_data, "data_sha256_end": data_end,
              "calibration_bound_inputs_sha256": mapped_calibration_inputs,
              "source_sha256_start": source_start, "source_sha256_end": source_end,
              "author_source_provenance": author_provenance,
              "runtime": {"python": sys.version, "torch": torch.__version__, "device": str(device),
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                          "cudnn_deterministic": torch.backends.cudnn.deterministic,
                          "cudnn_benchmark": torch.backends.cudnn.benchmark,
                          "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
              "elapsed_sec": time.monotonic()-start_time,
              "resource_probe":resource_probe,
              "teacher_accuracy": float(np.mean(teacher.argmax(1) == labels)),
              "rows": rows, "operation_ledger": {
                  "scope": "partial observed reference counters; dense arithmetic, memory traffic, hardware latency and energy remain unmeasured",
                  "per_method": {f"{mode}_T{t}": ledger[(mode, t)] for mode, t in method_keys},
                  "dense_mac_or_flop_count": None, "memory_traffic_bytes": None,
                  "hardware_latency_or_energy": None},
              "alias_and_scope": {
                  "generic": "the immutable helper's generic shared-input encoder mode (all four layers receive coded inputs); no separate per-consumer CSE-only implementation is present.",
                  "per_consumer_extension": "A/B/D/E/C require independent encoders at each consumer and a separately reviewed forward implementation; no costs or results are imputed here.",
                  "dc_comparator_fairness": "The historical DC adaptation codes four graph node-branch ReLUs plus two MLP hidden ReLUs, while these affine modes keep the same MLP head analog. It is not a matched-head comparator. A valid DC-adapt-analog-head control must code only the four gated sites and use this same analog MLP head; that adapter is outside this evaluator and must be added before cross-family accuracy/fidelity comparisons."},
              "interpretation": "Dense feasibility reference with the same analog GatedTeacher MLP readout; this is not an event-cache backend or cost-efficiency result."}

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=False)
    for mode, timestep in method_keys:
        filename = f"{mode}_T{timestep}.npz"
        with (out_dir / filename).open("xb") as stream:
            np.savez_compressed(stream, indices=indices_all, labels=labels, teacher_logits=teacher,
                                logits=predictions[(mode, timestep)])
        report_key = f"{mode}_T{timestep}"
        if "prediction_archives_sha256" not in report: report["prediction_archives_sha256"] = {}
        report["prediction_archives_sha256"][report_key] = sha256(out_dir / filename)
    with (out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "graphs": graphs, "teacher_accuracy": report["teacher_accuracy"],
                      "rows": rows}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn2snn-root", type=Path, default=Path(os.environ.get("GNN2SNN_ROOT", "")), required=not bool(os.environ.get("GNN2SNN_ROOT")))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GNN2SNN_DATA_ROOT", "")), required=not bool(os.environ.get("GNN2SNN_DATA_ROOT")))
    parser.add_argument("--teacher-dir", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--author-repo", type=Path, default=Path(os.environ.get("DCGS_REPO", "")), required=not bool(os.environ.get("DCGS_REPO")))
    parser.add_argument("--newrun-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate(parser.parse_args())


if __name__ == "__main__": main()
