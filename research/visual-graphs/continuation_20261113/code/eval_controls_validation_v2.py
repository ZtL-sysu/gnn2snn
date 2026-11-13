"""Official full-5000 validation for the remaining E1 matched-control rows.

Frozen inputs: PROTOCOL-v3.md, PROTOCOL-E1-head-addendum-v1.md, affine_cal1.json
(seed41 full-train label-free) and gcal_v1.json (six-site seed41 calibration).

Rows produced here: dc_adapt_analog_head (addendum), per_consumer and cse_only
(explicit generic alias). The historical six-site DC-adapt row and the three
affine rows keep their own evaluators and existing immutable evidence.

Seed 41 only. Scale 1 only. T in {1,2,4,8}. No test split, no masks, no tuning.
This evaluator emits dense reference measurements with partial observed
counters; it does NOT produce complete operation cost, matched wall-clock
latency or hardware energy evidence.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# matched_controls imports the frozen teacher helpers by module name, so the
# takeover source directory must be importable before that library is loaded.
_IMPORT_ROOT = Path(os.environ.get("GNN2SNN_ROOT", CODE_DIR.parents[3])).resolve()
_IMPORT_SOURCE_ROOT = _IMPORT_ROOT / "research" / "visual-graphs" / "takeover_20261112"
if str(_IMPORT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_SOURCE_ROOT))

import eval_affine_validation as base  # noqa: E402
import matched_controls as controls  # noqa: E402

MODES = ("dc_adapt_analog_head", "per_consumer", "cse_only")
TIMESTEPS = base.TIMESTEPS
EXPECTED_VAL = base.EXPECTED_VAL
BATCH_SIZE = base.BATCH_SIZE
EXPECTED_SIX_SITE_SHA256 = "b28485ddfa8ef259ca048f3272662fedc1e4f519b0324cfa514ccf82b2cf8a9e"
SIX_SITES = ("gated_0", "gated_1", "gated_2", "gated_3", "mlp_0", "mlp_1")


def verify_six_site(path: Path, teacher_dir: Path) -> dict:
    """Bind the six-site calibration to the same frozen seed-41 teacher."""
    if base.sha256(path) != EXPECTED_SIX_SITE_SHA256:
        raise RuntimeError("six-site calibration hash differs from E0 reference")
    evaluator = importlib.import_module("eval_teacher")
    metadata, done, doc, checkpoint = evaluator.validate_inputs(teacher_dir, path)
    if set(doc["calibration"]["channel_thresholds"]) != set(SIX_SITES):
        raise RuntimeError("six-site calibration sites differ from the frozen order")
    if doc["calibration"]["full_train_graph_count"] != base.EXPECTED_TRAIN:
        raise RuntimeError("six-site calibration did not use the full official train split")
    for site in SIX_SITES:
        values = doc["calibration"]["channel_thresholds"][site]
        if not values or not all(float(v) > 0 and np.isfinite(float(v)) for v in values):
            raise RuntimeError(f"non-positive or non-finite threshold at {site}")
    return {"doc": doc, "metadata": metadata, "done": done, "checkpoint": checkpoint}


def control_thresholds(mode: str, affine_doc: dict, six_site_doc: dict,
                       cal_helper, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    """Frozen, non-searchable threshold mapping. Never tuned against results."""
    if mode == "dc_adapt_analog_head":
        ct = six_site_doc["calibration"]["channel_thresholds"]
        vectors = [torch.as_tensor(ct[f"gated_{i}"], dtype=dtype, device=device) for i in range(4)]
        if any(v.numel() != 70 for v in vectors):
            raise RuntimeError("four gated node thresholds must match hidden width 70")
        return vectors
    if mode in ("per_consumer", "cse_only"):
        vectors = [t.to(device=device, dtype=dtype)
                   for t in cal_helper.profile_thresholds(affine_doc, "generic", 1)]
        if len(vectors) != 8:
            raise RuntimeError("generic raw coordinates require eight frozen threshold vectors")
        return vectors
    raise ValueError(f"unsupported control mode: {mode}")


@torch.no_grad()
def evaluate(args) -> None:
    source_root = args.gnn2snn_root.resolve() / "research" / "visual-graphs" / "takeover_20261112"
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    teacher_dir = args.teacher_dir.resolve()
    data_root = args.data_root.resolve()
    affine_cal = args.calibration.resolve()
    six_cal = args.six_site_calibration.resolve()
    out_dir = args.output_dir.resolve()
    allowed_root = args.newrun_root.resolve()
    try:
        out_dir.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("output directory must stay under --newrun-root") from exc
    if out_dir.exists():
        raise FileExistsError(f"Refusing existing result directory: {out_dir}")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    cal_helper = importlib.import_module("gated_affine_cal_v1")
    affine_helper = importlib.import_module("gated_affine_base_v1")
    teacher_helper = importlib.import_module("gated_frozen_gate_calibrate")
    MTH_diff, MTH_diff_line, author_provenance = affine_helper.require_author_sources(args.author_repo)

    for path in (teacher_dir / "metadata.json", teacher_dir / "done.json",
                 teacher_dir / "best.pt", affine_cal, six_cal):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads((teacher_dir / "metadata.json").read_text())
    done = json.loads((teacher_dir / "done.json").read_text())
    checkpoint = teacher_dir / "best.pt"
    checkpoint_hash = base.sha256(checkpoint)
    if (metadata.get("dataset") != "CIFAR10" or
            metadata.get("scope") != "all_official_pyg_train_and_val" or
            metadata.get("official_counts") != {"train": base.EXPECTED_TRAIN, "val": EXPECTED_VAL} or
            int(metadata.get("seed", -1)) != 41):
        raise RuntimeError("Expected full official CIFAR10 seed-41 teacher")
    if done.get("status") != "finished" or done.get("official_counts") != metadata["official_counts"]:
        raise RuntimeError("Teacher run is not complete")
    if done.get("best_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Teacher checkpoint hash differs from terminal marker")

    affine_doc = json.loads(affine_cal.read_text(encoding="utf-8"))
    mapped_affine_inputs = base.verify_calibration(
        affine_doc, affine_cal, teacher_dir, source_root, data_root,
        metadata, done, checkpoint_hash, cal_helper)
    six = verify_six_site(six_cal, teacher_dir)
    if six["metadata"] != metadata or six["done"] != done:
        raise RuntimeError("six-site calibration binds a different teacher")
    if six["doc"].get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("six-site calibration checkpoint hash mismatch")

    start_data = base.processed_hashes(data_root)
    if start_data != metadata["provenance"]["processed_data_sha256"]:
        raise RuntimeError("Processed data differ from teacher metadata")
    teacher_source_paths = {}
    for recorded, expected in metadata["provenance"]["source_sha256"].items():
        path = teacher_helper.resolve_teacher_source(source_root, recorded)
        if base.sha256(path) != expected:
            raise RuntimeError(f"Teacher source hash mismatch: {recorded}")
        teacher_source_paths[str(path.resolve())] = path.resolve()

    study_root = CODE_DIR.parent
    source_paths = {
        "evaluator": Path(__file__).resolve(),
        "base_evaluator": Path(base.__file__).resolve(),
        "control_library": Path(controls.__file__).resolve(),
        "frozen_protocol": study_root / "PROTOCOL-v3.md",
        "head_matching_addendum": study_root / "PROTOCOL-E1-head-addendum-v1.md",
        "canonical_baseline": study_root / "inputs" / "canonical-baseline-val.npz",
        "affine_calibration": affine_cal,
        "six_site_calibration": six_cal,
        "affine_helper": Path(affine_helper.__file__).resolve(),
        "calibration_helper": Path(cal_helper.__file__).resolve(),
        "teacher_helper": Path(teacher_helper.__file__).resolve(),
        "author_relu_source": args.author_repo.resolve() / author_provenance["source_files"]["relu_neuron"]["path"],
        "author_line_source": args.author_repo.resolve() / author_provenance["source_files"]["line_neuron"]["path"],
    }
    for path in mapped_affine_inputs:
        p = Path(path)
        if p.suffix.lower() in (".py", ".md", ".json"):
            source_paths[f"affine_bound:{p.name}"] = p
    source_paths.update({f"teacher_source:{Path(k).name}": v for k, v in teacher_source_paths.items()})
    source_start = {name: base.sha256(path) for name, path in source_paths.items()}

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    torch.manual_seed(41)
    np.random.seed(41)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("metadata") != metadata:
        raise RuntimeError("Checkpoint metadata differs from run metadata")
    model = teacher_helper.build_teacher(state, device).eval()
    if len(model.layers) != 4:
        raise RuntimeError("Expected the frozen four-layer GatedGCN teacher")

    dataset = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    if len(dataset) != EXPECTED_VAL:
        raise RuntimeError(f"Unexpected validation count: {len(dataset)}")
    labels_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model_loader = DataLoader(base.LabelFreeDataset(dataset), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    label_iter = iter(labels_loader)

    method_keys = [(mode, t) for mode in MODES for t in TIMESTEPS]
    prediction_parts = {key: [] for key in method_keys}
    events = {key: {} for key in method_keys}
    ledger = {key: {"source_warmup_calls": 0, "source_active_unit_calls": 0,
                    "analog_head_calls": 0, "graph_layer_evaluations": 0} for key in method_keys}
    label_parts, teacher_parts, indices = [], [], []
    graphs, start_time = 0, time.monotonic()
    resource_probe = None
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
            thresholds = control_thresholds(mode, affine_doc, six["doc"], cal_helper,
                                            device, batch.x.dtype)
            result = controls.affine_mode_forward(model, batch, thresholds, timestep, mode,
                                                  MTH_diff, MTH_diff_line)
            key = (mode, timestep)
            prediction_parts[key].append(result["logits"].detach().cpu().numpy())
            for name, stat in zip(result["site_names"], result["site_stats"]):
                events[key][name] = events[key].get(name, 0) + int(stat["nonzero_scalar_events"])
            ledger[key]["source_warmup_calls"] += int(result["source_warmup_calls"])
            ledger[key]["source_active_unit_calls"] += int(result["active_unit_calls"])
            ledger[key]["analog_head_calls"] += int(result["head_calls"])
            ledger[key]["graph_layer_evaluations"] += len(model.layers) * int(timestep)
        graphs += n
        if batch_no == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            resource_probe = {"first_batch_graphs": n, "elapsed_sec": time.monotonic() - start_time,
                              "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
                              if device.type == "cuda" else None,
                              "scope": "resource feasibility only; not a latency benchmark"}
            print(json.dumps({"resource_probe": resource_probe}), flush=True)
            if resource_probe["elapsed_sec"] * 40 > 80 * 3600:
                raise RuntimeError("Projected fixed control grid exceeds declared E1 compute ceiling")
        if batch_no % 10 == 9 or graphs == EXPECTED_VAL:
            print(json.dumps({"graphs": graphs, "elapsed_sec": time.monotonic() - start_time}), flush=True)

    labels = np.concatenate(label_parts)
    teacher = np.concatenate(teacher_parts)
    indices_all = np.concatenate(indices)
    if graphs != EXPECTED_VAL or not np.array_equal(indices_all, np.arange(EXPECTED_VAL)):
        raise RuntimeError("Incomplete or noncanonical validation traversal")
    if float(np.mean(teacher.argmax(1) == labels)) != 0.7066:
        raise RuntimeError("Seed-41 teacher validation accuracy does not match its frozen reference")
    with np.load(source_paths["canonical_baseline"], allow_pickle=False) as reference:
        if not np.array_equal(indices_all, reference["indices"]) or not np.array_equal(labels, reference["labels"]):
            raise RuntimeError("Canonical teacher reference index/label mismatch")
        if not np.array_equal(teacher, reference["teacher_logits"]):
            raise RuntimeError("Canonical teacher logits differ under fixed deterministic setup")

    predictions = {key: np.concatenate(parts) for key, parts in prediction_parts.items()}
    rows = []
    for mode, timestep in method_keys:
        rows.append({"mode": mode, "T": timestep,
                     **base.metric_row(predictions[(mode, timestep)], teacher, labels),
                     "per_site_nonzero_scalar_events": events[(mode, timestep)]})

    source_end = {name: base.sha256(path) for name, path in source_paths.items()}
    data_end = base.processed_hashes(data_root)
    if source_start != source_end or start_data != data_end:
        raise RuntimeError("Source, calibration, model, or data changed during evaluation")
    if author_provenance != affine_helper.validate_source_unchanged(args.author_repo, author_provenance):
        raise RuntimeError("Pinned author sources changed during evaluation")

    report = {
        "status": "complete", "dataset": "CIFAR10", "split": "all official validation",
        "graphs": graphs, "seed": 41, "threshold_scale": 1,
        "historical_validation_exposure": True, "historical_test_exposure": True,
        "test_inference_performed": False,
        "modes": list(MODES), "timesteps": list(TIMESTEPS),
        "affine_calibration_sha256": source_start["affine_calibration"],
        "six_site_calibration_sha256": source_start["six_site_calibration"],
        "checkpoint_sha256": checkpoint_hash,
        "data_sha256_start": start_data, "data_sha256_end": data_end,
        "calibration_bound_inputs_sha256": mapped_affine_inputs,
        "source_sha256_start": source_start, "source_sha256_end": source_end,
        "author_source_provenance": author_provenance,
        "runtime": {"python": sys.version, "torch": torch.__version__, "device": str(device),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
        "elapsed_sec": time.monotonic() - start_time, "resource_probe": resource_probe,
        "teacher_accuracy": float(np.mean(teacher.argmax(1) == labels)),
        "rows": rows,
        "operation_ledger": {
            "scope": "partial observed reference counters only; dense arithmetic, memory traffic, hardware latency and energy are NOT measured",
            "per_method": {f"{mode}_T{t}": ledger[(mode, t)] for mode, t in method_keys},
            "dense_mac_or_flop_count": None, "memory_traffic_bytes": None,
            "hardware_latency_or_energy": None},
        "threshold_binding": {
            "dc_adapt_analog_head": "four gated_0..gated_3 node vectors from the frozen six-site full-train calibration; the two MLP sites are intentionally not coded",
            "per_consumer": "eight raw_node_i/raw_edge_i vectors from affine_cal1.json via profile_thresholds(generic, 1)",
            "cse_only": "same eight vectors; explicitly an alias of the shared generic numerical trajectory"},
        "alias_and_scope": {
            "cse_only": "ordinary shared-generic alias; identical trajectory by construction, not an independent method row",
            "per_consumer": "five independent encoder states per layer; equal numerical trajectory to generic is expected and must not be credited as an independent win",
            "dc_adapt_analog_head": "matched-head control: four pinned graph branches, exact original analog MLP head once; the historical six-site DC-adapt remains a separate required row"},
        "limitations": [
            "Seed 41 only; seeds 42/43 require their own frozen full-train calibrations and unchanged validation.",
            "No complete known-operation cost, matched wall-clock latency or energy evidence; E1 method-advantage criterion is NOT evaluable from these rows alone.",
            "No Story decision, no new test inference, no paper or independent review claim."]}

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=False)
    for mode, timestep in method_keys:
        filename = f"{mode}_T{timestep}.npz"
        with (out_dir / filename).open("xb") as stream:
            np.savez_compressed(stream, indices=indices_all, labels=labels,
                                teacher_logits=teacher, logits=predictions[(mode, timestep)])
        report.setdefault("prediction_archives_sha256", {})[f"{mode}_T{timestep}"] = base.sha256(out_dir / filename)
    with (out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "graphs": graphs,
                      "teacher_accuracy": report["teacher_accuracy"],
                      "rows": [{k: r[k] for k in ("mode", "T", "accuracy", "teacher_agreement")}
                               for r in rows]}, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gnn2snn-root", type=Path,
                   default=Path(os.environ.get("GNN2SNN_ROOT", "")),
                   required=not bool(os.environ.get("GNN2SNN_ROOT")))
    p.add_argument("--data-root", type=Path,
                   default=Path(os.environ.get("GNN2SNN_DATA_ROOT", "")),
                   required=not bool(os.environ.get("GNN2SNN_DATA_ROOT")))
    p.add_argument("--teacher-dir", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path, help="affine_cal1.json")
    p.add_argument("--six-site-calibration", required=True, type=Path, help="gcal_v1.json")
    p.add_argument("--author-repo", type=Path,
                   default=Path(os.environ.get("DCGS_REPO", "")),
                   required=not bool(os.environ.get("DCGS_REPO")))
    p.add_argument("--newrun-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
