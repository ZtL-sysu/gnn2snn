"""Frozen E1 full-5000 validation for CIFAR10 GatedGCN teacher seeds 42/43.

This is the *unchanged* E1 procedure parameterised by teacher identity only.
It reuses, without modification:

  * eval_affine_validation.metric_row / LabelFreeDataset / processed_hashes
  * gated_affine_cal_v1.profile_thresholds + validate_statistics (threshold map)
  * gated_affine_base_v1.affine_mode_forward (centered / first_analog / generic)
  * matched_controls.affine_mode_forward (dc_adapt_analog_head / per_consumer / cse_only)
  * the pinned DCGS author sources through affine_helper.require_author_sources

Rows: six implemented rows x T in {1,2,4,8} = 24 groups per seed.
Scale 1 fixed. All 5000 official validation graphs, canonical fixed order,
batch 128, label-free converter inputs, deterministic CUDA, no test split.

Per-seed bindings (and only these) differ from the seed-41 reference:
  * teacher checkpoint + metadata + done marker (frozen terminal run)
  * that teacher's own frozen full-train label-free calibrations (14-site affine,
    six-site gcal) produced by calibrate_multiseed_v1.py with the SAME immutable
    calibration functions
  * teacher validation accuracy expected from that run's frozen done.json
    (0.703 for seed42, 0.7074 for seed43); never tuned against results
  * canonical indices/labels are still checked bitwise against the frozen
    canonical baseline (data order is teacher-independent). The seed-41
    canonical *teacher logits* are NOT a valid reference for other teachers;
    instead teacher logits are checked bitwise across two independent forward
    passes under the same deterministic setup.

The seed-41 reference artifacts and all prior evidence are never modified.
No complete operation cost, matched latency, energy, Story decision, new test
inference, paper, citation audit or independent review is claimed here.
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
STUDY_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

_IMPORT_ROOT = Path(os.environ.get("GNN2SNN_ROOT", CODE_DIR.parents[3])).resolve()
_IMPORT_SOURCE_ROOT = _IMPORT_ROOT / "research" / "visual-graphs" / "takeover_20261112"
if str(_IMPORT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_SOURCE_ROOT))

import eval_affine_validation as base  # noqa: E402
import matched_controls as controls  # noqa: E402

AFFINE_MODES = ("centered", "first_analog", "generic")
CONTROL_MODES = ("dc_adapt_analog_head", "per_consumer", "cse_only")
MODES = AFFINE_MODES + CONTROL_MODES
TIMESTEPS = base.TIMESTEPS
EXPECTED_VAL = base.EXPECTED_VAL
BATCH_SIZE = base.BATCH_SIZE
SIX_SITES = ("gated_0", "gated_1", "gated_2", "gated_3", "mlp_0", "mlp_1")
FROZEN_PROTOCOL = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
FROZEN_ADDENDUM = "8bde5832d556c8c7b04ee630f2390703c33a26fa750f196f4dc1c94da61291f6"


def verify_affine_calibration(path: Path, teacher_dir: Path, seed: int,
                              checkpoint_hash: str, metadata: dict, done: dict,
                              cal_helper) -> dict:
    """Same structure/statistics checks as seed41, bound to this seed's teacher."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("status") != "complete" or doc.get("dataset") != "CIFAR10" or
            int(doc.get("seed", -1)) != seed or
            doc.get("scope") != "all_45000_official_train_graphs_label_free_two_pass"):
        raise RuntimeError("wrong/incomplete affine calibration scope or seed")
    teacher_doc = doc.get("teacher", {})
    if (teacher_doc.get("checkpoint_sha256") != checkpoint_hash or
            teacher_doc.get("metadata") != metadata or teacher_doc.get("done") != done):
        raise RuntimeError("affine calibration does not bind this seed's teacher")
    if int(teacher_doc.get("epoch", -1)) < 1:
        raise RuntimeError("affine calibration lacks a valid teacher epoch")
    if not doc.get("input_hashes_start") or doc.get("input_hashes_start") != doc.get("input_hashes_end"):
        raise RuntimeError("affine calibration input hash ledger incomplete or changed")
    mapped = {}
    for recorded, expected in doc["input_hashes_start"].items():
        p = Path(recorded)
        if not p.is_file():
            raise FileNotFoundError(f"calibration-bound input missing: {p}")
        if base.sha256(p) != expected:
            raise RuntimeError(f"calibration-bound input hash mismatch: {p}")
        mapped[str(p)] = expected
    cal_helper.validate_statistics(doc["calibration"])
    if set(doc["calibration"]["channel_thresholds"]) != set(cal_helper.SITES):
        raise RuntimeError("affine calibration threshold sites do not match helper")
    binding = doc.get("algorithm_binding", {})
    if binding.get("calibration_function") != "gated_affine_cal_v1.calibrate_loader":
        raise RuntimeError("affine calibration was not produced by the frozen function")
    if binding.get("helper_sha256", {}).get("affine_helper") != "0fb4f431b142dca0829e8ca57f2a12199f80af89a839863d5ad0001eb4ac1b39":
        raise RuntimeError("affine calibration helper hash differs from frozen evidence")
    return doc


def verify_six_calibration(path: Path, seed: int, checkpoint_hash: str,
                           metadata: dict, done: dict) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("status") != "complete" or doc.get("dataset") != "CIFAR10" or
            int(doc.get("seed", -1)) != seed):
        raise RuntimeError("wrong/incomplete six-site calibration")
    if (doc.get("teacher_metadata") != metadata or doc.get("teacher_done") != done or
            doc.get("checkpoint_sha256") != checkpoint_hash):
        raise RuntimeError("six-site calibration binds a different teacher")
    cal = doc.get("calibration", {})
    if cal.get("full_train_graph_count") != base.EXPECTED_TRAIN:
        raise RuntimeError("six-site calibration did not use full 45000 train graphs")
    if cal.get("passes") != 2 or cal.get("pass1_graphs") != base.EXPECTED_TRAIN or cal.get("pass2_graphs") != base.EXPECTED_TRAIN:
        raise RuntimeError("six-site calibration is not two full passes")
    if cal.get("bins") != 512 or abs(float(cal.get("quantile", 0)) - 0.999) > 1e-12:
        raise RuntimeError("six-site bins/quantile changed")
    if set(cal.get("channel_thresholds", {})) != set(SIX_SITES):
        raise RuntimeError("six-site threshold sites differ from frozen order")
    for site in SIX_SITES:
        vals = cal["channel_thresholds"][site]
        if not vals or not all(float(v) > 0 and np.isfinite(float(v)) for v in vals):
            raise RuntimeError(f"invalid six-site threshold at {site}")
    if not doc.get("input_hashes_start") or doc.get("input_hashes_start") != doc.get("input_hashes_end"):
        raise RuntimeError("six-site calibration input hash ledger incomplete")
    binding = doc.get("algorithm_binding", {})
    if binding.get("calibration_function") != "gated_frozen_gate_calibrate.calibrate_loader":
        raise RuntimeError("six-site calibration was not produced by the frozen function")
    return doc


def thresholds_for(mode: str, affine_doc: dict, six_doc: dict, cal_helper,
                   device, dtype):
    """Unchanged frozen threshold mapping (no search, no tuning)."""
    if mode in AFFINE_MODES:
        return [t.to(device=device, dtype=dtype)
                for t in cal_helper.profile_thresholds(affine_doc, mode, 1)]
    if mode == "dc_adapt_analog_head":
        ct = six_doc["calibration"]["channel_thresholds"]
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
    raise ValueError(f"unsupported mode: {mode}")


def forward_for(mode, model, batch, thresholds, T, MTH_diff, MTH_diff_line,
                affine_helper):
    if mode in AFFINE_MODES:
        return affine_helper.affine_mode_forward(model, batch, thresholds, T, mode,
                                                 MTH_diff, MTH_diff_line)
    return controls.affine_mode_forward(model, batch, thresholds, T, mode,
                                        MTH_diff, MTH_diff_line)


@torch.no_grad()
def evaluate(args) -> None:
    source_root = args.gnn2snn_root.resolve() / "research" / "visual-graphs" / "takeover_20261112"
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    teacher_dir = args.teacher_dir.resolve()
    data_root = args.data_root.resolve()
    affine_cal = args.affine_calibration.resolve()
    six_cal = args.six_site_calibration.resolve()
    out_dir = args.output_dir.resolve()
    allowed_root = args.newrun_root.resolve()
    try:
        out_dir.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("output directory must stay under --newrun-root") from exc
    if out_dir.exists():
        raise FileExistsError(f"Refusing existing result directory: {out_dir}")

    for path in (teacher_dir / "metadata.json", teacher_dir / "done.json",
                 teacher_dir / "best.pt", affine_cal, six_cal):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads((teacher_dir / "metadata.json").read_text(encoding="utf-8"))
    done = json.loads((teacher_dir / "done.json").read_text(encoding="utf-8"))
    checkpoint = teacher_dir / "best.pt"
    checkpoint_hash = base.sha256(checkpoint)
    seed = args.seed
    if (metadata.get("dataset") != "CIFAR10" or
            metadata.get("scope") != "all_official_pyg_train_and_val" or
            metadata.get("official_counts") != {"train": base.EXPECTED_TRAIN, "val": EXPECTED_VAL} or
            int(metadata.get("seed", -1)) != seed):
        raise RuntimeError(f"Expected full official CIFAR10 seed-{seed} teacher")
    if done.get("status") != "finished" or done.get("official_counts") != metadata["official_counts"]:
        raise RuntimeError("Teacher run is not complete")
    if done.get("best_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Teacher checkpoint hash differs from terminal marker")
    expected_teacher_accuracy = float(done["best_val_accuracy"])

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    cal_helper = importlib.import_module("gated_affine_cal_v1")
    affine_helper = importlib.import_module("gated_affine_base_v1")
    teacher_helper = importlib.import_module("gated_frozen_gate_calibrate")
    MTH_diff, MTH_diff_line, author_provenance = affine_helper.require_author_sources(args.author_repo)

    affine_doc = verify_affine_calibration(affine_cal, teacher_dir, seed,
                                           checkpoint_hash, metadata, done, cal_helper)
    six_doc = verify_six_calibration(six_cal, seed, checkpoint_hash, metadata, done)

    start_data = base.processed_hashes(data_root)
    if start_data != metadata["provenance"]["processed_data_sha256"]:
        raise RuntimeError("Processed data differ from teacher metadata")
    teacher_source_paths = {}
    for recorded, expected in metadata["provenance"]["source_sha256"].items():
        p = teacher_helper.resolve_teacher_source(source_root, recorded)
        if base.sha256(p) != expected:
            raise RuntimeError(f"Teacher source hash mismatch: {recorded}")
        teacher_source_paths[str(p.resolve())] = p.resolve()

    source_paths = {
        "evaluator": Path(__file__).resolve(),
        "base_evaluator": Path(base.__file__).resolve(),
        "control_library": Path(controls.__file__).resolve(),
        "calibration_driver": CODE_DIR / "calibrate_multiseed_v1.py",
        "frozen_protocol": STUDY_ROOT / "PROTOCOL-v3.md",
        "head_matching_addendum": STUDY_ROOT / "PROTOCOL-E1-head-addendum-v1.md",
        "canonical_baseline": STUDY_ROOT / "inputs" / "canonical-baseline-val.npz",
        "affine_calibration": affine_cal, "six_site_calibration": six_cal,
        "affine_helper": Path(affine_helper.__file__).resolve(),
        "calibration_helper": Path(cal_helper.__file__).resolve(),
        "teacher_helper": Path(teacher_helper.__file__).resolve(),
        "author_relu_source": args.author_repo.resolve() / author_provenance["source_files"]["relu_neuron"]["path"],
        "author_line_source": args.author_repo.resolve() / author_provenance["source_files"]["line_neuron"]["path"],
    }
    if base.sha256(source_paths["frozen_protocol"]) != FROZEN_PROTOCOL:
        raise RuntimeError("frozen PROTOCOL-v3 hash changed")
    if base.sha256(source_paths["head_matching_addendum"]) != FROZEN_ADDENDUM:
        raise RuntimeError("frozen head addendum hash changed")
    source_paths.update({f"teacher_source:{Path(k).name}": v for k, v in teacher_source_paths.items()})
    source_start = {name: base.sha256(path) for name, path in source_paths.items()}

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    model_loader = DataLoader(base.LabelFreeDataset(dataset), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
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
        # Determinism check: a second independent teacher forward must be bitwise equal.
        teacher_logits_repeat = model(batch)
        if not torch.equal(teacher_logits, teacher_logits_repeat):
            raise RuntimeError("teacher logits are not bitwise deterministic across forwards")
        teacher_parts.append(teacher_logits.detach().cpu().numpy())
        for mode, timestep in method_keys:
            thresholds = thresholds_for(mode, affine_doc, six_doc, cal_helper,
                                        device, batch.x.dtype)
            result = forward_for(mode, model, batch, thresholds, timestep,
                                 MTH_diff, MTH_diff_line, affine_helper)
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
            resource_probe = {"first_batch_graphs": n,
                              "elapsed_sec": time.monotonic() - start_time,
                              "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
                              if device.type == "cuda" else None,
                              "scope": "resource feasibility only; not a latency benchmark"}
            print(json.dumps({"resource_probe": resource_probe}), flush=True)
            if resource_probe["elapsed_sec"] * 40 > 80 * 3600:
                raise RuntimeError("Projected fixed E1 grid exceeds declared E1 compute ceiling")
        if batch_no % 10 == 9 or graphs == EXPECTED_VAL:
            print(json.dumps({"graphs": graphs, "elapsed_sec": time.monotonic() - start_time}), flush=True)

    labels = np.concatenate(label_parts)
    teacher = np.concatenate(teacher_parts)
    indices_all = np.concatenate(indices)
    if graphs != EXPECTED_VAL or not np.array_equal(indices_all, np.arange(EXPECTED_VAL)):
        raise RuntimeError("Incomplete or noncanonical validation traversal")
    teacher_accuracy = float(np.mean(teacher.argmax(1) == labels))
    if teacher_accuracy != expected_teacher_accuracy:
        raise RuntimeError(f"seed-{seed} teacher validation accuracy {teacher_accuracy} "
                           f"does not match frozen reference {expected_teacher_accuracy}")
    with np.load(source_paths["canonical_baseline"], allow_pickle=False) as reference:
        if not np.array_equal(indices_all, reference["indices"]) or not np.array_equal(labels, reference["labels"]):
            raise RuntimeError("Canonical index/label mismatch (teacher-independent order)")
        if seed == 41 and not np.array_equal(teacher, reference["teacher_logits"]):
            raise RuntimeError("seed-41 canonical teacher logits differ")

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
        "graphs": graphs, "seed": seed, "threshold_scale": 1,
        "historical_validation_exposure": True, "historical_test_exposure": True,
        "test_inference_performed": False,
        "modes": list(MODES), "timesteps": list(TIMESTEPS),
        "affine_calibration_sha256": source_start["affine_calibration"],
        "six_site_calibration_sha256": source_start["six_site_calibration"],
        "checkpoint_sha256": checkpoint_hash,
        "frozen_expected_teacher_accuracy": expected_teacher_accuracy,
        "data_sha256_start": start_data, "data_sha256_end": data_end,
        "source_sha256_start": source_start, "source_sha256_end": source_end,
        "author_source_provenance": author_provenance,
        "runtime": {"python": sys.version, "torch": torch.__version__, "device": str(device),
                    "manual_seed": seed,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                    "teacher_logits_bitwise_double_forward": True},
        "elapsed_sec": time.monotonic() - start_time, "resource_probe": resource_probe,
        "teacher_accuracy": teacher_accuracy,
        "rows": rows,
        "operation_ledger": {
            "scope": "partial observed reference counters only; dense arithmetic, memory traffic, hardware latency and energy are NOT measured",
            "per_method": {f"{mode}_T{t}": ledger[(mode, t)] for mode, t in method_keys},
            "dense_mac_or_flop_count": None, "memory_traffic_bytes": None,
            "hardware_latency_or_energy": None},
        "threshold_binding": {
            "centered/first_analog/generic": "gated_affine_cal_v1.profile_thresholds(mode, 1) on this seed's frozen 14-site calibration",
            "dc_adapt_analog_head": "four gated_0..gated_3 vectors from this seed's frozen six-site full-train calibration; MLP sites not coded",
            "per_consumer": "eight raw vectors from profile_thresholds(generic, 1)",
            "cse_only": "same eight vectors; explicitly a shared-generic numerical alias"},
        "alias_and_scope": {
            "cse_only": "ordinary shared-generic alias; identical trajectory by construction, not an independent method row",
            "per_consumer": "five independent encoder states per layer; equal numerical trajectory to generic is expected, not an independent win",
            "dc_adapt_analog_head": "matched-head control, four pinned graph branches + exact original analog MLP head once; historical six-site DC-adapt remains a separate required row"},
        "canonical_reference_scope": {
            "indices_labels": "bitwise checked against the frozen canonical-baseline-val.npz (teacher-independent order)",
            "teacher_logits": "seed-41 canonical teacher logits are NOT a reference for other teachers; this run checks bitwise determinism across two independent forwards instead",
            "statement": "no historical-trajectory reproduction is claimed for seeds 42/43"},
        "limitations": [
            "Six implemented rows only; ANN row, historical six-site DC-adapt row aggregation and shared-theta controls remain outstanding.",
            "No complete known-operation cost, matched wall-clock latency or energy evidence; E1 method-advantage criterion is NOT evaluable from these rows alone.",
            "No Story decision, no new test inference, no paper, no citation audit, no independent review.",
            "Teacher accuracy is the frozen done.json value, not tuned or selected against results."],
    }

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
    print(json.dumps({"status": report["status"], "seed": seed, "graphs": graphs,
                      "teacher_accuracy": teacher_accuracy,
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
    p.add_argument("--affine-calibration", required=True, type=Path)
    p.add_argument("--six-site-calibration", required=True, type=Path)
    p.add_argument("--author-repo", type=Path,
                   default=Path(os.environ.get("DCGS_REPO", "")),
                   required=not bool(os.environ.get("DCGS_REPO")))
    p.add_argument("--seed", type=int, choices=(42, 43), required=True)
    p.add_argument("--newrun-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
