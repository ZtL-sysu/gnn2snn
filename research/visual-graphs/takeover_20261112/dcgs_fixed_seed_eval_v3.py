"""Full official val/test DCGS evaluation with matched ANN reference guards.

This version reuses the pinned source, fixed-GCN forward, and provenance
validators from immutable v2. It strips labels from the cloned batch before
event conversion. Test evaluation requires a saved same-seed teacher prediction
archive and checks exact ordered indices/labels plus teacher-logit agreement.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data" / "pyg_bench"
OUT_ROOT = HERE / "full_teacher_runs"
TIMESTEPS = (1, 2, 4, 8)
SPLIT_COUNTS = {"val": 5000, "test": 10000}
REFERENCE_RTOL = 1e-5
REFERENCE_ATOL = 1e-4


def _load_v2():
    path = HERE / "dcgs_fixed_seed_eval_v2.py"
    spec = importlib.util.spec_from_file_location("dcgs_fixed_seed_eval_immutable_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load_v2()


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _read_reference(path, dataset, seed, count, checkpoint_sha):
    path = Path(path).resolve()
    if not _inside(path, HERE):
        raise RuntimeError("reference predictions must be inside HERE")
    sidecar = path.with_name("teacher_full_test.json")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("reference predictions and teacher_full_test.json are required")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if meta.get("dataset") != dataset or meta.get("seed") != seed:
        raise RuntimeError("reference dataset/seed mismatch")
    if meta.get("graphs") != count or "test" not in meta.get("split", "").lower():
        raise RuntimeError("reference is not the expected complete official test split")
    if meta.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("reference teacher checkpoint hash mismatch")
    if meta.get("prediction_artifact_sha256") != _sha256(path):
        raise RuntimeError("reference prediction archive hash mismatch")
    archive = np.load(path, allow_pickle=False)
    required = {"index", "labels", "logits", "predictions"}
    if set(archive.files) != required:
        raise RuntimeError("reference archive keys must be index, labels, logits, predictions")
    indices = np.asarray(archive["index"])
    labels = np.asarray(archive["labels"])
    logits = np.asarray(archive["logits"])
    predictions = np.asarray(archive["predictions"])
    if indices.shape != (count,) or not np.array_equal(indices, np.arange(count, dtype=indices.dtype)):
        raise RuntimeError("reference ordered indices do not cover the full split")
    if labels.shape != (count,) or logits.shape != (count, 10) or predictions.shape != (count,):
        raise RuntimeError("reference archive has invalid row/class dimensions")
    if not np.isfinite(logits).all() or not np.array_equal(predictions, logits.argmax(axis=1)):
        raise RuntimeError("reference logits/predictions are invalid or inconsistent")
    return {"path": path, "sidecar": sidecar, "labels": labels,
            "indices": indices, "logits": logits,
            "archive_sha256": _sha256(path), "sidecar_sha256": _sha256(sidecar)}


def _compare_reference(indices, labels, teacher_logits, reference):
    if not np.array_equal(indices, reference["indices"]):
        raise RuntimeError("evaluation indices differ from reference ordering")
    if not np.array_equal(labels, reference["labels"]):
        raise RuntimeError("evaluation labels differ from teacher reference labels")
    if teacher_logits.shape != reference["logits"].shape:
        raise RuntimeError("teacher logits shape differs from reference")
    if not np.isfinite(teacher_logits).all():
        raise RuntimeError("non-finite evaluation teacher logits")
    delta = float(np.max(np.abs(teacher_logits - reference["logits"])))
    if not np.allclose(teacher_logits, reference["logits"], rtol=REFERENCE_RTOL,
                       atol=REFERENCE_ATOL):
        raise RuntimeError("teacher logits differ from saved reference (max_abs=%g)" % delta)
    return delta


def _strip_targets(batch):
    """Copy a PyG batch and remove labels before event conversion."""
    result = batch.clone()
    if "y" in result:
        del result.y
    return result


def _bound_inputs(dataset, seed, teacher_run, calibration, author_repo, provenance):
    done, meta, cal, checkpoint, checkpoint_sha, source_sha = V2.validate_inputs(
        dataset, seed, teacher_run, calibration, provenance)
    if source_sha != V2.PINNED_COMMIT:
        raise RuntimeError("pinned source mismatch")
    hashes = {"script_v3": _sha256(__file__),
              "forward_v2": _sha256(HERE / "dcgs_fixed_seed_eval_v2.py"),
              "model": _sha256(HERE / "full_teacher_gcn.py"),
              "bn_fold": _sha256(HERE / "fold_teacher_bn.py"),
              "source_oracle": _sha256(HERE / "dcgs_mth_diff_source_equivalence_v1.py"),
              "protocol_v3": _sha256(HERE / "dc-fixed-fulltest-protocol-v3.md"),
              "checkpoint": checkpoint_sha,
              "calibration": _sha256(calibration),
              "teacher_metadata": _sha256(Path(teacher_run) / "metadata.json"),
              "teacher_done": _sha256(Path(teacher_run) / "done.json"),
              "author_neuron_source": provenance["source_file_sha256"],
              "author_commit": source_sha}
    processed = DATA_ROOT / dataset / "processed"
    for name in cal.get("data", {}).get("processed_files_sha256", {}):
        hashes["processed:" + name] = _sha256(processed / name)
    return done, meta, cal, checkpoint, checkpoint_sha, hashes


def evaluate(dataset, seed, teacher_run, calibration, author_repo, output,
             reference_predictions, split, device_name="cpu", batch_size=128):
    if split not in SPLIT_COUNTS:
        raise ValueError("split must be val or test")
    if batch_size != 128:
        raise ValueError("locked evaluator batch size is 128")
    output = Path(output).resolve()
    pred_out = output.with_suffix(".npz")
    if output.suffix.lower() != ".json":
        raise ValueError("--output must name a .json file with a paired .npz")
    if not _inside(output, HERE):
        raise RuntimeError("output must be inside HERE")
    if output.exists() or pred_out.exists():
        raise FileExistsError("refusing existing or partial JSON/NPZ output")

    MTH_diff, commit, provenance = V2.require_author_source(author_repo)
    if commit != V2.PINNED_COMMIT:
        raise RuntimeError("author source commit mismatch")
    done, metadata, cal, checkpoint, checkpoint_sha, start_hashes = _bound_inputs(
        dataset, seed, teacher_run, calibration, author_repo, provenance)
    count = SPLIT_COUNTS[split]
    reference = None
    if split == "test":
        if reference_predictions is None:
            raise RuntimeError("test evaluation requires --reference-predictions")
        reference = _read_reference(reference_predictions, dataset, seed, count, checkpoint_sha)
        start_hashes["reference_predictions"] = reference["archive_sha256"]
        start_hashes["reference_sidecar"] = reference["sidecar_sha256"]
    elif reference_predictions is not None:
        raise ValueError("--reference-predictions is reserved for the full test split")

    from full_teacher_gcn import BenchmarkGCN
    from fold_teacher_bn import fold_model
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu":
        torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(state.get("epoch", -1)) != int(cal.get("teacher_epoch", -2)):
        raise RuntimeError("calibration epoch does not match checkpoint")
    model = BenchmarkGCN(input_dim=metadata["input_dim"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    folded = fold_model(model).eval()
    thresholds = [cal["channel_thresholds"][name] for name in V2.SITES]
    data = GNNBenchmarkDataset(root=str(DATA_ROOT), name=dataset, split=split)
    if len(data) != count:
        raise RuntimeError("official %s split count mismatch" % split)
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=0)
    labels_parts, teacher_parts, indices_parts = [], [], []
    converted = {T: [] for T in TIMESTEPS}
    event_totals = {T: [None] * len(V2.SITES) for T in TIMESTEPS}
    next_index = 0
    start = time.monotonic()
    for batch_num, batch in enumerate(loader, 1):
        batch = batch.to(device)
        batch_count = int(batch.num_graphs)
        with torch.no_grad():
            teacher_logits = model(batch)
        if teacher_logits.shape != (batch_count, 10) or not bool(torch.isfinite(teacher_logits).all()):
            raise RuntimeError("invalid terminal teacher logits")
        teacher_parts.append(teacher_logits.cpu().numpy())
        labels_parts.append(batch.y.view(-1).long().cpu().numpy())
        indices_parts.append(np.arange(next_index, next_index + batch_count, dtype=np.int64))
        next_index += batch_count
        # Keep labels available for metrics only. Neither teacher logits nor labels
        # enter the fixed event conversion forward.
        conversion_batch = _strip_targets(batch)
        for T in TIMESTEPS:
            with torch.no_grad():
                logits, stats, _ = V2.dcgs_event_forward(folded, conversion_batch, thresholds, T,
                                                        MTH_diff)
            if logits.shape != teacher_logits.shape or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("invalid/non-finite converted logits at T=%d" % T)
            converted[T].append(logits.cpu().numpy())
            for site, site_stats in enumerate(stats):
                if event_totals[T][site] is None:
                    event_totals[T][site] = json.loads(json.dumps(site_stats))
                else:
                    V2._accumulate_stats(event_totals[T][site], site_stats)
        if batch_num % 10 == 0:
            print(json.dumps({"split": split, "batch": batch_num,
                              "graphs": next_index,
                              "elapsed_sec": time.monotonic() - start}), flush=True)

    indices = np.concatenate(indices_parts)
    labels = np.concatenate(labels_parts)
    teacher_logits = np.concatenate(teacher_parts)
    if (len(indices) != count or not np.array_equal(indices, np.arange(count)) or
            labels.shape != (count,)):
        raise RuntimeError("evaluation failed full ordered split coverage")
    logits_by_t = {T: np.concatenate(converted[T]) for T in TIMESTEPS}
    teacher_reference_max_abs_diff = None
    if reference is not None:
        teacher_reference_max_abs_diff = _compare_reference(
            indices, labels, teacher_logits, reference)
    teacher_accuracy = float(np.mean(teacher_logits.argmax(1) == labels))
    rows = []
    for T in TIMESTEPS:
        logits = logits_by_t[T]
        rows.append({"T": T, "teacher_accuracy": teacher_accuracy,
                     "converted_accuracy": float(np.mean(logits.argmax(1) == labels)),
                     "teacher_agreement": float(np.mean(logits.argmax(1) == teacher_logits.argmax(1))),
                     "logit_mse": float(np.mean((logits - teacher_logits) ** 2)),
                     "sites": event_totals[T]})

    # Bind all loaded code/data/reference inputs again immediately before writing.
    _, _, _, _, checkpoint_sha_end, end_hashes = _bound_inputs(
        dataset, seed, teacher_run, calibration, author_repo, provenance)
    _, commit_end, provenance_end = V2.require_author_source(author_repo)
    if commit_end != commit or provenance_end != provenance:
        raise RuntimeError("pinned author source changed during evaluation")
    if reference is not None:
        end_hashes["reference_predictions"] = _sha256(reference["path"])
        end_hashes["reference_sidecar"] = _sha256(reference["sidecar"])
    if checkpoint_sha_end != checkpoint_sha or end_hashes != start_hashes:
        raise RuntimeError("a code, teacher, calibration, or processed-data input changed during evaluation")
    if reference is not None:
        if (_sha256(reference["path"]) != reference["archive_sha256"] or
                _sha256(reference["sidecar"]) != reference["sidecar_sha256"]):
            raise RuntimeError("reference archive or sidecar changed during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or pred_out.exists():
        raise FileExistsError("output appeared during evaluation; refusing overwrite")
    with pred_out.open("xb") as pred_file:
        np.savez_compressed(pred_file, indices=indices, labels=labels,
                            teacher_logits=teacher_logits,
                            **{"T%d_logits" % T: logits_by_t[T] for T in TIMESTEPS})
    result = {"status": "complete", "dataset": dataset, "seed": seed,
              "split": "complete official %s; historically exposed" % split,
              "graphs": count, "batch_size": 128, "checkpoint_epoch": int(state["epoch"]),
              "teacher_run": str(Path(teacher_run).resolve()),
              "calibration_path": str(Path(calibration).resolve()),
              "checkpoint_sha256": checkpoint_sha,
              "calibration_sha256": start_hashes["calibration"],
              "source_commit": commit, "author_source_provenance": provenance,
              "input_hashes_start": start_hashes, "input_hashes_end": end_hashes,
              "reference_predictions": str(reference["path"]) if reference else None,
              "reference_predictions_sha256": reference["archive_sha256"] if reference else None,
              "reference_comparison": {"rtol": REFERENCE_RTOL, "atol": REFERENCE_ATOL,
                                       "teacher_logits_max_abs_diff": teacher_reference_max_abs_diff},
              "script_sha256": _sha256(__file__),
              "predictions_sha256": _sha256(pred_out), "predictions_file": str(pred_out),
              "device": str(device),
              "runtime": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                          "numpy": np.__version__, "python": sys.version,
                          "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                          "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
              "elapsed_sec": time.monotonic() - start,
              "decoder": "logit = sum_{t=1..T} classifier_event_t / unit_internal_T_after_active_call; event biases once at t=1",
              "neuron_placement": "pinned MTH_diff source unit on project ReLU convolution branches before residual event addition; common full-training histogram calibration; this does not implement generic pre-linear identity placement or claim full DCGS Algorithm 2 reproduction",
              "unit_call_accounting": {
                  "source_warmup_calls_per_batch": len(V2.SITES),
                  "warmup_emits_output": False,
                  "warmup_is_active_tick": False,
                  "active_unit_calls_per_batch_by_T": {"T%d" % T: len(V2.SITES) * T for T in TIMESTEPS},
                  "warmup_calls_per_T_run": len(loader) * len(V2.SITES),
                  "active_calls_per_T_run_by_T": {"T%d" % T: len(loader) * len(V2.SITES) * T for T in TIMESTEPS}},
              "readout_rows": rows,
              "qualification": "Project MTH-ReLU-branch graph adaptation using pinned source units and common full-train histogram calibration; not full DCGS reproduction or author iteration. Test split has prior project exposure; no untouched-test claim."}
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=tuple(V2.EXPECTED_COUNTS), required=True)
    p.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    p.add_argument("--teacher-run", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--author-repo", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--reference-predictions", type=Path,
                   help="Required for test; same-seed saved teacher_full_test_predictions.npz")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args(argv)
    evaluate(args.dataset, args.seed, args.teacher_run, args.calibration,
             args.author_repo, args.output, args.reference_predictions,
             args.split, args.device, args.batch_size)


if __name__ == "__main__":
    main()
