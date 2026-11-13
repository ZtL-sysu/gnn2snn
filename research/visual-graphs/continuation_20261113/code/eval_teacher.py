"""Portable full validation evaluator for the frozen CIFAR10 gated teacher.

This records prediction fidelity and a deliberately partial operation ledger;
it is not a latency, energy, or hardware-cost benchmark.
"""
from __future__ import annotations

# Must be set before importing torch (and therefore before CUDA initializes).
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be unset or :4096:8 before importing Torch")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

EXPECTED_VAL = 5000
T_VALUES = (4, 8)
SITES = ("gated_0", "gated_1", "gated_2", "gated_3", "mlp_0", "mlp_1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class LabelFreeGraphView(torch.utils.data.Dataset):
    """Only return converter inputs, so labels never enter converter batches."""
    def __init__(self, source): self.source = source
    def __len__(self): return len(self.source)
    def __getitem__(self, index):
        item = self.source[index]
        return Data(**{k: getattr(item, k) for k in ("x", "edge_index", "edge_attr", "pos")
                       if getattr(item, k, None) is not None})


def load_helpers(root: Path):
    """Import the project model/converter/source verifier from configured repo."""
    helper = root / "research" / "visual-graphs" / "takeover_20261112"
    if not helper.is_dir():
        raise FileNotFoundError(f"Expected helper source directory under GNN2SNN_ROOT: {helper}")
    sys.path.insert(0, str(helper))
    cal = importlib.import_module("gated_frozen_gate_calibrate")
    validation = importlib.import_module("gated_frozen_gate_validate")
    hybrid = importlib.import_module("gated_differential_hybrid_v1")
    source = importlib.import_module("dcgs_fixed_residual_gcn_event_forward_v1")
    return cal, validation, hybrid, source


def validate_inputs(teacher_dir: Path, calibration_path: Path):
    meta_p, done_p, checkpoint = teacher_dir / "metadata.json", teacher_dir / "done.json", teacher_dir / "best.pt"
    for p in (meta_p, done_p, checkpoint, calibration_path):
        if not p.is_file(): raise FileNotFoundError(p)
    meta, done = json.loads(meta_p.read_text()), json.loads(done_p.read_text())
    cal = json.loads(calibration_path.read_text())
    if meta.get("dataset") != "CIFAR10" or meta.get("scope") != "all_official_pyg_train_and_val":
        raise RuntimeError("Expected the full official CIFAR10 gated teacher")
    if meta.get("official_counts") != {"train": 45000, "val": EXPECTED_VAL}:
        raise RuntimeError("Teacher split metadata mismatch")
    if int(meta.get("seed", -1)) != 41:
        raise RuntimeError("E0 is fixed to teacher seed 41")
    if done.get("status") != "finished" or done.get("official_counts") != meta["official_counts"]:
        raise RuntimeError("Teacher run is not complete")
    if done.get("best_checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("Teacher checkpoint hash differs from terminal marker")
    if cal.get("status") != "complete" or cal.get("dataset") != "CIFAR10":
        raise RuntimeError("Calibration is incomplete or for another dataset")
    if cal.get('scope') != 'all_45000_official_train_graphs_label_free_two_pass_fixed_order':
        raise RuntimeError('Calibration scope mismatch')
    if cal.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("Calibration checkpoint hash mismatch")
    if cal.get("calibration", {}).get("pass1_graphs") != 45000 or cal.get("calibration", {}).get("pass2_graphs") != 45000:
        raise RuntimeError("Calibration did not cover both full training passes")
    if set(cal.get("calibration", {}).get("channel_thresholds", {})) != set(SITES):
        raise RuntimeError("Calibration site thresholds are missing")
    return meta, done, cal, checkpoint


def kl_teacher_to_student(teacher, student):
    p = torch.softmax(teacher, dim=1).clamp_min(1e-12)
    logp = p.log()
    logq = torch.log_softmax(student, dim=1)
    return float(torch.sum(p * (logp - logq), dim=1).mean().item())


@torch.no_grad()
def evaluate(args):
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Test split is guarded; pass --allow-test only under an approved test protocol")
    if args.batch_size != 128:
        raise ValueError("Protocol fixes sequential batch size at 128")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    newrun = args.output_dir.resolve()
    if not newrun.is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError('Output must be under the current run root')
    if newrun.exists():
        raise FileExistsError('Refusing existing result directory')
    if newrun.exists():
        raise FileExistsError(f"Refusing existing newrun directory: {newrun}")
    report_path, archive_path = newrun / "report.json", newrun / "predictions.npz"
    if report_path.exists() or archive_path.exists():
        raise FileExistsError("Refusing to overwrite a newrun result")
    meta, done, cal, checkpoint = validate_inputs(args.teacher_dir, args.calibration)
    if int(meta.get("seed", -1)) != 41:
        raise RuntimeError("E0 evaluation requires teacher seed 41")
    gnn_root = Path(os.environ.get("GNN2SNN_ROOT", "")).resolve()
    if not os.environ.get("GNN2SNN_ROOT"):
        raise RuntimeError("Set GNN2SNN_ROOT to the project repository root")
    data_root = Path(os.environ.get("GNN2SNN_DATA_ROOT", str(gnn_root / "research" / "visual-graphs" / "data" / "pyg_bench"))).resolve()
    helpers, validation, hybrid, author = load_helpers(gnn_root)
    helpers.DATA_ROOT = data_root
    validation.DATA_ROOT = data_root
    # Keep the helper's teacher provenance verification, but do not require
    # the calibration's recorded absolute teacher path to survive relocation.
    _run, teacher_meta, _done, validated_checkpoint, checkpoint_hash = validation.validate_teacher_run(args.teacher_dir)
    if validated_checkpoint.resolve() != checkpoint.resolve() or cal.get("teacher_metadata") != teacher_meta:
        raise RuntimeError("Selected checkpoint/metadata does not match calibration")
    for filename, expected_hash in cal.get("code_sha256", {}).items():
        path = Path(helpers.__file__).resolve().parent / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Calibration source hash mismatch: {filename}")
    data_files_at_start = sorted((data_root / "CIFAR10" / "processed").glob("*.pt"))
    data_hashes_at_start = {p.name: sha256(p) for p in data_files_at_start}
    expected_data_hashes = cal.get("data", {}).get("processed_files_sha256", {})
    teacher_data_hashes = teacher_meta.get("provenance", {}).get("processed_data_sha256", {})
    if not data_hashes_at_start or data_hashes_at_start != expected_data_hashes or data_hashes_at_start != teacher_data_hashes:
        raise RuntimeError("Processed data hashes disagree with teacher/calibration")
    MTH_diff, source_commit, source_info = author.require_author_source(args.author_repo)
    source_file = Path(source_info["source_file"])
    source_dir = Path(helpers.__file__).resolve().parent
    source_paths = {"evaluator": Path(__file__).resolve(),
                    "protocol": Path(__file__).resolve().parents[1]/'PROTOCOL-v3.md',
                    "numerical_correction": Path(__file__).resolve().parents[1]/'numerical-correction-v3.md',
                    "author_source": args.author_repo.resolve() / source_file,
                    "teacher_builder": Path(helpers.__file__),
                    "calibration_validator": Path(validation.__file__),
                    "converter": Path(hybrid.__file__),
                    "hybrid_base": source_dir / "gated_causal_hybrid.py",
                    "teacher_model": source_dir / "gated_teacher_model.py",
                    "author_verifier": Path(author.__file__)}
    for filename in cal.get("code_sha256", {}):
        source_paths[f"calibration:{filename}"] = source_dir / filename
    if args.reference is not None:source_paths['historical_reference']=args.reference
    if args.canonical_reference is not None:source_paths['canonical_reference']=args.canonical_reference
    source_sha_start = {name: sha256(path) for name, path in source_paths.items()}
    calibration_hash_start = sha256(args.calibration)
    checkpoint_hash_start = sha256(checkpoint)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(41)
    np.random.seed(41)
    if args.device == "cpu": torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("metadata") != cal.get("teacher_metadata"):
        raise RuntimeError("Checkpoint and calibration teacher metadata mismatch")
    model = helpers.build_teacher(state, device).eval()
    thresholds = [cal["calibration"]["channel_thresholds"][site] for site in SITES]
    dataset = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split=args.split)
    expected = EXPECTED_VAL if args.split == "val" else 10000
    if len(dataset) != expected: raise RuntimeError(f"Unexpected {args.split} size: {len(dataset)}")
    eval_count = min(args.limit, expected) if args.limit is not None else expected
    selected = Subset(dataset, range(eval_count)) if args.limit is not None else dataset
    labels_loader = DataLoader(selected, batch_size=128, shuffle=False, num_workers=0)
    conversion_loader = DataLoader(LabelFreeGraphView(selected), batch_size=128, shuffle=False, num_workers=0)
    # Independent sequential loaders preserve canonical indices while keeping y out of conversion.
    label_iter = iter(labels_loader)
    teacher_parts, labels_parts = [], []
    method_keys = [f"ann_T{t}" for t in T_VALUES] + [f"dc_adapt_T{t}" for t in T_VALUES]
    predictions = {key: [] for key in method_keys}
    events, unit_calls, gate_evals, gate_elements = ({k: 0 for k in method_keys} for _ in range(4))
    start = time.monotonic()
    graphs = 0
    for batch_no, conv_batch in enumerate(conversion_loader):
        label_batch = next(label_iter)
        n = int(label_batch.num_graphs)
        labels_parts.append(label_batch.y.view(-1).long().cpu().numpy())
        conv_batch = conv_batch.to(device)
        # Teacher reference and converter both receive a label-free batch.
        teacher = model(conv_batch)
        teacher_parts.append(teacher.cpu().numpy())
        for t in T_VALUES:
            if args.converter in ("ann", "both"):
                # ANN control is exact teacher evaluated at fixed sequential tick count.
                predictions[f"ann_T{t}"].append(teacher.cpu().numpy())
            if args.converter in ("dc_adapt", "both"):
                logits, diag = hybrid.differential_hybrid_forward(model, conv_batch, thresholds, t, MTH_diff)
                predictions[f"dc_adapt_T{t}"].append(logits.cpu().numpy())
                events[f"dc_adapt_T{t}"] += sum(s["nonzero_scalar_events"] for s in diag["sites"])
                unit_calls[f"dc_adapt_T{t}"] += diag["source_warmup_calls"] + diag["active_unit_calls"]
                gate_evals[f"dc_adapt_T{t}"] += diag["analog_gate_evaluations"]
                gate_elements[f"dc_adapt_T{t}"] += diag["analog_gate_edge_channel_elements"]
        graphs += n
        if args.limit is not None and graphs >= args.limit: break
        if graphs % 1280 == 0: print(json.dumps({"graphs": graphs, "elapsed_sec": time.monotonic()-start}), flush=True)
    y, teacher = np.concatenate(labels_parts), np.concatenate(teacher_parts)
    partial = args.limit is not None
    if not partial and graphs != expected: raise RuntimeError(f"Incomplete split: {graphs} != {expected}")
    if partial and graphs != eval_count: raise RuntimeError("Smoke limit mismatch")
    arrays = {k: np.concatenate(v) for k, v in predictions.items() if v}
    teacher_acc = float(np.mean(teacher.argmax(1) == y))
    rows = []
    for key, pred in arrays.items():
        if pred.shape != (graphs,10) or not np.isfinite(pred).all():
            raise RuntimeError(f'Invalid logits for {key}')
        rows.append({"method": key, "accuracy": float(np.mean(pred.argmax(1) == y)),
                     "teacher_agreement": float(np.mean(pred.argmax(1) == teacher.argmax(1))),
                     "logit_mse": float(np.mean((pred-teacher)**2)),
                     "teacher_to_model_kl": float(kl_teacher_to_student(torch.from_numpy(teacher), torch.from_numpy(pred))),
                     "nonzero_scalar_events": events[key], "source_unit_calls": unit_calls[key],
                     "analog_gate_evaluations": gate_evals[key],
                     "analog_gate_edge_channel_elements": gate_elements[key]})
    # Preserve actual completed predictions even when an acceptance assertion
    # fails. These files explicitly have no accepted/complete-evidence status.
    newrun.parent.mkdir(parents=True,exist_ok=True)
    newrun.mkdir(exist_ok=False)
    np.savez_compressed(newrun/'unvalidated-predictions.npz',
                        indices=np.arange(graphs),labels=y,teacher_logits=teacher,**arrays)
    (newrun/'unvalidated-metrics.json').write_text(json.dumps({
        'status':'unvalidated','graphs':graphs,'teacher_accuracy':teacher_acc,
        'rows':rows,'acceptance':'not checked; inspect final report or failure log'},indent=2)+'\n')
    reference_hash = None
    reference_numerics = {}
    if args.split == "val" and not partial:
        if args.reference is None:
            raise RuntimeError("Full validation requires --reference gd_val1.npz for exact E0 reproduction checks")
        reference_hash = sha256(args.reference)
        with np.load(args.reference, allow_pickle=False) as ref:
            if not np.array_equal(ref["indices"], np.arange(EXPECTED_VAL)):
                raise RuntimeError("Reference validation indices are not canonical")
            if not np.array_equal(ref["labels"], y):
                raise RuntimeError("Reference validation labels differ")
            if not np.array_equal(ref["teacher_logits"].argmax(1), teacher.argmax(1)):
                raise RuntimeError("Reference teacher classes differ")
            np.testing.assert_allclose(teacher,ref['teacher_logits'],rtol=1e-5,atol=1e-4,
                                       err_msg='Reference teacher logits differ')
            for new_key, ref_key in (("dc_adapt_T4", "dc_T4"), ("dc_adapt_T8", "dc_T8")):
                if new_key not in arrays:
                    if args.converter=='ann':continue
                    raise RuntimeError("E0 converter rows are required with full validation")
                changed=np.flatnonzero(arrays[new_key].argmax(1)!=ref[ref_key].argmax(1))
                delta=arrays[new_key].astype(np.float64)-ref[ref_key].astype(np.float64)
                tight=np.isclose(arrays[new_key],ref[ref_key],rtol=1e-5,atol=1e-4)
                reference_numerics[new_key]={
                    'max_abs_logit_difference':float(np.abs(delta).max()),
                    'mean_abs_logit_difference':float(np.abs(delta).mean()),
                    'rms_logit_difference':float(np.sqrt(np.mean(delta**2))),
                    'outside_original_tight_tolerance':int((~tight).sum()),
                    'original_tight_logit_criterion_passed':bool(tight.all()),
                    'predicted_classes_exact':bool(len(changed)==0),
                    'historical_class_mismatches':int(len(changed)),
                    'historical_changed_indices':changed.tolist()}
                expected_acc = {"dc_adapt_T4": 0.7072, "dc_adapt_T8": 0.7066}[new_key]
                if float(np.mean(arrays[new_key].argmax(1) == y)) != expected_acc:
                    raise RuntimeError(f"E0 accuracy mismatch for {new_key}")
            ref_report = json.loads(str(ref["report_json"]))
            if ref_report.get("checkpoint_sha256") != checkpoint_hash:
                raise RuntimeError("Reference checkpoint hash differs")
            if ref_report.get("calibration_sha256") != sha256(args.calibration):
                raise RuntimeError("Reference calibration hash differs")
            if abs(teacher_acc - 0.7066) > 1e-12:
                raise RuntimeError(f"E0 teacher accuracy mismatch: {teacher_acc}")
        if args.canonical_reference is None:
            raise RuntimeError('v3 full validation requires frozen deterministic canonical reference')
        with np.load(args.canonical_reference,allow_pickle=False) as canonical:
            for name,actual in {'indices':np.arange(graphs),'labels':y,'teacher_logits':teacher,**arrays}.items():
                if not np.array_equal(actual,canonical[name]):
                    raise RuntimeError('Canonical bitwise repeatability failed: '+name)
    source_sha_end = {name: sha256(path) for name, path in source_paths.items()}
    if source_sha_start != source_sha_end: raise RuntimeError("Evaluation source changed during run")
    data_hashes_at_end = {p.name: sha256(p) for p in sorted((data_root / "CIFAR10" / "processed").glob("*.pt"))}
    if data_hashes_at_start != data_hashes_at_end: raise RuntimeError("Dataset files changed during evaluation")
    calibration_hash = sha256(args.calibration)
    if calibration_hash != calibration_hash_start: raise RuntimeError("Calibration changed during evaluation")
    checkpoint_hash_end = sha256(checkpoint)
    if checkpoint_hash_end != checkpoint_hash_start or checkpoint_hash_end != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation")
    report = {"status": "smoke_only" if partial else "complete", "dataset": "CIFAR10", "split": args.split,
              "graphs": graphs, "expected_graphs": expected, "seed": 41, "converter": args.converter,
              "teacher_accuracy": teacher_acc, "checkpoint_sha256_start": checkpoint_hash_start,
              "checkpoint_sha256_end": checkpoint_hash_end,
              "calibration_sha256_start": calibration_hash_start,
              "calibration_sha256_end": calibration_hash, "reference_sha256": reference_hash,
              "reference_comparison_policy": "PROTOCOL-v3 exact canonical arrays and historical aggregate metrics; historical DC numerical drift reported",
              "canonical_bitwise_verified": args.split=='val' and not partial,
              "reference_numerics": reference_numerics,
              "data_sha256_start": data_hashes_at_start, "data_sha256_end": data_hashes_at_end,
              "teacher_metadata_sha256": sha256(args.teacher_dir / "metadata.json"),
              "source_commit": source_commit, "source_sha256_start": source_sha_start,
              "source_sha256_end": source_sha_end,
              "runtime": {"python": sys.version, "torch": torch.__version__, "device": str(device),
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                          "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
              "elapsed_sec": time.monotonic()-start, "rows": rows,
              "operation_ledger": {"scope": "partial observed counts only; not full compute, latency, energy, or hardware cost",
                  "dense_affine_and_graph_aggregation_ops": None, "memory_traffic_bytes": None,
                  "wall_clock_breakdown_by_method": None},
              "interpretation": "DC adaptation uses analog gates and signed graded source events; ANN rows are the frozen teacher output. Labels are read only by the scoring loader.",
              "smoke_limit": eval_count if partial else None}
    with archive_path.open("xb") as f:
        np.savez_compressed(f, indices=np.arange(graphs), labels=y, teacher_logits=teacher,
                            report_json=np.array(json.dumps(report)), **arrays)
    report["predictions_sha256"] = sha256(archive_path)
    with report_path.open("x", encoding="utf-8") as f: json.dump(report, f, indent=2); f.write("\n")
    print(json.dumps({"status": report["status"], "teacher_accuracy": teacher_acc, "rows": rows}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-dir", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--converter", choices=("ann", "dc_adapt", "both"), default="both")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--author-repo", required=True, type=Path)
    p.add_argument("--reference", type=Path, help="gd_val1.npz; required for complete validation reproduction")
    p.add_argument('--canonical-reference',type=Path,
                   default=Path(__file__).resolve().parents[1]/'inputs/canonical-baseline-val.npz')
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit", type=int, help="explicit smoke run only; never complete evidence")
    evaluate(p.parse_args())


if __name__ == "__main__": main()
