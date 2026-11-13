"""R5: complete the remaining frozen E1 row families on the official CIFAR10 val.

Subcommands (all read-only with respect to prior evidence):

  precheck  - hash/structure/threshold binding gate, no GPU, no dataset pass
  run       - full 5000-graph validation for one seed, 3 rows x T{1,2,4,8}
  verify    - independent recheck of the produced per-sample NPZ and report

Rows added here (protocol-v3 + E1 head addendum, all T1/2/4/8, seeds 41/42/43):
  * dc_adapt_six_site        - historical six-site DC-adapt (E0 converter path),
                               six ReLU sites: four node residual branches + two
                               hidden MLP readout activations.
  * first_analog_shared_theta - first_analog trajectory with the frozen
                               same-vector theta controls.
  * centered_shared_theta    - centered trajectory with the SAME frozen
                               same-vector theta controls.

The shared vector set is produced by the immutable frozen helper
gated_affine_shared_profile_v1.shared_thresholds(doc, 1): elementwise max of
corresponding raw_abs (first_analog) and branch_cumsum (centered) per-channel
thresholds. It is fixed before validation, never tuned against results.

Unchanged frozen helpers used:
  * eval_affine_validation.metric_row / LabelFreeDataset / processed_hashes
  * gated_affine_base_v1.affine_mode_forward / require_author_sources
  * gated_differential_hybrid_v1.differential_hybrid_forward
  * dcgs_fixed_residual_gcn_event_forward_v1.require_author_source
  * gated_affine_shared_profile_v1.shared_thresholds

No complete operation cost, matched wall-clock, energy, Story decision, new test
inference, paper, citation audit or independent review is claimed here. P4 stays
running until all E1-E6 evidence exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.datasets import GNNBenchmarkDataset  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = STUDY.parents[2]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
RR = RUNROOT / "clean-restore-v1" / "research" / "visual-graphs" / "takeover_20261112"
SOURCE = ROOT / "research" / "visual-graphs" / "takeover_20261112"
CANON = RUNROOT / "inputs" / "canonical-baseline-val.npz"
ANN_DIR = RUNROOT / "results" / "e1-ann-multiseed-v1"

MODES = ("dc_adapt_six_site", "first_analog_shared_theta", "centered_shared_theta")
TIMESTEPS = (1, 2, 4, 8)
SIX_SITES = ("gated_0", "gated_1", "gated_2", "gated_3", "mlp_0", "mlp_1")
EXPECTED_VAL, EXPECTED_TRAIN, BATCH_SIZE = 5000, 45000, 128
PROTOCOL_SHA = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
ADDENDUM_SHA = "8bde5832d556c8c7b04ee630f2390703c33a26fa750f196f4dc1c94da61291f6"
CANON_SHA = "7c0727a2902bb2c05a0ec4930734b0bfa05e0905c4b86a6c976155a0c702fcc3"
DCGS_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"

# Exact frozen bindings, recomputed on 2026-11-14 and pinned here. Any drift aborts.
BINDINGS = {
    41: {
        "teacher": RR / "full_teacher_runs" / "gated_cifar10_seed41_full_v1",
        "affine": RR / "affine_cal1.json",
        "six": RR / "full_teacher_runs" / "gated_cifar10_seed41_full_v1" / "gcal_v1.json",
        "teacher_sha": "d26df76d724872edc91ea11055d114935ea173282a8ef049d1b85cc1f5598b40",
        "affine_sha": "1618e3b46f75646ab051229b0d954815b5e28bf1729cbbf2042e4d596766a511",
        "six_sha": "b28485ddfa8ef259ca048f3272662fedc1e4f519b0324cfa514ccf82b2cf8a9e",
        "accuracy": 0.7066,
    },
    42: {
        "teacher": RR / "full_teacher_runs" / "gated_cifar10_seed42_full_v2",
        "affine": RUNROOT / "results" / "e1-calibration-seed42-v1" / "affine_cal_seed42_v1.json",
        "six": RUNROOT / "results" / "e1-calibration-seed42-v1" / "gcal_seed42_v1.json",
        "teacher_sha": "0ad3f3146f94bf73f5f398ef44a1fbc46f0c5924332c4d8fca3d711d0fbff1b6",
        "affine_sha": "07bc86293fd719c683c2f8f74f5763bb35e261796f28ec9e74807f5448bcd81b",
        "six_sha": "9da8ad79fb76fe3e13f0df5fff0b72f54e5ad1bb87bc500858219188f9ca98da",
        "accuracy": 0.703,
    },
    43: {
        "teacher": RR / "full_teacher_runs" / "gated_cifar10_seed43_full_v2",
        "affine": RUNROOT / "results" / "e1-calibration-seed43-v1" / "affine_cal_seed43_v1.json",
        "six": RUNROOT / "results" / "e1-calibration-seed43-v1" / "gcal_seed43_v1.json",
        "teacher_sha": "9d194020ba105c31fc8956fbfc3f3c7556fd50f120f4bb4341eb883189ed8861",
        "affine_sha": "339dd7929a08f6f57bb4ab3ca73e9f1eb27c155421260e815c0ab4a8bd5db21a",
        "six_sha": "a3405ed1ee32a1d07f27ae67682447e95ac14223073be5d3d1aabc0081f21a87",
        "accuracy": 0.7074,
    },
}

CSV_COLUMNS = ("task_id", "agent_id", "run_id", "dataset", "split", "coverage",
               "method", "teacher_seed", "student_seed", "T", "metric", "value",
               "unit", "evidence", "protocol_sha256", "status")
METRIC_NAMES = ("accuracy", "teacher_agreement", "logit_mse", "teacher_to_output_kl")
METRIC_UNITS = {"accuracy": "fraction", "teacher_agreement": "fraction",
                "logit_mse": "mse", "teacher_to_output_kl": "kl"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def default_data_root() -> Path:
    value = os.environ.get("GNN2SNN_DATA_ROOT")
    if not value:
        raise RuntimeError("set GNN2SNN_DATA_ROOT to the official PyG data root")
    return Path(value)


def default_author_repo() -> Path:
    value = os.environ.get("DCGS_REPO")
    return Path(value) if value else Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"


def load_modules(author_repo: Path):
    for p in (CODE, SOURCE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    base = importlib.import_module("eval_affine_validation")
    affine = importlib.import_module("gated_affine_base_v1")
    cal = importlib.import_module("gated_affine_cal_v1")
    hybrid = importlib.import_module("gated_differential_hybrid_v1")
    author = importlib.import_module("dcgs_fixed_residual_gcn_event_forward_v1")
    shared = importlib.import_module("gated_affine_shared_profile_v1")
    teacher_helper = importlib.import_module("gated_frozen_gate_calibrate")
    return base, affine, cal, hybrid, author, shared, teacher_helper


def verify_bindings(seed: int, data_root: Path, modules) -> dict:
    """Hash-only + structural frozen gate; no validation data is opened."""
    base, affine, cal, hybrid, author, shared, teacher_helper = modules
    b = BINDINGS[seed]
    checks = {}
    for key, expected in (("teacher_sha", b["teacher"] / "best.pt"),
                          ("affine_sha", b["affine"]), ("six_sha", b["six"])):
        require(expected.is_file(), f"missing frozen input: {expected.name}")
        checks[rel(expected)] = sha256(expected)
        require(checks[rel(expected)] == b[key], f"frozen hash drift: {expected.name}")
    require(sha256(CANON) == CANON_SHA, "canonical baseline hash drift")
    require(sha256(STUDY / "PROTOCOL-v3.md") == PROTOCOL_SHA, "protocol hash drift")
    require(sha256(STUDY / "PROTOCOL-E1-head-addendum-v1.md") == ADDENDUM_SHA,
            "head addendum hash drift")
    checks[rel(CANON)] = CANON_SHA
    checks[rel(STUDY / "PROTOCOL-v3.md")] = PROTOCOL_SHA
    checks[rel(STUDY / "PROTOCOL-E1-head-addendum-v1.md")] = ADDENDUM_SHA

    metadata = json.loads((b["teacher"] / "metadata.json").read_text(encoding="utf-8"))
    done = json.loads((b["teacher"] / "done.json").read_text(encoding="utf-8"))
    require(metadata.get("dataset") == "CIFAR10", "teacher dataset drift")
    require(metadata.get("official_counts") == {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL},
            "teacher official split counts drift")
    require(int(metadata.get("seed", -1)) == seed, "teacher seed drift")
    require(done.get("status") == "finished", "teacher run not terminal")
    require(done.get("best_checkpoint_sha256") == b["teacher_sha"], "checkpoint/done mismatch")
    require(abs(float(done["best_val_accuracy"]) - b["accuracy"]) < 1e-12,
            "frozen teacher accuracy drift")
    data_hashes = base.processed_hashes(data_root)
    require(data_hashes == metadata["provenance"]["processed_data_sha256"],
            "processed data differ from teacher metadata")

    affine_doc = json.loads(b["affine"].read_text(encoding="utf-8"))
    six_doc = json.loads(b["six"].read_text(encoding="utf-8"))
    require(affine_doc.get("status") == "complete" and affine_doc.get("dataset") == "CIFAR10",
            "affine calibration incomplete")
    require(set(affine_doc["calibration"]["channel_thresholds"]) == set(cal.SITES),
            "affine calibration site set differs from frozen helper")
    require(six_doc.get("status") == "complete", "six-site calibration incomplete")
    require(set(six_doc["calibration"]["channel_thresholds"]) == set(SIX_SITES),
            "six-site calibration site set drift")
    require(int(six_doc["calibration"].get("full_train_graph_count", six_doc["calibration"].get("pass2_graphs", -1))) == EXPECTED_TRAIN,
            "six-site calibration did not cover all 45000 train graphs")
    require(int(six_doc["calibration"].get("passes", -1)) == 2,
            "six-site calibration is not two full passes")
    require(float(six_doc["calibration"].get("quantile", -1)) == 0.999,
            "six-site calibration quantile drift")
    for site in SIX_SITES:
        v = six_doc["calibration"]["channel_thresholds"][site]
        require(v and all(float(x) > 0 and np.isfinite(float(x)) for x in v),
                f"invalid six-site threshold at {site}")

    # Frozen shared-theta vectors: immutable helper AND independent recomputation.
    frozen_vectors = shared.shared_thresholds(affine_doc, 1)
    raw = cal.profile_thresholds(affine_doc, "first_analog", 1)
    cumulative = cal.profile_thresholds(affine_doc, "centered", 1)
    require(len(frozen_vectors) == len(raw) == len(cumulative) == 6,
            "shared-theta vector count drift")
    recomputed = [torch.maximum(a, c) for a, c in zip(raw, cumulative)]
    require(all(torch.equal(x, y) for x, y in zip(frozen_vectors, recomputed)),
            "shared-theta helper differs from independent channelwise max")
    require(all(int(v.numel()) == 70 for v in frozen_vectors), "shared-theta width drift")
    six_widths = [len(six_doc["calibration"]["channel_thresholds"][s]) for s in SIX_SITES]
    require(six_widths == [70, 70, 70, 70, 35, 17], f"six-site width drift: {six_widths}")

    return {
        "seed": seed,
        "teacher_dir": rel(b["teacher"]),
        "affine_calibration": rel(b["affine"]),
        "six_site_calibration": rel(b["six"]),
        "frozen_hashes": checks,
        "data_hashes": data_hashes,
        "metadata": metadata,
        "done": done,
        "affine_doc": affine_doc,
        "six_doc": six_doc,
        "shared_theta_vectors": frozen_vectors,
    }


def author_provenance(author_repo: Path, modules) -> dict:
    """Bind both pinned source verifiers and prove they resolve to one class."""
    _base, affine, _cal, _hybrid, author, _shared, _teacher_helper = modules
    MTH_diff, MTH_diff_line, prov = affine.require_author_sources(author_repo)
    MTH_diff_e0, commit, prov_e0 = author.require_author_source(author_repo)
    require(commit == DCGS_COMMIT, "pinned DCGS commit drift")
    require(MTH_diff is MTH_diff_e0,
            "E0 and E1 source verifiers resolved to different MTH_diff classes")
    safe = {
        "commit": commit,
        "affine_base_source_files": {
            k: {"path": v["path"], "sha256": v["sha256"]}
            for k, v in prov["source_files"].items()},
        "e0_source_file": {"path": prov_e0["source_file"],
                           "sha256": prov_e0["source_file_sha256"],
                           "head_blob_oid": prov_e0["source_head_blob_oid"]},
        "worktree_clean_except_ignored_bytecode": True,
        "e0_e1_class_identity": True,
    }
    return {"MTH_diff": MTH_diff, "MTH_diff_line": MTH_diff_line, "safe": safe}


def precheck(data_root: Path, author_repo: Path, out: Path) -> None:
    modules = load_modules(author_repo)
    gate = {}
    for seed in (41, 42, 43):
        b = verify_bindings(seed, data_root, modules)
        gate[str(seed)] = {
            "frozen_files_verified": len(b["frozen_hashes"]),
            "processed_files_verified": len(b["data_hashes"]),
            "teacher_accuracy": b["done"]["best_val_accuracy"],
            "affine_sites": len(b["affine_doc"]["calibration"]["channel_thresholds"]),
            "six_sites": list(SIX_SITES),
            "six_widths": [len(b["six_doc"]["calibration"]["channel_thresholds"][s]) for s in SIX_SITES],
            "shared_theta_vector_count": len(b["shared_theta_vectors"]),
            "shared_theta_channelwise_max_equivalent": True,
        }
    ap = author_provenance(author_repo, modules)
    gate["author_provenance"] = ap["safe"]
    gate["status"] = "PASS"
    gate["check"] = "e1_residual_precheck_v1"
    gate["note"] = "hash/structure/threshold binding only; no validation pass, no scientific result"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


def thresholds_for(mode: str, binding: dict, device, dtype) -> list[torch.Tensor]:
    if mode == "dc_adapt_six_site":
        ct = binding["six_doc"]["calibration"]["channel_thresholds"]
        return [torch.as_tensor(ct[s], dtype=dtype, device=device) for s in SIX_SITES]
    return [v.to(device=device, dtype=dtype) for v in binding["shared_theta_vectors"]]


@torch.no_grad()
def run(args) -> None:
    data_root = (args.data_root or default_data_root()).resolve()
    author_repo = (args.author_repo or default_author_repo()).resolve()
    seed = args.seed
    modules = load_modules(author_repo)
    base, affine, cal, hybrid, author, shared, teacher_helper = modules
    binding = verify_bindings(seed, data_root, modules)
    ap = author_provenance(author_repo, modules)
    MTH_diff, MTH_diff_line = ap["MTH_diff"], ap["MTH_diff_line"]

    out_dir = args.output_dir.resolve()
    require(not out_dir.exists(), f"refusing existing output directory: {out_dir.name}")
    require(out_dir.parent == (RUNROOT / "results").resolve(),
            "output must be directly under the run-root results directory")

    source_paths = {
        "evaluator": Path(__file__).resolve(),
        "base_evaluator": Path(base.__file__).resolve(),
        "affine_base": Path(affine.__file__).resolve(),
        "calibration_helper": Path(cal.__file__).resolve(),
        "differential_hybrid": Path(hybrid.__file__).resolve(),
        "e0_source_verifier": Path(author.__file__).resolve(),
        "shared_profile_helper": Path(shared.__file__).resolve(),
        "teacher_builder": Path(teacher_helper.__file__).resolve(),
        "teacher_model": SOURCE / "gated_teacher_model.py",
        "frozen_protocol": STUDY / "PROTOCOL-v3.md",
        "head_addendum": STUDY / "PROTOCOL-E1-head-addendum-v1.md",
        "canonical_baseline": CANON,
        "affine_calibration": BINDINGS[seed]["affine"],
        "six_site_calibration": BINDINGS[seed]["six"],
    }
    # The pinned DCGS checkout lives OUTSIDE the repository root, so its source
    # files cannot be expressed as ROOT-relative ledger keys (verify() resolves
    # every source_sha256_start key as ROOT / name). Bind them with de-identified
    # short keys instead; their full identity (relative path, sha256, head blob
    # oid, pinned commit, E0/E1 class identity) is recorded in
    # author_source_provenance and re-checked at the end of this run.
    external_source_paths = {
        "e0_relu_neuron": author_repo / "neurons" / "relu_neuron.py",
        "e0_line_neuron": author_repo / "neurons" / "line_neuron.py",
    }
    source_start = {rel(p): sha256(p) for p in source_paths.values()}
    external_start = {k: sha256(v) for k, v in external_source_paths.items()}
    data_start = base.processed_hashes(data_root)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    teacher_dir = BINDINGS[seed]["teacher"]
    state = torch.load(teacher_dir / "best.pt", map_location=device, weights_only=False)
    require(state.get("metadata") == binding["metadata"], "checkpoint metadata mismatch")
    model = teacher_helper.build_teacher(state, device).eval()
    require(len(model.layers) == 4, "expected frozen four-layer GatedGCN teacher")
    fc = model.MLP_layer.FC_layers
    actual_widths = [int(l.out_channels) for l in model.layers] + [int(x.out_features) for x in fc[:-1]]
    require(actual_widths == [70, 70, 70, 70, 35, 17],
            f"six-site model width mismatch: {actual_widths}")

    dataset = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    require(len(dataset) == EXPECTED_VAL, "unexpected official validation count")
    labels_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model_loader = DataLoader(base.LabelFreeDataset(dataset), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    label_iter = iter(labels_loader)

    keys = [(m, t) for m in MODES for t in TIMESTEPS]
    prediction_parts = {k: [] for k in keys}
    events = {k: {} for k in keys}
    ledger = {k: {"source_warmup_calls": 0, "source_active_unit_calls": 0,
                  "head_calls": 0, "analog_gate_evaluations": 0} for k in keys}
    label_parts, teacher_parts, indices_parts = [], [], []
    graphs, start_time = 0, time.monotonic()
    resource_probe = None

    for batch_no, batch in enumerate(model_loader):
        label_batch = next(label_iter)
        n = int(batch.num_graphs)
        y = label_batch.y.view(-1).long().cpu().numpy()
        label_parts.append(y)
        indices_parts.append(np.arange(graphs, graphs + n, dtype=np.int64))
        batch = batch.to(device)
        teacher_logits = model(batch)
        teacher_repeat = model(batch)
        require(torch.equal(teacher_logits, teacher_repeat),
                "teacher logits not bitwise deterministic across two forwards")
        teacher_parts.append(teacher_logits.detach().cpu().numpy())

        for mode, timestep in keys:
            th = thresholds_for(mode, binding, device, batch.x.dtype)
            key = (mode, timestep)
            if mode == "dc_adapt_six_site":
                logits, diag = hybrid.differential_hybrid_forward(
                    model, batch, th, timestep, MTH_diff)
                ledger[key]["source_warmup_calls"] += int(diag["source_warmup_calls"])
                ledger[key]["source_active_unit_calls"] += int(diag["active_unit_calls"])
                ledger[key]["analog_gate_evaluations"] += int(diag["analog_gate_evaluations"])
                ledger[key]["head_calls"] += int(timestep)
                site_names = [f"site_{i}" for i in range(len(diag["sites"]))]
                site_stats = diag["sites"]
            else:
                base_mode = "first_analog" if mode == "first_analog_shared_theta" else "centered"
                out = affine.affine_mode_forward(model, batch, th, timestep, base_mode,
                                                 MTH_diff, MTH_diff_line)
                logits = out["logits"]
                ledger[key]["source_warmup_calls"] += int(out["source_warmup_calls"])
                ledger[key]["source_active_unit_calls"] += int(out["active_unit_calls"])
                ledger[key]["head_calls"] += int(out["head_calls"])
                # Dense gates are recomputed for every layer at every tick, exactly
                # as in the E0 converter's analog_gate_evaluations definition.
                ledger[key]["analog_gate_evaluations"] += len(model.layers) * int(timestep)
                site_names, site_stats = list(out["site_names"]), list(out["site_stats"])
            prediction_parts[key].append(logits.detach().cpu().numpy())
            for name, stat in zip(site_names, site_stats):
                events[key][name] = events[key].get(name, 0) + int(stat["nonzero_scalar_events"])
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
            require(resource_probe["elapsed_sec"] * 40 < 80 * 3600,
                    "projected fixed grid exceeds the E1 compute ceiling")
        if batch_no % 10 == 9 or graphs == EXPECTED_VAL:
            print(json.dumps({"graphs": graphs, "elapsed_sec": time.monotonic() - start_time}),
                  flush=True)

    labels = np.concatenate(label_parts)
    teacher = np.concatenate(teacher_parts)
    indices_all = np.concatenate(indices_parts)
    require(graphs == EXPECTED_VAL and np.array_equal(indices_all, np.arange(EXPECTED_VAL)),
            "incomplete or noncanonical validation traversal")
    teacher_accuracy = float(np.mean(teacher.argmax(1) == labels))
    require(abs(teacher_accuracy - binding["done"]["best_val_accuracy"]) < 1e-12,
            "teacher accuracy differs from frozen done.json")
    with np.load(CANON, allow_pickle=False) as ref:
        require(np.array_equal(indices_all, ref["indices"]), "canonical index drift")
        require(np.array_equal(labels, ref["labels"]), "canonical label drift")
        canon_arrays = {k: ref[k].copy() for k in ("teacher_logits", "dc_adapt_T4", "dc_adapt_T8")}
    if seed == 41:
        require(np.array_equal(teacher, canon_arrays["teacher_logits"]),
                "seed41 canonical teacher logits differ")

    predictions = {k: np.concatenate(v) for k, v in prediction_parts.items()}
    canonical_dc = {}
    if seed == 41:
        for t in (4, 8):
            arr = predictions[("dc_adapt_six_site", t)]
            canonical_dc[f"T{t}"] = {
                "bitwise_equal_to_E0_canonical": bool(np.array_equal(arr, canon_arrays[f"dc_adapt_T{t}"])),
                "max_abs_difference": float(np.abs(arr.astype(np.float64) -
                                                  canon_arrays[f"dc_adapt_T{t}"].astype(np.float64)).max()),
            }
            require(canonical_dc[f"T{t}"]["bitwise_equal_to_E0_canonical"],
                    f"seed41 six-site DC-adapt T{t} differs bitwise from E0 canonical")

    source_end = {rel(p): sha256(p) for p in source_paths.values()}
    external_end = {k: sha256(v) for k, v in external_source_paths.items()}
    data_end = base.processed_hashes(data_root)
    require(source_start == source_end, "bound source/calibration changed during evaluation")
    require(external_start == external_end, "pinned DCGS source changed during evaluation")
    require(data_start == data_end, "official data changed during evaluation")
    require(author_provenance(author_repo, modules)["safe"] == ap["safe"],
            "pinned author source changed during evaluation")

    rows = []
    metric_lines = []
    for mode, timestep in keys:
        key = (mode, timestep)
        m = base.metric_row(predictions[key], teacher, labels)
        rows.append({"mode": mode, "T": timestep, **m,
                     "per_site_nonzero_scalar_events": events[key]})
        for name in METRIC_NAMES:
            metric_lines.append({
                "task_id": "visual-conversion", "agent_id": "20261113-wb-resume",
                "run_id": "e1-residual-v1", "dataset": "CIFAR10", "split": "val",
                "coverage": "5000_all_official_val", "method": mode,
                "teacher_seed": str(seed), "student_seed": "", "T": str(timestep),
                "metric": name, "value": repr(m[name]), "unit": METRIC_UNITS[name],
                "evidence": f"{rel(out_dir)}/{mode}_T{timestep}.npz",
                "protocol_sha256": PROTOCOL_SHA,
                "status": "measured_full_official_validation",
            })

    shared_json = json.dumps([v.tolist() for v in binding["shared_theta_vectors"]],
                             separators=(",", ":"), sort_keys=True)
    report = {
        "status": "complete",
        "run": "e1_residual_v1",
        "dataset": "CIFAR10", "split": "all official validation",
        "graphs": graphs, "seed": seed, "threshold_scale": 1,
        "historical_validation_exposure": True, "historical_test_exposure": True,
        "test_inference_performed": False,
        "modes": list(MODES), "timesteps": list(TIMESTEPS),
        "affine_calibration_sha256": BINDINGS[seed]["affine_sha"],
        "six_site_calibration_sha256": BINDINGS[seed]["six_sha"],
        "checkpoint_sha256": BINDINGS[seed]["teacher_sha"],
        "canonical_baseline_sha256": CANON_SHA,
        "frozen_expected_teacher_accuracy": binding["done"]["best_val_accuracy"],
        "data_sha256_start": data_start, "data_sha256_end": data_end,
        "source_sha256_start": source_start, "source_sha256_end": source_end,
        "external_author_source_sha256_start": external_start,
        "external_author_source_sha256_end": external_end,
        "author_source_provenance": ap["safe"],
        "shared_theta_binding": {
            "rule": "elementwise max(raw_abs first_analog, branch_cumsum centered) per channel, scale 1",
            "frozen_helper": "gated_affine_shared_profile_v1.shared_thresholds",
            "independent_recomputation_bitwise_equal": True,
            "vectors": 6, "width": 70,
            "vectors_sha256": hashlib.sha256(shared_json.encode("utf-8")).hexdigest(),
            "fixed_before_validation": True,
        },
        "threshold_binding": {
            "dc_adapt_six_site": "six frozen full-train ReLU-site vectors gated_0..3, mlp_0, mlp_1",
            "first_analog_shared_theta": "same six shared-theta vectors, signed MTH_diff_line raw coordinates, exact first analog layer",
            "centered_shared_theta": "same six shared-theta vectors, nonnegative MTH_diff cumulative residual coordinates",
        },
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
            "per_method": {f"{m}_T{t}": ledger[(m, t)] for m, t in keys},
            "dense_mac_or_flop_count": None, "memory_traffic_bytes": None,
            "hardware_latency_or_energy": None},
        "canonical_dc_adapt_reproduction": canonical_dc,
        "alias_and_scope": {
            "dc_adapt_six_site": "E0 historical converter; analog edge controller each tick; six graded source sites including two MLP readout sites; NOT the four-site analog-head control",
            "shared_theta_controls": "additional controls only; neither replaces the strongest matched conversion control nor counts as an independent method win",
        },
        "limitations": [
            "No complete known-operation cost, matched wall-clock latency or energy evidence; the E1 method-advantage criterion remains NOT EVALUABLE.",
            "No Story decision, no new test inference, no paper, no citation audit, no independent review.",
            "Accuracy is descriptive; no three-seed significance or confidence interval is claimed.",
        ],
        "prediction_archives_sha256": {},
    }

    out_dir.mkdir(parents=False, exist_ok=False)
    for mode, timestep in keys:
        npz = out_dir / f"{mode}_T{timestep}.npz"
        with npz.open("xb") as f:
            np.savez_compressed(f, indices=indices_all, labels=labels,
                                teacher_logits=teacher, logits=predictions[(mode, timestep)])
        report["prediction_archives_sha256"][f"{mode}_T{timestep}"] = sha256(npz)
    with (out_dir / "report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    with (out_dir / "METRICS.csv").open("x", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(metric_lines)
    print(json.dumps({"status": "complete", "seed": seed, "graphs": graphs,
                      "teacher_accuracy": teacher_accuracy,
                      "canonical_dc_adapt_reproduction": canonical_dc,
                      "rows": [{k: r[k] for k in ("mode", "T", "accuracy")} for r in rows]},
                     indent=2), flush=True)


@torch.no_grad()
def smoke(args) -> None:
    """Two-batch correctness probe only; writes no evidence and claims no result."""
    data_root = (args.data_root or default_data_root()).resolve()
    author_repo = (args.author_repo or default_author_repo()).resolve()
    modules = load_modules(author_repo)
    base, affine, cal, hybrid, author, shared, teacher_helper = modules
    seed = args.seed
    binding = verify_bindings(seed, data_root, modules)
    ap = author_provenance(author_repo, modules)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    state = torch.load(BINDINGS[seed]["teacher"] / "best.pt", map_location=device,
                       weights_only=False)
    model = teacher_helper.build_teacher(state, device).eval()
    dataset = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    loader = DataLoader(base.LabelFreeDataset(dataset), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    out = {}
    canonical_check = {}
    if seed == 41:
        with np.load(CANON, allow_pickle=False) as ref:
            canon_dc = {t: ref[f"dc_adapt_T{t}"].copy() for t in (4, 8)}
    dc_parts = {4: [], 8: []}
    for batch_no, batch in enumerate(loader):
        if batch_no >= 2:
            break
        batch = batch.to(device)
        teacher = model(batch)
        for mode in MODES:
            th = thresholds_for(mode, binding, device, batch.x.dtype)
            t = 4
            if mode == "dc_adapt_six_site":
                logits, diag = hybrid.differential_hybrid_forward(model, batch, th, t,
                                                                  ap["MTH_diff"])
                extra = {"sites": len(diag["sites"])}
                if seed == 41:
                    dc_parts[4].append(logits.detach().cpu().numpy())
                    logits8, _ = hybrid.differential_hybrid_forward(
                        model, batch, th, 8, ap["MTH_diff"])
                    dc_parts[8].append(logits8.detach().cpu().numpy())
            else:
                base_mode = "first_analog" if mode == "first_analog_shared_theta" else "centered"
                r = affine.affine_mode_forward(model, batch, th, t, base_mode,
                                               ap["MTH_diff"], ap["MTH_diff_line"])
                logits, extra = r["logits"], {"sites": len(r["site_names"])}
            require(logits.shape == teacher.shape and bool(torch.isfinite(logits).all()),
                    f"invalid smoke logits for {mode}")
            out[mode] = {"shape": list(logits.shape), "finite": True, **extra}
    require(len(out) == len(MODES), "smoke did not cover all three rows")
    if seed == 41:
        for t in (4, 8):
            actual = np.concatenate(dc_parts[t])
            expected = canon_dc[t][:actual.shape[0]]
            canonical_check[f"T{t}"] = {
                "bitwise_equal_to_E0_canonical_prefix": bool(np.array_equal(actual, expected)),
                "max_abs_difference": float(np.abs(actual.astype(np.float64) -
                                                   expected.astype(np.float64)).max()),
            }
    print(json.dumps({"smoke": "PASS", "seed": seed, "batches": 2,
                      "modes": out, "canonical_dc_prefix": canonical_check,
                      "note": "no evidence written; no result claimed"}, indent=2))


def verify(args) -> None:
    """Independent post-run recheck; writes a public-safe JSON report."""
    author_repo = (args.author_repo or default_author_repo()).resolve()
    modules = load_modules(author_repo)
    base = modules[0]
    external_paths = {
        "e0_relu_neuron": author_repo / "neurons" / "relu_neuron.py",
        "e0_line_neuron": author_repo / "neurons" / "line_neuron.py",
    }
    results = {}
    for seed in (41, 42, 43):
        out_dir = RUNROOT / "results" / f"e1-residual-seed{seed}-v1"
        doc = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        require(doc["status"] == "complete" and doc["graphs"] == EXPECTED_VAL,
                f"seed{seed} report incomplete")
        require(int(doc["seed"]) == seed, f"seed{seed} report seed mismatch")
        require(doc["source_sha256_start"] == doc["source_sha256_end"],
                f"seed{seed} source ledger changed")
        require(doc["data_sha256_start"] == doc["data_sha256_end"],
                f"seed{seed} data ledger changed")
        for name, expected in doc["source_sha256_start"].items():
            require(sha256(ROOT / name) == expected, f"seed{seed} source drift: {name}")
        ext_start = doc.get("external_author_source_sha256_start", {})
        require(ext_start == doc.get("external_author_source_sha256_end", {}),
                f"seed{seed} pinned DCGS source ledger changed")
        require(set(ext_start) == set(external_paths),
                f"seed{seed} unexpected external source key set")
        for name, expected in ext_start.items():
            require(sha256(external_paths[name]) == expected,
                    f"seed{seed} pinned DCGS source drift: {name}")
        with np.load(CANON, allow_pickle=False) as ref:
            ci, cl, ct = ref["indices"].copy(), ref["labels"].copy(), ref["teacher_logits"].copy()
            dc4, dc8 = ref["dc_adapt_T4"].copy(), ref["dc_adapt_T8"].copy()
        rows = {}
        for r in doc["rows"]:
            key = (r["mode"], int(r["T"]))
            npz = out_dir / f"{r['mode']}_T{r['T']}.npz"
            require(sha256(npz) == doc["prediction_archives_sha256"][f"{r['mode']}_T{r['T']}"],
                    f"seed{seed} NPZ hash mismatch: {npz.name}")
            with np.load(npz, allow_pickle=False) as z:
                require(set(z.files) == {"indices", "labels", "teacher_logits", "logits"},
                        f"seed{seed} unexpected NPZ keys")
                require(np.array_equal(z["indices"], ci) and np.array_equal(z["labels"], cl),
                        f"seed{seed} canonical index/label drift")
                tl, lg = z["teacher_logits"].copy(), z["logits"].copy()
            if seed == 41:
                require(np.array_equal(tl, ct), "seed41 canonical teacher logits drift")
            m = base.metric_row(lg, tl, cl)
            for name in METRIC_NAMES:
                require(abs(float(m[name]) - float(r[name])) < 1e-9,
                        f"seed{seed} {key} metric {name} not recomputable")
            rows[key] = {"accuracy": m["accuracy"], "logits": lg}
        if seed == 41:
            require(np.array_equal(rows[("dc_adapt_six_site", 4)]["logits"], dc4),
                    "seed41 six-site DC-adapt T4 differs from E0 canonical")
            require(np.array_equal(rows[("dc_adapt_six_site", 8)]["logits"], dc8),
                    "seed41 six-site DC-adapt T8 differs from E0 canonical")
        # Shared-theta controls must not be trivial aliases of the frozen rows.
        seed41_affine = STUDY / "results" / "e1-seed41-affine-v1"
        alias_checks = {}
        if seed == 41 and seed41_affine.is_dir():
            for mode, original in (("first_analog_shared_theta", "first_analog"),
                                   ("centered_shared_theta", "centered")):
                p = seed41_affine / f"{original}_T4.npz"
                if p.is_file():
                    with np.load(p, allow_pickle=False) as z:
                        alias_checks[mode] = bool(np.array_equal(
                            rows[(mode, 4)]["logits"], z["logits"]))
        results[str(seed)] = {
            "rows_verified": len(rows),
            "npz_hashes_verified": len(rows),
            "canonical_indices_labels_bitwise": True,
            "accuracy_recomputable": True,
            "shared_theta_equals_frozen_mode_T4": alias_checks,
            "seed41_teacher_canonical_bitwise": seed == 41,
        }
    report = {"check": "verify_e1_residual_v1", "status": "PASS",
              "per_seed": results,
              "note": "independent recheck of published artifacts only; no scientific acceptance claim"}
    out = RUNROOT / "logs" / "e1-residual-verify-v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("precheck")
    pc.add_argument("--data-root", type=Path, default=None)
    pc.add_argument("--author-repo", type=Path, default=None)
    pc.add_argument("--out", type=Path, default=RUNROOT / "logs" / "e1-residual-precheck-v1.json")
    rn = sub.add_parser("run")
    rn.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    rn.add_argument("--output-dir", type=Path, required=True)
    rn.add_argument("--data-root", type=Path, default=None)
    rn.add_argument("--author-repo", type=Path, default=None)
    rn.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    sm = sub.add_parser("smoke")
    sm.add_argument("--seed", type=int, choices=(41, 42, 43), default=41)
    sm.add_argument("--data-root", type=Path, default=None)
    sm.add_argument("--author-repo", type=Path, default=None)
    sm.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    vf = sub.add_parser("verify")
    vf.add_argument("--author-repo", type=Path, default=None)
    args = p.parse_args()
    if args.command == "smoke":
        smoke(args)
    elif args.command == "precheck":
        precheck((args.data_root or default_data_root()).resolve(),
                 (args.author_repo or default_author_repo()).resolve(), args.out)
    elif args.command == "run":
        run(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
