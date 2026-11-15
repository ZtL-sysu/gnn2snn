"""Fixed-budget six-arm temporal first-site controller calibration.

This runner fits only the 584 controller coefficients. It reuses the frozen
strong-local offsets and never loads the official test split.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
CRITICAL = HERE.parent / "20261114_critical"
sys.path.insert(0, str(CRITICAL))
sys.path.insert(0, str(HERE))
import analyze_allocation as historical
import calibration_core as core
from decision_objectives import decision_loss
from run_objectives import local_loss
from temporal_core import TemporalController, temporal_cache, temporal_forward

ARMS = (
    ("nongraph_local", "nongraph", "local"),
    ("graph_local", "graph", "local"),
    ("nongraph_decision", "nongraph", "decision"),
    ("graph_decision", "graph", "decision"),
    ("shuffled_decision", "shuffled", "decision"),
    ("toppin_decision", "toppin", "decision"),
)
ENDPOINTS = (1, 2)
SEED = 20261115
FIT_STEPS = 64
LR = 0.01
BATCH_SIZE = 32
TRAIN_COUNT = 128
VAL_COUNT = 512
RUN_CAP_SECONDS = 600


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _check_deadline(started: float, output: Path) -> None:
    if time.monotonic() - started > RUN_CAP_SECONDS:
        _save_json(output / "TIMEOUT.json", {"status": "ten-minute cap reached; do not restart", "elapsed_seconds": time.monotonic() - started})
        raise TimeoutError("temporal fit run exceeded its ten-minute cap")


def _load_controller_state(controller: torch.nn.Module, arrays: dict[str, np.ndarray]) -> None:
    state = {k: torch.as_tensor(v, device=next(controller.parameters()).device,
                                dtype=next(controller.parameters()).dtype)
             for k, v in arrays.items()}
    controller.load_state_dict(state, strict=True)


def _objective(kind: str, logits: torch.Tensor, sites, teacher_logits: torch.Tensor, teacher_sites):
    if kind == "local":
        return local_loss(sites, teacher_sites)
    if kind == "decision":
        return decision_loss(logits, teacher_logits, "logits")
    raise ValueError(kind)


def _pair_loss(kind: str, net, batch, cache, graph_ids, controller, teacher_logits, teacher_sites):
    z1, s1, tr1 = temporal_forward(net, batch, 1, controller, cache, graph_ids)
    z2, s2, tr2 = temporal_forward(net, batch, 2, controller, cache, graph_ids)
    l1 = _objective(kind, z1, s1, teacher_logits, teacher_sites)
    l2 = _objective(kind, z2, s2, teacher_logits, teacher_sites)
    if not torch.equal(tr1["choice_count"], tr2["choice_count"]):
        raise AssertionError("T1 and T2 prefix runs made different first-site replacement counts")
    _assert_first_decode(tr1, tr2, require_zero=True)
    return (l1 + l2) / 2, l1, l2, tr1, tr2


@torch.no_grad()
def _full_train_objective(kind, net, controller, batches, mode, cache_by_batch):
    losses = []
    for (batch, teacher_logits, teacher_sites, ids), cache in zip(batches, cache_by_batch):
        loss, _, _, tr1, tr2 = _pair_loss(kind, net, batch, cache, ids, controller, teacher_logits, teacher_sites)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite full-train hard objective")
        _assert_first_decode(tr1, tr2, require_zero=True)
        losses.append(float(loss))
    return float(np.mean(losses))


def _assert_first_decode(trace1, trace2, require_zero: bool) -> None:
    for trace in (trace1, trace2):
        value = trace["first_decode_error_max"]
        if not torch.isfinite(value).all():
            raise RuntimeError("nonfinite first-prefix decode error")
        if require_zero and torch.any(value != 0):
            raise RuntimeError("temporal intervention broke exact first-prefix decode")


def _cache(batch, mode, graph_ids):
    if mode == "nongraph":
        return None
    return temporal_cache(batch, mode, graph_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MNIST", "CIFAR10"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda")

    source_paths = [
        Path(__file__), HERE / "temporal_core.py", HERE / "prefix_codec.py",
        HERE / "TEMPORAL_METHOD_PROTOCOL.md", HERE / "intervention.py", HERE / "swap_predictor.py", HERE / "run_objectives.py",
        HERE / "decision_objectives.py", CRITICAL / "analyze_allocation.py",
        Path(core.__file__).resolve(),
    ]
    missing = [str(p) for p in source_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("required source snapshots missing: " + ", ".join(missing))
    source_dir = args.output / "source"
    source_dir.mkdir()
    for path in source_paths:
        shutil.copyfile(path, source_dir / path.name)
    source_hashes = {path.name: _sha(source_dir / path.name) for path in source_paths}

    teacher_dir = CRITICAL / "runs" / "allocation" / f"{args.dataset}41"
    meta, _, _, folded, loaded, calibration, data_root = historical.load_teacher(
        args.dataset, 41, historical.DEFAULT_ASSETS, teacher_dir, device)
    if len(meta["train_indices"]) < TRAIN_COUNT or len(meta["val_indices"]) != VAL_COUNT:
        raise ValueError("expected fixed first128 train and fixed validation512 indices")
    train_indices = list(meta["train_indices"][:TRAIN_COUNT])
    val_indices = list(meta["val_indices"])
    baseline_path = HERE / "runs" / "objectives" / f"{args.dataset}41" / "local_t1_parameters.npz"
    with np.load(baseline_path, allow_pickle=False) as stored:
        baseline_arrays = {key: stored[key].copy() for key in stored.files}
    if set(baseline_arrays) != {f"offset_{i}" for i in range(6)}:
        raise ValueError("strong-local parameter file must contain exactly six offsets")

    initial = core.CalibratedGCN(folded, loaded.thresholds).to(device).eval()
    for i, offset in enumerate(initial.offsets):
        value = torch.as_tensor(baseline_arrays[f"offset_{i}"], device=device, dtype=offset.dtype)
        if value.shape != offset.shape:
            raise ValueError(f"baseline offset_{i} shape mismatch")
        offset.data.copy_(value)
        offset.requires_grad_(False)
    initial.requires_grad_(False)

    train_data = GNNBenchmarkDataset(root=str(data_root), name=args.dataset, split="train")
    val_data = GNNBenchmarkDataset(root=str(data_root), name=args.dataset, split="val")
    train_batches = []
    for batch_no, raw in enumerate(DataLoader(train_data[train_indices], batch_size=BATCH_SIZE, shuffle=False)):
        batch = raw.cuda()
        graph_ids = train_indices[batch_no * BATCH_SIZE:batch_no * BATCH_SIZE + batch.num_graphs]
        if hasattr(batch, "y"):
            del batch.y
        with torch.no_grad():
            teacher_logits, teacher_sites = core.teacher_sites(folded, batch)
        train_batches.append((batch, teacher_logits.detach(), [x.detach() for x in teacher_sites], graph_ids))
    if len(train_batches) != 4:
        raise ValueError("fixed 128 graph training set must yield four batches of 32")
    val_batches = list(DataLoader(val_data[val_indices], batch_size=BATCH_SIZE, shuffle=False))

    # Verify inherited strong-local logits bitwise before fitting.
    baseline_root = HERE / "runs" / "objectives" / f"{args.dataset}41"
    expected_by_t = {}
    for baseline_T in ENDPOINTS:
        with np.load(baseline_root / f"local_t1_T{baseline_T}.npz", allow_pickle=False) as ref:
            if not np.array_equal(ref["indices"], np.asarray(val_indices)):
                raise ValueError(f"historical local T{baseline_T} endpoint indices differ from fixed validation indices")
            expected_by_t[baseline_T] = ref["logits"].copy()
    with torch.no_grad():
        for baseline_T in ENDPOINTS:
            offset = 0
            for raw in val_batches:
                batch = raw.clone().cuda()
                if hasattr(batch, "y"):
                    del batch.y
                baseline_logits, _ = initial(batch, baseline_T)
                count = batch.num_graphs
                if not torch.equal(baseline_logits.cpu(), torch.as_tensor(expected_by_t[baseline_T][offset:offset + count])):
                    raise AssertionError(f"frozen strong-local T{baseline_T} logits are not bitwise identical to historical baseline")
                offset += count

    meta_record = dict(
        dataset=args.dataset, teacher_seed=41, train_indices=train_indices, val_indices=val_indices,
        train_count=TRAIN_COUNT, val_count=VAL_COUNT, batch_size=BATCH_SIZE,
        checkpoint_sha256=meta["checkpoint_sha256"], calibration_sha256=meta["calibration_sha256"],
        data_hashes=calibration["data"]["processed_files_sha256"],
        baseline_parameters_sha256=_sha(baseline_path), source_sha256=source_hashes,
        test_loaded=False, labels_used_for_training=False, teacher_logits_sites_cached_offline=True,
        arms=[name for name, _, _ in ARMS], mode_and_objective={n: {"mode": m, "objective": o} for n, m, o in ARMS},
        controller_parameters=584, trainable_parameters_per_arm=584, optimizer="Adam", steps=FIT_STEPS,
        lr=LR, clamping=False, model_selection="fixed final 64-step parameters; no best-step selection",
        objective="mean of hard-forward T1 and T2 local normalized 6-site MSE or decision logits loss",
        all_prefixes_share_baseline_offsets=True, online_teacher=False, online_downstream_lookahead=False,
        fixed_initial_offsets=True, qualification="single fixed seed, validation512 selection-exposed; no independent confirmation")
    meta_record["prefix_diagnostic_semantics"] = (
        "first_decode_error_max is measured from decoded sites at T2; T1 trace is a zero placeholder because no next tick exists")
    _save_json(args.output / "metadata.json", meta_record)

    all_results = []
    trained = {}
    for arm_name, mode, objective in ARMS:
        _check_deadline(started, args.output)
        net = copy.deepcopy(initial).eval().requires_grad_(False)
        controller = TemporalController(mode=mode).to(device)
        params = list(controller.parameters())
        if len(params) != 1 or params[0].numel() != 584:
            raise ValueError(f"{arm_name}: controller must have exactly one trainable [4,146] weight")
        if params[0].shape != (4, 146) or not torch.equal(params[0].detach(), torch.zeros_like(params[0])):
            raise ValueError(f"{arm_name}: controller weight must initialize as zero [4,146]")
        caches = [_cache(batch, mode, ids) for batch, _, _, ids in train_batches]
        # Zero controller must reproduce the frozen strong-local path at both prefixes.
        probe_batch, _, _, probe_ids = train_batches[0]
        with torch.no_grad():
            for probe_T in ENDPOINTS:
                base_logits, base_sites = net(probe_batch, probe_T)
                probe_logits, probe_sites, probe_trace = temporal_forward(
                    net, probe_batch, probe_T, controller, caches[0], probe_ids)
                if not torch.equal(base_logits, probe_logits) or any(
                        not torch.equal(a, b) for a, b in zip(base_sites, probe_sites)):
                    raise AssertionError(f"{arm_name}: zero controller differs from baseline at T={probe_T}")
                if torch.any(probe_trace["choice_count"] != 0):
                    raise AssertionError(f"{arm_name}: zero controller changed events at T={probe_T}")
                _assert_first_decode(probe_trace, probe_trace, require_zero=True)
        with torch.no_grad():
            initial_train_loss = _full_train_objective(objective, net, controller, train_batches, mode, caches)
        optimizer = torch.optim.Adam(params, lr=LR)
        history = []
        fit_started = time.monotonic()
        for step in range(FIT_STEPS):
            _check_deadline(started, args.output)
            batch, teacher_logits, teacher_sites, graph_ids = train_batches[step % len(train_batches)]
            cache = caches[step % len(caches)]
            optimizer.zero_grad(set_to_none=True)
            loss, l1, l2, tr1, tr2 = _pair_loss(objective, net, batch, cache, graph_ids, controller,
                                                 teacher_logits, teacher_sites)
            _assert_first_decode(tr1, tr2, require_zero=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{arm_name}: nonfinite hard-forward training loss at step {step}")
            loss.backward()
            grad = torch.sqrt(sum((p.grad.square().sum() for p in params if p.grad is not None),
                                  torch.zeros((), device=device)))
            if not torch.isfinite(grad):
                raise RuntimeError(f"{arm_name}: nonfinite controller gradient at step {step}")
            optimizer.step()
            history.append(dict(step=step, loss=float(loss.detach()), T1=float(l1.detach()), T2=float(l2.detach()),
                                gradient_norm=float(grad.detach())))
        fit_seconds = time.monotonic() - fit_started
        with torch.no_grad():
            final_train_loss = _full_train_objective(objective, net, controller, train_batches, mode, caches)
        controller_np = {k: v.detach().cpu().numpy() for k, v in controller.state_dict().items()}
        if len(controller_np) != 1 or next(iter(controller_np.values())).shape != (4, 146):
            raise ValueError("saved controller state is not exactly one [4,146] tensor")
        np.savez_compressed(args.output / f"{arm_name}_controller.npz", **controller_np)
        _save_json(args.output / f"{arm_name}_training.json", dict(
            arm=arm_name, mode=mode, objective=objective, optimizer="Adam", lr=LR, steps=FIT_STEPS,
            parameters=584, clamping=False, initial_full_train_hard_objective=initial_train_loss,
            final_full_train_hard_objective=final_train_loss, fit_seconds=fit_seconds, history=history))
        trained[arm_name] = controller

        arm_choice_counts = {}
        for T in ENDPOINTS:
            _check_deadline(started, args.output)
            parts = {key: [] for key in ("logits", "teacher", "labels", "events", "amplitude", "deliveries",
                                         "choice_count", "first_decode_error_max")}
            offset = 0
            evaluation_started = time.monotonic()
            with torch.no_grad():
                for raw in val_batches:
                    _check_deadline(started, args.output)
                    batch = raw.clone().cuda()
                    labels = batch.y.flatten().detach().cpu().numpy()
                    del batch.y
                    ids = val_indices[offset:offset + batch.num_graphs]
                    cache = _cache(batch, mode, ids)
                    teacher_logits, _ = core.teacher_sites(folded, batch)
                    logits, _, trace = temporal_forward(net, batch, T, controller, cache, ids)
                    if T == 2 and torch.any(trace["first_decode_error_max"] != 0):
                        raise AssertionError(f"{arm_name} T2 first-tick decoded prefix changed")
                    for key, value in dict(logits=logits, teacher=teacher_logits, labels=torch.as_tensor(labels),
                        events=trace["events"], amplitude=trace["amplitude"], deliveries=trace["deliveries"],
                        choice_count=trace["choice_count"], first_decode_error_max=trace["first_decode_error_max"]).items():
                        parts[key].append(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
                    offset += batch.num_graphs
            arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
            arrays["indices"] = np.asarray(val_indices, dtype=np.int64)
            if len(arrays["indices"]) != len(arrays["labels"]) or not np.array_equal(arrays["indices"], np.asarray(val_indices)):
                raise AssertionError("validation output/index alignment failure")
            np.savez_compressed(args.output / f"{arm_name}_T{T}.npz", **arrays)
            if T == 1:
                arm_choice_counts["T1"] = arrays["choice_count"].copy()
            else:
                if not np.array_equal(arm_choice_counts["T1"], arrays["choice_count"]):
                    raise AssertionError(f"{arm_name}: T1/T2 replacement counts differ")
            pred = arrays["logits"].argmax(1)
            row = dict(arm=arm_name, T=T, accuracy_percent=float(100 * np.mean(pred == arrays["labels"])),
                       teacher_agreement_percent=float(100 * np.mean(pred == arrays["teacher"].argmax(1))),
                       events=int(arrays["events"].sum(dtype=np.int64)), amplitude_sum=float(arrays["amplitude"].sum()),
                       deliveries=int(arrays["deliveries"].sum(dtype=np.int64)),
                       changed_coordinates=int(arrays["choice_count"].sum(dtype=np.int64)),
                       first_decode_error_max=(float(arrays["first_decode_error_max"].max(initial=0)) if T == 2 else None),
                       controller_parameters=584, evaluation_seconds=time.monotonic() - evaluation_started)
            all_results.append(row)
            _save_json(args.output / "results.json", all_results)
            print(json.dumps(row), flush=True)

    # Replay the unmodified strong-local baseline through the same instrumentation.
    for T in ENDPOINTS:
        parts = {key: [] for key in ("logits", "teacher", "labels", "events", "amplitude", "deliveries",
                                     "choice_count", "first_decode_error_max")}
        offset = 0
        with torch.no_grad():
            for raw in val_batches:
                batch = raw.clone().cuda()
                labels = batch.y.flatten().detach().cpu().numpy()
                del batch.y
                ids = val_indices[offset:offset + batch.num_graphs]
                teacher_logits, _ = core.teacher_sites(folded, batch)
                logits, _, trace = temporal_forward(initial, batch, T, controller=None, cache=None,
                                                     graph_ids=ids, instrument=True)
                for key, value in dict(logits=logits, teacher=teacher_logits, labels=torch.as_tensor(labels),
                    events=trace["events"], amplitude=trace["amplitude"], deliveries=trace["deliveries"],
                    choice_count=trace["choice_count"], first_decode_error_max=trace["first_decode_error_max"]).items():
                    parts[key].append(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
                offset += batch.num_graphs
        arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
        arrays["indices"] = np.asarray(val_indices, dtype=np.int64)
        if not np.array_equal(arrays["indices"], np.asarray(val_indices)):
            raise AssertionError("strong-local replay indices mismatch")
        if not np.array_equal(arrays["logits"], expected_by_t[T]):
            raise AssertionError(f"baseline temporal replay T{T} logits differ from historical baseline")
        np.savez_compressed(args.output / f"baseline_T{T}.npz", **arrays)
        all_results.append(dict(arm="baseline", T=T,
            accuracy_percent=float(100 * np.mean(arrays["logits"].argmax(1) == arrays["labels"])),
            teacher_agreement_percent=float(100 * np.mean(arrays["logits"].argmax(1) == arrays["teacher"].argmax(1))),
            events=int(arrays["events"].sum(dtype=np.int64)), amplitude_sum=float(arrays["amplitude"].sum()),
            deliveries=int(arrays["deliveries"].sum(dtype=np.int64)), changed_coordinates=int(arrays["choice_count"].sum(dtype=np.int64)),
            first_decode_error_max=None if T == 1 else float(arrays["first_decode_error_max"].max(initial=0)),
            controller_parameters=0, evaluation_seconds=None, source="temporal_forward replay of frozen strong-local baseline"))
    _save_json(args.output / "results.json", all_results)
    if len(all_results) != 14:
        raise AssertionError(f"expected 14 endpoints, wrote {len(all_results)}")

    # One batch: warm includes only temporal forward; cold also includes cache construction.
    first_raw = val_batches[0]
    timing_batch = first_raw.clone().cuda()
    del timing_batch.y
    timing_ids = val_indices[:timing_batch.num_graphs]
    timing_caches = {name: _cache(timing_batch, mode, timing_ids) for name, mode, _ in ARMS}
    configs = [("baseline", "plain")] + [(name, temp) for name, _, _ in ARMS for temp in ("warm", "cold")]
    schedule = [(name, temp) for name, temp in configs for _ in range(7)]
    random.Random(SEED).shuffle(schedule)
    samples = {f"{name}/{temp}": [] for name, temp in configs}
    counts = {key: 0 for key in samples}
    mode_for = {name: mode for name, mode, _ in ARMS}
    for name, temp in schedule:
        _check_deadline(started, args.output)
        torch.cuda.synchronize()
        then = time.perf_counter()
        if name == "baseline":
            with torch.no_grad():
                initial(timing_batch, 1)
        else:
            mode = mode_for[name]
            cache = timing_caches[name] if temp == "warm" else _cache(timing_batch, mode, timing_ids)
            with torch.no_grad():
                temporal_forward(initial, timing_batch, 1, trained[name], cache, timing_ids, instrument=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - then) * 1000
        key = f"{name}/{temp}"
        counts[key] += 1
        if counts[key] > 2:
            samples[key].append(elapsed_ms)
    _save_json(args.output / "timing.json", dict(
        batch_graphs=timing_batch.num_graphs, T=1, warmups=2, measured_repeats=5,
        teacher_in_timed_block=False, diagnostic_counters_in_timed_block=False,
        cold_includes_temporal_cache_build=True, warm_cache_build_excluded=True,
        metric="synchronized CUDA+host wall time including controller, local source probes, and CPU copies; not energy",
        results={key: dict(raw_ms=values, median_ms_per_graph=float(np.median(values) / timing_batch.num_graphs))
                 for key, values in samples.items()}))
    _save_json(args.output / "DONE.json", dict(endpoints=len(all_results), elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
