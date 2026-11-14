"""Fixed-GCN DCGS event forward and guarded seed-general full-val evaluator.

MTH_diff is imported unchanged from the pinned author checkout. The mapping
from its unit into this PyG residual GCN is an explicit project adaptation.
The default mode performs only synthetic CPU source-oracle acceptance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

HERE = Path(__file__).resolve().parent
RUNROOT = HERE.parent
DATA_ROOT = RUNROOT / "data" / "pyg_bench"
OUT_ROOT = HERE / "full_teacher_runs"
PINNED_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
SITES = ["gcn_0", "gcn_1", "gcn_2", "gcn_3", "mlp_0", "mlp_1"]
TIMESTEPS = (1, 2, 4, 8)
EXPECTED_COUNTS = {"MNIST": {"train": 55000, "val": 5000},
                   "CIFAR10": {"train": 45000, "val": 5000}}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode() + b"\0")
    h.update(json.dumps(list(t.shape), separators=(",", ":")).encode() + b"\0")
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def require_author_source(author_repo):
    author_repo = Path(author_repo).resolve()
    commit = subprocess.check_output(["git", "-C", str(author_repo), "rev-parse", "HEAD"],
                                     universal_newlines=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError("author source commit mismatch: %s != %s" % (commit, PINNED_COMMIT))
    status = subprocess.check_output(["git", "-C", str(author_repo), "status", "--porcelain", "--untracked-files=all"],
                                     universal_newlines=True)
    untracked_bytecode = []
    dirty_sources = []
    for line in status.splitlines():
        if line.startswith("?? "):
            rel = line[3:].replace("\\", "/")
            if "/__pycache__/" in "/" + rel and rel.endswith(".pyc"):
                untracked_bytecode.append(rel)
            else:
                dirty_sources.append(line)
        else:
            dirty_sources.append(line)
    if dirty_sources:
        raise RuntimeError("author source worktree has tracked or non-bytecode changes: %r" % dirty_sources[:10])
    source_rel = "neurons/relu_neuron.py"
    source_path = author_repo / "neurons" / "relu_neuron.py"
    blob = subprocess.check_output(["git", "-C", str(author_repo), "rev-parse", "HEAD:" + source_rel],
                                   universal_newlines=True).strip()
    work_blob = subprocess.check_output(["git", "-C", str(author_repo), "hash-object",
                                         "--path=" + source_rel, str(source_path)],
                                        universal_newlines=True).strip()
    if work_blob != blob:
        raise RuntimeError("MTH_diff source file bytes do not match the pinned git blob")
    source_provenance = {"commit": commit, "worktree_clean_except_ignored_bytecode": True,
                         "untracked_bytecode_entries": len(untracked_bytecode),
                         "source_file": source_rel, "source_file_sha256": sha256(source_path),
                         "source_head_blob_oid": blob, "source_worktree_blob_oid": work_blob}
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(author_repo))
    from neurons.relu_neuron import MTH_diff
    return MTH_diff, commit, source_provenance


def load_oracle():
    import importlib.util
    path = HERE / "dcgs_mth_diff_source_equivalence_v1.py"
    spec = importlib.util.spec_from_file_location("dcgs_source_oracle_v1", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def graph_weights(edge_index, n, dtype, device):
    source, target = edge_index
    outdegree = torch.bincount(source, minlength=n).clamp(min=1).to(device=device, dtype=dtype)
    indegree = torch.bincount(target, minlength=n).clamp(min=1).to(device=device, dtype=dtype)
    return outdegree[source].rsqrt() * indegree[target].rsqrt()


def fresh_mth(MTH_diff, theta, device):
    mod = MTH_diff(T=0, thresh=torch.as_tensor(theta, dtype=torch.float32, device=device),
                   step_mode="s", num_thresholds=8).to(device)
    return mod


def _event_counts(values, theta, channels):
    flat = values.detach().reshape(-1)
    theta = torch.as_tensor(theta, dtype=flat.dtype, device=flat.device).reshape(-1)
    if channels < 1 or theta.numel() != channels or flat.numel() % channels:
        raise ValueError("event/threshold channel shape mismatch")
    expanded_theta = theta.repeat(flat.numel() // channels)
    positive_mask = flat > 0
    negative_mask = flat < 0
    nonzero_mask = positive_mask | negative_mask
    level_count = 8
    levels = torch.round(torch.log2(expanded_theta[nonzero_mask] / flat[nonzero_mask].abs()))
    levels = levels.to(torch.int64).clamp(0, level_count - 1)
    signs = negative_mask[nonzero_mask].to(torch.int64)
    level_hist = torch.bincount(signs * level_count + levels,
                                minlength=2 * level_count)
    metrics = torch.stack((positive_mask.sum(), negative_mask.sum(), (~nonzero_mask).sum(),
                           nonzero_mask.sum(), flat.abs().sum())).to(torch.float64)
    summary = torch.cat((metrics, level_hist.to(torch.float64))).cpu().tolist()
    positive, negative, zero, nonzero, abs_sum = summary[:5]
    levels = {}
    for level in range(level_count):
        pos_n = int(summary[5 + level])
        neg_n = int(summary[5 + level_count + level])
        if pos_n:
            levels["+2^-" + str(level)] = pos_n
        if neg_n:
            levels["-2^-" + str(level)] = neg_n
    return {"positive": int(positive), "negative": int(negative), "zero": int(zero),
            "nonzero_scalar_events": int(nonzero),
            "sum_abs_graded_amplitude": float(abs_sum),
            "signed_threshold_level_events": levels}


def _event_counts_scalar_reference(values, theta, channels):
    """Small synthetic-only oracle retained to validate vectorized statistics."""
    flat = values.detach().reshape(-1).cpu().tolist()
    theta = torch.as_tensor(theta).reshape(-1).cpu().tolist()
    positive = negative = zero = 0
    abs_sum = 0.0
    levels = {}
    for i, value in enumerate(flat):
        if value > 0:
            positive += 1
        elif value < 0:
            negative += 1
        else:
            zero += 1
        abs_sum += abs(value)
        if value:
            level = int(round(math.log(theta[i % channels] / abs(value), 2)))
            sign_level = ("-" if value < 0 else "+") + "2^-" + str(max(0, min(7, level)))
            levels[sign_level] = levels.get(sign_level, 0) + 1
    return {"positive": positive, "negative": negative, "zero": zero,
            "nonzero_scalar_events": positive + negative,
            "sum_abs_graded_amplitude": abs_sum,
            "signed_threshold_level_events": levels}


def _assert_stats_equal(vectorized, scalar, atol=1e-6):
    for key in ("positive", "negative", "zero", "nonzero_scalar_events",
                "signed_threshold_level_events"):
        if vectorized[key] != scalar[key]:
            raise AssertionError("vector event statistic mismatch in %s" % key)
    error = abs(vectorized["sum_abs_graded_amplitude"] - scalar["sum_abs_graded_amplitude"])
    if error > atol:
        raise AssertionError("graded amplitude sum mismatch %g > %g" % (error, atol))


def dcgs_event_forward(model, batch, thresholds, T, MTH_diff, capture_oracle=False):
    """Emit one graph/model event per tick and decode final logits by Σ e_t/t.

    The static raw input is emitted once at t=1. Dense embedding bias, all
    folded graph-convolution affine offsets, both MLP biases, and the final
    classifier bias are included only in that first event. ReLU sites use the
    author MTH_diff unit. Residual bypasses add the aligned current event.
    """
    if len(thresholds) != 6 or T not in TIMESTEPS:
        raise ValueError("requires six site thresholds and T in {1,2,4,8}")
    device = batch.x.device
    threshold_tensors = [torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1)
                         for x in thresholds]
    modules = [fresh_mth(MTH_diff, x, device) for x in threshold_tensors]
    raw = torch.cat([batch.x.float(), batch.pos.float()], dim=-1)
    n = raw.size(0)
    edge_weight = graph_weights(batch.edge_index, n, raw.dtype, device)
    currents = [[] for _ in range(6)] if capture_oracle else None
    stats = [{"positive": 0, "negative": 0, "zero": 0, "nonzero_scalar_events": 0,
              "sum_abs_graded_amplitude": 0.0, "signed_threshold_level_events": {}}
             for _ in range(6)]
    logits = None

    # The MTH source T=-1 initialization call latches a zero baseline current.
    for i, mod in enumerate(modules):
        init_shape = (n, threshold_tensors[i].numel()) if i < 4 else None
        if i < 4:
            mod(torch.zeros(init_shape, dtype=torch.float32, device=device))
        elif i == 4:
            mod(torch.zeros((batch.num_graphs, threshold_tensors[i].numel()), dtype=torch.float32, device=device))
        else:
            mod(torch.zeros((batch.num_graphs, threshold_tensors[i].numel()), dtype=torch.float32, device=device))

    for t in range(1, T + 1):
        first = (t == 1)
        raw_event = raw if first else torch.zeros_like(raw)
        h = F.linear(raw_event, model.embedding.weight,
                     model.embedding.bias if first else None)

        for i, conv in enumerate(model.convs):
            # PyG GCNConv adds its folded bias on every call. Remove it after
            # t=1 so the DCGS affine offset is delivered exactly once.
            z = conv(h, batch.edge_index, edge_weight)
            if not first and conv.bias is not None:
                z = z - conv.bias
            if capture_oracle:
                currents[i].append(z.detach().cpu().clone())
            branch = modules[i](z)
            event_count = _event_counts(branch, threshold_tensors[i], branch.shape[-1])
            _accumulate_stats(stats[i], event_count)
            h = h + branch

        pooled = global_mean_pool(h, batch.batch)
        first_linear = model.mlp[0]
        z = F.linear(pooled, first_linear.weight,
                     first_linear.bias if first else None)
        if capture_oracle:
            currents[4].append(z.detach().cpu().clone())
        h = modules[4](z)
        _accumulate_stats(stats[4], _event_counts(h, threshold_tensors[4], h.shape[-1]))

        second_linear = model.mlp[2]
        z = F.linear(h, second_linear.weight,
                     second_linear.bias if first else None)
        if capture_oracle:
            currents[5].append(z.detach().cpu().clone())
        h = modules[5](z)
        _accumulate_stats(stats[5], _event_counts(h, threshold_tensors[5], h.shape[-1]))

        unit_ticks = [int(mod.T) for mod in modules]
        if any(unit_tick != unit_ticks[0] for unit_tick in unit_ticks) or unit_ticks[0] != t:
            raise RuntimeError("MTH_diff active tick count desynchronized: %r vs %d" % (unit_ticks, t))
        classifier = model.mlp[4]
        event_logits = F.linear(h, classifier.weight,
                                classifier.bias if first else None)
        # Use the unit's own active-T state for the DC decoder denominator.
        contribution = event_logits / float(unit_ticks[5])
        logits = contribution if logits is None else logits + contribution

    trace = None
    if capture_oracle:
        trace = {"currents": currents, "modules": modules,
                 "thresholds": threshold_tensors}
    return logits, stats, trace


def _accumulate_stats(total, current):
    for key in ("positive", "negative", "zero", "nonzero_scalar_events"):
        total[key] += current[key]
    total["sum_abs_graded_amplitude"] += current["sum_abs_graded_amplitude"]
    for key, value in current["signed_threshold_level_events"].items():
        total["signed_threshold_level_events"][key] = total["signed_threshold_level_events"].get(key, 0) + value


def synthetic_acceptance(MTH_diff, oracle):
    from full_teacher_gcn import BenchmarkGCN
    from fold_teacher_bn import fold_model
    torch.manual_seed(20261112)
    device = torch.device("cpu")
    graphs = [
        Data(x=torch.tensor([[0.2, 0.4], [0.8, -0.1], [0.3, 0.7]], dtype=torch.float32),
             pos=torch.tensor([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2]], dtype=torch.float32),
             edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long),
             y=torch.tensor([0])),
        Data(x=torch.tensor([[-0.4, 0.6], [0.5, 0.2], [0.7, 0.3], [0.1, -0.2]], dtype=torch.float32),
             pos=torch.tensor([[0.1, 0.0], [0.3, 0.1], [0.5, 0.3], [0.7, 0.2]], dtype=torch.float32),
             edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long),
             y=torch.tensor([1])),
    ]
    batch = Batch.from_data_list(graphs).to(device)
    model = BenchmarkGCN(input_dim=4, width=8, layers=4, num_classes=3).to(device).eval()
    folded = fold_model(model).eval()
    with torch.no_grad():
        teacher = model(batch)
        teacher_fold = folded(batch)
    fold_error = float((teacher - teacher_fold).abs().max().item())
    if not math.isfinite(fold_error) or fold_error > 2e-5:
        raise AssertionError("BN-folded teacher mismatch: %g" % fold_error)
    theta = [[0.23] * 8] * 4 + [[0.25] * 4, [0.27] * 2]
    output = {}
    for T in TIMESTEPS:
        with torch.no_grad():
            logits, stats, trace = dcgs_event_forward(folded, batch, theta, T, MTH_diff,
                                                     capture_oracle=True)
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("non-finite synthetic logits")
        captured = {}
        for i, (site_currents, theta_site) in enumerate(zip(trace["currents"], theta)):
            flat_theta = torch.as_tensor(theta_site).repeat(site_currents[0].shape[0]).tolist()
            frames = [[0.0] * len(flat_theta)] + [f.reshape(-1).tolist() for f in site_currents]
            source_trace = oracle.source_exact_stepper(frames, flat_theta, 8)
            # Author module was executed by dcgs_event_forward; current-by-current
            # comparison is made by a separate direct source replay in CPU below.
            from neurons.relu_neuron import MTH_diff as SourceMTH
            author_trace = []
            m2 = SourceMTH(T=0, thresh=torch.as_tensor(theta_site), step_mode="s", num_thresholds=8).cpu()
            m2(torch.zeros_like(site_currents[0]))
            for current in site_currents:
                out = m2(current.cpu())
                author_trace.append({"output": out.reshape(-1).tolist(),
                                     "membrane": m2.mem.reshape(-1).tolist(),
                                     "cumulative_out": m2.cumulative_out.reshape(-1).tolist(),
                                     "exp_in": m2.exp_in.reshape(-1).tolist(),
                                     "exp_out": m2.exp_out.reshape(-1).tolist()})
            report = oracle._assert_trace_equal(SITES[i], source_trace, author_trace, atol=3e-6)
            stat_checks = []
            for tick in author_trace:
                tick_values = torch.tensor(tick["output"], dtype=torch.float32)
                vector = _event_counts(tick_values, theta_site, len(theta_site))
                scalar = _event_counts_scalar_reference(tick_values, theta_site,
                                                        len(theta_site))
                _assert_stats_equal(vector, scalar)
                stat_checks.append(vector)
            captured[SITES[i]] = report
            captured[SITES[i]]["vector_stats_equal_scalar_oracle_all_ticks"] = len(stat_checks) == T
        output["T%d" % T] = {"logits_shape": list(logits.shape), "finite": True,
                             "bn_fold_max_abs_error": fold_error,
                             "source_oracle": captured, "event_stats": stats,
                             "warmup_calls": len(SITES),
                             "active_unit_calls": len(SITES) * T,
                             "effective_model_ticks": T,
                             "decoder_denominator_source": "MTH_diff.T after each active call"}
    return {"device": "cpu", "graphs": int(batch.num_graphs), "nodes": int(batch.num_nodes),
            "pinned_source_commit": PINNED_COMMIT, "results": output,
            "warmup_semantics": warmup_acceptance(MTH_diff, oracle),
            "affine_differential_identity": verify_affine_dc_identity(),
            "interpretation": "synthetic source-equivalence and project-level event mapping; no accuracy/energy claim"}


def warmup_acceptance(MTH_diff, oracle):
    """Verify zero warmup and the first active event, and expose skipped-first-call case."""
    theta = torch.tensor([0.3, 0.4], dtype=torch.float32)
    bias = torch.tensor([0.08, -0.06], dtype=torch.float32)
    signal = torch.zeros(2, dtype=torch.float32)
    module = MTH_diff(T=0, thresh=theta, step_mode="s", num_thresholds=4).cpu()
    warm = module(torch.zeros_like(signal))
    if bool((warm != 0).any()) or module.T != 0 or bool((module.exp_in != 0).any()):
        raise AssertionError("zero warmup did not latch zero and emit zero")
    first_current = signal + bias
    first_out = module(first_current)
    if module.T != 1:
        raise AssertionError("first active event should advance unit T to 1")
    if not bool((first_out != 0).any()):
        raise AssertionError("first-tick affine bias failed to produce its expected event")
    reference = oracle.source_exact_stepper([[0.0, 0.0], first_current.tolist()],
                                              theta.tolist(), 4)
    direct = oracle.author_source_trace(torch, MTH_diff, [[0.0, 0.0], first_current.tolist()],
                                        theta.tolist(), 4)
    oracle._assert_trace_equal("zero_warmup_first_bias_event", reference, direct)
    if not bool(torch.isfinite(first_out).all()):
        raise AssertionError("non-finite first active event")
    # Without the zero warmup, source T=-1 consumes this first current as its
    # non-emitting latch call. This is a genuine lost first event, not T=1.
    no_warm = MTH_diff(T=0, thresh=theta, step_mode="s", num_thresholds=4).cpu()
    lost = no_warm(first_current)
    if bool((lost != 0).any()) or no_warm.T != 0:
        raise AssertionError("expected initial call without warmup to be non-emitting")
    return {"warmup_input": "zeros_like(first current)", "warmup_output_nonzero": False,
            "warmup_is_counted_as_active_T": False, "warmup_calls_per_batch": len(SITES),
            "active_calls_per_unit": "exactly T; verified each MTH_diff.T equals 1..T",
            "first_current_includes_affine_bias": True,
            "no_warmup_counterexample": "first affine current is latched and emits zero at T=-1; it must not be treated as active tick 1",
            "post_active_internal_T": int(module.T)}


def verify_affine_dc_identity():
    """Algebraic control for bias-once versus bias-on-every-tick decoding."""
    x = torch.tensor([0.35, -0.2], dtype=torch.float64)
    w = torch.tensor([[0.7, -0.3], [0.2, 0.8], [-0.5, 0.4]], dtype=torch.float64)
    b = torch.tensor([0.08, -0.04, 0.11], dtype=torch.float64)
    errors = {}
    naive_bias_error = {}
    for T in TIMESTEPS:
        decoded = torch.zeros_like(b)
        naive = torch.zeros_like(b)
        for t in range(1, T + 1):
            e = (w @ x + b) if t == 1 else torch.zeros_like(b)
            decoded += e / float(t)
            wrong = (w @ x + b) if t == 1 else b
            naive += wrong / float(t)
        target = w @ x + b
        errors[str(T)] = float((decoded - target).abs().max().item())
        naive_bias_error[str(T)] = float((naive - target).abs().max().item())
        if errors[str(T)] > 1e-12:
            raise AssertionError("one-time affine bias failed DC identity at T=%d" % T)
        if T > 1 and naive_bias_error[str(T)] <= 1e-12:
            raise AssertionError("repeated bias counterexample did not diverge")
    return {"once_at_t1_max_abs_error": errors,
            "bias_every_tick_max_abs_error_counterexample": naive_bias_error,
            "identity": "sum_t (W e_t + b*1[t=1])/t = W*sum_t(e_t/t) + b"}


def _is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_inputs(dataset, seed, teacher_run, calibration_path, author_source):
    teacher_run = Path(teacher_run).resolve()
    calibration_path = Path(calibration_path).resolve()
    allowed_runs = (HERE / "full_teacher_runs").resolve()
    if not _is_within(teacher_run, allowed_runs):
        raise RuntimeError("teacher run must be inside HERE/full_teacher_runs")
    if not _is_within(calibration_path, HERE):
        raise RuntimeError("calibration must be inside HERE")
    if dataset not in EXPECTED_COUNTS:
        raise ValueError("dataset must be MNIST or CIFAR10")
    done_path = teacher_run / "done.json"
    metadata_path = teacher_run / "metadata.json"
    checkpoint = teacher_run / "best.pt"
    for path in (done_path, metadata_path, checkpoint, calibration_path):
        if not path.is_file():
            raise RuntimeError("missing required provenance input: %s" % path)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    cal = json.loads(calibration_path.read_text(encoding="utf-8"))
    if done.get("status") != "finished":
        raise RuntimeError("teacher run is not terminal")
    expected = EXPECTED_COUNTS[dataset]
    if meta.get("dataset") != dataset or meta.get("seed") != seed:
        raise RuntimeError("teacher dataset/seed mismatch")
    if meta.get("scope") != "full_official_train_val" or meta.get("official_counts") != expected:
        raise RuntimeError("teacher is not the complete official train/val run")
    if cal.get("status") != "complete" or cal.get("dataset") != dataset:
        raise RuntimeError("calibration is incomplete or for a different dataset")
    expected_teacher_run = Path(cal.get("teacher_run", "")).resolve()
    if expected_teacher_run != teacher_run:
        raise RuntimeError("calibration teacher_run path does not match selected teacher run")
    if cal.get("teacher_epoch") is None:
        raise RuntimeError("calibration does not bind to a teacher epoch")
    if cal.get("scope") != "all_official_training_graphs_two_pass_original_order":
        raise RuntimeError("calibration is not full-train two-pass original-order")
    cal_meta = cal.get("calibration", {})
    for key in ("full_train_graph_count", "pass1_graphs", "pass2_graphs"):
        if cal_meta.get(key) != expected["train"]:
            raise RuntimeError("calibration %s does not cover full official train" % key)
    if cal_meta.get("quantile") != 0.999 or cal_meta.get("bins") != 512:
        raise RuntimeError("calibration threshold rule does not match locked project protocol")
    ck_sha = sha256(checkpoint)
    if cal.get("checkpoint_sha256") != ck_sha:
        raise RuntimeError("calibration checkpoint hash does not match teacher checkpoint")
    done_checkpoint_sha = done.get("best_checkpoint_sha256", done.get("checkpoint_sha256"))
    if done_checkpoint_sha is not None and done_checkpoint_sha != ck_sha:
        raise RuntimeError("terminal teacher record checkpoint hash mismatch")
    for filename, expected_hash in cal.get("code_sha256", {}).items():
        path = HERE / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError("calibration code hash mismatch: %s" % filename)
    if meta.get("script_sha256") != sha256(HERE / "full_teacher_gcn.py"):
        raise RuntimeError("terminal teacher implementation hash mismatch")
    if meta.get("protocol_sha256") != sha256(HERE / "teacher-full-protocol-v1.md"):
        raise RuntimeError("terminal teacher protocol hash mismatch")
    processed = DATA_ROOT / dataset / "processed"
    for filename, expected_hash in cal.get("data", {}).get("processed_files_sha256", {}).items():
        path = processed / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError("processed official data hash mismatch: %s" % filename)
    source_sha = author_source["commit"]
    if source_sha != PINNED_COMMIT or author_source["source_worktree_blob_oid"] != author_source["source_head_blob_oid"]:
        raise RuntimeError("pinned MTH_diff author source mismatch")
    for name in SITES:
        if name not in cal.get("channel_thresholds", {}) or name not in cal.get("global_thresholds", {}):
            raise RuntimeError("missing calibrated threshold site %s" % name)
    dims = [146, 146, 146, 146, 73, 36]
    for name, dim in zip(SITES, dims):
        if len(cal["channel_thresholds"][name]) != dim:
            raise RuntimeError("threshold shape mismatch at %s" % name)
        values = np.asarray(cal["channel_thresholds"][name], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise RuntimeError("non-positive/non-finite threshold at %s" % name)
    return done, meta, cal, checkpoint, ck_sha, source_sha


def full_validation(dataset, seed, teacher_run, calibration_path, author_repo, author_source, MTH_diff, output, device_name="cpu", batch_size=128):
    teacher_run = Path(teacher_run).resolve()
    calibration_path = Path(calibration_path).resolve()
    output = Path(output).resolve()
    predictions_path = output.with_suffix(".npz")
    if not _is_within(output, HERE):
        raise RuntimeError("output must be inside HERE")
    if output.exists() or predictions_path.exists():
        raise FileExistsError("refusing to overwrite output or predictions")
    done, metadata, cal, checkpoint, checkpoint_sha, source_sha = validate_inputs(
        dataset, seed, teacher_run, calibration_path, author_source)
    from full_teacher_gcn import BenchmarkGCN
    from fold_teacher_bn import fold_model
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu":
        torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(state.get("epoch", -1)) != int(calibration_epoch := json.loads(
            calibration_path.read_text(encoding="utf-8")).get("teacher_epoch", -2)):
        raise RuntimeError("calibration teacher epoch does not match selected checkpoint")
    model = BenchmarkGCN(input_dim=metadata["input_dim"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    folded = fold_model(model).eval()
    thresholds = [cal["channel_thresholds"][name] for name in SITES]
    val = GNNBenchmarkDataset(root=str(DATA_ROOT), name=dataset, split="val")
    if len(val) != 5000:
        raise RuntimeError("official full validation split size mismatch")
    loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    labels = []
    teacher_parts = []
    converted = {T: [] for T in TIMESTEPS}
    event_totals = {T: [None] * 6 for T in TIMESTEPS}
    index_parts = []
    next_index = 0
    start = time.monotonic()
    for batch_num, batch in enumerate(loader, 1):
        batch = batch.to(device)
        bsz = int(batch.num_graphs)
        with torch.no_grad():
            teacher_logits = model(batch)
        if not bool(torch.isfinite(teacher_logits).all()):
            raise RuntimeError("non-finite terminal teacher logits")
        teacher_parts.append(teacher_logits.cpu().numpy())
        labels.append(batch.y.view(-1).long().cpu().numpy())
        index_parts.append(np.arange(next_index, next_index + bsz, dtype=np.int64))
        next_index += bsz
        for T in TIMESTEPS:
            with torch.no_grad():
                logits, stats, _ = dcgs_event_forward(folded, batch, thresholds, T, MTH_diff)
            if logits.shape != (bsz, 10) or logits.shape != teacher_logits.shape or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("invalid/non-finite converted logits at T=%d" % T)
            converted[T].append(logits.cpu().numpy())
            for i, site_stats in enumerate(stats):
                if event_totals[T][i] is None:
                    event_totals[T][i] = json.loads(json.dumps(site_stats))
                else:
                    _accumulate_stats(event_totals[T][i], site_stats)
        if batch_num % 10 == 0:
            print(json.dumps({"batch": batch_num, "graphs": next_index,
                              "elapsed_sec": time.monotonic() - start}), flush=True)
    idx = np.concatenate(index_parts)
    y = np.concatenate(labels)
    teacher_logits = np.concatenate(teacher_parts)
    if len(idx) != 5000 or not np.array_equal(idx, np.arange(5000)):
        raise RuntimeError("validation did not cover all 5000 ordered examples exactly once")
    logits_by_t = {T: np.concatenate(converted[T]) for T in TIMESTEPS}
    teacher_acc = float(np.mean(teacher_logits.argmax(1) == y))
    rows = []
    for T in TIMESTEPS:
        pred = logits_by_t[T]
        rows.append({"T": T, "teacher_accuracy": teacher_acc,
                     "converted_accuracy": float(np.mean(pred.argmax(1) == y)),
                     "teacher_agreement": float(np.mean(pred.argmax(1) == teacher_logits.argmax(1))),
                     "logit_mse": float(np.mean((pred - teacher_logits) ** 2)),
                     "sites": event_totals[T]})
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or predictions_path.exists():
        raise FileExistsError("output appeared during run; refusing overwrite")
    np.savez_compressed(predictions_path, indices=idx, labels=y, teacher_logits=teacher_logits,
                        **{"T%d_logits" % T: logits_by_t[T] for T in TIMESTEPS})
    result = {"dataset": dataset, "seed": seed, "split": "complete official validation",
              "teacher_run": str(teacher_run), "calibration_path": str(calibration_path),
              "graphs": 5000, "checkpoint_epoch": state["epoch"],
              "checkpoint_sha256": checkpoint_sha, "calibration_sha256": sha256(calibration_path),
              "teacher_done": done, "teacher_metadata_sha256": sha256(teacher_run / "metadata.json"),
              "source_commit": source_sha, "author_source_provenance": author_source,
              "script_sha256": sha256(__file__),
              "predictions_sha256": sha256(predictions_path), "device": str(device),
              "runtime": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                          "numpy": np.__version__, "python": sys.version},
              "elapsed_sec": time.monotonic() - start,
              "decoder": "logit = sum_{t=1..T} classifier_event_t / unit_internal_T_after_active_call; event biases once at t=1",
              "unit_call_accounting": {
                  "source_warmup_calls_per_batch": len(SITES),
                  "warmup_emits_output": False,
                  "warmup_is_an_active_simulation_tick": False,
                  "active_unit_calls_per_batch_by_T": {"T%d" % T: len(SITES) * T for T in TIMESTEPS},
                  "source_warmup_calls_per_T_run": len(loader) * len(SITES),
                  "source_warmup_calls_all_T_runs": len(loader) * len(SITES) * len(TIMESTEPS),
                  "warmup_policy": "one zero-input latch call per site before any active current; all site modules latch no ANN hidden activation",
              },
              "warmup_semantics": warmup_acceptance(MTH_diff, load_oracle()),
              "scope": "project fixed-GCN adaptation using source MTH_diff units; not author full-model reproduction",
              "readout_rows": rows}
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    syn = sub.add_parser("synthetic")
    syn.add_argument("--author-repo", required=True, type=Path)
    syn.add_argument("--device", choices=("cpu",), default="cpu")
    val = sub.add_parser("full-val")
    val.add_argument("--author-repo", required=True, type=Path)
    val.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    val.add_argument("--dataset", choices=tuple(EXPECTED_COUNTS), required=True)
    val.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    val.add_argument("--teacher-run", required=True, type=Path)
    val.add_argument("--calibration", required=True, type=Path)
    val.add_argument("--batch-size", type=int, default=128)
    val.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    MTH_diff, commit, source_provenance = require_author_source(args.author_repo)
    if args.mode == "synthetic":
        result = synthetic_acceptance(MTH_diff, load_oracle())
        result["pinned_source_commit"] = commit
        result["author_source_provenance"] = source_provenance
        print(json.dumps(result, indent=2))
    else:
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        full_validation(args.dataset, args.seed, args.teacher_run, args.calibration,
                        args.author_repo, source_provenance,
                        MTH_diff, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
