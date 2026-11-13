"""Synthetic/reference implementation of shared-consumer affine-base modes.

This is a dense feasibility reference, not an event-cache backend or energy
claim. The generic and first-analog modes use signed MTH_diff_line input
coders. The centered mode retains raw residual skips around MTH_diff-coded
nonnegative residual inputs. All modes use the same analog MLP head.
"""
import hashlib
import copy
import importlib
import inspect
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

HERE = Path(__file__).resolve().parent
PINNED_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
DEFAULT_AUTHOR_REPO = Path(r"C:\Users\xhf\AppData\Local\Temp\dcgs-5e0e4c73")
MODES = ("generic", "first_analog", "centered")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(author_repo, *args):
    return subprocess.check_output(["git", "-C", str(author_repo), *args],
                                   universal_newlines=True).strip()


def require_author_sources(author_repo=DEFAULT_AUTHOR_REPO):
    """Load exact pinned MTH_diff and MTH_diff_line classes and bind their files."""
    repo = Path(author_repo).resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != PINNED_COMMIT:
        raise RuntimeError("author source commit mismatch: %s != %s" % (commit, PINNED_COMMIT))
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    dirty = []
    for line in status.splitlines():
        if line.startswith("?? "):
            rel = line[3:].replace("\\", "/")
            if "/__pycache__/" in "/" + rel and rel.endswith(".pyc"):
                continue
        dirty.append(line)
    if dirty:
        raise RuntimeError("author worktree has tracked or non-bytecode changes: %r" % dirty[:10])

    files = {"relu_neuron": "neurons/relu_neuron.py",
             "line_neuron": "neurons/line_neuron.py"}
    provenance = {"commit": commit, "author_repo": str(repo),
                  "worktree_clean_except_ignored_bytecode": True,
                  "source_files": {}}
    for key, rel in files.items():
        path = (repo / rel).resolve()
        if not path.is_file() or not path.is_relative_to(repo):
            raise RuntimeError("missing/out-of-repository pinned source file: %s" % rel)
        head_blob = _git(repo, "rev-parse", "HEAD:" + rel)
        work_blob = _git(repo, "hash-object", "--path=" + rel, str(path))
        if work_blob != head_blob:
            raise RuntimeError("author source file differs from pinned commit: %s" % rel)
        provenance["source_files"][key] = {"path": rel, "sha256": sha256(path),
                                             "head_blob_oid": head_blob,
                                             "worktree_blob_oid": work_blob}

    sys.dont_write_bytecode = True
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    importlib.invalidate_caches()
    relu_module = importlib.import_module("neurons.relu_neuron")
    line_module = importlib.import_module("neurons.line_neuron")
    MTH_diff = relu_module.MTH_diff
    MTH_diff_line = line_module.MTH_diff_line
    for name, cls, rel in (("MTH_diff", MTH_diff, files["relu_neuron"]),
                           ("MTH_diff_line", MTH_diff_line, files["line_neuron"])):
        actual = Path(inspect.getfile(cls)).resolve()
        if actual != (repo / rel).resolve():
            raise RuntimeError("%s imported from unexpected path: %s" % (name, actual))
    provenance["loaded_classes"] = {"MTH_diff": inspect.getfile(MTH_diff),
                                    "MTH_diff_line": inspect.getfile(MTH_diff_line)}
    return MTH_diff, MTH_diff_line, provenance


def site_specs(mode, model):
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (MODES,))
    layers = list(model.layers)
    start = 0 if mode == "generic" else 1
    specs = []
    for layer_index in range(start, len(layers)):
        width = int(layers[layer_index].in_channels)
        neuron_kind = "line" if mode in ("generic", "first_analog") else "relu"
        specs.append(("layer%d.node" % layer_index, width, neuron_kind))
        specs.append(("layer%d.edge" % layer_index, width, neuron_kind))
    return specs


def validate_batch_model_thresholds(model, batch, thresholds, T, mode):
    if mode not in MODES:
        raise ValueError("unsupported mode")
    if model.training:
        raise ValueError("model must be in eval mode")
    if any(module.training for module in model.modules()):
        raise ValueError("all model submodules must be in eval mode")
    if T not in (1, 2, 4, 8):
        raise ValueError("T must be one of 1, 2, 4, 8")
    if "y" in batch and batch.y is not None:
        raise ValueError("unlabeled batches required; labels are not inputs to this forward")
    if not hasattr(batch, "x") or batch.x.ndim != 2 or batch.x.shape[0] < 1:
        raise ValueError("batch must contain nonempty node features")
    if not bool(torch.isfinite(batch.x).all()):
        raise ValueError("input node features must be finite")
    if getattr(batch, "edge_attr", None) is None:
        raise ValueError("GatedGCN requires edge_attr")
    if not bool(torch.isfinite(batch.edge_attr).all()):
        raise ValueError("input edge features must be finite")
    edge_index = batch.edge_index
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] < 1:
        raise ValueError("edge_index must be [2,E] with E>0")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("edge_index must use an integer dtype")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(batch.x.shape[0]):
        raise ValueError("edge_index contains node outside batch")
    if int(batch.edge_attr.reshape(-1, getattr(model, "edge_dim", 1)).shape[0]) != int(edge_index.shape[1]):
        raise ValueError("edge feature row count must match edge_index")
    if getattr(batch, "pos", None) is not None and not bool(torch.isfinite(batch.pos).all()):
        raise ValueError("input positions must be finite")
    layers = list(model.layers)
    if not layers:
        raise ValueError("at least one GatedGCN layer is required")
    widths = {int(layer.in_channels) for layer in layers} | {int(layer.out_channels) for layer in layers}
    if len(widths) != 1:
        raise ValueError("all layers must have equal width")
    if any(not bool(layer.residual) or layer.in_channels != layer.out_channels for layer in layers):
        raise ValueError("all layers must use equal-width identity residuals")
    if any(layer.training for layer in layers):
        raise ValueError("all graph layers must be in eval mode")
    if getattr(batch, "batch", None) is None or batch.batch.numel() != batch.x.shape[0]:
        raise ValueError("batch assignment is missing or malformed")
    specs = site_specs(mode, model)
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) != len(specs):
        raise ValueError("threshold list must have one full channel vector per coded site")
    for (name, width, _), threshold in zip(specs, thresholds):
        value = torch.as_tensor(threshold)
        if value.ndim != 1 or value.numel() != width:
            raise ValueError("threshold shape mismatch at %s: expected [%d]" % (name, width))
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("thresholds must be finite and positive at %s" % name)
    return specs


def _site_stats(event, threshold):
    flat = event.detach().reshape(-1)
    theta = torch.as_tensor(threshold, dtype=flat.dtype, device=flat.device).reshape(-1)
    if flat.numel() % theta.numel():
        raise ValueError("event and threshold channel dimensions do not align")
    expanded = theta.repeat(flat.numel() // theta.numel())
    pos, neg = flat > 0, flat < 0
    nz = pos | neg
    nonzero_rows = int(nz.reshape_as(event).any(dim=-1).sum())
    levels = torch.round(torch.log2(expanded[nz] / flat[nz].abs())).long().clamp(0, 7)
    signs = neg[nz].long()
    histogram = torch.bincount(signs * 8 + levels, minlength=16).cpu().tolist()
    partition = {}
    for level in range(8):
        if histogram[level]:
            partition["+2^-%d" % level] = int(histogram[level])
        if histogram[8 + level]:
            partition["-2^-%d" % level] = int(histogram[8 + level])
    return {"positive": int(pos.sum()), "negative": int(neg.sum()),
            "zero": int((~nz).sum()), "nonzero_scalar_events": int(nz.sum()),
            "nonzero_rows": nonzero_rows,
            "sum_abs_graded_amplitude": float(flat.abs().sum().item()),
            "signed_threshold_level_events": partition}


def _sum_stats(total, new):
    for key in ("positive", "negative", "zero", "nonzero_scalar_events", "nonzero_rows"):
        total[key] += new[key]
    total["sum_abs_graded_amplitude"] += new["sum_abs_graded_amplitude"]
    for key, count in new["signed_threshold_level_events"].items():
        total["signed_threshold_level_events"][key] = total["signed_threshold_level_events"].get(key, 0) + count


class SourcePrefixEncoder:
    """Unchanged source unit fed t*(raw_t-raw_(t-1)), decoded by sum(event/t)."""
    def __init__(self, reference, threshold, neuron_class, site):
        device = reference.device
        theta = torch.as_tensor(threshold, dtype=reference.dtype, device=device)
        self.module = neuron_class(T=0, thresh=theta, step_mode="s", num_thresholds=8).to(device)
        self.previous_raw = torch.zeros_like(reference)
        self.decoded = torch.zeros_like(reference)
        self.active_calls = 0
        self.warmup_calls = 0
        warm = self.module(torch.zeros_like(reference))
        self.warmup_calls = 1
        if bool(warm.any()) or int(self.module.T) != 0:
            raise RuntimeError("source coder zero latch changed at %s" % site)

    def step(self, raw, tick):
        if tick != self.active_calls + 1:
            raise RuntimeError("source coder tick sequence reset/mismatch")
        if raw.shape != self.previous_raw.shape or not bool(torch.isfinite(raw).all()):
            raise ValueError("source coder raw prefix shape/nonfinite mismatch")
        source_input = tick * (raw - self.previous_raw)
        if not bool(torch.isfinite(source_input).all()):
            raise ValueError("source coder t*delta input must be finite")
        self.previous_raw = raw.detach().clone()
        event = self.module(source_input)
        self.active_calls += 1
        if int(self.module.T) != tick:
            raise RuntimeError("source active T did not equal current tick")
        if not bool(torch.isfinite(event).all()):
            raise RuntimeError("source coder returned a nonfinite event")
        self.decoded = self.decoded + event / float(tick)
        if not bool(torch.isfinite(self.decoded).all()):
            raise RuntimeError("source coder decoded prefix became nonfinite")
        persistent = list(vars(self.module).items())
        persistent += [("parameter." + name, value)
                       for name, value in self.module._parameters.items()]
        persistent += [("buffer." + name, value)
                       for name, value in self.module._buffers.items()]
        for name, value in persistent:
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise RuntimeError("source coder persistent state %s became nonfinite" % name)
        return self.decoded, event


def _new_encoder(reference, threshold, neuron_class, site, encoder_factory):
    if encoder_factory is None:
        return SourcePrefixEncoder(reference, threshold, neuron_class, site)
    return encoder_factory(reference, threshold, neuron_class, site)


def _raw_embeddings(model, batch):
    x = batch.x.float()
    if x.shape[-1] != model.embedding_h.in_features:
        if getattr(batch, "pos", None) is None:
            raise ValueError("positions required to complete node inputs")
        x = torch.cat((x, batch.pos.float()), dim=-1)
    if x.shape[-1] != model.embedding_h.in_features:
        raise ValueError("node embedding input width mismatch")
    e = batch.edge_attr.float().reshape(-1, model.edge_dim)
    return model.embedding_h(x), model.embedding_e(e)


def _graph_layer(layer, h, e, edge_index):
    src, dst = edge_index.long()
    b_source = layer.B(h)[src]
    z = layer.C(e) + layer.D(h)[src] + layer.E(h)[dst]
    gates = torch.sigmoid(z)
    numerator = h.new_zeros((h.shape[0], layer.out_channels))
    denominator = h.new_zeros((h.shape[0], layer.out_channels))
    numerator.index_add_(0, dst, gates * b_source)
    denominator.index_add_(0, dst, gates)
    node_current = layer.A(h) + numerator / (denominator + layer.eps)
    edge_current = z
    if layer.use_batch_norm:
        node_current = layer.bn_node_h(node_current)
        edge_current = layer.bn_node_e(edge_current)
    if not bool(torch.isfinite(node_current).all()) or not bool(torch.isfinite(edge_current).all()):
        raise RuntimeError("post-BN graph node/edge current became nonfinite")
    node_branch = F.relu(node_current)
    edge_branch = F.relu(edge_current)
    return node_branch, edge_branch


def _pool(model, h, batch):
    pool = {"mean": global_mean_pool, "sum": global_add_pool,
            "max": global_max_pool}[model.readout]
    return pool(h, batch.batch)


def _analog_head(model, graph_h):
    return model.MLP_layer(graph_h)


@torch.no_grad()
def affine_mode_forward(model, batch, thresholds, T, mode, MTH_diff, MTH_diff_line,
                        capture_trace=False, encoder_factory=None,
                        capture_prefix_stats=False, capture_prefix_logits=False):
    """Evaluate one of the three matched modes; never calls model.forward.

    `encoder_factory` is a synthetic-test hook with signature
    `(reference, threshold, neuron_class, site) -> object(step(raw,tick))`.
    The normal path always instantiates the pinned source neuron classes.
    """
    specs = validate_batch_model_thresholds(model, batch, thresholds, T, mode)
    base_h, base_e = _raw_embeddings(model, batch)
    if not bool(torch.isfinite(base_h).all()) or not bool(torch.isfinite(base_e).all()):
        raise ValueError("initial embeddings must be finite")
    spec_map = {name: (threshold, kind) for (name, _, kind), threshold in zip(specs, thresholds)}
    encoders = {}
    for name, width, kind in specs:
        if kind == "line":
            neuron_class = MTH_diff_line
        else:
            neuron_class = MTH_diff
        reference = base_h if name.endswith(".node") else base_e
        threshold = spec_map[name][0]
        encoders[name] = _new_encoder(reference, threshold, neuron_class, name, encoder_factory)
    site_stats = {name: {"positive": 0, "negative": 0, "zero": 0,
                         "nonzero_scalar_events": 0, "nonzero_rows": 0,
                         "sum_abs_graded_amplitude": 0.0,
                         "signed_threshold_level_events": {}}
                  for name, _, _ in specs}

    # source input trace order mirrors graph depth; depth0 is exact in modes2/3.
    trace_ticks, tick_logits, head_calls = [], [], 0
    prefix_site_stats, prefix_logits = {}, {}
    prefix_ticks = [value for value in (1, 2, 4, 8) if value <= T]
    final_graph_h = None
    layers = list(model.layers)
    for tick in range(1, T + 1):
        raw_h, raw_e = base_h, base_e
        centered_q_h = centered_q_e = None
        if mode == "centered":
            centered_q_h = torch.zeros_like(base_h)
            centered_q_e = torch.zeros_like(base_e)
        tick_layers = []
        for layer_index, layer in enumerate(layers):
            if mode == "centered" and layer_index > 0:
                raw_node_input = base_h + centered_q_h
                raw_edge_input = base_e + centered_q_e
                node_name = "layer%d.node" % layer_index
                edge_name = "layer%d.edge" % layer_index
                decoded_q_h, event_h = encoders[node_name].step(centered_q_h, tick)
                decoded_q_e, event_e = encoders[edge_name].step(centered_q_e, tick)
                consumer_h, consumer_e = base_h + decoded_q_h, base_e + decoded_q_e
                _sum_stats(site_stats[node_name], _site_stats(event_h, spec_map[node_name][0]))
                _sum_stats(site_stats[edge_name], _site_stats(event_e, spec_map[edge_name][0]))
            else:
                raw_node_input, raw_edge_input = raw_h, raw_e
                is_analog_first = layer_index == 0 and mode in ("first_analog", "centered")
                if is_analog_first:
                    consumer_h, consumer_e = raw_h, raw_e
                else:
                    node_name = "layer%d.node" % layer_index
                    edge_name = "layer%d.edge" % layer_index
                    consumer_h, event_h = encoders[node_name].step(raw_h, tick)
                    consumer_e, event_e = encoders[edge_name].step(raw_e, tick)
                    _sum_stats(site_stats[node_name], _site_stats(event_h, spec_map[node_name][0]))
                    _sum_stats(site_stats[edge_name], _site_stats(event_e, spec_map[edge_name][0]))

            node_branch, edge_branch = _graph_layer(layer, consumer_h, consumer_e,
                                                   batch.edge_index)
            if mode == "centered":
                if layer_index == 0:
                    centered_q_h, centered_q_e = node_branch, edge_branch
                else:
                    # Preserve the raw q/r skip. Never substitute decoded q/r.
                    centered_q_h = centered_q_h + node_branch
                    centered_q_e = centered_q_e + edge_branch
                raw_h, raw_e = base_h + centered_q_h, base_e + centered_q_e
            else:
                # Generic and first-analog raw skip forwards its own graded prefix.
                raw_h = raw_h + node_branch
                raw_e = raw_e + edge_branch
            if capture_trace:
                tick_layers.append({"layer": layer_index,
                                    "raw_node_input": raw_node_input.detach().clone(),
                                    "raw_edge_input": raw_edge_input.detach().clone(),
                                    "consumer_node_input": consumer_h.detach().clone(),
                                    "consumer_edge_input": consumer_e.detach().clone(),
                                    "raw_node_output": raw_h.detach().clone(),
                                    "raw_edge_output": raw_e.detach().clone()})
        final_graph_h = _pool(model, raw_h, batch)
        if capture_prefix_stats and tick in prefix_ticks:
            prefix_site_stats[str(tick)] = copy.deepcopy([site_stats[name] for name, _, _ in specs])
        should_compute_head = capture_trace or (capture_prefix_logits and tick in prefix_ticks)
        current_logits = None
        if should_compute_head:
            current_logits = _analog_head(model, final_graph_h)
            head_calls += 1
        if capture_trace:
            tick_logits.append(current_logits)
        if capture_prefix_logits and tick in prefix_ticks:
            prefix_logits[str(tick)] = current_logits
        if capture_trace:
            trace_ticks.append({"tick": tick, "base_node": base_h.detach().clone(),
                                "base_edge": base_e.detach().clone(), "layers": tick_layers})

    if capture_trace:
        logits = tick_logits[-1]
    elif capture_prefix_logits:
        logits = prefix_logits[str(T)]
    else:
        logits = _analog_head(model, final_graph_h)
        head_calls = 1
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("nonfinite affine-base output")
    coders = list(encoders.values())
    source_ticks = [int(getattr(coder, "T", getattr(getattr(coder, "module", None), "T", -1)))
                    for coder in coders]
    warmup_calls = sum(int(getattr(coder, "warmup_calls", 1)) for coder in coders)
    active_calls = sum(int(getattr(coder, "active_calls", getattr(coder, "calls", T))) for coder in coders)
    ordered_stats = [site_stats[name] for name, _, _ in specs]
    return {"mode": mode, "T": int(T), "logits": logits,
            "tick_logits": tick_logits, "head_calls": head_calls,
            "source_warmup_calls": warmup_calls, "active_unit_calls": active_calls,
            "source_ticks": source_ticks, "site_names": [name for name, _, _ in specs],
            "site_stats": ordered_stats,
            "prefix_site_stats": prefix_site_stats,
            "prefix_logits": prefix_logits,
            "trace": trace_ticks if capture_trace else [],
            "encoder_kind": "signed MTH_diff_line" if mode in ("generic", "first_analog")
                            else "nonnegative MTH_diff residual code",
            "analog_head": "same GatedTeacher MLPReadout; final head once unless capture_trace",
            "scope": "dense synthetic/reference adaptation; no cache backend, accuracy, novelty, or energy claim"}


def validate_source_unchanged(author_repo, provenance):
    _, _, now = require_author_sources(author_repo)
    if now != provenance:
        raise RuntimeError("pinned author source changed since it was bound")
    return now

