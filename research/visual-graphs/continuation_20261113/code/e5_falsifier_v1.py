"""R6 E5: real-data normalized-gate bound and two falsifiers.

This is a DIAGNOSTIC, not a new converter. It reuses the frozen helpers from
gated_affine_base_v1.py and the frozen seed-41 teacher/thresholds. The
instrumented centered forward is identical to affine_mode_forward when both
counterfactual switches are off; that equality is CHECKED on real data before
any falsifier result is accepted.

Evidence produced (all on the official CIFAR10 validation order, label-free):

  1. Real-data gate-bound: at every T in {1,2,4,8} and all four graph layers,
     compare the raw-skip trajectory against the decoded consumer trajectory:
         |m(z',b') - m(z,b)| <= max_i|b_i'-b_i| + R(b') tanh(eta/2)
     with eta = per-target-channel max |z'-z| over incoming edges and R(b')
     the converted message range including the zero dummy item. Actual message
     error, pre-BN layer error and logit error are reported separately.
     Degree (in-degree median split) and sharpness (teacher logit margin median
     split) subsets are reported.
  2. Fresh-vs-stale gates falsifier: gates are frozen from tick 1 and reused.
     The same bound is checked against the observed stale/fresh message error,
     and the logit-level change is measured.
  3. Raw-vs-decoded skip falsifier: the raw accumulated prefix is replaced by
     the decoder output. The logit-level change is measured.

A bound is local (message-level); no accuracy, energy, novelty, floating-point
certificate or method-advantage claim is made. Loose bounds are labelled.
No new test inference, no teacher hidden-state oracle inside the converter.

Env: GNN2SNN_DATA_ROOT (official PyG root), DCGS_REPO (pinned checkout).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = STUDY.parents[2]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
RR = "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix"
OUT = RUNROOT / "results" / "e5-falsifier-v1"
LOGDIR = RUNROOT / "logs"

PROTOCOL_SHA = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
ADDENDUM_SHA = "8bde5832d556c8c7b04ee630f2390703c33a26fa750f196f4dc1c94da61291f6"
DCGS_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
TIMESTEPS = (1, 2, 4, 8)
TOL_ABS, TOL_REL = 1e-4, 1e-3

CSV_COLUMNS = ("task_id", "agent_id", "run_id", "dataset", "split", "coverage",
               "method", "teacher_seed", "student_seed", "T", "metric", "value",
               "unit", "evidence", "protocol_sha256", "status")
TASK, AGENT, RUN_ID = "visual-conversion", "20261113-wb-resume", "e5-falsifier-v1"


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def default_data_root() -> Path:
    v = os.environ.get("GNN2SNN_DATA_ROOT")
    require(bool(v), "set GNN2SNN_DATA_ROOT to the official PyG data root")
    return Path(v)


def default_author_repo() -> Path:
    import tempfile
    v = os.environ.get("DCGS_REPO")
    return Path(v) if v else Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"


# ----------------------------------------------------------------- primitives
def scatter(src, dst, n, reduce, init=0.0):
    """scatter_reduce over incoming-edge index dst (E,) into (n, C) tensor."""
    import torch
    out = torch.full((n, src.shape[1]), init, dtype=src.dtype, device=src.device)
    idx = dst.view(-1, 1).expand_as(src)
    out.scatter_reduce_(0, idx, src, reduce=reduce, include_self=True)
    return out


def gate_parts(layer, h, e, edge_index, gates_override=None):
    """Frozen GatedGCN gate aggregate, exposed for the bound check."""
    import torch
    src, dst = edge_index.long()
    b = layer.B(h)[src]
    z = layer.C(e) + layer.D(h)[src] + layer.E(h)[dst]
    g = torch.sigmoid(z) if gates_override is None else gates_override
    num = scatter(g * b, dst, h.shape[0], "sum")
    den = scatter(g, dst, h.shape[0], "sum")
    m = num / (den + layer.eps)
    return z, g, b, m, src, dst


def custom_layer(layer, h, e, edge_index, gates_override):
    """Only used for the stale-gate counterfactual; all other ops frozen."""
    import torch
    import torch.nn.functional as F
    src, dst = edge_index.long()
    b = layer.B(h)[src]
    z = layer.C(e) + layer.D(h)[src] + layer.E(h)[dst]
    num = scatter(gates_override * b, dst, h.shape[0], "sum")
    den = scatter(gates_override, dst, h.shape[0], "sum")
    cur = layer.A(h) + num / (den + layer.eps)
    ecur = z
    if layer.use_batch_norm:
        cur = layer.bn_node_h(cur)
        ecur = layer.bn_node_e(ecur)
    return F.relu(cur), F.relu(ecur)


def bound_terms(z_ref, z_per, b_ref, b_per, dst, n):
    """Return (eta, bound) for |m(z',b')-m(z,b)| <= max|db| + R(b')tanh(eta/2)."""
    import torch
    eta = scatter((z_per - z_ref).abs(), dst, n, "amax")
    dmax = scatter((b_per - b_ref).abs(), dst, n, "amax")
    amax = scatter(b_per, dst, n, "amax", 0.0)
    amin = scatter(b_per, dst, n, "amin", 0.0)
    r = amax - amin                      # converted message range, dummy zero included
    return eta, dmax + r * torch.tanh(eta / 2)


class Acc:
    """Running scalar accumulator for node- or graph-level reductions."""
    def __init__(self):
        self.count = 0
        self.violations = 0
        self.max_actual = 0.0
        self.max_bound = 0.0
        self.max_excess = 0.0
        self.max_logit_error = 0.0
        self.max_layer_error = 0.0

    def add(self, actual, bound, tol, mask=None, layer_err=0.0):
        import torch
        a = actual if mask is None else actual[mask]
        b = bound if mask is None else bound[mask]
        if a.numel() == 0:
            return
        excess = (a - (b + tol)).clamp(min=0)
        self.count += int(a.numel())
        self.violations += int((a > b + tol).sum().item())
        self.max_actual = max(self.max_actual, float(a.max().item()))
        self.max_bound = max(self.max_bound, float(b.max().item()))
        self.max_excess = max(self.max_excess, float(excess.max().item()))
        self.max_layer_error = max(self.max_layer_error, float(layer_err))

    def as_dict(self):
        return {"elements": self.count, "bound_violations": self.violations,
                "max_actual": self.max_actual, "max_bound": self.max_bound,
                "max_excess_over_bound": self.max_excess,
                "max_pre_bn_layer_error": self.max_layer_error}


def graph_reduce(node_scalar, batch_index, n_graphs):
    """(N,) -> (G,) max over nodes of the same graph."""
    return scatter(node_scalar.view(-1, 1), batch_index.long(), n_graphs, "amax").view(-1)


# ------------------------------------------------------- instrumented forward
@torch.no_grad()
def instrumented_centered(model, batch, thresholds, T, MTH_diff, MTH_diff_line,
                          affine, decoded_skip=False, stale_gates=False,
                          collect=False, deg_low=None, sharp_low_graph=None,
                          accum=None):
    """Copy of the frozen centered loop with two counterfactual switches.

    With decoded_skip=False and stale_gates=False this reproduces
    affine.affine_mode_forward(..., mode='centered') exactly: the layer call is
    the frozen affine._graph_layer and the skip recurrence is unchanged.
    """
    import torch
    specs = affine.validate_batch_model_thresholds(model, batch, thresholds, T, "centered")
    base_h, base_e = affine._raw_embeddings(model, batch)
    spec_map = {name: th for (name, _, _), th in zip(specs, thresholds)}
    encoders = {}
    for name, width, kind in specs:
        ref = base_h if name.endswith(".node") else base_e
        neuron_class = MTH_diff_line if kind == "line" else MTH_diff
        encoders[name] = affine._new_encoder(ref, spec_map[name], neuron_class, name, None)

    edge_index = batch.edge_index
    src, dst = edge_index.long()
    n_nodes = int(base_h.shape[0])
    first_z, first_g = {}, {}
    raw_skip_diff = 0.0
    logits = None

    for tick in range(1, T + 1):
        centered_q_h = torch.zeros_like(base_h)
        centered_q_e = torch.zeros_like(base_e)
        raw_h, raw_e = base_h, base_e
        for li, layer in enumerate(model.layers):
            if li == 0:
                raw_node_input, raw_edge_input = base_h, base_e
                consumer_h, consumer_e = base_h, base_e
                decoded_q_h = decoded_q_e = None
            else:
                raw_node_input = base_h + centered_q_h
                raw_edge_input = base_e + centered_q_e
                nh, ne = "layer%d.node" % li, "layer%d.edge" % li
                decoded_q_h, _ = encoders[nh].step(centered_q_h, tick)
                decoded_q_e, _ = encoders[ne].step(centered_q_e, tick)
                consumer_h = base_h + decoded_q_h
                consumer_e = base_e + decoded_q_e

            # Tick-1 gate logits/gates are cached for BOTH the stale-gate
            # counterfactual and the real-data fresh-vs-stale bound check.
            # This is computed on consumer (converter) inputs only.
            if tick == 1:
                z0 = layer.C(consumer_e) + layer.D(consumer_h)[src] + layer.E(consumer_h)[dst]
                first_z[li] = z0.detach().clone()
                first_g[li] = torch.sigmoid(z0).detach().clone()

            if stale_gates:
                node_branch, edge_branch = custom_layer(layer, consumer_h, consumer_e,
                                                        edge_index, first_g[li])
            else:
                node_branch, edge_branch = affine._graph_layer(layer, consumer_h,
                                                               consumer_e, edge_index)

            raw_skip_diff = max(raw_skip_diff,
                                float((consumer_h - raw_node_input).abs().max().item()),
                                float((consumer_e - raw_edge_input).abs().max().item()))

            # Bound evidence is collected at the FINAL tick of each T (the tick
            # whose logits are the converted output), for all four layers.
            if collect and tick == T and accum is not None:
                z_r, g_r, b_r, m_r, _src, dst = gate_parts(layer, raw_node_input,
                                                           raw_edge_input, edge_index)
                z_d, g_d, b_d, m_d, _src, dst = gate_parts(layer, consumer_h,
                                                           consumer_e, edge_index)
                eta1, bound1 = bound_terms(z_r, z_d, b_r, b_d, dst, n_nodes)
                actual1 = (m_d - m_r).abs()
                tol1 = TOL_ABS + TOL_REL * bound1
                layer_err = float((layer.A(consumer_h) + m_d -
                                   (layer.A(raw_node_input) + m_r)).abs().max().item())
                A = accum["raw_vs_decoded"]
                A["all"].add(actual1, bound1, tol1, layer_err=layer_err)
                A["degree_low"].add(actual1, bound1, tol1, mask=deg_low, layer_err=layer_err)
                A["degree_high"].add(actual1, bound1, tol1, mask=~deg_low, layer_err=layer_err)
                gmax = graph_reduce(actual1.max(dim=1).values, batch.batch,
                                    int(batch.num_graphs))
                bmax = graph_reduce(bound1.max(dim=1).values, batch.batch,
                                    int(batch.num_graphs))
                viol = graph_reduce((actual1 > bound1 + tol1).float().max(dim=1).values,
                                    batch.batch, int(batch.num_graphs))
                for key, mask in (("sharpness_low", sharp_low_graph),
                                  ("sharpness_high", ~sharp_low_graph)):
                    a = A[key]
                    a.add(gmax[mask].view(-1, 1), bmax[mask].view(-1, 1),
                          TOL_ABS, layer_err=layer_err)
                    a.violations += int((viol[mask] > 0).sum().item())

                # Fresh vs stale: b is unchanged, gates come from tick 1.
                if li in first_z:
                    _z, _g, _b, m_stale, _s, _d = gate_parts(
                        layer, consumer_h, consumer_e, edge_index,
                        gates_override=first_g[li])
                    eta2, bound2 = bound_terms(first_z[li], z_d, b_d, b_d, dst, n_nodes)
                    actual2 = (m_stale - m_d).abs()
                    tol2 = TOL_ABS + TOL_REL * bound2
                    B = accum["fresh_vs_stale"]
                    B["all"].add(actual2, bound2, tol2)
                    B["degree_low"].add(actual2, bound2, tol2, mask=deg_low)
                    B["degree_high"].add(actual2, bound2, tol2, mask=~deg_low)

            if decoded_skip and li > 0:
                centered_q_h, centered_q_e = decoded_q_h, decoded_q_e
            else:
                if li == 0:
                    centered_q_h, centered_q_e = node_branch, edge_branch
                else:
                    centered_q_h = centered_q_h + node_branch
                    centered_q_e = centered_q_e + edge_branch
            raw_h, raw_e = base_h + centered_q_h, base_e + centered_q_e

        graph_h = affine._pool(model, raw_h, batch)
        logits = affine._analog_head(model, graph_h)

    return logits, raw_skip_diff


def new_accum():
    keys = ("all", "degree_low", "degree_high", "sharpness_low", "sharpness_high")
    return {name: {k: Acc() for k in keys} for name in ("raw_vs_decoded", "fresh_vs_stale")}


# ------------------------------------------------------------------ subcommands
def precheck() -> None:
    import torch  # noqa: F401
    data_root = default_data_root()
    author_repo = default_author_repo()
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e1_residual_v1 as R
    mods = R.load_modules(author_repo)
    b = R.verify_bindings(41, data_root, mods)
    ap = R.author_provenance(author_repo, mods)
    require(ap["safe"]["commit"] == DCGS_COMMIT, "pinned DCGS commit drift")
    gate = {"check": "e5_falsifier_precheck_v1", "status": "PASS",
            "frozen_files_verified": len(b["frozen_hashes"]),
            "processed_files_verified": len(b["data_hashes"]),
            "shared_theta_vectors": len(b["shared_theta_vectors"]),
            "pinned_commit": ap["safe"]["commit"],
            "note": "hash/structure gate only; no GPU, no scientific result"}
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (LOGDIR / "e5-falsifier-precheck-v1.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


def run(args) -> None:
    import numpy as np
    import torch
    from torch_geometric.datasets import GNNBenchmarkDataset
    from torch_geometric.loader import DataLoader

    data_root = default_data_root().resolve()
    author_repo = default_author_repo().resolve()
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e1_residual_v1 as R
    mods = R.load_modules(author_repo)
    import eval_affine_validation as base
    import gated_affine_base_v1 as affine
    import gated_affine_cal_v1 as cal
    binding = R.verify_bindings(41, data_root, mods)
    ap = R.author_provenance(author_repo, mods)
    MTH_diff, MTH_diff_line = ap["MTH_diff"], ap["MTH_diff_line"]
    require(ap["safe"]["commit"] == DCGS_COMMIT, "pinned DCGS commit drift")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    torch.manual_seed(41)
    np.random.seed(41)
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")

    td = R.BINDINGS[41]["teacher"]
    state = torch.load(td / "best.pt", map_location=device, weights_only=False)
    require(state.get("metadata") == binding["metadata"], "checkpoint metadata mismatch")
    _base, _affine, _cal, _hybrid, _author, _shared, teacher_helper = mods
    model = teacher_helper.build_teacher(state, device).eval()
    require(len(model.layers) == 4, "frozen four-layer teacher drift")

    thresholds = [t.to(device=device, dtype=torch.float32)
                  for t in cal.profile_thresholds(binding["affine_doc"], "centered", 1)]
    require(len(thresholds) == 6, "centered requires six frozen threshold vectors")

    ds = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    require(len(ds) == 5000, "official validation count drift")
    loader = DataLoader(base.LabelFreeDataset(ds), batch_size=128, shuffle=False,
                        num_workers=0)
    batches = []
    for i, b in enumerate(loader):
        if i >= args.batches:
            break
        batches.append(b.to(device))
    require(len(batches) == args.batches, "not enough batches")

    accum = {str(T): new_accum() for T in TIMESTEPS}
    fidelity = []
    fals = {str(T): {"decoded_skip_logit_max_abs_change": 0.0,
                     "stale_gate_logit_max_abs_change": 0.0,
                     "teacher_logit_max_abs_error": 0.0,
                     "raw_vs_decoded_input_max_abs_difference": 0.0,
                     "graphs": 0} for T in TIMESTEPS}
    t_start = time.monotonic()
    for bi, batch in enumerate(batches):
        require(getattr(batch, "y", None) is None, "labels leaked into converter input")
        dst = batch.edge_index[1].long()
        n_nodes = int(batch.x.shape[0])
        deg = scatter(torch.ones((dst.numel(), 1), device=device), dst,
                      n_nodes, "sum").view(-1)
        deg_low = deg <= deg.median()
        with torch.no_grad():
            teacher_logits = model(batch)
        top2 = torch.topk(teacher_logits, 2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        sharp_low_graph = margin <= margin.median()

        for T in TIMESTEPS:
            acc = accum[str(T)]
            # Frozen reference (truth) for this (batch, T).
            ref = affine.affine_mode_forward(model, batch, thresholds, T, "centered",
                                             MTH_diff, MTH_diff_line)
            # Fidelity: switches off must reproduce the frozen forward.
            plain, raw_diff = instrumented_centered(
                model, batch, thresholds, T, MTH_diff, MTH_diff_line, affine)
            fd = float((plain - ref["logits"]).abs().max().item())
            fidelity.append({"batch": bi, "T": T, "max_abs_diff": fd})
            require(fd <= 1e-4, "instrumented forward is not faithful to frozen converter")

            # Bound evidence collected on the faithful (switches-off) forward.
            instrumented_centered(model, batch, thresholds, T, MTH_diff,
                                  MTH_diff_line, affine, collect=True,
                                  deg_low=deg_low, sharp_low_graph=sharp_low_graph,
                                  accum=acc)

            dec_logits, _ = instrumented_centered(model, batch, thresholds, T,
                                                  MTH_diff, MTH_diff_line, affine,
                                                  decoded_skip=True)
            stale_logits, _ = instrumented_centered(model, batch, thresholds, T,
                                                    MTH_diff, MTH_diff_line, affine,
                                                    stale_gates=True)
            f = fals[str(T)]
            f["decoded_skip_logit_max_abs_change"] = max(
                f["decoded_skip_logit_max_abs_change"],
                float((dec_logits - ref["logits"]).abs().max().item()))
            f["stale_gate_logit_max_abs_change"] = max(
                f["stale_gate_logit_max_abs_change"],
                float((stale_logits - ref["logits"]).abs().max().item()))
            f["teacher_logit_max_abs_error"] = max(
                f["teacher_logit_max_abs_error"],
                float((ref["logits"] - teacher_logits).abs().max().item()))
            f["raw_vs_decoded_input_max_abs_difference"] = max(
                f["raw_vs_decoded_input_max_abs_difference"], raw_diff)
            f["graphs"] += int(batch.num_graphs)
        if bi % 8 == 0 or bi == len(batches) - 1:
            print(json.dumps({"batch": bi + 1, "of": len(batches),
                              "elapsed_sec": round(time.monotonic() - t_start, 1)}),
                  flush=True)

    data_end = base.processed_hashes(data_root)
    require(data_end == binding["metadata"]["provenance"]["processed_data_sha256"],
            "processed data changed during falsifier job")

    def dump_acc(a):
        return {k: {s: v.as_dict() for s, v in sub.items()} for k, sub in a.items()}

    report = {
        "status": "real_data_gate_bound_and_falsifiers_no_accuracy_or_energy_claim",
        "check": "e5_gate_bound_v1",
        "protocol_sha256": PROTOCOL_SHA, "head_addendum_sha256": ADDENDUM_SHA,
        "teacher_seed": 41, "mode": "centered",
        "device": str(device), "gpu_name": (torch.cuda.get_device_name(device)
                                            if device.type == "cuda" else "cpu"),
        "torch": torch.__version__,
        "coverage": {"batches": args.batches, "graphs": sum(int(b.num_graphs) for b in batches),
                     "split": "official validation, fixed sequential order, label-free",
                     "timesteps": list(TIMESTEPS), "layers_per_T": 4,
                     "bound_collected_at": "final tick of each T"},
        "bound_statement": "|m(z',b')-m(z,b)| <= max_i|b_i'-b_i| + R(b')tanh(eta/2); "
                           "eta is the per-target-channel max incoming |z'-z|, "
                           "R(b') is the converted message range with the zero dummy item",
        "tolerance": {"absolute": TOL_ABS, "relative_to_bound": TOL_REL,
                      "reason": "float32 scatter/index_add accumulation; declared, not loosened silently"},
        "subsets": {"degree": "node in-degree median split (low = <= median)",
                    "sharpness": "teacher top-1 minus top-2 logit margin median split "
                                 "(low = <= median); margin is label-free"},
        "fidelity_check": {"frozen_vs_instrumented_max_abs_diff": max(x["max_abs_diff"] for x in fidelity),
                           "checked_pairs": len(fidelity), "threshold": 1e-4,
                           "note": "counterfactual implementation is faithful only if this passes"},
        "gate_bound": {str(T): dump_acc(accum[str(T)]["raw_vs_decoded"]) for T in TIMESTEPS},
        "fresh_vs_stale_bound": {str(T): dump_acc(accum[str(T)]["fresh_vs_stale"]) for T in TIMESTEPS},
        "measured_errors_and_falsifiers": fals,
        "falsifier_verdicts": {
            "raw_vs_decoded_skip": {
                "interpretation": "if substituting decoded skips produces zero logit change, "
                                  "the raw-skip distinction is falsified for this trajectory",
                "per_T_distinction_falsified": {str(T): bool(fals[str(T)]["decoded_skip_logit_max_abs_change"] == 0.0)
                                                 for T in TIMESTEPS}},
            "fresh_vs_stale_gates": {
                "interpretation": "if reusing tick-1 gates produces zero logit change, "
                                  "fresh gate recomputation is falsified for this trajectory",
                "per_T_gate_refresh_falsified": {str(T): bool(fals[str(T)]["stale_gate_logit_max_abs_change"] == 0.0)
                                                  for T in TIMESTEPS}},
        },
        "limitations": [
            "The bound is a LOCAL real-arithmetic message bound, not a floating-point certificate, "
            "accuracy guarantee, logit certificate, efficiency or novelty result.",
            "Bounds computed with absolute-value propagation are loose; loose is reported as loose.",
            "Logit errors are measured, not bounded; a final logit certificate would need "
            "pooling/head weight-norm propagation and a class-margin condition.",
            "Counterfactual switches are diagnostics, not new conversion methods and not baselines.",
            "No new test inference, no threshold/seed/scale tuning, no event-count-as-energy claim.",
        ],
        "total_job_seconds": time.monotonic() - t_start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / "e5_gate_bound_v1.json"
    require(not rp.exists(), "refusing to overwrite published gate-bound report")
    rp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                  encoding="utf-8")
    metric_rows = []
    for T in TIMESTEPS:
        for kind in ("raw_vs_decoded", "fresh_vs_stale"):
            for subset, a in accum[str(T)][kind].items():
                d = a.as_dict()
                for name, value in (("max_actual", d["max_actual"]),
                                    ("max_bound", d["max_bound"]),
                                    ("bound_violations", d["bound_violations"])):
                    metric_rows.append(dict(zip(CSV_COLUMNS, (
                        TASK, AGENT, RUN_ID, "CIFAR10", "val",
                        f"{sum(int(b.num_graphs) for b in batches)}_fixed_prefix_val",
                        f"gate_bound_{kind}_{subset}", 41, "", T, name, value,
                        "scalar" if name != "bound_violations" else "count",
                        f"{RR}/results/e5-falsifier-v1/e5_gate_bound_v1.json",
                        PROTOCOL_SHA, "measured"))))
    with (OUT / "falsifier_METRICS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(metric_rows)
    print(json.dumps({"status": report["status"],
                      "fidelity_max_abs_diff": report["fidelity_check"]["frozen_vs_instrumented_max_abs_diff"],
                      "total_job_seconds": round(report["total_job_seconds"], 1)}, indent=2))


def verify() -> None:
    p = OUT / "e5_gate_bound_v1.json"
    require(p.is_file(), "missing gate-bound report")
    d = json.loads(p.read_text(encoding="utf-8"))
    require(d["fidelity_check"]["frozen_vs_instrumented_max_abs_diff"] <= 1e-4,
            "fidelity check fails on re-read")
    for T in TIMESTEPS:
        for k in ("gate_bound", "fresh_vs_stale_bound"):
            require(str(T) in d[k], f"missing {k} T{T}")
        for s in ("all", "degree_low", "degree_high", "sharpness_low", "sharpness_high"):
            require(s in d["gate_bound"][str(T)], f"missing subset {s} T{T}")
    require(bool(d.get("limitations")), "limitations missing")
    require(d["coverage"]["timesteps"] == list(TIMESTEPS), "timestep coverage drift")
    print(json.dumps({"check": "e5_falsifier_verify_v1", "status": "PASS",
                      "report_sha256": sha256(p),
                      "timesteps": list(TIMESTEPS)}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("precheck")
    sub.add_parser("verify")
    r = sub.add_parser("run")
    r.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    r.add_argument("--batches", type=int, default=40)
    a = p.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        {"precheck": precheck, "verify": verify}[a.cmd]()


if __name__ == "__main__":
    main()
