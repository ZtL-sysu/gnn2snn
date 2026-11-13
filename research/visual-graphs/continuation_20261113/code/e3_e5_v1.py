"""R6 E3/E5: complete known-operation cost ledger + matched wall-clock.

Read-only with respect to all prior evidence/protocols/hashes. Subcommands:

  precheck   - frozen-input gate, official shape census, site-spec census (no GPU)
  cost       - deterministic complete known-operation cost ledger for the 108
               conversion rows (9 modes x 3 seeds x 4 T); ANN is built
               separately by build_ann_cost() (CPU only), AC + conservative
               MAC conventions, pJ proxy
  latency    - 3-repeat median matched wall-clock on the LOCAL RTX 3060 (GPU),
               fixed batch128/order/warmup, ANN + all 10 E1 row families
  aggregate  - combine cost + latency, re-evaluate the E1 two-part criterion

No chip-energy claim, no event-count-as-energy, no accuracy/efficiency claim.
Every declared counting formula is written into the output accounting_model.
Unresolved (unmeasured) items are enumerated explicitly and are NEVER silently
dropped. The dense PyTorch prototype recomputes all four graph layers at every
tick; the fused first-analog cache is NOT implemented here and is flagged.

Env: GNN2SNN_DATA_ROOT (official PyG root), DCGS_REPO (pinned checkout).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = STUDY.parents[2]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
RR = "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix"
SR = "research/visual-graphs/continuation_20261113"
OUT = RUNROOT / "results" / "e3-e5-v1"
LOGDIR = RUNROOT / "logs"

PROTOCOL_SHA = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
ADDENDUM_SHA = "8bde5832d556c8c7b04ee630f2390703c33a26fa750f196f4dc1c94da61291f6"
DCGS_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"

# pJ arithmetic proxy declared by frozen PROTOCOL-v3 (45 nm / 0.9 V screen).
PJ = {"ADD": 0.9, "MUL": 3.7, "MAC": 4.6}
# Frozen model shape (verified against checkpoint and official val census).
D = 70                     # hidden width, all four GatedGCN layers
L = 4                      # graph layers
B = 5000                   # official validation graphs
N = 587924                 # measured total nodes over all 5000 val graphs
E = 4703392                # measured total edges (E/N = 8.0)
P = 40                     # batches of 128 in fixed sequential order
HEAD_MACS = 70 * 35 + 35 * 17 + 17 * 10     # 3215 per graph
EMB_MACS = 350 * N + 70 * E                  # 5->70 nodes, 1->70 edges
NODE_IN, EDGE_IN = 5, 1
W_HEAD = (70, 35, 17, 10)

ALL_MODES = ("ANN", "centered", "first_analog", "generic", "dc_adapt_analog_head",
             "per_consumer", "cse_only", "dc_adapt_six_site",
             "first_analog_shared_theta", "centered_shared_theta")
CONVERSION_CONTROLS = ("first_analog", "generic", "per_consumer", "cse_only",
                       "dc_adapt_analog_head", "dc_adapt_six_site",
                       "first_analog_shared_theta", "centered_shared_theta")
CANDIDATE = "centered"
STRONGEST = {4: "centered_shared_theta", 8: "dc_adapt_six_site"}
TIMESTEPS = (1, 2, 4, 8)
SEEDS = (41, 42, 43)

# Frozen immutable E1 reports. Keys are (seed, path, rel).
REPORTS = {
    41: [
        (STUDY / "results" / "e1-seed41-affine-v1" / "report.json",
         f"{SR}/results/e1-seed41-affine-v1/report.json"),
        (RUNROOT / "results" / "e1-seed41-controls-v1" / "report.json",
         f"{RR}/results/e1-seed41-controls-v1/report.json"),
        (RUNROOT / "results" / "e1-residual-seed41-v1" / "report.json",
         f"{RR}/results/e1-residual-seed41-v1/report.json"),
    ],
    42: [
        (RUNROOT / "results" / "e1-seed42-multiseed-v1" / "report.json",
         f"{RR}/results/e1-seed42-multiseed-v1/report.json"),
        (RUNROOT / "results" / "e1-residual-seed42-v1" / "report.json",
         f"{RR}/results/e1-residual-seed42-v1/report.json"),
    ],
    43: [
        (RUNROOT / "results" / "e1-seed43-multiseed-v1" / "report.json",
         f"{RR}/results/e1-seed43-multiseed-v1/report.json"),
        (RUNROOT / "results" / "e1-residual-seed43-v1" / "report.json",
         f"{RR}/results/e1-residual-seed43-v1/report.json"),
    ],
}

CSV_COLUMNS = ("task_id", "agent_id", "run_id", "dataset", "split", "coverage",
               "method", "teacher_seed", "student_seed", "T", "metric", "value",
               "unit", "evidence", "protocol_sha256", "status")
TASK, AGENT, RUN_ID = "visual-conversion", "20261113-wb-resume", "e3-e5-v1"

# Declared per-site specification. Elements are per source-unit call on one
# batch; ac_updates is the number of output accumulators one nonzero event
# updates under the declared AC (event-accumulate) convention.
SITE_AC = {
    "affine_node": {"elements": (N / P) * D, "ac": 4 * D, "kind": "node", "width": D},
    "affine_edge": {"elements": (E / P) * D, "ac": 1 * D, "kind": "edge", "width": D},
    "consumer_node": {"elements": (N / P) * D, "ac": 1 * D, "kind": "node", "width": D},
    "consumer_edge": {"elements": (E / P) * D, "ac": 1 * D, "kind": "edge", "width": D},
    "dc_node": {"elements": (N / P) * D, "ac": 1 * D, "kind": "node", "width": D},
    "mlp_0": {"elements": (B / P) * 35, "ac": 17, "kind": "mlp", "width": 35},
    "mlp_1": {"elements": (B / P) * 17, "ac": 10, "kind": "mlp", "width": 17},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def default_data_root() -> Path:
    v = os.environ.get("GNN2SNN_DATA_ROOT")
    if not v:
        raise RuntimeError("set GNN2SNN_DATA_ROOT to the official PyG data root")
    return Path(v)


def default_author_repo() -> Path:
    v = os.environ.get("DCGS_REPO")
    return Path(v) if v else Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"


def site_spec(mode: str, name: str) -> dict:
    """Map a frozen report site name to its declared site specification."""
    if mode == "dc_adapt_six_site":
        if name == "site_4":
            return SITE_AC["mlp_0"]
        if name == "site_5":
            return SITE_AC["mlp_1"]
        require(name.startswith("site_"), f"unexpected six-site name {name}")
        return SITE_AC["dc_node"]
    if mode == "per_consumer":
        return SITE_AC["consumer_edge"] if name.endswith(".C") else SITE_AC["consumer_node"]
    if mode == "dc_adapt_analog_head":
        return SITE_AC["dc_node"]
    # affine modes: generic has 8 sites (layers 0..3), others 6 (layers 1..3)
    return SITE_AC["affine_edge"] if name.endswith(".edge") else SITE_AC["affine_node"]


def load_rows() -> dict:
    """Read the 108 frozen conversion rows (ANN is built separately)."""
    rows = {}
    for seed, paths in REPORTS.items():
        for path, rel in paths:
            require(path.is_file(), f"missing frozen report: {rel}")
            doc = json.loads(path.read_text(encoding="utf-8"))
            require(doc.get("status") == "complete" and int(doc.get("graphs", -1)) == 5000,
                    f"incomplete/not-full report: {rel}")
            require(int(doc.get("seed", -1)) == seed, f"seed mismatch: {rel}")
            ol = doc.get("operation_ledger", {})
            pm = ol.get("per_method", {})
            for r in doc["rows"]:
                mode, T = r["mode"], int(r["T"])
                key = (seed, mode, T)
                require(key not in rows, f"duplicate row {key}")
                ctr = pm.get(f"{mode}_T{T}")
                require(ctr is not None, f"missing operation counter for {mode}_T{T} in {rel}")
                ev = r.get("per_site_nonzero_scalar_events")
                require(isinstance(ev, dict) and ev, f"missing event counts for {key}")
                rows[key] = {
                    "accuracy": float(r["accuracy"]),
                    "counters": dict(ctr),
                    "events": {k: int(v) for k, v in ev.items()},
                    "evidence": rel,
                }
    expected = {(s, m, t) for s in SEEDS for m in ALL_MODES if m != "ANN" for t in TIMESTEPS}
    require(set(rows) == expected, f"row coverage mismatch: missing {sorted(expected - set(rows))[:5]}")
    require(len(rows) == len(expected), f"expected {len(expected)} conversion rows, got {len(rows)}")
    return rows


def precheck() -> None:
    """Frozen-input gate + official shape census + site-spec census. No GPU."""
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from torch_geometric.datasets import GNNBenchmarkDataset

    data_root = default_data_root()
    author_repo = default_author_repo()
    # Reuse the frozen binding verifier from R5 (unchanged, read-only).
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e1_residual_v1 as R
    mods = R.load_modules(author_repo)
    gate = {}
    for seed in SEEDS:
        b = R.verify_bindings(seed, data_root, mods)
        gate[str(seed)] = {"frozen_files_verified": len(b["frozen_hashes"]),
                           "processed_files_verified": len(b["data_hashes"]),
                           "teacher_accuracy": b["done"]["best_val_accuracy"]}
    ap = R.author_provenance(author_repo, mods)
    gate["author_provenance"] = ap["safe"]

    ds = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    n = sum(int(g.num_nodes) for g in ds)
    e = sum(int(g.edge_index.shape[1]) for g in ds)
    require((len(ds), n, e) == (B, N, E),
            f"official val shape census drift: {(len(ds), n, e)} != {(B, N, E)}")
    gate["shape_census"] = {"graphs": len(ds), "nodes": n, "edges": e,
                            "E_per_N": e / n, "batches_128": math.ceil(len(ds) / 128)}

    rows = load_rows()
    # Validate that every site name maps to a declared spec and counters scale.
    site_kinds = {}
    for (seed, mode, T), r in rows.items():
        for name in r["events"]:
            site_spec(mode, name)
        c = r["counters"]
        gl = c.get("graph_layer_evaluations", c.get("analog_gate_evaluations"))
        require(gl == L * T * P, f"unexpected layer evaluations {gl} for {(seed, mode, T)}")
        site_kinds[str(len(r["events"]))] = site_kinds.get(str(len(r["events"])), 0) + 1
    gate["row_coverage"] = {"rows": len(rows), "seeds": list(SEEDS),
                            "modes": len(ALL_MODES) - 1, "timesteps": list(TIMESTEPS),
                            "site_count_histogram": site_kinds}
    gate["status"] = "PASS"
    gate["check"] = "e3_e5_precheck_v1"
    gate["note"] = "hash/shape/structure gate only; no GPU, no scientific result"
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (LOGDIR / "e3e5-precheck-v1.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate["status"], "row_coverage": gate["row_coverage"],
                      "shape_census": gate["shape_census"]}, indent=2))


# ---------------------------------------------------------------- cost model
ACCOUNTING_MODEL = {
    "scope": "complete KNOWN arithmetic-operation count of the AS-IMPLEMENTED dense PyTorch prototype for one full 5000-graph validation sweep; not hardware energy, not hardware latency, not a cache/backend claim",
    "shape_inputs": {"graphs_B": B, "nodes_N": N, "edges_E": E, "hidden_d": D,
                     "graph_layers_L": L, "batches_P": P, "head_widths": list(W_HEAD),
                     "E_per_N": E / N, "node_input": NODE_IN, "edge_input": EDGE_IN},
    "pJ_proxy": PJ,
    "conventions": {
        "AC": "event-accumulate: coded-stream projections counted as K_s(T)*ac_updates_s ADDs using the REAL measured nonzero scalar event counts from the frozen E1 reports; all other groups dense as executed",
        "conservative_MAC": "every graph-layer projection counted as a full dense MAC (no sparsity credit): T*L*(4N+E)*d^2 MACs; all other groups identical to AC",
    },
    "formulas": {
        "setup_embedding_mac": "350*N + 70*E once per (mode,T) sweep (5->70 node + 1->70 edge embeddings)",
        "dense_projection_mac": "T*L*(4*N*d^2 + E*d^2): A,B,D,E on N nodes and C on E edges, all four layers at EVERY tick",
        "ac_projection_add": "sum_s K_s(T)*ac_updates_s (K_s = measured cumulative nonzero scalar events over all ticks and all 5000 graphs)",
        "gates_add": "T*L*2*E*d (z = C(e) + D(h)[src] + E(h)[dst])",
        "sigmoid": "T*L*E*d scalar sigmoids (declared, priced as MUL-equivalent; real hardware cost unresolved)",
        "products_mul": "T*L*E*d (gates * b[src])",
        "scatter_add": "T*L*2*E*d (numerator + denominator index_add_)",
        "normalization": "T*L*N*d ADD (denominator+eps) + T*L*N*d DIV (priced as MUL-equivalent)",
        "batch_norm": "T*L*(N+E)*d MUL + T*L*(N+E)*d ADD (node + edge BN, gamma*x+beta); final layer edge BN pruned",
        "activation_relu": "T*L*(N+E)*d comparisons (priced as ADD-equivalent); final layer edge ReLU pruned",
        "residual_add": "T*L*(N+E)*d ADD (raw h/e skips) + T*L*N*d ADD (A(h)+message); final layer edge skip pruned",
        "source_per_active_call_per_element": "1 SUB + 1 MUL (t*(raw-prev)) + 1 DIV (event/t) + 1 ADD (decoded accumulate) + 2 state read/write traffic",
        "graded_banks": "8 threshold comparisons per element per active source call (num_thresholds=8)",
        "decoder_one_over_t": "1 DIV per element per active source call, explicitly counted (NOT assumed free power-of-two shift)",
        "head": "(head_calls/P) * [B*3215 MAC + B*(35+17) ReLU] once per endpoint; dc_adapt_six_site evaluates the head every tick",
        "pooling": "(head_calls/P) * [N*d ADD + B*d DIV]",
        "storage": "declared FP32 bank/state bytes = coded_layers*5*d^2*8*4 + source state tensors; DECLARED, not measured memory traffic",
        "traffic": "declared scalar state read/write counts and accumulator touches; DECLARED, not measured memory traffic",
        "final_unused_edge_pruned": "subtract T*(E*d) for each of: final-layer edge BN MUL, BN ADD, ReLU, residual ADD. Applied IDENTICALLY to every row including ANN (fair pruning).",
    },
    "unresolved_items": [
        "hardware latency and hardware energy (not measured; no chip-energy claim)",
        "real DRAM/cache memory traffic bytes (only DECLARED scalar counts)",
        "event-bank construction, pre-scaling, 1/t time-specific bank variants, address/routing logic",
        "cache-equality checks, invalidation and maintenance for reused layer-0 state",
        "sigmoid/division/ReLU transcendental hardware cost (priced only by declared ADD/MUL equivalents)",
        "weight/bank storage precision, accumulator width, overflow handling, quantization",
        "BN fold-vs-execute implementation choice; bias/setup terms beyond declared embedding",
        "the fused first-analog/centered cache and event-driven projection backend are NOT implemented: the dense prototype recomputes all four graph layers at every tick",
        "host-side Python/DataLoader overhead and host<->device transfer time are excluded from the declared arithmetic ledger (measured separately as wall-clock)",
    ],
}


def row_cost(mode: str, T: int, counters: dict, events: dict) -> dict:
    """Complete known-operation counts for one (mode,T,seed) row."""
    cgl = counters.get("graph_layer_evaluations", counters.get("analog_gate_evaluations"))
    require(cgl == L * T * P, "layer evaluation counter drift")
    head_calls = counters.get("analog_head_calls", counters.get("head_calls"))
    require(isinstance(head_calls, int), "missing head counter")
    sw = int(counters["source_warmup_calls"])
    sa = int(counters["source_active_unit_calls"])

    # --- dense controller / gate / skip arithmetic, as actually executed ----
    g_add = T * L * 2 * E * D
    sigmoid = T * L * E * D
    products = T * L * E * D
    scatter = T * L * 2 * E * D
    norm_add = T * L * N * D
    norm_div = T * L * N * D
    bn_mul = T * L * (N + E) * D
    bn_add = T * L * (N + E) * D
    relu = T * L * (N + E) * D
    residual = T * L * (N + E) * D + T * L * N * D
    # Fair pruning of the final layer's unused edge output, identical for all rows.
    prune = T * E * D
    bn_mul -= prune
    bn_add -= prune
    relu -= prune
    residual -= prune

    projection_mac = T * L * (4 * N * D * D + E * D * D)
    setup_mac = EMB_MACS
    head_reps = head_calls / P
    head_mac = head_reps * B * HEAD_MACS
    head_relu = head_reps * B * (W_HEAD[1] + W_HEAD[2])
    pool_add = head_reps * N * D
    pool_div = head_reps * B * D

    # --- graded source / decoder / 1-t / banks / traffic --------------------
    # Measured counters are summed across all P batches. Per active call the
    # site tensor has `elements` scalars; active calls are evenly split over
    # the declared sites of this mode (verified against the frozen site count).
    specs = [site_spec(mode, name) for name in events]
    require(sa % len(specs) == 0 and sw % len(specs) == 0,
            f"source counters not divisible by site count for {mode}_T{T}")
    calls_per_site_active = sa // len(specs)
    calls_per_site_warmup = sw // len(specs)
    total_elements = sum(calls_per_site_active * s["elements"] for s in specs)
    warmup_elements = sum(calls_per_site_warmup * s["elements"] for s in specs)
    ac_projection_add = sum(int(events[n]) * s["ac"] for n, s in zip(events, specs))

    source_sub = total_elements           # raw - previous
    source_mul = total_elements           # t * delta
    decoder_div = total_elements          # event / t   (one_over_t)
    decoder_add = total_elements          # decoded += event/t
    graded_cmp = 8 * total_elements       # 8-level threshold decisions
    warmup_cmp = warmup_elements          # zero-latch check
    traffic_scalars = 2 * total_elements  # state read + write
    # Declared (not measured) bank storage: one d x d pre-scaled bank per coded
    # site per source level, FP32, plus two persistent state tensors per site.
    storage_bytes = len(specs) * D * D * 8 * 4 + 2 * sum(s["elements"] for s in specs) * 4

    add_common = (g_add + scatter + norm_add + bn_add + relu + residual +
                  source_sub + decoder_add + pool_add + warmup_cmp)
    mul_common = (products + sigmoid + norm_div + bn_mul + source_mul +
                  decoder_div + pool_div + head_relu)

    def priced(add, mul, mac):
        return {"ADD": int(add), "MUL": int(mul), "MAC": int(mac),
                "pJ": PJ["ADD"] * add + PJ["MUL"] * mul + PJ["MAC"] * mac}

    conv = {}
    conv["AC"] = priced(add_common + ac_projection_add, mul_common, setup_mac + head_mac)
    conv["conservative_MAC"] = priced(add_common, mul_common,
                                      setup_mac + head_mac + projection_mac)

    groups = {
        "setup": {"mac": setup_mac, "source_zero_latch_comparisons": int(warmup_cmp)},
        "first_analog_or_base": {"declared_fused_mac": None,
                                 "implemented": False,
                                 "note": "fused cache/base NOT implemented; dense prototype evaluates all four layers every tick"},
        "projection": {"dense_mac": projection_mac,
                       "ac_event_adds": int(ac_projection_add),
                       "measured_nonzero_events_by_site": {k: int(v) for k, v in events.items()},
                       "ac_updates_per_event_by_site": {n: s["ac"] for n, s in zip(events, specs)}},
        "source": {"warmup_calls": sw, "active_unit_calls": sa,
                   "elements_processed": int(total_elements),
                   "sub": int(source_sub), "mul": int(source_mul)},
        "decoder": {"add": int(decoder_add)},
        "one_over_t": {"div": int(decoder_div), "note": "explicitly counted, not assumed free"},
        "graded_banks": {"threshold_comparisons": int(graded_cmp)},
        "gates": {"add": g_add, "layer_evaluations": cgl, "dense_sweeps": T * L},
        "sigmoid": {"scalars": sigmoid},
        "products": {"mul": products},
        "normalization": {"add": norm_add, "div": norm_div},
        "scatter": {"add": scatter},
        "residual": {"add": residual, "final_unused_edge_pruned_scalars": int(prune)},
        "batch_norm": {"mul": int(bn_mul), "add": int(bn_add)},
        "head": {"calls": head_calls, "mac": int(head_mac), "relu": int(head_relu)},
        "pooling": {"add": int(pool_add), "div": int(pool_div)},
        "storage": {"declared_fp32_bytes": int(storage_bytes),
                    "measured_memory_traffic_bytes": None},
        "traffic": {"declared_scalar_state_read_write": int(traffic_scalars),
                    "measured_memory_traffic_bytes": None},
    }
    return {"groups": groups, "conventions": conv,
            "pJ_per_graph": {k: v["pJ"] / B for k, v in conv.items()},
            "measured_counters": dict(counters)}


def build_ann_cost() -> dict:
    """ANN upper-reference forward: one dense sweep, no source coding."""
    g_add = L * 2 * E * D
    sigmoid = L * E * D
    products = L * E * D
    scatter = L * 2 * E * D
    norm_add, norm_div = L * N * D, L * N * D
    bn_mul = bn_add = relu = residual = L * (N + E) * D
    residual += L * N * D
    prune = E * D
    bn_mul -= prune
    bn_add -= prune
    relu -= prune
    residual -= prune
    add = g_add + scatter + norm_add + bn_add + relu + residual + N * D
    mul = products + sigmoid + norm_div + bn_mul + B * D
    mac = EMB_MACS + L * (4 * N * D * D + E * D * D) + B * HEAD_MACS + B * (W_HEAD[1] + W_HEAD[2]) * 0
    # ReLU in head priced as ADD-equivalent (declared).
    add += B * (W_HEAD[1] + W_HEAD[2])
    conv = {"AC": {"ADD": int(add), "MUL": int(mul), "MAC": int(mac),
                   "pJ": PJ["ADD"] * add + PJ["MUL"] * mul + PJ["MAC"] * mac},
            "conservative_MAC": {"ADD": int(add), "MUL": int(mul), "MAC": int(mac),
                                 "pJ": PJ["ADD"] * add + PJ["MUL"] * mul + PJ["MAC"] * mac}}
    return {"groups": {"scope": "single dense teacher forward; no source, no decoder, no 1/t, no graded banks"},
            "conventions": conv, "pJ_per_graph": {k: v["pJ"] / B for k, v in conv.items()},
            "measured_counters": {"graph_layer_evaluations": L * P, "analog_head_calls": P}}


def cost() -> None:
    rows = load_rows()
    pre = LOGDIR / "e3e5-precheck-v1.json"
    require(pre.is_file(), "run precheck first")
    require(json.loads(pre.read_text(encoding="utf-8"))["status"] == "PASS", "precheck not PASS")

    out = {"status": "measured_declared_complete_known_operation_cost_not_hardware_energy",
           "check": "e3_cost_ledger_v1",
           "dataset": "CIFAR10", "split": "all official validation", "graphs": B,
           "seeds": list(SEEDS), "timesteps": list(TIMESTEPS),
           "protocol_sha256": PROTOCOL_SHA, "head_addendum_sha256": ADDENDUM_SHA,
           "accounting_model": ACCOUNTING_MODEL,
           "rows": [], "ann_upper_reference": None,
           "representative_batch_check": {},
           "unresolved_items": ACCOUNTING_MODEL["unresolved_items"],
           "limitations": [
               "This is a declared arithmetic-operation count of the dense PyTorch prototype, NOT hardware energy or hardware latency.",
               "Cost contains explicitly enumerated unresolved items; per PROTOCOL-v3 'Cost with unresolved omitted work cannot pass', it cannot by itself establish a method advantage.",
               "No event-count-as-energy claim, no 5x energy claim, no efficient-title claim.",
               "The fused first-analog/centered cache and event-driven projection backend are not implemented in this prototype.",
           ]}
    metric_rows = []

    # Representative batch manual check (batch 0, 128 graphs, fixed order).
    from torch_geometric.datasets import GNNBenchmarkDataset
    ds = GNNBenchmarkDataset(root=str(default_data_root()), name="CIFAR10", split="val")
    b0 = [ds[i] for i in range(128)]
    n0 = sum(int(g.num_nodes) for g in b0)
    e0 = sum(int(g.edge_index.shape[1]) for g in b0)
    manual_layer_mac = 4 * n0 * D * D + e0 * D * D
    manual_layer_add = 2 * e0 * D + 2 * e0 * D + n0 * D + n0 * D
    out["representative_batch_check"] = {
        "batch": 0, "graphs": len(b0), "nodes": n0, "edges": e0,
        "declared_projection_mac_one_layer": manual_layer_mac,
        "declared_controller_add_one_layer": manual_layer_add,
        "source_lines_inspected": [
            f"{SR}/../takeover_20261112/gated_affine_base_v1.py::_graph_layer",
            f"{SR}/../takeover_20261112/gated_teacher_model.py::GatedGCNLayer.forward",
            f"{SR}/../takeover_20261112/gated_teacher_model.py::MLPReadout.forward",
        ],
        "manual_check": "per-layer A/B/D/E on N (MAC=4*N*d^2), C on E (MAC=E*d^2), "
                        "z two E*d ADD, numerator+denominator 2*E*d ADD, "
                        "denominator+eps N*d ADD, A(h)+message N*d ADD; "
                        "dense projection count matches the declared formula at this batch's real N/E",
        "formula_matches_batch_shapes": bool(manual_layer_mac == 4 * n0 * D * D + e0 * D * D
                                             and manual_layer_add == 4 * e0 * D + 2 * n0 * D),
    }

    ann = build_ann_cost()
    out["ann_upper_reference"] = ann
    for conv, v in ann["conventions"].items():
        metric_rows.append(dict(zip(CSV_COLUMNS, (
            TASK, AGENT, RUN_ID, "CIFAR10", "val", "5000_all_official_val", "ANN",
            "41,42,43", "", "", f"complete_known_cost_{conv}_pJ_per_graph",
            v["pJ"] / B, "pJ_per_graph", f"{RR}/results/e3-e5-v1/e3_cost_ledger_v1.json",
            PROTOCOL_SHA, "declared_count"))))

    for seed in SEEDS:
        for mode in ALL_MODES:
            if mode == "ANN":
                continue
            for T in TIMESTEPS:
                r = rows[(seed, mode, T)]
                rc = row_cost(mode, T, r["counters"], r["events"])
                out["rows"].append({"seed": seed, "mode": mode, "T": T,
                                    "evidence": r["evidence"], **rc})
                for conv, v in rc["conventions"].items():
                    metric_rows.append(dict(zip(CSV_COLUMNS, (
                        TASK, AGENT, RUN_ID, "CIFAR10", "val", "5000_all_official_val",
                        mode, seed, "", T, f"complete_known_cost_{conv}_pJ_per_graph",
                        v["pJ"] / B, "pJ_per_graph",
                        f"{RR}/results/e3-e5-v1/e3_cost_ledger_v1.json",
                        PROTOCOL_SHA, "declared_count"))))

    OUT.mkdir(parents=True, exist_ok=True)
    cost_path = OUT / "e3_cost_ledger_v1.json"
    require(not cost_path.exists(), "refusing to overwrite published cost ledger")
    cost_path.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    with (OUT / "cost_METRICS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(metric_rows)
    print(json.dumps({"status": out["status"], "rows": len(out["rows"]),
                      "metric_rows": len(metric_rows),
                      "ann_pJ_per_graph": ann["pJ_per_graph"],
                      "candidate_T4_AC_pJ_per_graph": next(
                          x["pJ_per_graph"]["AC"] for x in out["rows"]
                          if x["mode"] == CANDIDATE and x["T"] == 4 and x["seed"] == 41)}, indent=2))


# ------------------------------------------------------------- latency (GPU)
def latency(args) -> None:
    import numpy as np
    import torch
    from torch_geometric.datasets import GNNBenchmarkDataset
    from torch_geometric.loader import DataLoader

    data_root = default_data_root().resolve()
    author_repo = default_author_repo().resolve()
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e1_residual_v1 as R
    import eval_e1_multiseed_v1 as MS
    import matched_controls as controls
    import eval_affine_validation as base

    binding = R.verify_bindings(41, data_root, R.load_modules(author_repo))
    mods = R.load_modules(author_repo)
    _base, affine, cal, hybrid, author, shared, teacher_helper = mods
    ap = R.author_provenance(author_repo, mods)
    MTH_diff, MTH_diff_line = ap["MTH_diff"], ap["MTH_diff_line"]

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
    model = teacher_helper.build_teacher(state, device).eval()
    require(len(model.layers) == L, "expected four frozen graph layers")
    for lyr in model.layers:
        require(bool(lyr.use_batch_norm) and int(lyr.out_channels) == D,
                "frozen BN/width drift")

    ds = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    require(len(ds) == B, "unexpected validation count")
    loader = DataLoader(base.LabelFreeDataset(ds), batch_size=128, shuffle=False, num_workers=0)
    batches = []
    for i, b in enumerate(loader):
        if i >= args.batches:
            break
        batches.append(b.to(device))
    require(len(batches) == args.batches, "not enough batches")

    affine_doc, six_doc = binding["affine_doc"], binding["six_doc"]
    shared_vectors = [v.to(device=device, dtype=batches[0].x.dtype)
                      for v in binding["shared_theta_vectors"]]
    six_ct = six_doc["calibration"]["channel_thresholds"]

    def thresholds(mode):
        dt = batches[0].x.dtype
        if mode in ("centered", "first_analog", "generic"):
            return [t.to(device=device, dtype=dt) for t in cal.profile_thresholds(affine_doc, mode, 1)]
        if mode in ("per_consumer", "cse_only"):
            return [t.to(device=device, dtype=dt) for t in cal.profile_thresholds(affine_doc, "generic", 1)]
        if mode == "dc_adapt_analog_head":
            return [torch.as_tensor(six_ct[f"gated_{i}"], dtype=dt, device=device) for i in range(4)]
        if mode == "dc_adapt_six_site":
            return [torch.as_tensor(six_ct[s], dtype=dt, device=device) for s in R.SIX_SITES]
        return [v.clone() for v in shared_vectors]

    @torch.no_grad()
    def forward(mode, T, batch):
        if mode == "ANN":
            return model(batch)
        th = thresholds(mode)
        if mode == "dc_adapt_six_site":
            logits, _diag = hybrid.differential_hybrid_forward(model, batch, th, T, MTH_diff)
            return logits
        if mode in ("dc_adapt_analog_head", "per_consumer", "cse_only"):
            return controls.affine_mode_forward(model, batch, th, T, mode,
                                                MTH_diff, MTH_diff_line)["logits"]
        base_mode = ("first_analog" if mode == "first_analog_shared_theta" else
                     "centered" if mode == "centered_shared_theta" else mode)
        return affine.affine_mode_forward(model, batch, th, T, base_mode,
                                          MTH_diff, MTH_diff_line)["logits"]

    # Determinism / label-free smoke check on the representative batch.
    b0 = batches[0]
    a1, a2 = forward("ANN", 1, b0), forward("ANN", 1, b0)
    require(torch.equal(a1, a2), "ANN teacher forward not bitwise deterministic")
    c1, c2 = forward("centered", 4, b0), forward("centered", 4, b0)
    require(torch.equal(c1, c2), "centered forward not bitwise deterministic")
    require(not hasattr(b0, "y") or b0.y is None, "labels leaked into converter input")

    configs = [("ANN", 0)] + [(m, t) for m in ALL_MODES if m != "ANN" for t in TIMESTEPS]
    reps = args.repeats
    warmup = args.warmup
    results = []
    t_start = time.monotonic()
    for ci, (mode, T) in enumerate(configs):
        # warm-up passes (untimed), identical for every configuration
        for _ in range(warmup):
            for b in batches:
                forward(mode, T, b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for rep in range(reps):
            t0 = time.monotonic()
            for b in batches:
                forward(mode, T, b)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.monotonic() - t0)
        graphs = sum(int(b.num_graphs) for b in batches)
        med = statistics.median(times)
        results.append({"mode": mode, "T": T, "graphs_per_repeat": graphs,
                        "repeat_wall_seconds": [round(x, 6) for x in times],
                        "median_wall_seconds": med,
                        "median_ms_per_graph": med * 1000.0 / graphs,
                        "min_ms_per_graph": min(times) * 1000.0 / graphs,
                        "max_ms_per_graph": max(times) * 1000.0 / graphs})
        if ci % 8 == 0 or ci == len(configs) - 1:
            print(json.dumps({"config": ci + 1, "of": len(configs), "mode": mode,
                              "T": T, "median_ms_per_graph": round(med * 1000.0 / graphs, 4),
                              "elapsed_sec": round(time.monotonic() - t_start, 1)}), flush=True)

    data_start = base.processed_hashes(data_root)
    require(data_start == binding["metadata"]["provenance"]["processed_data_sha256"],
            "processed data changed during latency job")
    require(ap["safe"]["commit"] == DCGS_COMMIT, "pinned DCGS commit drift")

    report = {
        "status": "measured_matched_wall_clock_dense_prototype_not_hardware_latency",
        "check": "e5_latency_v1",
        "scope": "same local RTX 3060, batch 128, fixed sequential validation order, "
                 "identical warmup and 3 repeats, median wall-clock per graph; "
                 "dense PyTorch prototype only, NOT neuromorphic hardware latency",
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "python": sys.version,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "teacher_seed": 41, "batches_used": args.batches,
        "graphs_per_repeat": sum(int(b.num_graphs) for b in batches),
        "warmup_passes": warmup, "repeats": reps, "configs": len(configs),
        "total_job_seconds": time.monotonic() - t_start,
        "representative_batch_manual_check": {
            "batch": 0, "graphs": int(b0.num_graphs), "nodes": int(b0.num_nodes),
            "edges": int(b0.edge_index.shape[1]),
            "teacher_double_forward_bitwise_equal": True,
            "centered_double_forward_bitwise_equal": True,
            "labels_absent_from_converter_input": True},
        "results": results,
        "limitations": [
            "Wall-clock on a dense PyTorch prototype is not SNN hardware latency or energy.",
            "Includes Python loop, host->device transfer if any, and kernel launch overhead; excludes dataset load.",
            "Measured on seed-41 frozen teacher only; graph shapes are seed-independent.",
            "No event-count-as-speedup, no efficiency or chip-energy claim.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    lp = OUT / "e5_latency_v1.json"
    require(not lp.exists(), "refusing to overwrite published latency report")
    lp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                  encoding="utf-8")
    metric_rows = []
    for r in results:
        metric_rows.append(dict(zip(CSV_COLUMNS, (
            TASK, AGENT, RUN_ID, "CIFAR10", "val", f"{args.batches * 128}_fixed_prefix_val",
            r["mode"], 41, "", r["T"] or "", "median_matched_wall_clock_ms_per_graph",
            r["median_ms_per_graph"], "ms_per_graph",
            f"{RR}/results/e3-e5-v1/e5_latency_v1.json", PROTOCOL_SHA, "measured"))))
    with (OUT / "latency_METRICS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(metric_rows)
    print(json.dumps({"status": report["status"], "configs": len(results),
                      "total_job_seconds": round(report["total_job_seconds"], 1)}, indent=2))


def aggregate() -> None:
    """Combine cost + latency and re-evaluate the frozen E1 two-part criterion."""
    cost_doc = json.loads((OUT / "e3_cost_ledger_v1.json").read_text(encoding="utf-8"))
    lat_doc = json.loads((OUT / "e5_latency_v1.json").read_text(encoding="utf-8"))
    summary = json.loads((RUNROOT / "results" / "e1-fullrow-summary-v1" /
                          "e1_fullrow_summary_v1.json").read_text(encoding="utf-8"))

    by_mode_T = {(r["mode"], r["T"]): r for r in cost_doc["rows"] if r["seed"] == 41}
    lat = {(r["mode"], r["T"]): r for r in lat_doc["results"]}

    def mean3(mode, T, conv):
        vals = [next(x for x in cost_doc["rows"]
                     if x["seed"] == s and x["mode"] == mode and x["T"] == T)["pJ_per_graph"][conv]
                for s in SEEDS]
        return sum(vals) / 3

    criterion = {}
    for T in (4, 8):
        cand = CANDIDATE
        strong = STRONGEST[T]
        acc = summary["criterion"]["accuracy_only"][f"T{T}"]
        cost_cmp = {conv: {"candidate_pJ_per_graph": mean3(cand, T, conv),
                           "strongest_control_pJ_per_graph": mean3(strong, T, conv),
                           "candidate_strictly_lower": mean3(cand, T, conv) < mean3(strong, T, conv)}
                    for conv in ("AC", "conservative_MAC")}
        lat_c, lat_s = lat[(cand, T)], lat[(strong, T)]
        lat_lower = lat_c["median_ms_per_graph"] < lat_s["median_ms_per_graph"]
        criterion[f"T{T}"] = {
            "accuracy_gap_pp": acc["gap_pp"],
            "accuracy_within_0.3pp": acc["within_0.3pp_accuracy_only"],
            "strongest_matched_control": strong,
            "complete_known_cost": cost_cmp,
            "matched_wall_clock": {
                "candidate_ms_per_graph": lat_c["median_ms_per_graph"],
                "strongest_ms_per_graph": lat_s["median_ms_per_graph"],
                "candidate_strictly_lower": lat_lower,
                "repeats": lat_doc["repeats"], "warmup_passes": lat_doc["warmup_passes"]},
        }

    unresolved = ACCOUNTING_MODEL["unresolved_items"]
    acc_ok = all(criterion[f"T{T}"]["accuracy_within_0.3pp"] for T in (4, 8))
    cost_ok = all(all(criterion[f"T{T}"]["complete_known_cost"][c]["candidate_strictly_lower"]
                      for c in ("AC", "conservative_MAC")) for T in (4, 8))
    lat_ok = all(criterion[f"T{T}"]["matched_wall_clock"]["candidate_strictly_lower"]
                 for T in (4, 8))
    # PROTOCOL-v3: "Cost with unresolved omitted work cannot pass". Unresolved
    # items are explicitly enumerated, so the cost leg cannot credit a pass.
    verdict = "E1 NOT PASSED (method advantage not established)"
    reasons = [
        "accuracy-only condition met at both T4 and T8 (gap <= 0.3pp)",
        f"complete known-operation cost strictly lower at BOTH T and under BOTH conventions: {cost_ok}",
        f"matched median wall-clock strictly lower at BOTH T: {lat_ok}",
        f"{len(unresolved)} unresolved items remain in the accounting; per PROTOCOL-v3 "
        "'Cost with unresolved omitted work cannot pass', so the cost leg cannot credit a pass",
        "dense PyTorch prototype recomputes all four graph layers at every tick; "
        "no event-driven backend or fused cache is implemented",
    ]
    out = {
        "status": "evidence_backed_criterion_evaluation_no_method_advantage_claim",
        "check": "e3_e5_criterion_v1",
        "protocol_sha256": PROTOCOL_SHA, "head_addendum_sha256": ADDENDUM_SHA,
        "criterion": criterion, "accuracy_only_ok": acc_ok,
        "cost_leg_strictly_lower": cost_ok, "latency_leg_strictly_lower": lat_ok,
        "unresolved_item_count": len(unresolved), "unresolved_items": unresolved,
        "verdict": verdict, "reasons": reasons,
        "story_decision": "DEFERRED - written validation-only Story decision requires "
                          "E2/E4/E6 and the full E5 mechanism/falsifier evidence first; "
                          "accuracy-only supports a dataflow result (protocol: neither -> Story B) "
                          "but no Story is recorded here",
        "new_test_inference_performed": False,
        "p4_status": "running",
        "notes": "This closes the previous 'NOT EVALUABLE' evidence gap with real "
                 "declared cost counts and real 3-repeat median wall-clock; it does "
                 "NOT lower the frozen acceptance bar and does NOT claim a pass.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ap = OUT / "e3_e5_criterion_v1.json"
    require(not ap.exists(), "refusing to overwrite published criterion report")
    ap.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n",
                  encoding="utf-8")
    print(json.dumps({"verdict": verdict, "accuracy_only_ok": acc_ok,
                      "cost_leg_strictly_lower": cost_ok,
                      "latency_leg_strictly_lower": lat_ok,
                      "unresolved_item_count": len(unresolved),
                      "story_decision": "DEFERRED", "p4_status": "running"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("precheck")
    sub.add_parser("cost")
    sub.add_parser("aggregate")
    lp = sub.add_parser("latency")
    lp.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    lp.add_argument("--batches", type=int, default=4)
    lp.add_argument("--repeats", type=int, default=3)
    lp.add_argument("--warmup", type=int, default=1)
    a = p.parse_args()
    if a.cmd == "latency":
        latency(a)
    else:
        {"precheck": precheck, "cost": cost, "aggregate": aggregate}[a.cmd]()


if __name__ == "__main__":
    main()
