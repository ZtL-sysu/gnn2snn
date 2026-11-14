"""E2 faithful NaiveIF comparator, stage 2: IF forward + full validation eval.

NaiveIF here is the standard data-normalised ANN-to-SNN conversion applied
WITHOUT any operator-specific adaptation (this is what makes it "naive" and
is exactly the comparison PROTOCOL-v3 asks for):

  * each ReLU site of the frozen four-layer GatedGCN teacher and its two-layer
    MLP head is replaced by an integrate-and-fire neuron;
  * the per-channel firing threshold is the maximum positive pre-activation
    measured label-free on ALL 45000 official training graphs by
    e2_naiveif_v1.py (stage 1, one frozen report per teacher seed);
  * constant real-valued input is injected at every tick, membrane potential
    is maintained across ticks, spikes use subtractive reset;
  * sigmoid edge gates, BatchNorm, residual connections, dropout (eval mode)
    and mean pooling are left EXACTLY as in the ANN (no adaptation, no folding,
    no cache).  Any degradation is reported as a real measured result, never
    assumed;
  * T in {1,2,4,8}; all 5000 official validation graphs, fixed sequential
    order, batch 128, label-free converter input, deterministic CUDA.

Metrics are the same four as E1 (accuracy, teacher agreement, logit MSE,
teacher-to-output KL), with one per-sample NPZ per (seed, T) carrying
indices / labels / teacher_logits / logits and a SHA-256 manifest.

REPAIR NOTE (v1 draft was NEVER published, so no manifest binds it):
the unreviewed draft is archived byte-for-byte in
logs/e2-naiveif-eval-draft-v1/e2_naiveif_eval_v1.draft.py
(SHA-256 2d26d9479023fcb44d859f4840d8ef33d7dde6b3d5ebf62519e3a55462ded9b0).
Static review found four TOOLING defects, all fixed here:
  (1) it reimplemented GatedGCNLayer by calling layer.A/B/C/D/E directly,
      bypassing the author's layer(edge_index, h, e) forward (which also owns
      the edge BN, dropout, global_mean_pool and the MLP's own F.relu calls).
      Such a rewrite cannot be claimed faithful to the frozen teacher;
  (2) LabelFreeDataset strips y, and the fallback produced ONE label per
      128-graph batch, so concatenation could never match the logits;
  (3) --limit smoke wrote into the canonical OUT/seedN directory, causing the
      later full-split run to be refused;
  (4) coverage.graphs used a vacuous ``sum(... for o in [])`` expression.
This version instead registers forward hooks on the EXACT ten stage-1 capture
sites (bn_node_h, bn_node_e, FC_layers[0], FC_layers[1]) and calls the author's
model(batch) unchanged.  The author's own F.relu then applies to the returned
non-negative IF output, so it is an identity on spikes.  A no-op-hook
bitwise fidelity check against the uninstrumented forward is mandatory.
No scientific parameter changed: seeds 41/42/43, T in {1,2,4,8}, all 5000
official validation graphs, batch128, fixed order, label-free input.

Hard gates: frozen inputs re-verified through e1_residual_v1.verify_bindings;
stage-1 calibration reports verified before use; canonical indices/labels
checked bitwise (and seed41 teacher logits bitwise on the first batch);
processed hashes checked before and after; output directories refused if they
already exist.  No accuracy-only pass claim, no energy, hardware-latency,
speedup, novelty or method-advantage claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.datasets import GNNBenchmarkDataset  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
RR = "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix"
CAL = RUNROOT / "results" / "e2-naiveif-v1"
OUT = RUNROOT / "results" / "e2-naiveif-eval-v1"
SMOKE = RUNROOT / "logs" / "e2_naiveif_eval_smoke_v1"
CANON = RUNROOT / "inputs" / "canonical-baseline-val.npz"
CANON_SHA = "7c0727a2902bb2c05a0ec4930734b0bfa05e0905c4b86a6c976155a0c702fcc3"

PROTOCOL_SHA = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
ADDENDUM_SHA = "8bde5832d556c8c7b04ee630f2390703c33a26fa750f196f4dc1c94da61291f6"
DCGS_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
SEEDS = (41, 42, 43)
TIMESTEPS = (1, 2, 4, 8)
EXPECTED_VAL = 5000
BATCH_SIZE = 128
SITE_NAMES = tuple(
    f"gated{i}.{kind}" for i in range(4) for kind in ("node", "edge")
) + ("mlp0", "mlp1")
ZERO_THRESHOLD_FLOOR = 1e-6

CSV_COLUMNS = ("task_id", "agent_id", "run_id", "dataset", "split", "coverage",
               "method", "teacher_seed", "student_seed", "T", "metric", "value",
               "unit", "evidence", "protocol_sha256", "status")
TASK, AGENT, RUN_ID = "visual-conversion", "20261113-wb-resume", "e2-naiveif-eval-v1"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    v = os.environ.get("DCGS_REPO")
    return Path(v) if v else Path(tempfile.gettempdir()) / "dcgs-5e0e4c73"


def cal_path(seed: int) -> Path:
    return CAL / f"naiveif_calibration_seed{seed}_v1.json"


def row(seed: int, T: int, metric: str, value: float, unit: str,
        evidence: str, status: str = "measured") -> dict:
    return dict(zip(CSV_COLUMNS, (
        TASK, AGENT, RUN_ID, "CIFAR10", "val", "5000_fixed_sequential_val",
        "NaiveIF", str(seed), "", str(T), metric, value, unit, evidence,
        PROTOCOL_SHA, status)))


def load_calibration(seed: int, device, dtype) -> dict:
    p = cal_path(seed)
    require(p.is_file(), f"missing stage-1 calibration for seed {seed}")
    d = json.loads(p.read_text(encoding="utf-8"))
    require(d["status"] == "fulltrain_label_free_threshold_calibration",
            "stage-1 calibration is not full-split evidence")
    require(d["coverage"]["full_split"] is True
            and d["coverage"]["graphs"] == 45000, "stage-1 coverage drift")
    require(d["teacher_seed"] == seed and d["dcgs_commit"] == DCGS_COMMIT,
            "stage-1 binding drift")
    require(set(d["sites"]) == set(SITE_NAMES), "stage-1 site drift")
    out = {}
    for name in SITE_NAMES:
        v = torch.tensor(d["sites"][name]["per_channel_max"],
                         dtype=dtype, device=device)
        # Channels with no positive pre-activation on train cannot be normalised;
        # freeze them with a documented floor instead of fishing a threshold.
        out[name] = v.clamp(min=ZERO_THRESHOLD_FLOOR)
    return out


def if_step(pre: torch.Tensor, threshold: torch.Tensor,
            membrane: torch.Tensor | None):
    """One IF tick. Constant input, subtractive reset, per-channel threshold."""
    if membrane is None:
        membrane = torch.zeros_like(pre)
    v = membrane + pre / threshold.view(1, -1)
    spikes = torch.floor(v.clamp(min=0))       # integer bursts allowed
    membrane = v - spikes
    return spikes * threshold.view(1, -1), membrane


def install_if_hooks(model, thresholds):
    """Substitute IF at the exact ten stage-1 capture sites, nothing else.

    The author's GatedTeacher applies F.relu immediately after bn_node_h /
    bn_node_e, and MLPReadout applies F.relu after FC_layers[0]/[1].  Returning
    a non-negative spike train from the hook therefore passes through the
    author's ReLU unchanged.  With active=False the hook is a strict no-op,
    which makes the bitwise instrumentation fidelity check possible.
    """
    mem: dict[str, torch.Tensor] = {}
    state = {"active": False, "touched": set()}

    def make(name):
        def hook(_module, _inp, out):
            if not state["active"]:
                return out
            state["touched"].add(name)
            spikes, m = if_step(out, thresholds[name], mem.get(name))
            mem[name] = m
            return spikes
        return hook

    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.bn_node_h.register_forward_hook(make(f"gated{i}.node")))
        handles.append(layer.bn_node_e.register_forward_hook(make(f"gated{i}.edge")))
    handles.append(model.MLP_layer.FC_layers[0].register_forward_hook(make("mlp0")))
    handles.append(model.MLP_layer.FC_layers[1].register_forward_hook(make("mlp1")))
    return handles, mem, state


def naiveif_forward(model, batch, thresholds, T, hooks):
    """T ticks of the author's own forward with IF substitution at ten sites."""
    _handles, mem, state = hooks
    mem.clear()
    state["touched"].clear()
    state["active"] = True
    try:
        acc = None
        for _tick in range(T):
            logits = model(batch)
            acc = logits if acc is None else acc + logits
        missing = set(SITE_NAMES) - state["touched"]
        require(not missing, f"IF hooks did not cover all sites at T={T}: {sorted(missing)}")
        return acc / float(T)
    finally:
        state["active"] = False


def metric_row(logits: np.ndarray, teacher: np.ndarray, labels: np.ndarray) -> dict:
    pred = logits.argmax(1)
    tpred = teacher.argmax(1)
    mse = float(np.mean((logits - teacher) ** 2))
    tp = np.exp(teacher - teacher.max(1, keepdims=True))
    tp /= tp.sum(1, keepdims=True)
    lp = np.exp(logits - logits.max(1, keepdims=True))
    lp /= lp.sum(1, keepdims=True)
    kl = float(np.mean(np.sum(tp * (np.log(tp + 1e-12) - np.log(lp + 1e-12)), 1)))
    return {"accuracy": float(np.mean(pred == labels)),
            "teacher_agreement": float(np.mean(pred == tpred)),
            "logit_mse": mse, "teacher_to_output_kl": kl}


def precheck(_args=None) -> None:
    data_root = default_data_root().resolve()
    author_repo = default_author_repo().resolve()
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e1_residual_v1 as R
    mods = R.load_modules(author_repo)
    for seed in SEEDS:
        R.verify_bindings(seed, data_root, mods)
        d = json.loads(cal_path(seed).read_text(encoding="utf-8"))
        require(d["status"] == "fulltrain_label_free_threshold_calibration"
                and d["coverage"]["graphs"] == 45000,
                f"seed {seed}: stage-1 calibration missing or incomplete")
        require(d["dcgs_commit"] == DCGS_COMMIT, f"seed {seed}: stage-1 DCGS drift")
    require(sha256(CANON) == CANON_SHA, "canonical baseline hash drift")
    ap = R.author_provenance(author_repo, mods)
    require(ap["safe"]["commit"] == DCGS_COMMIT, "pinned DCGS commit drift")
    print(json.dumps({
        "check": "e2_naiveif_eval_precheck_v1", "status": "PASS",
        "seeds_verified": list(SEEDS), "timesteps": list(TIMESTEPS),
        "val_graphs": EXPECTED_VAL, "batch_size": BATCH_SIZE,
        "sites_per_tick": len(SITE_NAMES),
        "pinned_commit": ap["safe"]["commit"],
        "canonical_baseline_sha256": CANON_SHA,
        "calibration_reports": {str(s): sha256(cal_path(s)) for s in SEEDS},
        "note": "hash/structure gate only; no GPU pass, no scientific result",
    }, indent=2))


def run(args) -> None:
    seed = int(args.seed)
    require(seed in SEEDS, "seed must be one of the three frozen teachers")
    smoke = bool(args.limit)
    # A --limit run is a plumbing smoke probe and must NEVER occupy the
    # canonical output directory.
    destdir = (SMOKE / f"seed{seed}" if smoke else OUT / f"seed{seed}")
    require(not destdir.exists(), f"output dir already exists, refusing: {destdir}")
    data_root = default_data_root().resolve()
    author_repo = default_author_repo().resolve()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")

    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e1_residual_v1 as R
    mods = R.load_modules(author_repo)
    binding = R.verify_bindings(seed, data_root, mods)
    ap = R.author_provenance(author_repo, mods)
    require(ap["safe"]["commit"] == DCGS_COMMIT, "pinned DCGS commit drift")
    import eval_affine_validation as base
    data_start = base.processed_hashes(data_root)
    require(data_start == binding["data_hashes"], "processed data drift")
    require(sha256(CANON) == CANON_SHA, "canonical baseline hash drift")
    with np.load(CANON, allow_pickle=False) as ref:
        canon_indices = np.array(ref["indices"], copy=True)
        canon_labels = np.array(ref["labels"], copy=True).reshape(-1)
        canon_teacher = np.array(ref["teacher_logits"], copy=True)
    require(canon_indices.shape[0] == EXPECTED_VAL
            and np.array_equal(canon_indices, np.arange(EXPECTED_VAL)),
            "canonical index drift")
    require(canon_labels.shape[0] == EXPECTED_VAL, "canonical label drift")

    td = R.BINDINGS[seed]["teacher"]
    state = torch.load(td / "best.pt", map_location=device, weights_only=False)
    require(state.get("metadata") == binding["metadata"], "metadata mismatch")
    model = mods[-1].build_teacher(state, device).eval()
    require(len(model.layers) == 4, "frozen four-layer teacher drift")
    thresholds = load_calibration(seed, device, torch.float32)

    ds = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    require(len(ds) == EXPECTED_VAL, "official validation count drift")
    # Labels are read from the ORIGINAL dataset through a separate loader, so
    # the converter input stays strictly label-free.
    label_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    labels_all = np.concatenate(
        [b.y.view(-1).long().cpu().numpy() for b in label_loader]).reshape(-1)
    require(labels_all.shape[0] == EXPECTED_VAL, "label count drift")
    require(np.array_equal(labels_all, canon_labels), "canonical label drift")

    loader = DataLoader(base.LabelFreeDataset(ds), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    batches = []
    for i, b in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        require(getattr(b, "y", None) is None, "labels leaked into converter input")
        batches.append(b.to(device))
    require(bool(batches), "no batches")
    if not smoke:
        require(len(batches) == (EXPECTED_VAL + BATCH_SIZE - 1) // BATCH_SIZE,
                "unexpected batch count")

    out_by_T = {str(T): {"indices": [], "labels": [], "teacher_logits": [],
                         "logits": []} for T in TIMESTEPS}
    t0 = time.monotonic()
    total_graphs = 0
    fidelity = None
    canonical_teacher_bitwise = None
    hooks = None
    try:
        with torch.no_grad():
            # Instrumentation fidelity gate: a no-op hooked forward must be
            # bitwise identical to the uninstrumented frozen teacher forward.
            first = batches[0]
            plain = model(first)
            hooks = install_if_hooks(model, thresholds)
            identity = model(first)
            fidelity = float((plain - identity).abs().max().item())
            require(fidelity == 0.0,
                    f"instrumented identity forward drift: {fidelity}")

            for bi, batch in enumerate(batches):
                teacher = model(batch)
                require(torch.isfinite(teacher).all(), "nonfinite teacher logits")
                n = int(batch.num_graphs)
                start = bi * BATCH_SIZE
                idx = np.arange(start, start + n, dtype=np.int64)
                y = labels_all[start:start + n]
                require(y.shape[0] == n, "labels/batch length mismatch")
                if not smoke and bi == 0 and seed == 41:
                    canonical_teacher_bitwise = bool(np.array_equal(
                        teacher.detach().cpu().numpy(), canon_teacher[:n]))
                    require(canonical_teacher_bitwise,
                            "seed41 canonical teacher logits drift on first batch")
                for T in TIMESTEPS:
                    logits = naiveif_forward(model, batch, thresholds, T, hooks)
                    require(torch.isfinite(logits).all(), "nonfinite NaiveIF logits")
                    o = out_by_T[str(T)]
                    o["indices"].append(idx)
                    o["labels"].append(y)
                    o["teacher_logits"].append(teacher.detach().cpu().numpy())
                    o["logits"].append(logits.detach().cpu().numpy())
                total_graphs += n
                if args.limit or (bi + 1) % 10 == 0 or bi + 1 == len(batches):
                    print(json.dumps({"seed": seed, "batch": bi + 1, "of": len(batches),
                                      "graphs": total_graphs,
                                      "elapsed_sec": round(time.monotonic() - t0, 1)}),
                          flush=True)
    finally:
        if hooks is not None:
            for h in hooks[0]:
                h.remove()

    data_end = base.processed_hashes(data_root)
    require(data_end == data_start, "processed data changed during eval")
    if not smoke:
        require(total_graphs == EXPECTED_VAL, "did not cover all official val graphs")

    destdir.mkdir(parents=True, exist_ok=True)
    metrics, npz_manifest = [], {}
    for T in TIMESTEPS:
        o = out_by_T[str(T)]
        idx = np.concatenate(o["indices"])
        logits = np.concatenate(o["logits"])
        teacher = np.concatenate(o["teacher_logits"])
        labels = np.concatenate(o["labels"]).reshape(-1)
        require(labels.shape[0] == logits.shape[0] == idx.shape[0],
                "labels/logits/indices length mismatch")
        m = metric_row(logits, teacher, labels)
        npz = destdir / f"naiveif_seed{seed}_T{T}_v1.npz"
        np.savez(npz, indices=idx, labels=labels,
                 teacher_logits=teacher, logits=logits)
        npz_manifest[npz.name] = sha256(npz)
        ev = f"{RR}/{'logs/e2_naiveif_eval_smoke_v1' if smoke else 'results/e2-naiveif-eval-v1'}/seed{seed}/{npz.name}"
        for k, v in m.items():
            unit = {"accuracy": "fraction", "teacher_agreement": "fraction",
                    "logit_mse": "mse", "teacher_to_output_kl": "kl"}[k]
            metrics.append(row(seed, T, k, v, unit, ev))

    with (destdir / "naiveif_METRICS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(metrics)
    report = {
        "check": "e2_naiveif_full_validation_eval_v1",
        "status": ("smoke_probe_not_evidence" if smoke else
                   "real_naiveif_measurement_no_accuracy_or_energy_claim"),
        "method": "NaiveIF (data-normalised IF, NO operator-specific adaptation)",
        "protocol_sha256": PROTOCOL_SHA, "head_addendum_sha256": ADDENDUM_SHA,
        "dcgs_commit": DCGS_COMMIT,
        "teacher_seed": seed, "checkpoint_sha256": R.BINDINGS[seed]["teacher_sha"],
        "stage1_calibration_sha256": sha256(cal_path(seed)),
        "dataset": "CIFAR10", "split": "val",
        "coverage": {"graphs": total_graphs, "expected_graphs": EXPECTED_VAL,
                     "batch_size": BATCH_SIZE,
                     "order": "fixed sequential, label-free converter input",
                     "full_split": bool(not smoke), "timesteps": list(TIMESTEPS)},
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "torch": torch.__version__, "deterministic_algorithms": True,
        "instrumented_vs_frozen_max_abs_diff": fidelity,
        "canonical_index_label_bitwise": True,
        "seed41_canonical_teacher_bitwise": canonical_teacher_bitwise,
        "metrics": {str(T): {k: v for k, v in
                             [(m["metric"], m["value"]) for m in metrics
                              if m["T"] == str(T)]} for T in TIMESTEPS},
        "npz_sha256": npz_manifest,
        "zero_threshold_floor": ZERO_THRESHOLD_FLOOR,
        "if_neuron": {"input": "constant pre/_threshold each tick",
                      "reset": "subtractive", "membrane": "persistent across ticks",
                      "spikes": "floor(v.clamp(min=0)), integer bursts allowed",
                      "output_scale": "spike_count * per_channel_threshold"},
        "limitations": [
            "NaiveIF deliberately does NOT adapt sigmoid gates, BN, residual or pooling; degradation is a measured result, not an assumption.",
            "Dense PyTorch prototype; no event-driven backend, no hardware energy or latency claim.",
            "Thresholds come only from the 45000 official train graphs, never from validation/test.",
            "No event-count-as-energy, no accuracy-only method-advantage claim, no Story decision.",
        ],
        "total_job_seconds": round(time.monotonic() - t0, 1),
        "created_at_utc": utc(),
    }
    (destdir / "naiveif_report_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "seed": seed, "smoke": smoke,
                      "report": str((destdir / "naiveif_report_v1.json").relative_to(ROOT)),
                      "metric_rows": len(metrics), "npz": len(npz_manifest),
                      "fidelity": fidelity}, indent=2), flush=True)


def verify(_args=None) -> None:
    require(sha256(CANON) == CANON_SHA, "canonical baseline hash drift")
    with np.load(CANON, allow_pickle=False) as ref:
        canon_indices = np.array(ref["indices"], copy=True)
        canon_labels = np.array(ref["labels"], copy=True).reshape(-1)
    reports = {}
    for seed in SEEDS:
        d = OUT / f"seed{seed}"
        p = d / "naiveif_report_v1.json"
        require(p.is_file(), f"seed {seed}: missing report")
        r = json.loads(p.read_text(encoding="utf-8"))
        require(r["status"] == "real_naiveif_measurement_no_accuracy_or_energy_claim",
                f"seed {seed}: smoke probe is not evidence")
        require(r["coverage"]["graphs"] == EXPECTED_VAL
                and r["coverage"]["full_split"] is True,
                f"seed {seed}: incomplete validation coverage")
        require(r["teacher_seed"] == seed and r["dcgs_commit"] == DCGS_COMMIT,
                f"seed {seed}: binding drift")
        require(r["instrumented_vs_frozen_max_abs_diff"] == 0.0,
                f"seed {seed}: instrumentation fidelity drift")
        require(set(r["metrics"]) == {str(T) for T in TIMESTEPS},
                f"seed {seed}: timestep drift")
        for name, digest in r["npz_sha256"].items():
            npz = d / name
            require(sha256(npz) == digest, f"seed {seed}: NPZ hash mismatch: {name}")
            with np.load(npz, allow_pickle=False) as z:
                require(set(z.files) >= {"indices", "labels", "teacher_logits", "logits"},
                        f"seed {seed}: NPZ schema drift: {name}")
                require(np.array_equal(z["indices"], canon_indices),
                        f"seed {seed}: canonical index drift: {name}")
                require(np.array_equal(z["labels"].reshape(-1), canon_labels),
                        f"seed {seed}: canonical label drift: {name}")
        with (d / "naiveif_METRICS.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        require(len(rows) == len(TIMESTEPS) * 4, f"seed {seed}: metric row count drift")
        reports[str(seed)] = sha256(p)
    print(json.dumps({"check": "e2_naiveif_eval_verify_v1", "status": "PASS",
                      "seeds": list(SEEDS), "reports": reports}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("precheck")
    r = sub.add_parser("run")
    r.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--limit", type=int, default=0,
                   help="smoke probe only; writes to a separate smoke dir")
    sub.add_parser("verify")
    a = ap.parse_args()
    {"precheck": precheck, "run": run, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    main()
