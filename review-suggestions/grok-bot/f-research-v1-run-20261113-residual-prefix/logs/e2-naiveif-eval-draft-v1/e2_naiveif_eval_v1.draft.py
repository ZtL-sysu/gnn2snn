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
  * sigmoid edge gates, BatchNorm, residual connections and mean pooling are
    left EXACTLY as in the ANN (no adaptation, no folding, no cache).  Any
    degradation is reported as a real measured result, never assumed;
  * T in {1,2,4,8}; all 5000 official validation graphs, fixed sequential
    order, batch 128, label-free converter input, deterministic CUDA.

Metrics are the same four as E1 (accuracy, teacher agreement, logit MSE,
teacher-to-output KL), with one per-sample NPZ per (seed, T) carrying
indices / labels / teacher_logits / logits and a SHA-256 manifest.

Hard gates: frozen inputs re-verified through e1_residual_v1.verify_bindings;
stage-1 calibration reports verified before use; canonical indices/labels
checked bitwise; processed hashes checked before and after; output directories
are refused if they already exist.  No accuracy-only pass claim, no energy,
hardware-latency, speedup, novelty or method-advantage claim.
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
import torch.nn.functional as F  # noqa: E402
from torch_geometric.datasets import GNNBenchmarkDataset  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
RR = "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix"
CAL = RUNROOT / "results" / "e2-naiveif-v1"
OUT = RUNROOT / "results" / "e2-naiveif-eval-v1"

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


def naiveif_forward(model, batch, thresholds, T, device):
    """Dense NaiveIF forward; gates/BN/residual/pool unchanged (no adaptation)."""
    x = batch.x.float()
    if x.shape[-1] != model.embedding_h.in_features:
        x = torch.cat([x, batch.pos.float()], dim=-1)
    h0 = model.embedding_h(x)
    e0 = model.embedding_e(batch.edge_attr.float().reshape(-1, model.edge_dim))
    src, dst = batch.edge_index.long()
    mem = {}
    logits_sum = None
    for _tick in range(T):
        h, e = h0, e0
        for i, layer in enumerate(model.layers):
            a = layer.A(h)
            b = layer.B(h)
            z = layer.C(e) + layer.D(h)[src] + layer.E(h)[dst]
            gates = torch.sigmoid(z)
            num = torch.zeros_like(a)
            den = torch.zeros_like(a)
            num.index_add_(0, dst, gates * b[src])
            den.index_add_(0, dst, gates)
            pre_h = layer.bn_node_h(a + num / (den + layer.eps))
            pre_e = layer.bn_node_e(z)
            nh, mem[f"gated{i}.node"] = if_step(
                pre_h, thresholds[f"gated{i}.node"], mem.get(f"gated{i}.node"))
            ne, mem[f"gated{i}.edge"] = if_step(
                pre_e, thresholds[f"gated{i}.edge"], mem.get(f"gated{i}.edge"))
            if layer.residual:
                h, e = h + nh, e + ne
            else:
                h, e = nh, ne
        ones = torch.ones(h.shape[0], dtype=torch.long, device=device)
        idx = batch.batch.long().to(device)
        graph_h = torch.zeros((int(idx.max().item()) + 1, h.shape[1]),
                              dtype=h.dtype, device=device)
        graph_h.index_add_(0, idx, h)
        counts = torch.zeros(graph_h.shape[0], dtype=h.dtype, device=device)
        counts.index_add_(0, idx, ones.to(h.dtype))
        graph_h = graph_h / counts.clamp(min=1).view(-1, 1)
        z = graph_h
        for j in (0, 1):
            pre = model.MLP_layer.FC_layers[j](z)
            z, mem[f"mlp{j}"] = if_step(pre, thresholds[f"mlp{j}"],
                                        mem.get(f"mlp{j}"))
        logits = model.MLP_layer.FC_layers[2](z)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / float(T)


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
    ap = R.author_provenance(author_repo, mods)
    require(ap["safe"]["commit"] == DCGS_COMMIT, "pinned DCGS commit drift")
    print(json.dumps({
        "check": "e2_naiveif_eval_precheck_v1", "status": "PASS",
        "seeds_verified": list(SEEDS), "timesteps": list(TIMESTEPS),
        "val_graphs": EXPECTED_VAL, "batch_size": BATCH_SIZE,
        "sites_per_tick": len(SITE_NAMES),
        "pinned_commit": ap["safe"]["commit"],
        "calibration_reports": {str(s): sha256(cal_path(s)) for s in SEEDS},
        "note": "hash/structure gate only; no GPU pass, no scientific result",
    }, indent=2))


def run(args) -> None:
    seed = int(args.seed)
    require(seed in SEEDS, "seed must be one of the three frozen teachers")
    destdir = OUT / f"seed{seed}"
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

    td = R.BINDINGS[seed]["teacher"]
    state = torch.load(td / "best.pt", map_location=device, weights_only=False)
    require(state.get("metadata") == binding["metadata"], "metadata mismatch")
    model = mods[-1].build_teacher(state, device).eval()
    require(len(model.layers) == 4, "frozen four-layer teacher drift")
    thresholds = load_calibration(seed, device, torch.float32)

    ds = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="val")
    require(len(ds) == EXPECTED_VAL, "official validation count drift")
    loader = DataLoader(base.LabelFreeDataset(ds), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    batches = []
    for i, b in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        require(getattr(b, "y", None) is None, "labels leaked into converter input")
        batches.append(b.to(device))

    out_by_T = {str(T): {"indices": [], "labels": [], "teacher_logits": [],
                         "logits": []} for T in TIMESTEPS}
    t0 = time.monotonic()
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            teacher = model(batch)
            require(torch.isfinite(teacher).all(), "nonfinite teacher logits")
            for T in TIMESTEPS:
                logits = naiveif_forward(model, batch, thresholds, T, device)
                require(torch.isfinite(logits).all(), "nonfinite NaiveIF logits")
                o = out_by_T[str(T)]
                o["indices"].append(np.arange(bi * BATCH_SIZE,
                                              bi * BATCH_SIZE + logits.shape[0]))
                o["labels"].append(batch.y.detach().cpu().numpy() if hasattr(batch, "y") else
                                   ds[bi * BATCH_SIZE].y.numpy().reshape(1))
                o["teacher_logits"].append(teacher.detach().cpu().numpy())
                o["logits"].append(logits.detach().cpu().numpy())
            if args.limit or (bi + 1) % 10 == 0 or bi + 1 == len(batches):
                print(json.dumps({"seed": seed, "batch": bi + 1, "of": len(batches),
                                  "elapsed_sec": round(time.monotonic() - t0, 1)}), flush=True)

    data_end = base.processed_hashes(data_root)
    require(data_end == data_start, "processed data changed during eval")

    destdir.mkdir(parents=True, exist_ok=True)
    metrics, npz_manifest = [], {}
    for T in TIMESTEPS:
        o = out_by_T[str(T)]
        idx = np.concatenate(o["indices"])
        logits = np.concatenate(o["logits"])
        teacher = np.concatenate(o["teacher_logits"])
        labels = np.concatenate(o["labels"]).reshape(-1)
        require(labels.shape[0] == logits.shape[0], "labels/logits length mismatch")
        m = metric_row(logits, teacher, labels)
        npz = destdir / f"naiveif_seed{seed}_T{T}_v1.npz"
        np.savez(npz, indices=idx, labels=labels,
                 teacher_logits=teacher, logits=logits)
        npz_manifest[npz.name] = sha256(npz)
        ev = f"{RR}/results/e2-naiveif-eval-v1/seed{seed}/{npz.name}"
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
        "status": ("smoke_probe_not_evidence" if args.limit else
                   "real_naiveif_measurement_no_accuracy_or_energy_claim"),
        "method": "NaiveIF (data-normalised IF, NO operator-specific adaptation)",
        "protocol_sha256": PROTOCOL_SHA, "head_addendum_sha256": ADDENDUM_SHA,
        "dcgs_commit": DCGS_COMMIT,
        "teacher_seed": seed, "checkpoint_sha256": R.BINDINGS[seed]["teacher_sha"],
        "stage1_calibration_sha256": sha256(cal_path(seed)),
        "dataset": "CIFAR10", "split": "val",
        "coverage": {"graphs": int(sum(len(o["indices"][0]) for o in []) or
                                   sum(len(x) for x in out_by_T[str(TIMESTEPS[0])]["logits"])),
                     "expected_graphs": EXPECTED_VAL, "batch_size": BATCH_SIZE,
                     "order": "fixed sequential, label-free converter input",
                     "full_split": bool(not args.limit),
                     "timesteps": list(TIMESTEPS)},
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "torch": torch.__version__, "deterministic_algorithms": True,
        "metrics": {str(T): {k: v for k, v in
                             [(m["metric"], m["value"]) for m in metrics
                              if m["T"] == str(T)]} for T in TIMESTEPS},
        "npz_sha256": npz_manifest,
        "zero_threshold_floor": ZERO_THRESHOLD_FLOOR,
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
    print(json.dumps({"status": "PASS", "seed": seed,
                      "report": str((destdir / "naiveif_report_v1.json").relative_to(ROOT)),
                      "metric_rows": len(metrics), "npz": len(npz_manifest)}, indent=2),
          flush=True)


def verify(_args=None) -> None:
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
        require(set(r["metrics"]) == {str(T) for T in TIMESTEPS},
                f"seed {seed}: timestep drift")
        for name, digest in r["npz_sha256"].items():
            require(sha256(d / name) == digest, f"seed {seed}: NPZ hash mismatch: {name}")
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
