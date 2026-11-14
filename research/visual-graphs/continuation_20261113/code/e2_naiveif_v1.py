"""E2 faithful NaiveIF comparator, stage 1: full-train label-free calibration.

PROTOCOL-v3 requires a *faithful* NaiveIF baseline (plus official DCGS, QCFS,
FTBC, SNN Calibration, One-Spike and directly trained SpikingGCN).  NaiveIF is
the standard data-normalised ANN-to-SNN conversion: replace each ReLU with an
integrate-and-fire neuron whose firing threshold is the maximum pre-activation
observed on the TRAINING split, with no calibration trick, no gate-specific
adaptation, no threshold/bias fishing, and no teacher hidden-state oracle.

Stage 1 produces the necessary, independently verifiable input to that
conversion: per-channel maximum pre-activation at every ReLU site, measured
label-free on ALL 45000 official CIFAR10 training graphs, in fixed sequential
order, for each of the three frozen teachers (seeds 41/42/43).  Stage 2 (the
IF forward and full-validation evaluation) consumes these frozen calibration
reports; it is NOT claimed here.

Hard gates:
  * frozen teacher / calibration / protocol / 5 official processed hashes are
    re-verified through e1_residual_v1.verify_bindings before any data pass;
  * labels are stripped by LabelFreeDataset before they reach the model;
  * processed-data hashes are checked before and after the pass;
  * deterministic CUDA, fixed batch order, batch128, no seed/T selection;
  * reports are written exclusively (existing files are refused), and every
    numeric value is recomputed from the model weights, never copied.

This is a diagnostic/calibration artifact. It makes NO accuracy, energy,
latency, speedup, novelty or method-advantage claim.
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

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
RR = "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix"
OUT = RUNROOT / "results" / "e2-naiveif-v1"

PROTOCOL_SHA = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
ADDENDUM_SHA = "8bde5832d556c8c7b04ee630f2390703c33a26fa750f196f4dc1c94da61291f6"
DCGS_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
SEEDS = (41, 42, 43)
EXPECTED_TRAIN = 45000
BATCH_SIZE = 128
SITE_NAMES = tuple(
    f"gated{i}.{kind}" for i in range(4) for kind in ("node", "edge")
) + ("mlp0", "mlp1")

CSV_COLUMNS = ("task_id", "agent_id", "run_id", "dataset", "split", "coverage",
               "method", "teacher_seed", "student_seed", "T", "metric", "value",
               "unit", "evidence", "protocol_sha256", "status")
TASK, AGENT, RUN_ID = "visual-conversion", "20261113-wb-resume", "e2-naiveif-v1"


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


def report_path(seed: int) -> Path:
    return OUT / f"naiveif_calibration_seed{seed}_v1.json"


def load_modules(author_repo: Path):
    import importlib
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import e1_residual_v1 as R
    mods = R.load_modules(author_repo)
    import eval_affine_validation as base
    return R, base, mods


def build_model(seed: int, data_root: Path, author_repo: Path, device):
    import torch
    R, base, mods = load_modules(author_repo)
    binding = R.verify_bindings(seed, data_root, mods)
    ap = R.author_provenance(author_repo, mods)
    require(ap["safe"]["commit"] == DCGS_COMMIT, "pinned DCGS commit drift")
    td = R.BINDINGS[seed]["teacher"]
    state = torch.load(td / "best.pt", map_location=device, weights_only=False)
    require(state.get("metadata") == binding["metadata"], "checkpoint metadata mismatch")
    teacher_helper = mods[-1]
    model = teacher_helper.build_teacher(state, device).eval()
    require(len(model.layers) == 4, "frozen four-layer teacher drift")
    require(not model.training, "model must be in eval mode")
    return R, base, mods, binding, model


def capture_sites(model):
    """Register forward hooks on the exact pre-ReLU tensors.

    GatedGCNLayer applies bn_node_h / bn_node_e then F.relu, so the BN output
    IS the ReLU pre-activation. The MLP head applies Linear then F.relu for
    its first two layers. Hook outputs only; nothing is modified.
    """
    captured = {}
    handles = []

    def make(name):
        def hook(_module, _inp, out):
            captured[name] = out.detach()
        return hook

    for i, layer in enumerate(model.layers):
        handles.append(layer.bn_node_h.register_forward_hook(make(f"gated{i}.node")))
        handles.append(layer.bn_node_e.register_forward_hook(make(f"gated{i}.edge")))
    handles.append(model.MLP_layer.FC_layers[0].register_forward_hook(make("mlp0")))
    handles.append(model.MLP_layer.FC_layers[1].register_forward_hook(make("mlp1")))
    return captured, handles


def precheck(_args=None) -> None:
    data_root = default_data_root().resolve()
    author_repo = default_author_repo().resolve()
    R, base, mods = load_modules(author_repo)
    for seed in SEEDS:
        R.verify_bindings(seed, data_root, mods)
    ap = R.author_provenance(author_repo, mods)
    require(ap["safe"]["commit"] == DCGS_COMMIT, "pinned DCGS commit drift")
    require(tuple(SITE_NAMES) == tuple(SITE_NAMES), "site names")
    print(json.dumps({
        "check": "e2_naiveif_calibration_precheck_v1", "status": "PASS",
        "seeds_verified": list(SEEDS), "train_graphs": EXPECTED_TRAIN,
        "batch_size": BATCH_SIZE, "sites": list(SITE_NAMES),
        "pinned_commit": ap["safe"]["commit"],
        "protocol_sha256": PROTOCOL_SHA,
        "note": "hash/structure gate only; no GPU pass, no scientific result",
    }, indent=2))


def calibrate(args) -> None:
    import numpy as np
    import torch
    from torch_geometric.datasets import GNNBenchmarkDataset
    from torch_geometric.loader import DataLoader

    seed = int(args.seed)
    require(seed in SEEDS, "seed must be one of the three frozen teachers")
    # A --limit run is a plumbing smoke probe: it must NEVER write the canonical
    # report path, otherwise the later full-split calibration is refused.
    smoke = bool(args.limit)
    out = (RUNROOT / "logs" / f"e2_naiveif_smoke_seed{seed}_v1.json"
           if smoke else report_path(seed))
    require(not out.exists(), f"report already exists, refusing to overwrite: {out}")

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

    R, base, mods, binding, model = build_model(seed, data_root, author_repo, device)
    data_start = base.processed_hashes(data_root)
    require(data_start == binding["data_hashes"],
            "processed data differ from frozen teacher metadata")

    ds = GNNBenchmarkDataset(root=str(data_root), name="CIFAR10", split="train")
    require(len(ds) == EXPECTED_TRAIN, "official train count drift")
    loader = DataLoader(base.LabelFreeDataset(ds), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    batches = []
    for i, b in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        require(getattr(b, "y", None) is None, "labels leaked into calibration input")
        batches.append(b.to(device))
    if args.limit:
        require(len(batches) == args.limit, "not enough batches")
    else:
        require(len(batches) == (EXPECTED_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE,
                "unexpected batch count")

    captured, handles = capture_sites(model)
    maxima = {name: None for name in SITE_NAMES}
    graphs = 0
    t0 = time.monotonic()
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            captured.clear()
            logits = model(batch)
            require(torch.isfinite(logits).all(), "nonfinite teacher logits")
            require(set(captured) == set(SITE_NAMES),
                    f"missing hook sites: {sorted(set(SITE_NAMES) - set(captured))}")
            for name in SITE_NAMES:
                v = captured[name]
                # ReLU passes non-negative values; threshold balancing uses the
                # maximum positive pre-activation per output channel.
                channel_max = v.clamp(min=0).amax(dim=0).to(torch.float64)
                m = maxima[name]
                maxima[name] = channel_max if m is None else torch.maximum(m, channel_max)
            graphs += int(batch.num_graphs)
            if args.limit or (bi + 1) % 50 == 0 or bi + 1 == len(batches):
                print(json.dumps({"seed": seed, "batch": bi + 1, "of": len(batches),
                                  "graphs": graphs,
                                  "elapsed_sec": round(time.monotonic() - t0, 1)}), flush=True)
    for h in handles:
        h.remove()

    data_end = base.processed_hashes(data_root)
    require(data_end == data_start, "processed data changed during calibration")
    if not args.limit:
        require(graphs == EXPECTED_TRAIN, "did not cover all official train graphs")

    payload = {
        name: {"per_channel_max": [float(x) for x in maxima[name].tolist()],
               "global_max": float(maxima[name].max().item()),
               "channels": int(maxima[name].numel())}
        for name in SITE_NAMES
    }
    report = {
        "check": "e2_naiveif_fulltrain_label_free_calibration_v1",
        "status": ("smoke_probe_not_evidence" if args.limit else
                   "fulltrain_label_free_threshold_calibration"),
        "method": "NaiveIF stage 1 (data-based threshold balancing input)",
        "protocol_sha256": PROTOCOL_SHA,
        "head_addendum_sha256": ADDENDUM_SHA,
        "dcgs_commit": DCGS_COMMIT,
        "teacher_seed": seed,
        "checkpoint_sha256": R.BINDINGS[seed]["teacher_sha"],
        "dataset": "CIFAR10", "split": "train",
        "coverage": {"graphs": graphs, "expected_graphs": EXPECTED_TRAIN,
                     "batch_size": BATCH_SIZE, "order": "fixed sequential, label-free",
                     "full_split": bool(not args.limit)},
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "torch": torch.__version__,
        "deterministic_algorithms": True,
        "sites": payload,
        "site_names": list(SITE_NAMES),
        "processed_data_sha256": data_start,
        "total_job_seconds": round(time.monotonic() - t0, 1),
        "limitations": [
            "Threshold calibration only; no IF forward, no accuracy, no energy, no latency claim.",
            "Max pre-activation is the standard NaiveIF normalisation basis; no percentile or threshold fishing.",
            "Gate/BN/residual/pooling operators are NOT adapted here; stage 2 must document real adaptation limits rather than assume them.",
        ],
        "created_at_utc": utc(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(json.dumps({"status": "PASS", "report": str(out.relative_to(ROOT)),
                      "report_sha256": sha256(out), "graphs": graphs,
                      "sites": len(SITE_NAMES)}, indent=2), flush=True)


def verify(_args=None) -> None:
    expected = {seed: report_path(seed) for seed in SEEDS}
    for seed, p in expected.items():
        require(p.is_file(), f"missing calibration report for seed {seed}")
        d = json.loads(p.read_text(encoding="utf-8"))
        require(d["status"] == "fulltrain_label_free_threshold_calibration",
                f"seed {seed}: smoke probe is not evidence")
        require(d["coverage"]["full_split"] is True and d["coverage"]["graphs"] == EXPECTED_TRAIN,
                f"seed {seed}: incomplete train coverage")
        require(d["teacher_seed"] == seed, f"seed {seed}: seed mismatch")
        require(set(d["sites"]) == set(SITE_NAMES), f"seed {seed}: site mismatch")
        require(d["dcgs_commit"] == DCGS_COMMIT, f"seed {seed}: DCGS drift")
        require(d["protocol_sha256"] == PROTOCOL_SHA, f"seed {seed}: protocol drift")
        for name in SITE_NAMES:
            s = d["sites"][name]
            require(len(s["per_channel_max"]) == s["channels"] > 0,
                    f"seed {seed} {name}: channel width mismatch")
            require(all(x >= 0 and x < 1e9 for x in s["per_channel_max"]),
                    f"seed {seed} {name}: non-finite/absurd threshold")
            require(abs(s["global_max"] - max(s["per_channel_max"])) < 1e-6,
                    f"seed {seed} {name}: global max inconsistent")
    print(json.dumps({"check": "e2_naiveif_calibration_verify_v1", "status": "PASS",
                      "seeds": list(SEEDS),
                      "reports": {str(k): sha256(v) for k, v in expected.items()}}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("precheck")
    c = sub.add_parser("calibrate")
    c.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--limit", type=int, default=0,
                   help="smoke probe only; writes NO canonical report")
    sub.add_parser("verify")
    a = ap.parse_args()
    {"precheck": precheck, "calibrate": calibrate, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    main()
