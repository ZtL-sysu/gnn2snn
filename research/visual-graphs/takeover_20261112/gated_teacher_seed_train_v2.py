"""Seed-specific CIFAR10 PyG GatedGCN teacher training, v2.

Full mode is CUDA-only and accepts teacher seeds 42/43. --smoke uses only
small synthetic CPU graphs and never constructs a benchmark dataset loader.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from gated_teacher_model import GatedTeacher

HERE = Path(__file__).resolve().parent
RUNROOT = HERE.parent
DATA_ROOT = RUNROOT / "data" / "pyg_bench"
OUT_ROOT = HERE / "full_teacher_runs"
EXPECTED = {"train": 45000, "val": 5000}
TEACHER_SEEDS = (42, 43)
STUDENT_SEED_REFERENCE = 41
EPOCH_CAP = 200
TIME_CAP_HOURS = 12.0
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
MIN_LR = 1e-5

SOURCE_PATHS = (
    "gated_teacher_seed_train_v2.py",
    "gated-teacher-seed-protocol-v2.md",
    "gated_teacher_train.py",
    "gated_teacher_model.py",
    "gated_teacher_train_protocol_v1.md",
    "public_baselines/gated_net.py",
    "public_baselines/gated_layer.py",
    "public_baselines/benchmark_mlp_readout.py",
    "public_baselines/gated_cifar.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest():
    """Pin this runner, its protocol, and the unchanged v1/model/config basis."""
    missing = [name for name in SOURCE_PATHS if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing source-manifest files: " + ", ".join(missing))
    hashes = {name: sha256(HERE / name) for name in SOURCE_PATHS}
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"files": hashes, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def seed_all(seed, include_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_dimensions(sample):
    if getattr(sample, "edge_attr", None) is None:
        raise RuntimeError("PyG CIFAR10 sample has no edge_attr; gated teacher requires edge features")
    xdim = int(sample.x.shape[-1])
    pos = getattr(sample, "pos", None)
    posdim = int(pos.shape[-1]) if pos is not None else 0
    node_dim = xdim if xdim == 5 else xdim + posdim
    edge_dim = int(sample.edge_attr.reshape(sample.edge_attr.shape[0], -1).shape[-1])
    if node_dim < 1 or edge_dim < 1:
        raise RuntimeError("Invalid node/edge feature dimensions")
    return node_dim, edge_dim


def build_model(node_dim, edge_dim, device):
    return GatedTeacher(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=70, out_dim=70,
                        n_classes=10, layers=4, dropout=0.0, batch_norm=True,
                        residual=True, readout="mean").to(device)


def initial_state_digest(seed, node_dim=5, edge_dim=1):
    """Deterministic model-init fingerprint used only by local CPU checks."""
    seed_all(seed, include_cuda=False)
    model = build_model(node_dim, edge_dim, torch.device("cpu"))
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()
        logits = model(batch)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    if total == 0:
        raise RuntimeError("Empty evaluation loader")
    return {"graphs": total, "loss": loss_sum / total, "accuracy": correct / total}


def train_epoch(model, loader, optimizer, device, expected_graphs=None):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        total += count
        loss_sum += float(loss.item()) * count
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    if expected_graphs is not None and total != expected_graphs:
        raise RuntimeError(f"Incomplete training epoch: {total} != {expected_graphs}")
    if total == 0:
        raise RuntimeError("Empty training loader")
    return {"graphs": total, "loss": loss_sum / total, "accuracy": correct / total}


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def synthetic_smoke(seed):
    """CPU-only synthetic optimization/evaluation; no dataset access or output dir."""
    seed_all(seed, include_cuda=False)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10000)
    graphs = []
    for i in range(4):
        n = 3 + (i % 2)
        nodes = torch.arange(n, dtype=torch.long)
        src = nodes.repeat_interleave(2)
        dst = torch.stack((torch.roll(nodes, 1), torch.roll(nodes, -1)), dim=1).reshape(-1)
        edge_index = torch.stack((src, dst))
        graphs.append(Data(x=torch.randn((n, 5), generator=generator), edge_index=edge_index,
                           edge_attr=torch.rand((edge_index.shape[1], 1), generator=generator),
                           y=torch.tensor([i % 2])))
    batches = [Batch.from_data_list(graphs[:2]), Batch.from_data_list(graphs[2:])]
    model = GatedTeacher(node_dim=5, edge_dim=1, hidden_dim=70, out_dim=70,
                         n_classes=10, layers=4, dropout=0.0, batch_norm=True,
                         residual=True, readout="mean").cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_stats = train_epoch(model, batches, optimizer, torch.device("cpu"), expected_graphs=4)
    val_stats = evaluate(model, batches, torch.device("cpu"))
    manifest = source_manifest()
    return {"status": "synthetic_smoke_passed", "scope": "synthetic_cpu_only",
            "dataset_loaded": False, "device": "cpu", "seed": seed,
            "teacher_seed_role": "teacher initialization and synthetic RNG only",
            "student_seed": STUDENT_SEED_REFERENCE,
            "student_seed_role": "separate existing student-run seed; not applied to this teacher",
            "source_manifest": manifest, "train": train_stats, "val": val_stats}


def run_dir_for(seed):
    if seed not in TEACHER_SEEDS:
        raise ValueError(f"Unsupported teacher seed: {seed}")
    return OUT_ROOT / f"gated_cifar10_seed{seed}_full_v2"


def refuse_existing_output(run_dir):
    if Path(run_dir).exists():
        raise FileExistsError(f"Refusing existing output directory: {run_dir}")


def train_full(seed):
    run_dir = run_dir_for(seed)
    refuse_existing_output(run_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("Full CIFAR teacher training requires CUDA; use --smoke for CPU engineering checks")

    seed_all(seed)
    train = GNNBenchmarkDataset(root=str(DATA_ROOT), name="CIFAR10", split="train")
    val = GNNBenchmarkDataset(root=str(DATA_ROOT), name="CIFAR10", split="val")
    if len(train) != EXPECTED["train"] or len(val) != EXPECTED["val"]:
        raise RuntimeError(f"Unexpected official split sizes: train={len(train)}, val={len(val)}")
    node_dim, edge_dim = infer_dimensions(train[0])
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    device = torch.device("cuda")
    model = build_model(node_dim, edge_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=MIN_LR)
    processed = DATA_ROOT / "CIFAR10" / "processed"
    manifest = source_manifest()
    metadata = {
        "dataset": "CIFAR10", "seed": seed, "teacher_seed": seed,
        "teacher_seed_role": "teacher parameter initialization and training-order RNG",
        "student_seed_reference": STUDENT_SEED_REFERENCE,
        "student_seed_role": "separate student experiment seed; not changed or consumed by this teacher run",
        "rng_seeds": {"python": seed, "numpy": seed, "torch_cpu": seed, "torch_cuda_all": seed,
                      "train_order": seed, "train_order_mechanism": "DataLoader shuffle uses seeded Torch RNG; num_workers=0"},
        "scope": "all_official_pyg_train_and_val",
        "official_counts": dict(EXPECTED),
        "train_indices": "all official train indices, shuffled each epoch",
        "val_indices": "all official validation indices, deterministic order",
        "split_source": "PyG GNNBenchmarkDataset CIFAR10 train/val; test split is never loaded",
        "model": {"class": "GatedTeacher", "node_dim": node_dim, "edge_dim": edge_dim,
                  "hidden_dim": 70, "out_dim": 70, "layers": 4, "n_classes": 10,
                  "readout": "mean", "residual": True, "batch_norm": True, "dropout": 0.0,
                  "parameters": sum(p.numel() for p in model.parameters())},
        "optimization": {"optimizer": "Adam", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                         "scheduler": "ReduceLROnPlateau", "scheduler_mode": "min_val_loss",
                         "factor": 0.5, "patience": 10, "min_lr": MIN_LR,
                         "epochs_cap": EPOCH_CAP, "wall_time_hours_cap": TIME_CAP_HOURS,
                         "batch_size": BATCH_SIZE,
                         "stopping": "stop when learning rate <= min_lr, epoch cap reached, or wall-time cap reached"},
        "provenance": {
            "adaptation_statement": "PyG data/operator adaptation of public GatedGCN configuration; not a reproduction of original DGL pipeline",
            "public_architecture_commit": "b6c407712fa576e9699555e1e035d1e327ccae6c",
            "public_edge_definition": "scalar affinity from color/coordinate RBF similarity and per-node top-k graph construction",
            "pyg_edge_semantics": "uses stored edge_attr as provided; does not reconstruct public DGL affinity edges",
            "public_cifar_node_inputs": "mean pixel channels concatenated with 2D coordinates",
            "source_manifest": manifest,
            # Existing calibration source validation consumes this legacy key.
            # Keeping the same relative-path/hash mapping makes v2 provenance
            # fail closed under that validator without editing it.
            "source_sha256": manifest["files"],
            "processed_data_sha256": {p.name: sha256(p) for p in sorted(processed.glob("*.pt")) if p.is_file()},
        },
        "runtime": {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                    "pyg": torch_geometric.__version__, "numpy": np.__version__, "python": sys.version},
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    best_accuracy = -1.0
    start = time.monotonic()
    with (run_dir / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, EPOCH_CAP + 1):
            epoch_start = time.monotonic()
            train_stats = train_epoch(model, train_loader, optimizer, device, len(train))
            val_stats = evaluate(model, val_loader, device)
            if val_stats["graphs"] != len(val):
                raise RuntimeError(f"Incomplete validation: {val_stats['graphs']} != {len(val)}")
            scheduler.step(val_stats["loss"])
            elapsed = time.monotonic() - start
            record = {"epoch": epoch, "train": train_stats, "val": val_stats,
                      "lr": optimizer.param_groups[0]["lr"],
                      "epoch_sec": time.monotonic() - epoch_start, "elapsed_sec": elapsed}
            log.write(json.dumps(record) + "\n")
            log.flush()
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "best_val_accuracy": max(best_accuracy, val_stats["accuracy"]),
                     "metadata": metadata, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
                     "python_rng": random.getstate()}
            atomic_torch_save(state, run_dir / "last.pt")
            if val_stats["accuracy"] > best_accuracy:
                best_accuracy = val_stats["accuracy"]
                atomic_torch_save(state, run_dir / "best.pt")
            print(json.dumps(record), flush=True)
            if optimizer.param_groups[0]["lr"] <= MIN_LR or elapsed >= TIME_CAP_HOURS * 3600:
                break
    with (run_dir / "done.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "finished", "best_val_accuracy": best_accuracy,
                   "official_counts": dict(EXPECTED), "epochs_completed": epoch,
                   "best_checkpoint_sha256": sha256(run_dir / "best.pt"),
                   "last_checkpoint_sha256": sha256(run_dir / "last.pt"),
                   "source_manifest_sha256": manifest["manifest_sha256"],
                   "elapsed_sec": time.monotonic() - start}, stream, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=TEACHER_SEEDS, default=42,
                        help="teacher initialization and train-order seed (42 or 43)")
    parser.add_argument("--smoke", action="store_true",
                        help="CPU synthetic engineering check; no official data or run directory")
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(synthetic_smoke(args.seed), sort_keys=True))
    else:
        train_full(args.seed)


if __name__ == "__main__":
    main()

