"""Full-split PyG adaptation of Benchmarking GNNs' superpixel GCN.

See teacher-full-protocol-v1.md. This script trains on all official train
graphs and uses all official validation graphs; it never loads the test split.
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
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

HERE = Path(__file__).resolve().parent
RUNROOT = HERE.parent
DATA_ROOT = RUNROOT / "data" / "pyg_bench"
OUT_ROOT = HERE / "full_teacher_runs"


class BenchmarkGCN(nn.Module):
    """PyG port of the reference GCNNet with DGL GraphConv normalization."""
    def __init__(self, input_dim, width=146, layers=4, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, width)
        self.convs = nn.ModuleList([GCNConv(width, width, normalize=False, add_self_loops=False)
                                    for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(layers)])
        self.mlp = nn.Sequential(nn.Linear(width, width // 2), nn.ReLU(),
                                 nn.Linear(width // 2, width // 4), nn.ReLU(),
                                 nn.Linear(width // 4, num_classes))

    def forward(self, batch):
        x = torch.cat([batch.x.float(), batch.pos.float()], dim=-1)
        h = self.embedding(x)
        source, target = batch.edge_index
        n = h.shape[0]
        deg_out = torch.bincount(source, minlength=n).clamp(min=1).to(h.dtype)
        deg_in = torch.bincount(target, minlength=n).clamp(min=1).to(h.dtype)
        edge_weight = deg_out[source].rsqrt() * deg_in[target].rsqrt()
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.relu(norm(conv(h, batch.edge_index, edge_weight)))
        return self.mlp(global_mean_pool(h, batch.batch))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        labels = batch.y.view(-1).long()
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return {"graphs": total, "loss": loss_sum / total, "accuracy": correct / total}


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MNIST", "CIFAR10"), required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--smoke", action="store_true", help="one batch engineering check; not a scientific result")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; protocol requires the user's GPU")
    if args.epochs < 1 or args.epochs > 200 or args.max_hours <= 0 or args.max_hours > 12:
        raise ValueError("Requested budget outside frozen protocol")
    run_dir = OUT_ROOT / f"{args.dataset}_seed{args.seed}_{'smoke' if args.smoke else 'full'}"
    run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = torch.device("cuda")
    train = GNNBenchmarkDataset(root=str(DATA_ROOT), name=args.dataset, split="train")
    val = GNNBenchmarkDataset(root=str(DATA_ROOT), name=args.dataset, split="val")
    expected = {"MNIST": (55000, 5000), "CIFAR10": (45000, 5000)}[args.dataset]
    if (len(train), len(val)) != expected:
        raise RuntimeError(f"Unexpected official split sizes: {(len(train), len(val))}")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    input_dim = train[0].x.size(-1) + train[0].pos.size(-1)
    model = BenchmarkGCN(input_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    processed = DATA_ROOT / args.dataset / "processed"
    metadata = {"dataset": args.dataset, "seed": args.seed, "scope": "engineering_smoke" if args.smoke else "full_official_train_val",
                "official_counts": {"train": len(train), "val": len(val)},
                "train_indices": "all official train indices in randomized order each epoch",
                "val_indices": "all official val indices", "input_dim": input_dim,
                "parameters": sum(p.numel() for p in model.parameters()),
                "original_repo_commit": "b6c407712fa576e9699555e1e035d1e327ccae6c",
                "script_sha256": file_sha(__file__), "protocol_sha256": file_sha(HERE / "teacher-full-protocol-v1.md"),
                "processed_files_sha256": {p.name: file_sha(p) for p in sorted(processed.glob("*.pt")) if p.is_file()},
                "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                "pyg": torch_geometric.__version__, "numpy": np.__version__, "python": sys.version,
                "epochs_cap": args.epochs, "batch_size": args.batch_size, "max_hours": args.max_hours}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    best_acc = -1.0
    start = time.monotonic()
    with (run_dir / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_start = time.monotonic()
            n_seen, n_correct, train_loss = 0, 0, 0.0
            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = batch.to(dev)
                labels = batch.y.view(-1).long()
                opt.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                opt.step()
                count = labels.numel()
                n_seen += count
                n_correct += (logits.argmax(1) == labels).sum().item()
                train_loss += loss.item() * count
                if not args.smoke and batch_idx % 100 == 0:
                    print(json.dumps({"epoch": epoch, "batch": batch_idx,
                                      "train_graphs_seen": n_seen,
                                      "epoch_elapsed_sec": time.monotonic()-epoch_start}), flush=True)
                if args.smoke:
                    break
            if args.smoke:
                print(json.dumps({"smoke_graphs": n_seen, "loss": train_loss / n_seen, "device": str(dev)}), flush=True)
                return
            if n_seen != len(train):
                raise RuntimeError(f"Incomplete train epoch: {n_seen} != {len(train)}")
            metrics = validate(model, val_loader, dev)
            if metrics["graphs"] != len(val):
                raise RuntimeError("Incomplete validation")
            scheduler.step(metrics["loss"])
            record = {"epoch": epoch, "train_graphs": n_seen, "train_loss": train_loss/n_seen,
                      "train_accuracy": n_correct/n_seen, "val": metrics,
                      "lr": opt.param_groups[0]["lr"], "epoch_sec": time.monotonic()-epoch_start,
                      "elapsed_sec": time.monotonic()-start}
            log.write(json.dumps(record) + "\n")
            log.flush()
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": scheduler.state_dict(), "best_val_accuracy": max(best_acc, metrics["accuracy"]),
                     "metadata": metadata, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
                     "python_rng": random.getstate()}
            atomic_save(state, run_dir / "last.pt")
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                atomic_save(state, run_dir / "best.pt")
            print(json.dumps(record), flush=True)
            if opt.param_groups[0]["lr"] < 1e-5 or (time.monotonic()-start) >= args.max_hours*3600:
                break
    (run_dir / "done.json").write_text(json.dumps({"status": "finished", "best_val_accuracy": best_acc,
                                                   "elapsed_sec": time.monotonic()-start}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
