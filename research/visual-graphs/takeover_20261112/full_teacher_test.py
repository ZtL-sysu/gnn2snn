"""One full official test evaluation for a completed, frozen teacher.

Historic project experiments exposed these public test sets. The output must
not be called an untouched confirmation set.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from full_teacher_gcn import BenchmarkGCN, DATA_ROOT, OUT_ROOT, file_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("MNIST", "CIFAR10"), required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    run = OUT_ROOT / f"{args.dataset}_seed{args.seed}_full"
    out = run / "teacher_full_test.json"
    pred_out = run / "teacher_full_test_predictions.npz"
    if out.exists() or pred_out.exists():
        raise SystemExit("Test output already exists; refusing to rerun or overwrite")
    if not (run / "done.json").is_file():
        raise SystemExit("Teacher training has not reached its recorded terminal state")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    checkpoint_path = run / "best.pt"
    checkpoint_hash = file_sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["metadata"]["dataset"] != args.dataset or checkpoint["metadata"]["seed"] != args.seed:
        raise RuntimeError("Checkpoint identity mismatch")
    model = BenchmarkGCN(input_dim=checkpoint["metadata"]["input_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = GNNBenchmarkDataset(root=str(DATA_ROOT), name=args.dataset, split="test")
    if len(dataset) != 10000:
        raise RuntimeError(f"Unexpected official test size: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    logits, labels = [], []
    start = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits.append(model(batch).cpu().numpy())
            labels.append(batch.y.view(-1).cpu().numpy())
    logits, labels = np.concatenate(logits), np.concatenate(labels)
    assert len(labels) == 10000 and logits.shape == (10000, 10)
    np.savez_compressed(pred_out, index=np.arange(len(labels)), labels=labels, logits=logits,
                        predictions=logits.argmax(axis=1))
    result = {"dataset": args.dataset, "seed": args.seed, "split": "full official test, historically exposed",
              "graphs": len(labels), "checkpoint_epoch": checkpoint["epoch"],
              "checkpoint_sha256": checkpoint_hash, "accuracy": float(np.mean(logits.argmax(axis=1) == labels)),
              "prediction_artifact_sha256": file_sha(pred_out), "device": str(device),
              "elapsed_sec": time.monotonic() - start,
              "qualification": "Teacher selected on full validation, but prior project test exposure remains."}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
