"""Two-pass full official-train activation calibration for the frozen full teacher."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from activation_histogram import ActivationHistogram
from fold_teacher_bn import fold_model
from full_teacher_gcn import BenchmarkGCN, file_sha
from full_teacher_conversion import reference_activations

HERE = Path(__file__).resolve().parent
RUNROOT = HERE.parent
DATA_ROOT = RUNROOT / "data" / "pyg_bench"
RUNS_ROOT = HERE / "full_teacher_runs"
EXPECTED = {"MNIST": 55000, "CIFAR10": 45000}
SITE_NAMES = ["gcn_0", "gcn_1", "gcn_2", "gcn_3", "mlp_0", "mlp_1"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_terminal_run(run_dir, dataset):
    done_path = run_dir / "done.json"
    best_path = run_dir / "best.pt"
    metadata_path = run_dir / "metadata.json"
    for path in (done_path, best_path, metadata_path):
        if not path.is_file():
            raise RuntimeError(f"Required terminal teacher file missing: {path.name}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if done.get("status") != "finished":
        raise RuntimeError("Teacher is not terminal with status=finished")
    if metadata.get("dataset") != dataset or metadata.get("scope") != "full_official_train_val":
        raise RuntimeError("Refusing non-full or mismatched teacher run")
    if metadata.get("official_counts", {}).get("train") != EXPECTED[dataset]:
        raise RuntimeError("Teacher metadata has unexpected official train count")
    if metadata.get("script_sha256") != file_sha(HERE / "full_teacher_gcn.py"):
        raise RuntimeError("Teacher script hash does not match recorded metadata")
    return done, metadata, best_path


@torch.no_grad()
def collect_first_pass(model, loader, device):
    maxima = None
    rows = [0] * len(SITE_NAMES)
    graphs = 0
    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        acts = reference_activations(model, batch)
        if len(acts) != 6:
            raise RuntimeError(f"Expected six activation sites, got {len(acts)}")
        if maxima is None:
            maxima = [torch.zeros(a.shape[1], dtype=torch.float64, device=device) for a in acts]
        for i, a in enumerate(acts):
            if a.ndim != 2 or not bool(torch.isfinite(a).all()) or bool((a < 0).any()):
                raise RuntimeError(f"Invalid activation at {SITE_NAMES[i]}")
            maxima[i] = torch.maximum(maxima[i], a.max(dim=0).values.double())
            rows[i] += int(a.shape[0])
        graphs += int(batch.num_graphs)
        if batch_index % 100 == 0:
            print(json.dumps({"calibration_pass": 1, "graphs": graphs}), flush=True)
    if graphs == 0:
        raise RuntimeError("Empty calibration loader")
    return maxima, graphs, rows


@torch.no_grad()
def collect_histograms(model, loader, device, maxima, expected_graphs, bins=512):
    histograms = [ActivationHistogram(m.float(), bins=bins) for m in maxima]
    graphs = 0
    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        acts = reference_activations(model, batch)
        for hist, activation in zip(histograms, acts):
            hist.update(activation)
        graphs += int(batch.num_graphs)
        if batch_index % 100 == 0:
            print(json.dumps({"calibration_pass": 2, "graphs": graphs}), flush=True)
    if graphs != expected_graphs:
        raise RuntimeError(f"Pass graph count mismatch: {graphs} != {expected_graphs}")
    return histograms, graphs


def histogram_record(hist, quantile):
    per_channel, global_threshold = hist.thresholds(quantile=quantile)
    return {
        "rows": hist.rows,
        "channels": int(hist.maxima.numel()),
        "maxima": hist.maxima.cpu().tolist(),
        "channel_counts": hist.channel.cpu().tolist(),
        "global_counts": hist.global_counts.cpu().tolist(),
        "per_channel_thresholds": per_channel.cpu().tolist(),
        "global_threshold": float(global_threshold.item()),
    }


def run_calibration(dataset, teacher_run, output, batch_size=128, device="cpu", bins=512):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    done, teacher_meta, best_path = validate_terminal_run(Path(teacher_run), dataset)
    if bins != 512:
        raise ValueError("Protocol fixes bins=512")
    dev = torch.device(device)
    if dev.type == "cpu":
        torch.set_num_threads(4)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train = GNNBenchmarkDataset(root=str(DATA_ROOT), name=dataset, split="train")
    if len(train) != EXPECTED[dataset]:
        raise RuntimeError(f"Unexpected official train count: {len(train)}")
    loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
    input_dim = train[0].x.size(-1) + train[0].pos.size(-1)
    model = BenchmarkGCN(input_dim).to(dev)
    state = torch.load(best_path, map_location=dev, weights_only=False)
    if "model" not in state:
        raise RuntimeError("Best checkpoint lacks model state")
    model.load_state_dict(state["model"])
    model.eval()
    folded = fold_model(model).eval()
    # Two independent passes, each in the original official split order.
    maxima, graphs1, pass1_rows = collect_first_pass(folded, loader, dev)
    histograms, graphs2 = collect_histograms(folded, loader, dev, maxima, graphs1, bins)
    if graphs1 != EXPECTED[dataset] or graphs2 != EXPECTED[dataset]:
        raise RuntimeError("Calibration did not cover the full official training split")
    sites = {name: histogram_record(hist, 0.999) for name, hist in zip(SITE_NAMES, histograms)}
    code_files = ["full_teacher_calibrate.py", "activation_histogram.py", "full_teacher_gcn.py",
                  "full_teacher_conversion.py", "fold_teacher_bn.py", "full-conversion-protocol-v1.md"]
    processed = DATA_ROOT / dataset / "processed"
    global_thresholds = {name: value["global_threshold"] for name, value in sites.items()}
    channel_thresholds = {name: value["per_channel_thresholds"] for name, value in sites.items()}
    result = {
        "status": "complete", "dataset": dataset,
        "scope": "all_official_training_graphs_two_pass_original_order",
        "teacher_run": str(Path(teacher_run).resolve()), "teacher_epoch": int(state.get("epoch", -1)),
        "teacher_done": done, "teacher_metadata": teacher_meta,
        "checkpoint": {"path": str(best_path.resolve()), "sha256": sha256(best_path)},
        "checkpoint_sha256": sha256(best_path),
        "data": {"root": str(DATA_ROOT.resolve()), "graphs": len(train),
                 "processed_files_sha256": {p.name: sha256(p) for p in sorted(processed.glob("*.pt")) if p.is_file()}},
        "code_sha256": {name: sha256(HERE / name) for name in code_files},
        "calibration": {"passes": 2, "pass1_graphs": graphs1, "pass2_graphs": graphs2,
                        "full_train_graph_count": len(train),
                        "activation_rows_per_site": {k: v["rows"] for k, v in sites.items()},
                        "pass1_activation_rows_per_site": dict(zip(SITE_NAMES, pass1_rows)),
                        "bins": bins, "quantile": 0.999, "threshold_rule": "upper edge of first cumulative bin reaching ceil(N*q)",
                        "inactive_channel_floor": 1e-6, "sites": sites},
        "global_thresholds": global_thresholds,
        "channel_thresholds": channel_thresholds,
        "runtime": {"device": str(dev), "torch": torch.__version__, "python": sys.version,
                    "batch_size": batch_size},
    }
    # Exclusive create prevents clobbering an existing result, including races.
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    result = run_calibration(args.dataset, args.teacher_run, args.output, args.batch_size, args.device)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "graphs": result["calibration"]["pass2_graphs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
