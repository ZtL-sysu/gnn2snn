"""Two-pass label-free channel-threshold calibration for frozen-gate GatedTeacher."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from activation_histogram import ActivationHistogram
from gated_teacher_model import GatedTeacher

HERE = Path(__file__).resolve().parent
RUNROOT = HERE.parent
DATA_ROOT = RUNROOT / "data" / "pyg_bench"
OUT_ROOT = HERE / "full_teacher_runs"
RUN_NAME = "gated_cifar10_seed41_full_v1"
EXPECTED_TRAIN, EXPECTED_VAL = 45000, 5000
SITE_NAMES = ("gated_0", "gated_1", "gated_2", "gated_3", "mlp_0", "mlp_1")
CALIBRATION_FILES = (
    "gated_frozen_gate_calibrate.py", "gated_frozen_gate_validate.py",
    "activation_histogram.py", "gated_teacher_model.py",
    "gated_frozen_gate_prototype.py", "gated_teacher_train.py",
    "gated_teacher_train_protocol_v1.md", "gated-frozen-gate-prototype-protocol.md",
    "gated-frozen-gate-calibration-validation-protocol.md",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def processed_file_hashes(processed_dir):
    return {p.name: sha256(p) for p in sorted(Path(processed_dir).glob("*.pt")) if p.is_file()}


def resolve_teacher_source(root, recorded_name):
    """Resolve metadata's recorded basename to its unique training source file."""
    root = Path(root)
    recorded = Path(recorded_name)
    candidates = [root / recorded]
    if recorded.parent == Path("."):
        candidates.append(root / "public_baselines" / recorded.name)
    existing = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if len(existing) != 1:
        raise RuntimeError(f"Teacher source name is missing or ambiguous: {recorded_name}")
    return existing[0]


class LabelFreeGraphView(torch.utils.data.Dataset):
    """Expose only model inputs; labels are never copied into calibration batches."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original = self.dataset[index]
        values = {key: getattr(original, key) for key in ("x", "edge_index", "edge_attr", "pos")
                  if getattr(original, key, None) is not None}
        return Data(**values)


def validate_teacher_run(teacher_run):
    run = Path(teacher_run)
    metadata_path, done_path, checkpoint = run / "metadata.json", run / "done.json", run / "best.pt"
    for path in (metadata_path, done_path, checkpoint):
        if not path.is_file():
            raise RuntimeError(f"Missing completed-teacher artifact: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != "CIFAR10" or metadata.get("scope") != "all_official_pyg_train_and_val":
        raise RuntimeError("Expected the completed full official CIFAR10 gated teacher")
    if metadata.get("official_counts") != {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL}:
        raise RuntimeError("Teacher metadata has unexpected official split counts")
    if done.get("status") != "finished" or done.get("official_counts") != metadata["official_counts"]:
        raise RuntimeError("Teacher run is not a matching terminal full-split run")
    checkpoint_hash = sha256(checkpoint)
    if done.get("best_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("best.pt hash does not match done.json")
    # Verify every source recorded by the completed training run, including the
    # model definition and data-adaptation protocol used to select this teacher.
    for rel, expected in metadata.get("provenance", {}).get("source_sha256", {}).items():
        path = resolve_teacher_source(HERE, rel)
        if sha256(path) != expected:
            raise RuntimeError(f"Teacher source hash mismatch: {rel}")
    return run, metadata, done, checkpoint, checkpoint_hash


def build_teacher(state, device):
    meta = state.get("metadata", {}).get("model", {})
    if not meta:
        raise RuntimeError("Checkpoint lacks model metadata")
    model = GatedTeacher(
        node_dim=int(meta["node_dim"]), edge_dim=int(meta["edge_dim"]),
        hidden_dim=int(meta["hidden_dim"]), out_dim=int(meta["out_dim"]),
        n_classes=int(meta["n_classes"]), layers=int(meta["layers"]),
        dropout=float(meta["dropout"]), batch_norm=bool(meta["batch_norm"]),
        residual=bool(meta["residual"]), readout=str(meta["readout"]),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def reference_activations(model, batch):
    """Capture the six actual eval forward ReLU outputs, preserving teacher order."""
    captured = [None] * (len(model.layers) + 2)
    handles = []
    for i, layer in enumerate(model.layers):
        def hook(_module, _inputs, output, site=i):
            captured[site] = F.relu(output.detach())
        handles.append(layer.bn_node_h.register_forward_hook(hook))
    fc_layers = model.MLP_layer.FC_layers
    for i, linear in enumerate(fc_layers[:-1], start=len(model.layers)):
        def hook(_module, _inputs, output, site=i):
            captured[site] = F.relu(output.detach())
        handles.append(linear.register_forward_hook(hook))
    try:
        model(batch)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("Could not capture all six teacher activation sites")
    for name, activation in zip(SITE_NAMES, captured):
        if activation.ndim != 2 or not bool(torch.isfinite(activation).all()) or bool((activation < 0).any()):
            raise RuntimeError(f"Invalid activation at {name}")
    return captured


@torch.no_grad()
def calibrate_loader(model, loader, expected_graphs=None, bins=512, quantile=0.999, device=None):
    """Two ordered passes: per-channel maxima, then 512-bin histograms."""
    if bins != 512 or quantile != 0.999:
        raise ValueError("Frozen protocol fixes bins=512 and quantile=0.999")
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    maxima, rows, graphs1 = None, [0] * len(SITE_NAMES), 0
    for batch in loader:
        batch = batch.to(dev)
        acts = reference_activations(model, batch)
        if maxima is None:
            maxima = [torch.zeros(a.shape[1], dtype=torch.float64, device=a.device) for a in acts]
        for i, activation in enumerate(acts):
            maxima[i] = torch.maximum(maxima[i], activation.max(dim=0).values.double())
            rows[i] += int(activation.shape[0])
        graphs1 += int(batch.num_graphs)
    if not graphs1 or (expected_graphs is not None and graphs1 != expected_graphs):
        raise RuntimeError(f"First pass graph count mismatch: {graphs1} != {expected_graphs}")

    histograms = [ActivationHistogram(maximum.float(), bins=bins) for maximum in maxima]
    graphs2 = 0
    for batch in loader:
        batch = batch.to(dev)
        for hist, activation in zip(histograms, reference_activations(model, batch)):
            hist.update(activation)
        graphs2 += int(batch.num_graphs)
    if graphs2 != graphs1 or (expected_graphs is not None and graphs2 != expected_graphs):
        raise RuntimeError(f"Second pass graph count mismatch: {graphs2} != {expected_graphs}")

    sites = {}
    for name, hist in zip(SITE_NAMES, histograms):
        per_channel, global_threshold = hist.thresholds(quantile=quantile)
        sites[name] = {
            "rows": hist.rows, "channels": int(hist.maxima.numel()),
            "maxima": hist.maxima.cpu().tolist(),
            "channel_counts": hist.channel.cpu().tolist(),
            "global_counts": hist.global_counts.cpu().tolist(),
            "per_channel_thresholds": per_channel.cpu().tolist(),
            "global_threshold": float(global_threshold.item()),
        }
    return {"passes": 2, "pass1_graphs": graphs1, "pass2_graphs": graphs2,
            "activation_rows_per_site": dict(zip(SITE_NAMES, rows)), "bins": bins,
            "quantile": quantile,
            "threshold_rule": "upper edge of first cumulative bin reaching ceil(N*q)",
            "inactive_channel_floor": 1e-6, "sites": sites,
            "channel_thresholds": {k: v["per_channel_thresholds"] for k, v in sites.items()},
            "global_thresholds": {k: v["global_threshold"] for k, v in sites.items()}}


def run_calibration(teacher_run, output, batch_size=128, device="cpu"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing calibration output: {output}")
    run, metadata, done, checkpoint, checkpoint_hash = validate_teacher_run(teacher_run)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if dev.type == "cpu":
        torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location=dev, weights_only=False)
    if int(state.get("epoch", -1)) < 1 or state.get("metadata") != metadata:
        raise RuntimeError("best.pt metadata/epoch mismatch")
    model = build_teacher(state, dev)
    train = GNNBenchmarkDataset(root=str(DATA_ROOT), name="CIFAR10", split="train")
    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(f"Unexpected official train graph count: {len(train)}")
    processed = DATA_ROOT / "CIFAR10" / "processed"
    data_hashes = processed_file_hashes(processed)
    expected_data_hashes = metadata.get("provenance", {}).get("processed_data_sha256", {})
    if not expected_data_hashes or data_hashes != expected_data_hashes:
        raise RuntimeError("Current processed CIFAR10 data hashes do not match teacher metadata")
    # This loader never carries y or any other target field into calibration.
    loader = DataLoader(LabelFreeGraphView(train), batch_size=batch_size, shuffle=False, num_workers=0)
    calibration = calibrate_loader(model, loader, expected_graphs=EXPECTED_TRAIN, device=dev)
    result = {
        "status": "complete", "dataset": "CIFAR10",
        "scope": "all_45000_official_train_graphs_label_free_two_pass_fixed_order",
        "teacher_run": str(run.resolve()), "teacher_epoch": int(state["epoch"]),
        "teacher_metadata": metadata, "teacher_done": done,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_hash},
        "checkpoint_sha256": checkpoint_hash,
        "data": {"root": str(DATA_ROOT.resolve()), "train_graphs": len(train),
                 "processed_files_sha256": data_hashes},
        "code_sha256": {name: sha256(HERE / name) for name in CALIBRATION_FILES},
        "calibration": {"full_train_graph_count": EXPECTED_TRAIN, **calibration},
        "runtime": {"device": str(dev), "torch": torch.__version__, "python": sys.version,
                    "batch_size": batch_size},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, default=OUT_ROOT / RUN_NAME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    result = run_calibration(args.teacher_run, args.output, args.batch_size, args.device)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "graphs": result["calibration"]["pass2_graphs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
