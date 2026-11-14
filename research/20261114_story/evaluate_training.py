"""Reevaluate frozen calibration parameters on their training graphs only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

# Match the screen's CUDA reproducibility settings before importing Torch.
_CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, _CUBLAS):
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must match run_screen.py (:4096:8)")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS

import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "visual-graphs" / "takeover_20261112"
DEFAULT_ASSETS = Path(
    "C:/code/phd-snn/auto-paper/20261110-152407/"
    "Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/"
    "f-research-v1-run-20261110-172308-superpixel-graph-vision"
)
SITES = ["gcn_0", "gcn_1", "gcn_2", "gcn_3", "mlp_0", "mlp_1"]
PINNED_AUTHOR_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
AUTHOR_REPO = Path("C:/code/gnn2snn-dcgs-reference")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="completed run_screen.py output directory")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON path; existing files are never overwritten")
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261114)
    np.random.seed(20261114)
    if not torch.cuda.is_available():
        raise RuntimeError("training-set reevaluation requires CUDA")
    device = torch.device("cuda")

    run = args.run.resolve()
    metadata_path = run / "metadata.json"
    done_path = run / "DONE.json"
    if not metadata_path.is_file() or not done_path.is_file():
        raise FileNotFoundError("run must contain metadata.json and DONE.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete" or done.get("test_loaded") is not False:
        raise RuntimeError("run is incomplete or test access is not explicitly false")
    dataset_name = metadata["dataset"]
    seed = int(metadata["seed"])
    if dataset_name not in ("MNIST", "CIFAR10"):
        raise ValueError("unsupported dataset in run metadata")

    sys.path.insert(0, str(LEGACY))
    from full_teacher_gcn import BenchmarkGCN
    from fold_teacher_bn import fold_model
    from calibration_core import CalibratedGCN, teacher_sites
    import dcgs_fixed_seed_eval_v2 as dcgs_reference
    sys.path.insert(0, str(HERE))
    from run_screen import resolve_calibration

    _, source_commit, source_provenance = dcgs_reference.require_author_source(AUTHOR_REPO)
    if source_commit != PINNED_AUTHOR_COMMIT or source_commit != metadata.get("source_commit"):
        raise RuntimeError("MTH_diff author source differs from the saved screen run")

    assets = args.assets.resolve()
    teacher_run = assets / "takeover_20261112" / "full_teacher_runs" / f"{dataset_name}_seed{seed}_full"
    checkpoint = teacher_run / "best.pt"
    calibration_path = resolve_calibration(assets, dataset_name, seed)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("dataset") != dataset_name:
        raise RuntimeError("resolved calibration dataset differs from saved run metadata")
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise RuntimeError("teacher checkpoint hash differs from saved run metadata")
    if sha256(calibration_path) != metadata["calibration_sha256"]:
        raise RuntimeError("calibration hash differs from saved run metadata")
    if sha256(HERE / "calibration_core.py") != metadata["core_sha256"]:
        raise RuntimeError("calibration_core.py differs from the code used by this run")
    if metadata.get("source_commit") != PINNED_AUTHOR_COMMIT:
        raise RuntimeError("run metadata does not name the pinned MTH_diff source")
    for filename, digest in calibration["code_sha256"].items():
        if sha256(LEGACY / filename) != digest:
            raise RuntimeError("calibration dependency hash mismatch: " + filename)

    data_root = assets / "data" / "pyg_bench"
    for filename, digest in calibration["data"]["processed_files_sha256"].items():
        if sha256(data_root / dataset_name / "processed" / filename) != digest:
            raise RuntimeError("processed train data hash mismatch: " + filename)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != int(metadata["checkpoint_epoch"]):
        raise RuntimeError("teacher checkpoint epoch differs from saved run metadata")
    teacher = BenchmarkGCN(int(json.loads((teacher_run / "metadata.json").read_text())["input_dim"]))
    teacher.load_state_dict(state["model"])
    teacher.to(device).eval().requires_grad_(False)
    folded = fold_model(teacher).eval()
    thresholds = [calibration["channel_thresholds"][site] for site in SITES]
    net = CalibratedGCN(folded, thresholds).to(device).eval()

    train = GNNBenchmarkDataset(root=str(data_root), name=dataset_name, split="train")
    expected = {"MNIST": 55000, "CIFAR10": 45000}[dataset_name]
    if len(train) != expected:
        raise RuntimeError("official training split count mismatch")
    indices = metadata.get("train_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("metadata must contain its exact nonempty train_indices")
    indices = np.asarray(indices, dtype=np.int64)
    if (indices.min() < 0 or indices.max() >= len(train)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("saved train indices are duplicated or out of range")
    batch_size = int(metadata.get("batch_size", 32))
    if batch_size != 32:
        raise RuntimeError("this evaluator is frozen to the screen's batch size 32")
    batches = []
    for batch in DataLoader(train[indices.tolist()], batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        if hasattr(batch, "y"):
            del batch.y
        with torch.no_grad():
            teacher_logits, target_sites = teacher_sites(folded, batch)
        batches.append((batch, teacher_logits.detach(), [x.detach() for x in target_sites]))
    expected_batches = (len(indices) + batch_size - 1) // batch_size
    if len(batches) != expected_batches:
        raise RuntimeError("training batch count does not match saved index list")

    parameter_files = sorted(run.glob("*_T*_parameters.npz"))
    saved = []
    for path in parameter_files:
        match = re.fullmatch(r"(.+)_T([1248])_parameters\.npz", path.name)
        if match is None:
            continue
        method, timestep = match.group(1), int(match.group(2))
        if method == "baseline":
            raise ValueError("baseline must be reconstructed from zero offsets")
        with np.load(path, allow_pickle=False) as archive:
            expected_keys = {f"offset_{i}" for i in range(6)}
            if not expected_keys.issubset(archive.files):
                raise ValueError(f"missing offset arrays in {path.name}")
            offsets = [np.asarray(archive[f"offset_{i}"]) for i in range(6)]
            output_bias = np.asarray(archive["output_bias"]) if "output_bias" in archive else None
        saved.append((method, timestep, path, offsets, output_bias))
    if not saved:
        raise FileNotFoundError("run contains no saved method_T*_parameters.npz files")

    def evaluate(method, timestep, offsets, output_bias=None):
        if len(offsets) != 6:
            raise ValueError("six offset vectors are required")
        with torch.no_grad():
            for param, array in zip(net.offsets, offsets):
                tensor = torch.as_tensor(array, dtype=param.dtype, device=device)
                if tensor.shape != param.shape or not torch.isfinite(tensor).all():
                    raise ValueError(f"invalid {method} T{timestep} offset shape or values")
                param.copy_(tensor)
            bias = None
            if output_bias is not None:
                bias = torch.as_tensor(output_bias, dtype=torch.float32, device=device)
                if bias.shape != (teacher.mlp[4].out_features,) or not torch.isfinite(bias).all():
                    raise ValueError("invalid saved output bias")
            graph_logit_square_sum = 0.0
            graph_logit_element_count = 0
            graph_norm_batches = []
            node_norm_batches = []
            node_site_norm_batches = [[] for _ in range(6)]
            agree = 0
            graph_count = 0
            for batch, target_logits, target_sites in batches:
                logits, sites = net(batch, timestep)
                if bias is not None:
                    logits = logits + bias
                diff = logits - target_logits
                graph_logit_square_sum += float(diff.square().sum())
                graph_logit_element_count += diff.numel()
                graph_norm_batches.append(float(
                    diff.square().mean() / target_logits.square().mean().clamp_min(1e-6)))
                per_site = []
                for i, (candidate, target) in enumerate(zip(sites, target_sites)):
                    ratio = (candidate - target).square().mean() / target.square().mean().clamp_min(1e-6)
                    value = float(ratio)
                    per_site.append(value)
                    node_site_norm_batches[i].append(value)
                node_norm_batches.append(float(np.mean(per_site)))
                agree += int((logits.argmax(1) == target_logits.argmax(1)).sum())
                graph_count += int(logits.shape[0])
        return {
            "method": method,
            "T": timestep,
            "graphs": graph_count,
            "batches": len(batches),
            "graph_logit_mse_unnormalized": graph_logit_square_sum / graph_logit_element_count,
            "graph_logit_mse_normalized_batch_mean": float(np.mean(graph_norm_batches)),
            "node_mse_normalized_equal_site_batch_mean": float(np.mean(node_norm_batches)),
            "node_mse_normalized_by_site_batch_mean": [float(np.mean(site)) for site in node_site_norm_batches],
            "teacher_agreement_percent": 100.0 * agree / graph_count,
            "output_bias_applied": bias is not None,
        }

    results = []
    timesteps = sorted(set(int(t) for t in metadata.get("T", [])))
    if not timesteps:
        raise ValueError("metadata must contain timesteps")
    for timestep in timesteps:
        results.append(evaluate("baseline", timestep, [np.zeros_like(p.detach().cpu().numpy())
                                                          for p in net.offsets]))
    for method, timestep, path, offsets, output_bias in saved:
        results.append(evaluate(method, timestep, offsets, output_bias))

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    payload = {
        "scope": "label-free saved-training-indices reevaluation; no validation or test split loaded",
        "dataset": dataset_name,
        "teacher_seed": seed,
        "teacher_checkpoint_sha256": metadata["checkpoint_sha256"],
        "calibration_sha256": metadata["calibration_sha256"],
        "core_sha256": metadata["core_sha256"],
        "author_source_commit": metadata["source_commit"],
        "author_source_provenance": source_provenance,
        "train_indices_count": int(len(indices)),
        "train_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "test_loaded": False,
        "parameter_files": [{"file": path.name, "sha256": sha256(path)}
                            for _, _, path, _, _ in saved],
        "results": results,
    }
    with out.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
