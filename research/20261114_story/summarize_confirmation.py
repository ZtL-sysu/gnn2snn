"""Strictly aggregate the six-seed CUDA calibration confirmation runs."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

DATASETS = ("MNIST", "CIFAR10")
SEEDS = (41, 42, 43)
TIMESTEPS = (1, 2, 4, 8)
METHODS = ("baseline", "node", "graph", "mix", "output", "shuffled")
METRICS = ("accuracy_percent", "teacher_agreement_percent", "logit_mse",
           "teacher_accuracy_percent")
PAIRS = (("graph-minus-node", "graph", "node"),
         ("graph-minus-output", "graph", "output"),
         ("mix-minus-node", "mix", "node"))
T95_DF2 = 4.30265273
REQUIRED_SNAPSHOTS = {
    "run_screen.py", "calibration_core.py", "full_teacher_gcn.py",
    "fold_teacher_bn.py", "dcgs_fixed_seed_eval_v2.py",
}
RUNNER_VERSION_NOTE = (
    "run_screen.py snapshot hashes may differ only for the recorded calibration asset-resolver/metadata-path repair; "
    "each archived runner must match its own metadata script_sha256, while core and mathematical dependency snapshots "
    "must be identical across runs."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual, expected, what, atol=1e-5):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError(f"non-finite {what}")
    if abs(float(actual) - float(expected)) > atol:
        raise ValueError(f"{what} disagrees with saved result: {actual} vs {expected}")


def read_run(run_dir):
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metadata.json"
    done_path = run_dir / "DONE.json"
    results_path = run_dir / "results.json"
    for path in (metadata_path, done_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete run directory: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    dataset = metadata.get("dataset")
    seed = metadata.get("seed")
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError(f"unexpected dataset/seed in {run_dir}")
    if "smoke" in str(run_dir).lower() or metadata.get("smoke"):
        raise ValueError(f"smoke run is not eligible: {run_dir}")
    if done.get("status") != "complete" or done.get("test_loaded") is not False:
        raise ValueError(f"run is incomplete or test access is not false: {run_dir}")
    if metadata.get("test_loaded") is not False:
        raise ValueError(f"metadata does not certify test_loaded=false: {run_dir}")
    if (metadata.get("val_graphs_official") != 5000
            or metadata.get("val_graphs") not in (None, 5000)):
        raise ValueError(f"unexpected validation split scope in {run_dir}")
    if metadata.get("T") != list(TIMESTEPS):
        raise ValueError(f"run does not contain exact T1/2/4/8 grid: {run_dir}")
    if not metadata.get("deterministic_algorithms") or metadata.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError(f"run is missing frozen deterministic CUDA configuration: {run_dir}")
    if metadata.get("source_commit") != "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be":
        raise ValueError(f"unexpected author source commit in {run_dir}")

    snapshot_hashes = metadata.get("source_snapshots")
    if not isinstance(snapshot_hashes, dict) or set(snapshot_hashes) != REQUIRED_SNAPSHOTS:
        raise ValueError(f"missing or unexpected source snapshots in {run_dir}")
    snapshot_dir = run_dir / "source"
    for name, expected_hash in snapshot_hashes.items():
        snapshot = snapshot_dir / name
        if not snapshot.is_file() or sha256(snapshot) != expected_hash:
            raise ValueError(f"source snapshot hash mismatch: {snapshot}")
    if metadata.get("script_sha256") != snapshot_hashes["run_screen.py"]:
        raise ValueError(f"archived run_screen.py hash differs from metadata script_sha256: {run_dir}")

    result_by_key = {}
    for row in results:
        key = (row.get("method"), row.get("T"))
        if key in result_by_key:
            raise ValueError(f"duplicate result row {key} in {run_dir}")
        result_by_key[key] = row
    expected_keys = set(itertools.product(METHODS, TIMESTEPS))
    if set(result_by_key) != expected_keys or done.get("rows") != len(expected_keys):
        raise ValueError(f"run does not have exactly 24 method/T results: {run_dir}")

    indices_meta = np.asarray(metadata.get("val_indices", []), dtype=np.int64)
    if indices_meta.shape != (5000,) or not np.array_equal(np.sort(indices_meta), np.arange(5000)):
        raise ValueError(f"metadata does not contain a complete ordered validation index set: {run_dir}")
    train_indices = np.asarray(metadata.get("train_indices", []), dtype=np.int64)
    if train_indices.ndim != 1 or train_indices.size == 0 or len(np.unique(train_indices)) != train_indices.size:
        raise ValueError(f"invalid train index manifest in {run_dir}")

    archives = {}
    reference = None
    for method, timestep in sorted(expected_keys, key=lambda x: (x[1], METHODS.index(x[0]))):
        archive_path = run_dir / f"{method}_T{timestep}.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(f"missing prediction archive {archive_path}")
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {"indices", "labels", "logits", "teacher"}
            if not required.issubset(archive.files):
                raise ValueError(f"missing arrays in {archive_path}")
            arrays = {name: np.asarray(archive[name]) for name in required}
        indices = arrays["indices"].astype(np.int64, copy=False)
        labels = arrays["labels"].reshape(-1).astype(np.int64, copy=False)
        logits = arrays["logits"]
        teacher = arrays["teacher"]
        if (indices.shape != (5000,) or not np.array_equal(indices, indices_meta)
                or len(np.unique(indices)) != 5000):
            raise ValueError(f"archive does not cover exact validation indices: {archive_path}")
        if labels.shape != (5000,) or logits.shape != (5000, 10) or teacher.shape != (5000, 10):
            raise ValueError(f"unexpected saved array shape in {archive_path}")
        if not all(np.isfinite(x).all() for x in (logits, teacher)):
            raise ValueError(f"non-finite logits in {archive_path}")
        if reference is None:
            reference = {"indices": indices.copy(), "labels": labels.copy(), "teacher": teacher.copy()}
        else:
            for name in ("indices", "labels", "teacher"):
                if not np.array_equal(arrays[name].reshape(-1) if name == "labels" else arrays[name],
                                      reference[name]):
                    raise ValueError(f"method/T rows do not share identical {name}: {archive_path}")
        pred = logits.argmax(axis=1)
        teacher_pred = teacher.argmax(axis=1)
        recomputed = {
            "accuracy_percent": 100.0 * np.mean(pred == labels),
            "teacher_agreement_percent": 100.0 * np.mean(pred == teacher_pred),
            "logit_mse": float(np.mean((logits.astype(np.float64) - teacher.astype(np.float64)) ** 2)),
            "teacher_accuracy_percent": 100.0 * np.mean(teacher_pred == labels),
        }
        saved = result_by_key[(method, timestep)]
        if saved.get("val_graphs") != 5000:
            raise ValueError(f"saved result does not report full validation: {archive_path}")
        for metric in recomputed:
            assert_close(recomputed[metric], saved[metric], f"{archive_path.name}:{metric}")
        archives[(method, timestep)] = {**arrays, **recomputed}

    # Ensure every trainable candidate has its exact saved parameter and optimizer record.
    costs = []
    parameter_files = sorted(run_dir.glob("*_T*_parameters.npz"))
    expected_parameter_names = {f"{m}_T{t}_parameters.npz" for m in METHODS if m != "baseline" for t in TIMESTEPS}
    if {p.name for p in parameter_files} != expected_parameter_names:
        raise ValueError(f"parameter artifact set mismatch in {run_dir}")
    for method in METHODS:
        if method == "baseline":
            continue
        for timestep in TIMESTEPS:
            training_path = run_dir / f"{method}_T{timestep}_training.json"
            parameter_path = run_dir / f"{method}_T{timestep}_parameters.npz"
            if not training_path.is_file():
                raise FileNotFoundError(training_path)
            training = json.loads(training_path.read_text(encoding="utf-8"))
            steps = int(training.get("steps", -1))
            params = int(training.get("trainable_parameters", -1))
            expected_params = 10 if method == "output" else 693
            trace = training.get("loss", [])
            if (steps != int(metadata.get("steps", -2)) or params != expected_params
                    or len(trace) != steps or not np.isfinite(np.asarray(trace, dtype=float)).all()):
                raise ValueError(f"training budget record mismatch: {training_path}")
            costs.append({
                "dataset": dataset, "seed": seed, "method": method, "T": timestep,
                "optimizer_steps": steps, "trainable_parameters": params,
                "scalar_update_budget": steps * params,
                "calibration_seconds": training.get("calibration_seconds"),
                "peak_allocated_bytes": training.get("peak_allocated_bytes"),
                "parameter_file_sha256": sha256(parameter_path),
                "runner_script_sha256": metadata["script_sha256"],
                "runner_version_note": RUNNER_VERSION_NOTE,
            })

    config = {
        key: metadata.get(key) for key in (
            "T", "steps", "batch_size", "optimizer", "lr", "offset_bound",
            "deterministic_algorithms", "CUBLAS_WORKSPACE_CONFIG", "gpu",
            "val_indices", "dependency_sha256", "core_sha256",
        )
    }
    return {
        "path": run_dir, "dataset": dataset, "seed": int(seed), "metadata": metadata,
        "config": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "train_indices_digest": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "snapshots": snapshot_hashes,
        "common_snapshots": {name: digest for name, digest in snapshot_hashes.items()
                              if name != "run_screen.py"},
        "runner_version": metadata["script_sha256"],
        "reference": reference,
        "archives": archives,
        "results": result_by_key,
        "costs": costs,
    }


def metric_triplet(logits, teacher, labels):
    prediction = logits.argmax(axis=1)
    teacher_prediction = teacher.argmax(axis=1)
    return {
        "accuracy_percent": float(100.0 * np.mean(prediction == labels)),
        "teacher_agreement_percent": float(100.0 * np.mean(prediction == teacher_prediction)),
        "logit_mse": float(np.mean((logits.astype(np.float64) - teacher.astype(np.float64)) ** 2)),
        "teacher_accuracy_percent": float(100.0 * np.mean(teacher_prediction == labels)),
    }


def sample_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("aggregate requires exactly three finite paired seed values")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = T95_DF2 * sd / math.sqrt(3.0)
    return mean, sd, mean - half, mean + half


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True,
                        help="root containing exactly six complete run directories")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for the three CSV summaries")
    args = parser.parse_args()
    root = args.runs.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")

    metadata_files = sorted(root.rglob("metadata.json"))
    if len(metadata_files) != 6:
        raise ValueError(f"expected exactly six run metadata files under {root}, found {len(metadata_files)}")
    runs = [read_run(path.parent) for path in metadata_files]
    identity = [(run["dataset"], run["seed"]) for run in runs]
    if len(set(identity)) != 6:
        raise ValueError("duplicate dataset/seed runs found")
    expected_identity = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if set(identity) != expected_identity:
        raise ValueError(f"expected seed 41/42/43 for each dataset; found {identity}")
    if len({run["config"] for run in runs}) != 1:
        raise ValueError("mixed protocol/configuration fields across the six runs")
    if len({json.dumps(run["common_snapshots"], sort_keys=True) for run in runs}) != 1:
        raise ValueError("core/model/mathematical dependency snapshots differ across confirmation runs")
    runner_versions = sorted({run["runner_version"] for run in runs})
    for dataset in DATASETS:
        dataset_runs = [run for run in runs if run["dataset"] == dataset]
        if len({run["train_indices_digest"] for run in dataset_runs}) != 1:
            raise ValueError(f"training subset differs across teacher seeds for {dataset}")
        first_ref = dataset_runs[0]["reference"]
        for run in dataset_runs[1:]:
            if (not np.array_equal(run["reference"]["indices"], first_ref["indices"])
                    or not np.array_equal(run["reference"]["labels"], first_ref["labels"])):
                raise ValueError(f"validation ordering/labels differ across teacher seeds for {dataset}")

    by_key = {(run["dataset"], run["seed"]): run for run in runs}
    excluded = np.random.default_rng(20261115).permutation(5000)[:512]
    excluded_set = set(int(i) for i in excluded)

    per_seed = []
    aggregate = []
    paired_field_names = []
    for name, _, _ in PAIRS:
        paired_field_names.extend((f"{name}_accuracy_pp", f"{name}_agreement_pp",
                                   f"{name}_logit_mse"))
    for dataset in DATASETS:
        for seed in SEEDS:
            run = by_key[(dataset, seed)]
            included_mask = np.asarray([int(i) not in excluded_set for i in run["reference"]["indices"]])
            if int(included_mask.sum()) != 4488:
                raise AssertionError("screen-excluded subset must contain 4488 examples")
            for scope, mask in (("full_validation", np.ones(5000, dtype=bool)),
                                ("screen_excluded", included_mask)):
                n_examples = int(mask.sum())
                for timestep in TIMESTEPS:
                    values = {}
                    for method in METHODS:
                        arrays = run["archives"][(method, timestep)]
                        values[method] = metric_triplet(
                            arrays["logits"][mask], arrays["teacher"][mask],
                            arrays["labels"].reshape(-1)[mask])
                    deltas = {}
                    for pair_name, left, right in PAIRS:
                        deltas[pair_name] = {
                            "accuracy_pp": values[left]["accuracy_percent"] - values[right]["accuracy_percent"],
                            "agreement_pp": values[left]["teacher_agreement_percent"] - values[right]["teacher_agreement_percent"],
                            "logit_mse": values[left]["logit_mse"] - values[right]["logit_mse"],
                        }
                    for method in METHODS:
                        row = {"dataset": dataset, "seed": seed, "scope": scope,
                               "method": method, "T": timestep, "n_examples": n_examples,
                               "runner_script_sha256": run["runner_version"],
                               "runner_version_note": RUNNER_VERSION_NOTE,
                               **values[method],
                               "qualification": "historically exposed validation; descriptive only, not untouched-test evidence"}
                        # Attach each paired difference once to the left-hand method row.
                        for pair_name, left, _ in PAIRS:
                            for suffix in ("accuracy_pp", "agreement_pp", "logit_mse"):
                                col = f"{pair_name}_{suffix}"
                                row[col] = deltas[pair_name][suffix] if method == left else ""
                        per_seed.append(row)
                    for pair_name, left, right in PAIRS:
                        diffrow = {"dataset": dataset, "seed": seed, "scope": scope,
                                   "method": f"paired_{pair_name}", "T": timestep,
                                   "n_examples": n_examples,
                                   "runner_script_sha256": run["runner_version"],
                                   "runner_version_note": RUNNER_VERSION_NOTE,
                                   "accuracy_percent": deltas[pair_name]["accuracy_pp"],
                                   "teacher_agreement_percent": deltas[pair_name]["agreement_pp"],
                                   "logit_mse": deltas[pair_name]["logit_mse"],
                                   "teacher_accuracy_percent": 0.0,
                                   "qualification": f"paired {left}-{right} difference within seed; descriptive only"}
                        diffrow.update({col: "" for col in paired_field_names})
                        per_seed.append(diffrow)

    # Per-seed method rows and paired-difference rows use the same plotting keys.
    aggregate_methods = list(METHODS) + [f"paired_{name}" for name, _, _ in PAIRS]
    for dataset in DATASETS:
        for scope in ("full_validation", "screen_excluded"):
            for timestep in TIMESTEPS:
                for method in aggregate_methods:
                    seed_rows = [r for r in per_seed if r["dataset"] == dataset and r["scope"] == scope
                                 and r["T"] == timestep and r["method"] == method]
                    row = {"dataset": dataset, "scope": scope, "method": method, "T": timestep,
                           "n_seeds": len(seed_rows),
                           "n_examples": seed_rows[0]["n_examples"],
                           "runner_versions": json.dumps(runner_versions),
                           "runner_version_note": RUNNER_VERSION_NOTE,
                           "qualification": "mean and sample SD across three teacher seeds; t95 df=2 assumes independent seed-level values; shared validation examples, descriptive only"}
                    for metric in METRICS:
                        mean, sd, low, high = sample_summary([r[metric] for r in seed_rows])
                        row[f"{metric}_mean"] = mean
                        row[f"{metric}_sample_sd"] = sd
                        row[f"{metric}_t95_low"] = low
                        row[f"{metric}_t95_high"] = high
                    aggregate.append(row)

    costs = [cost for run in runs for cost in run["costs"]]
    if len(costs) != 6 * 5 * 4:
        raise ValueError("expected 120 training/calibration cost rows")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "aggregate.csv", aggregate)
    write_csv(output / "cost_calibration.csv", costs)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    fields = list(rows[0].keys())
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
