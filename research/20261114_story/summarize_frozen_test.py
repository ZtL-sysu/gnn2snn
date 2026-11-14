"""Recompute and aggregate frozen, historically exposed full-test endpoints."""
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
PAIRS = (("graph-minus-node", "graph", "node"),
         ("graph-minus-output", "graph", "output"),
         ("mix-minus-node", "mix", "node"))
METRICS = ("accuracy_percent", "teacher_agreement_percent", "logit_mse",
           "cross_entropy", "teacher_accuracy_percent")
T95_DF2 = 4.30265273
REQUIRED_SOURCE_FILES = {
    "run_screen.py", "calibration_core.py", "full_teacher_gcn.py",
    "fold_teacher_bn.py", "dcgs_fixed_seed_eval_v2.py", "evaluate_frozen_test.py",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value, what):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"invalid SHA-256 value for {what}")


def stable_cross_entropy(logits, labels):
    z = logits.astype(np.float64)
    zmax = z.max(axis=1, keepdims=True)
    lse = zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(labels)), labels]))


def close(actual, expected, what, atol=2e-5):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError(f"non-finite {what}")
    if abs(float(actual) - float(expected)) > atol:
        raise ValueError(f"recomputed {what} differs from evaluator result: {actual} vs {expected}")


def recompute(logits, teacher, labels):
    pred = logits.argmax(axis=1)
    teacher_pred = teacher.argmax(axis=1)
    return {
        "accuracy_percent": float(100.0 * np.mean(pred == labels)),
        "teacher_agreement_percent": float(100.0 * np.mean(pred == teacher_pred)),
        "logit_mse": float(np.mean((logits.astype(np.float64) - teacher.astype(np.float64)) ** 2)),
        "cross_entropy": stable_cross_entropy(logits, labels),
        "teacher_accuracy_percent": float(100.0 * np.mean(teacher_pred == labels)),
    }


def sample_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("aggregate requires exactly three finite teacher-seed values")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = T95_DF2 * sd / math.sqrt(3.0)
    return mean, sd, mean - half, mean + half


def read_run(run_dir):
    run_dir = run_dir.resolve()
    for name in ("metadata.json", "results.json", "DONE.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"incomplete frozen-test run: {run_dir / name}")
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    result_rows = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    done = json.loads((run_dir / "DONE.json").read_text(encoding="utf-8"))
    dataset, seed = meta.get("dataset"), int(meta.get("seed", -1))
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError(f"unexpected dataset/seed in {run_dir}")
    if "smoke" in str(run_dir).lower() or meta.get("smoke"):
        raise ValueError(f"smoke run is ineligible: {run_dir}")
    expected_keys = set(itertools.product(METHODS, TIMESTEPS))
    actual_keys = [(row.get("method"), int(row.get("T", -1))) for row in result_rows]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected_keys:
        raise ValueError(f"expected exactly 24 unique method/T conditions in {run_dir}")
    if (meta.get("status") != "complete" or done.get("status") != "complete"
            or done.get("test_loaded") is not True or done.get("training_or_optimization") is not False
            or meta.get("test_loaded") is not True or meta.get("training_or_optimization") is not False
            or done.get("rows") != 24 or meta.get("graphs") != 10000):
        raise ValueError(f"run is not a complete frozen 10k test evaluation: {run_dir}")
    if tuple(meta.get("T", [])) != TIMESTEPS or tuple(meta.get("methods", [])) != METHODS:
        raise ValueError(f"mixed or incomplete condition grid in {run_dir}")

    # Require the evaluator's recorded teacher/reference guard to have passed.
    if meta.get("teacher_original_prediction_differences_vs_archive") != 0:
        raise ValueError(f"archived teacher prediction reference guard failed in {run_dir}")
    if meta.get("teacher_fold_prediction_differences") != 0:
        raise ValueError(f"folded teacher prediction reference guard failed in {run_dir}")
    for field in ("teacher_original_max_abs_difference_vs_archive", "teacher_fold_max_abs_difference"):
        if not math.isfinite(float(meta.get(field, float("nan")))):
            raise ValueError(f"missing/nonfinite teacher reference guard {field} in {run_dir}")

    # Sidecar integrity: test evaluator verifies hashes before loading the test split.
    source_hashes = meta.get("source_snapshots")
    if not isinstance(source_hashes, dict):
        raise ValueError(f"missing source snapshot manifest in {run_dir}")
    for field in ("checkpoint_sha256", "calibration_sha256", "teacher_reference_sha256",
                  "confirmation_metadata_sha256", "confirmation_DONE_sha256",
                  "confirmation_results_sha256", "core_sha256"):
        require_sha(meta.get(field), f"{field} in {run_dir}")
    param_hashes = meta.get("parameter_sha256")
    expected_parameter_keys = {f"{method}_T{t}" for method in METHODS if method != "baseline" for t in TIMESTEPS}
    if not isinstance(param_hashes, dict) or set(param_hashes) != expected_parameter_keys:
        raise ValueError(f"incomplete frozen-parameter hash manifest in {run_dir}")
    for condition, digest in param_hashes.items():
        require_sha(digest, f"parameter hash {condition}")
        parameter_path = run_dir / f"{condition}_parameters.npz"
        if not parameter_path.is_file() or sha256(parameter_path) != digest:
            raise ValueError(f"frozen parameter archive hash mismatch: {parameter_path}")

    # The frozen evaluator preserves the exact validation inputs and fitted
    # parameter archives so the test summary can independently bind its inputs.
    input_dir = run_dir / "input_confirmation"
    input_paths = {"metadata": input_dir / "metadata.json",
                   "DONE": input_dir / "DONE.json",
                   "results": input_dir / "results.json"}
    sidecar_fields = {"metadata": "confirmation_metadata_sha256",
                      "DONE": "confirmation_DONE_sha256",
                      "results": "confirmation_results_sha256"}
    for key, path in input_paths.items():
        if not path.is_file() or sha256(path) != meta[sidecar_fields[key]]:
            raise ValueError(f"copied confirmation {key} hash mismatch: {path}")
    confirmation_meta = json.loads(input_paths["metadata"].read_text(encoding="utf-8"))
    confirmation_done = json.loads(input_paths["DONE"].read_text(encoding="utf-8"))
    confirmation_results = json.loads(input_paths["results"].read_text(encoding="utf-8"))
    if (confirmation_done.get("status") != "complete"
            or confirmation_done.get("test_loaded") is not False
            or confirmation_meta.get("test_loaded") is not False):
        raise ValueError(f"copied confirmation inputs are not a completed validation-only run: {run_dir}")
    if (confirmation_meta.get("dataset") != dataset
            or int(confirmation_meta.get("seed", -1)) != seed
            or confirmation_meta.get("checkpoint_sha256") != meta.get("checkpoint_sha256")
            or confirmation_meta.get("calibration_sha256") != meta.get("calibration_sha256")
            or confirmation_meta.get("core_sha256") != meta.get("core_sha256")
            or confirmation_meta.get("dependency_sha256") != meta.get("dependency_sha256")
            or confirmation_done.get("rows") != 24
            or len(confirmation_results) != 24):
        raise ValueError(f"copied confirmation identity/condition count mismatch: {run_dir}")
    confirmation_sources = confirmation_meta.get("source_snapshots", {})
    for name, digest in confirmation_sources.items():
        if source_hashes.get(name) != digest:
            raise ValueError(f"test output source differs from copied confirmation source {name}: {run_dir}")
    confirmation_keys = {(row.get("method"), int(row.get("T", -1))) for row in confirmation_results}
    if confirmation_keys != set(itertools.product(METHODS, TIMESTEPS)):
        raise ValueError(f"copied confirmation condition grid mismatch: {run_dir}")
    expected_calibration_config = {"steps": 64, "optimizer": "Adam", "lr": 0.001,
                                   "batch_size": 32, "offset_bound": 0.5}
    calibration_config = meta.get("calibration_config")
    copied_config = {key: confirmation_meta.get(key) for key in expected_calibration_config}
    if calibration_config != copied_config or copied_config != expected_calibration_config:
        raise ValueError(f"test run calibration config differs from frozen validation metadata: {run_dir}")
    train_indices = np.asarray(confirmation_meta.get("train_indices", []), dtype=np.int64)
    train_size = 55000 if dataset == "MNIST" else 45000
    if (train_indices.shape != (512,) or len(np.unique(train_indices)) != 512
            or train_indices.min() < 0 or train_indices.max() >= train_size):
        raise ValueError(f"copied confirmation training indices are invalid: {run_dir}")
    copied_param_dir = run_dir
    for condition, digest in param_hashes.items():
        copied_path = copied_param_dir / f"{condition}_parameters.npz"
        if not copied_path.is_file() or sha256(copied_path) != digest:
            raise ValueError(f"copied frozen parameter input hash mismatch: {copied_path}")

    source_hashes = meta.get("source_snapshots")
    if not isinstance(source_hashes, dict) or set(source_hashes) != REQUIRED_SOURCE_FILES:
        raise ValueError(f"incomplete frozen source snapshot manifest in {run_dir}")
    source_dir = run_dir / "source"
    for name, digest in source_hashes.items():
        require_sha(digest, f"source snapshot {name}")
        path = source_dir / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"frozen source snapshot mismatch: {path}")
    if source_hashes["evaluate_frozen_test.py"] == "":
        raise ValueError("missing evaluator version hash")
    if source_hashes["calibration_core.py"] != meta.get("core_sha256"):
        raise ValueError(f"core hash disagrees with snapshot in {run_dir}")
    deps = meta.get("dependency_sha256")
    if not isinstance(deps, dict) or not deps:
        raise ValueError(f"missing mathematical dependency hashes in {run_dir}")
    for name, digest in deps.items():
        if name not in source_hashes or source_hashes[name] != digest:
            raise ValueError(f"dependency snapshot hash mismatch: {name} in {run_dir}")

    result_by_key = {(row["method"], int(row["T"])): row for row in result_rows}
    archive_data = {}
    common = None
    labels_reference = None
    for method, timestep in sorted(expected_keys, key=lambda k: (k[1], METHODS.index(k[0]))):
        path = run_dir / f"{method}_T{timestep}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            required = {"indices", "labels", "teacher_logits", "converted_logits"}
            if not required.issubset(z.files):
                raise ValueError(f"saved test archive missing arrays: {path}")
            indices = np.asarray(z["indices"])
            labels = np.asarray(z["labels"]).reshape(-1)
            teacher = np.asarray(z["teacher_logits"])
            converted = np.asarray(z["converted_logits"])
        if (indices.shape != (10000,) or not np.array_equal(indices, np.arange(10000))
                or labels.shape != (10000,) or teacher.shape != (10000, 10)
                or converted.shape != (10000, 10)):
            raise ValueError(f"test archive does not contain all ordered 10k examples: {path}")
        if not np.isfinite(teacher).all() or not np.isfinite(converted).all():
            raise ValueError(f"nonfinite logits in {path}")
        if common is None:
            common = {"indices": indices.copy(), "labels": labels.copy(), "teacher": teacher.copy()}
            labels_reference = labels.copy()
        else:
            if not np.array_equal(indices, common["indices"]):
                raise ValueError(f"method/T indices differ in {path}")
            if not np.array_equal(labels, common["labels"]):
                raise ValueError(f"method/T labels differ in {path}")
            if not np.array_equal(teacher, common["teacher"]):
                raise ValueError(f"method/T teacher logits differ in {path}")
        metrics = recompute(converted, teacher, labels.astype(np.int64, copy=False))
        saved = result_by_key[(method, timestep)]
        if int(saved.get("graphs", -1)) != 10000:
            raise ValueError(f"result row is not full 10k: {path}")
        for metric, value in metrics.items():
            close(value, saved.get(metric), f"{path.name}:{metric}")
        archive_data[(method, timestep)] = {
            "indices": indices, "labels": labels.astype(np.int64, copy=False),
            "teacher": teacher, "converted": converted, **metrics,
        }
    return {
        "path": run_dir, "dataset": dataset, "seed": seed, "meta": meta,
        "results": result_by_key, "archives": archive_data,
        "reference": common, "source_hashes": source_hashes,
        "common_source_hashes": {k: v for k, v in source_hashes.items() if k != "run_screen.py"},
        "train_indices": train_indices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True,
                        help="root containing exactly six completed evaluator output directories")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for per_seed.csv and aggregate.csv")
    args = parser.parse_args()
    root = args.runs.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    metadata_files = sorted(path for path in root.rglob("metadata.json")
                            if (path.parent / "baseline_T1.npz").is_file()
                            and (path.parent / "DONE.json").is_file())
    if len(metadata_files) != 6:
        raise ValueError(f"expected six completed frozen-test run directories, found {len(metadata_files)}")
    runs = [read_run(path.parent) for path in metadata_files]
    identities = [(r["dataset"], r["seed"]) for r in runs]
    expected_ids = {(d, s) for d in DATASETS for s in SEEDS}
    if len(set(identities)) != 6 or set(identities) != expected_ids:
        raise ValueError(f"expected exactly datasets×seeds 41/42/43, found {identities}")
    configurations = []
    for run in runs:
        m = run["meta"]
        configs = {key: m.get(key) for key in ("split", "graphs", "batch_size", "T", "methods",
                    "deterministic_algorithms", "CUBLAS_WORKSPACE_CONFIG", "core_sha256", "dependency_sha256",
                    "device", "torch", "training_or_optimization")}
        configurations.append(json.dumps(configs, sort_keys=True, separators=(",", ":")))
    if len(set(configurations)) != 1:
        raise ValueError("mixed frozen-test evaluation configuration across six runs")
    if len({json.dumps(r["common_source_hashes"], sort_keys=True) for r in runs}) != 1:
        raise ValueError("mathematical/core source snapshots differ across runs")
    for dataset in DATASETS:
        ds_runs = [r for r in runs if r["dataset"] == dataset]
        data_hashes = {json.dumps(r["meta"].get("processed_data_sha256", {}), sort_keys=True) for r in ds_runs}
        if len(data_hashes) != 1:
            raise ValueError(f"processed test-data hash manifests differ across teacher seeds for {dataset}")
        train_sets = {tuple(int(i) for i in r["train_indices"]) for r in ds_runs}
        if len(train_sets) != 1:
            raise ValueError(f"frozen validation training indices differ across teacher seeds for {dataset}")
    # Dataset-specific official labels must be stable across teacher seeds.
    for dataset in DATASETS:
        druns = [r for r in runs if r["dataset"] == dataset]
        for other in druns[1:]:
            if (not np.array_equal(other["reference"]["indices"], druns[0]["reference"]["indices"])
                    or not np.array_equal(other["reference"]["labels"], druns[0]["reference"]["labels"])):
                raise ValueError(f"test ordering/labels differ across teacher seeds for {dataset}")

    per_seed = []
    paired_names = [(f"paired_{name}", left, right) for name, left, right in PAIRS]
    for dataset in DATASETS:
        for seed in SEEDS:
            run = next(r for r in runs if (r["dataset"], r["seed"]) == (dataset, seed))
            for timestep in TIMESTEPS:
                values = {method: {metric: run["archives"][(method, timestep)][metric]
                                   for metric in METRICS} for method in METHODS}
                deltas = {}
                for name, left, right in PAIRS:
                    deltas[name] = {metric: values[left][metric] - values[right][metric]
                                    for metric in METRICS}
                for method in METHODS:
                    row = {"dataset": dataset, "seed": seed, "scope": "historically_exposed_test",
                           "method": method, "T": timestep, "n_examples": 10000,
                           **values[method],
                           "runner_version": run["source_hashes"]["evaluate_frozen_test.py"],
                           "qualification": "historically exposed full test; descriptive only, not independent confirmation"}
                    for name, left, _ in PAIRS:
                        for metric in METRICS:
                            col = f"{name}_{metric}"
                            row[col] = deltas[name][metric] if method == left else ""
                    per_seed.append(row)
                for name, left, right in PAIRS:
                    row = {"dataset": dataset, "seed": seed, "scope": "historically_exposed_test",
                           "method": f"paired_{name}", "T": timestep, "n_examples": 10000,
                           **deltas[name],
                           "runner_version": run["source_hashes"]["evaluate_frozen_test.py"],
                           "qualification": f"paired {left}-{right} difference within seed; historically exposed test, descriptive only"}
                    for pair_name, _, _ in PAIRS:
                        for metric in METRICS:
                            row[f"{pair_name}_{metric}"] = ""
                    per_seed.append(row)

    aggregate = []
    for dataset in DATASETS:
        for method in list(METHODS) + [name for name, _, _ in paired_names]:
            for timestep in TIMESTEPS:
                seed_rows = [r for r in per_seed if r["dataset"] == dataset and r["method"] == method
                             and r["T"] == timestep]
                row = {"dataset": dataset, "scope": "historically_exposed_test",
                       "method": method, "T": timestep, "n_seeds": len(seed_rows),
                       "n_examples": 10000,
                       "runner_versions": json.dumps(sorted({r["runner_version"] for r in seed_rows})),
                       "qualification": "mean/sample SD across three teacher seeds; t95 df=2 assumes seed-level independence; test is historically exposed, descriptive only"}
                for metric in METRICS:
                    mean, sd, low, high = sample_summary([r[metric] for r in seed_rows])
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_sample_sd"] = sd
                    row[f"{metric}_t95_low"] = low
                    row[f"{metric}_t95_high"] = high
                aggregate.append(row)

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "aggregate.csv", aggregate)


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
