"""Summarize completed ideal-site validation diagnostics, without reopening datasets."""
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
TIMESTEPS = (1, 2)
CONDITIONS = {
    "baseline": (), "site0": (0,), "site1": (1,), "site2": (2,), "site3": (3,),
    "site4": (4,), "site5": (5,), "nodes0_3": (0, 1, 2, 3),
    "head4_5": (4, 5), "all6": (0, 1, 2, 3, 4, 5),
}
SITE_NAMES = ("gcn_0", "gcn_1", "gcn_2", "gcn_3", "mlp_0", "mlp_1")
T95_DF2 = 4.30265273
METRICS = ("accuracy_percent", "teacher_agreement_percent", "logit_mse",
           "teacher_accuracy_percent")
QUALIFICATION = (
    "continuous-valued causal ReLU oracle diagnostic on historically exposed validation; "
    "not a deployable conversion; group sizes differ and effects are non-additive"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value, what):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"invalid SHA-256 for {what}")


def metric_values(logits, teacher, labels):
    pred = logits.argmax(1)
    teacher_pred = teacher.argmax(1)
    return {
        "accuracy_percent": float(100.0 * np.mean(pred == labels)),
        "teacher_agreement_percent": float(100.0 * np.mean(pred == teacher_pred)),
        "logit_mse": float(np.mean((logits.astype(np.float64) - teacher.astype(np.float64)) ** 2)),
        "teacher_accuracy_percent": float(100.0 * np.mean(teacher_pred == labels)),
    }


def close(actual, expected, what, atol=2e-5):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError(f"non-finite {what}")
    if abs(float(actual) - float(expected)) > atol:
        raise ValueError(f"recomputed {what} differs from evaluator result: {actual} vs {expected}")


def sample_summary(values, allow_undefined=False):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) == 0 or (not allow_undefined and len(vals) != 3) or len(vals) < 2:
        if allow_undefined and len(vals) < 2:
            return len(vals), None, None, None, None
        raise ValueError("aggregate requires three finite teacher-seed values")
    x = np.asarray(vals, dtype=np.float64)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if len(vals) != 3:
        return len(vals), mean, sd, None, None
    half = T95_DF2 * sd / math.sqrt(3.0)
    return len(vals), mean, sd, mean - half, mean + half


def read_run(run_dir):
    run_dir = run_dir.resolve()
    meta_path, results_path, done_path = (run_dir / "metadata.json", run_dir / "results.json", run_dir / "DONE.json")
    for path in (meta_path, results_path, done_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete ideal-site run: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    dataset, seed = meta.get("dataset"), int(meta.get("seed", -1))
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError(f"unexpected dataset/seed in {run_dir}")
    if "smoke" in str(run_dir).lower() or meta.get("smoke"):
        raise ValueError(f"smoke run is ineligible: {run_dir}")
    if (meta.get("status") != "complete" or done.get("status") != "complete"
            or done.get("test_loaded") is not False or meta.get("test_loaded", False) is not False
            or done.get("optimization_or_training") is not False
            or meta.get("optimization_or_training") is not False):
        raise ValueError(f"run is incomplete, used test data, or trained/optimized: {run_dir}")
    if (meta.get("split") != "full official validation; historically exposed"
            or meta.get("graphs") != 5000 or tuple(meta.get("T", [])) != TIMESTEPS):
        raise ValueError(f"unexpected validation scope or T grid in {run_dir}")
    if set(meta.get("conditions", [])) != set(CONDITIONS):
        raise ValueError(f"mask condition set mismatch in {run_dir}")
    expected = set(itertools.product(CONDITIONS, TIMESTEPS))
    keys = [(r.get("condition"), int(r.get("T", -1))) for r in results]
    if len(results) != 20 or len(set(keys)) != 20 or set(keys) != expected or done.get("rows") != 20:
        raise ValueError(f"expected exactly 20 mask/T results in {run_dir}")
    result_map = {(r["condition"], int(r["T"])): r for r in results}

    confirmation_hash = meta.get("confirmation_metadata_sha256")
    require_sha(confirmation_hash, "confirmation_metadata_sha256")
    source_hashes = meta.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"source hash manifest missing in {run_dir}")
    source_dir = run_dir / "source"
    for name, digest in source_hashes.items():
        require_sha(digest, f"source {name}")
        path = source_dir / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"source snapshot hash mismatch: {path}")
    for name in ("calibration_core.py", "full_teacher_gcn.py", "fold_teacher_bn.py",
                 "dcgs_fixed_seed_eval_v2.py", "ideal_site_diagnostic.py"):
        if name not in source_hashes:
            raise ValueError(f"missing scientific source snapshot {name} in {run_dir}")
    if source_hashes["calibration_core.py"] != meta.get("core_sha256"):
        raise ValueError(f"core snapshot differs from metadata in {run_dir}")

    max_abs = float(meta.get("all6_max_abs_vs_folded_teacher", float("nan")))
    if not math.isfinite(max_abs) or max_abs > 1e-4:
        raise ValueError(f"all-six oracle fails folded-teacher recovery tolerance in {run_dir}: {max_abs}")
    if meta.get("original_quantizer_restored") is not True or meta.get("zero_offsets") is not True:
        raise ValueError(f"ideal-site hooks/offset guard failed in {run_dir}")

    indices_ref = np.asarray(meta.get("indices", []), dtype=np.int64)
    if (indices_ref.shape != (5000,) or len(np.unique(indices_ref)) != 5000
            or indices_ref.min() < 0 or indices_ref.max() >= 5000):
        raise ValueError(f"invalid saved validation index ordering in {run_dir}")
    archive_data = {}
    shared = None
    for condition, timestep in sorted(expected, key=lambda x: (x[1], list(CONDITIONS).index(x[0]))):
        path = run_dir / f"{condition}_T{timestep}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {"indices", "labels", "teacher_logits", "logits"}
            if not required.issubset(archive.files):
                raise ValueError(f"missing prediction arrays in {path}")
            indices = np.asarray(archive["indices"], dtype=np.int64)
            labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
            teacher = np.asarray(archive["teacher_logits"])
            logits = np.asarray(archive["logits"])
        if (indices.shape != (5000,) or not np.array_equal(indices, indices_ref)
                or labels.shape != (5000,) or teacher.shape != (5000, 10) or logits.shape != (5000, 10)):
            raise ValueError(f"shape/order mismatch in {path}")
        if not np.isfinite(teacher).all() or not np.isfinite(logits).all():
            raise ValueError(f"nonfinite logits in {path}")
        if shared is None:
            shared = {"indices": indices.copy(), "labels": labels.copy(), "teacher": teacher.copy()}
        else:
            if (not np.array_equal(indices, shared["indices"])
                    or not np.array_equal(labels, shared["labels"])
                    or not np.array_equal(teacher, shared["teacher"])):
                raise ValueError(f"condition indices/labels/teacher logits differ in {path}")
        values = metric_values(logits, teacher, labels)
        saved = result_map[(condition, timestep)]
        if int(saved.get("graphs", -1)) != 5000:
            raise ValueError(f"result row lacks full validation coverage: {path}")
        for metric, value in values.items():
            close(value, saved.get(metric), f"{path.name}:{metric}")
        archive_data[(condition, timestep)] = {
            "indices": indices, "labels": labels, "teacher": teacher,
            "logits": logits, **values,
        }

    # If the frozen confirmation output is a sibling, verify baseline arrays
    # against it directly; otherwise the diagnostic's completion guard and
    # confirmation metadata SHA remain the provenance record.
    confirmation = run_dir.parent.parent / "confirmation" / f"{dataset}{seed}"
    if confirmation.is_dir():
        confirmation_meta = confirmation / "metadata.json"
        if not confirmation_meta.is_file() or sha256(confirmation_meta) != confirmation_hash:
            raise ValueError(f"sibling confirmation metadata hash mismatch: {confirmation_meta}")
        confirmation_doc = json.loads(confirmation_meta.read_text(encoding="utf-8"))
        expected_parameters = {f'{method}_T{t}' for method in ('node','graph','mix','output','shuffled') for t in (1,2,4,8)}
        if set(meta.get("parameter_sha256", {})) != expected_parameters:
            raise ValueError(f"incomplete confirmation parameter hash manifest: {run_dir}")
        for condition, digest in meta.get("parameter_sha256", {}).items():
            require_sha(digest, f"confirmation parameter {condition}")
            param_path = confirmation / f"{condition}_parameters.npz"
            if not param_path.is_file() or sha256(param_path) != digest:
                raise ValueError(f"confirmation parameter hash mismatch: {param_path}")
        for timestep in TIMESTEPS:
            ref_path = confirmation / f"baseline_T{timestep}.npz"
            if not ref_path.is_file():
                raise FileNotFoundError(ref_path)
            with np.load(ref_path, allow_pickle=False) as ref:
                for ref_name, archive_name in (("indices", "indices"), ("labels", "labels"),
                                               ("teacher", "teacher"), ("logits", "logits")):
                    if not np.array_equal(ref[ref_name], archive_data[("baseline", timestep)][archive_name]):
                        raise ValueError(f"baseline {ref_name} differs from frozen confirmation at T{timestep}")

    return {"path": run_dir, "dataset": dataset, "seed": seed, "meta": meta,
            "source_hashes": source_hashes,
            "common_sources": {k: v for k, v in source_hashes.items() if k != "run_screen.py"},
            "reference": shared, "archives": archive_data}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.runs.resolve(), args.output.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata_files = sorted(p for p in root.rglob("metadata.json")
                            if (p.parent / "DONE.json").is_file()
                            and (p.parent / "baseline_T1.npz").is_file())
    if len(metadata_files) != 6:
        raise ValueError(f"expected exactly six ideal-site run dirs, found {len(metadata_files)}")
    runs = [read_run(p.parent) for p in metadata_files]
    identities = [(r["dataset"], r["seed"]) for r in runs]
    expected_ids = {(d, s) for d in DATASETS for s in SEEDS}
    if len(set(identities)) != 6 or set(identities) != expected_ids:
        raise ValueError(f"expected six dataset×seed runs for seeds 41/42/43, found {identities}")
    if len({json.dumps(r["common_sources"], sort_keys=True) for r in runs}) != 1:
        raise ValueError("scientific/core source hashes differ across ideal-site runs")
    common_configs = []
    for run in runs:
        m = run["meta"]
        cfg = {key: m.get(key) for key in ("split", "graphs", "batch_size", "T", "conditions",
              "zero_offsets", "optimization_or_training", "core_sha256", "device", "torch")}
        common_configs.append(json.dumps(cfg, sort_keys=True, separators=(",", ":")))
    if len(set(common_configs)) != 1:
        raise ValueError("ideal-site diagnostic configurations differ across six runs")
    for dataset in DATASETS:
        ds_runs = [r for r in runs if r["dataset"] == dataset]
        if len({json.dumps(r["reference"]["indices"].tolist()) for r in ds_runs}) != 1:
            raise ValueError(f"validation indices differ across teacher seeds for {dataset}")
        if any(not np.array_equal(ds_runs[0]["reference"]["labels"], r["reference"]["labels"]) for r in ds_runs[1:]):
            raise ValueError(f"validation labels differ across teacher seeds for {dataset}")

    excluded = set(int(i) for i in np.random.default_rng(20261115).permutation(5000)[:512])
    per_seed = []
    for run in runs:
        indices = run["reference"]["indices"]
        masks = {
            "full_validation": np.ones(5000, dtype=bool),
            "screen_excluded": np.asarray([int(i) not in excluded for i in indices]),
        }
        if int(masks["screen_excluded"].sum()) != 4488:
            raise AssertionError("screen-excluded validation subset must contain 4488 graphs")
        for scope, mask in masks.items():
            for timestep in TIMESTEPS:
                values = {}
                for condition in CONDITIONS:
                    archive = run["archives"][(condition, timestep)]
                    values[condition] = metric_values(archive["logits"][mask],
                                                      archive["teacher"][mask],
                                                      archive["labels"][mask])
                baseline_agreement = values["baseline"]["teacher_agreement_percent"]
                for condition in CONDITIONS:
                    value = values[condition]
                    delta_accuracy = value["accuracy_percent"] - values["baseline"]["accuracy_percent"]
                    delta_agreement = value["teacher_agreement_percent"] - baseline_agreement
                    delta_mse = value["logit_mse"] - values["baseline"]["logit_mse"]
                    denom = 100.0 - baseline_agreement
                    closed = delta_agreement / denom if denom > 0 else None
                    per_seed.append({
                        "dataset": run["dataset"], "seed": run["seed"], "scope": scope,
                        "method": condition, "T": timestep, "n_examples": int(mask.sum()),
                        "ideal_site_count": len(CONDITIONS[condition]), **value,
                        "accuracy_delta_vs_baseline_pp": delta_accuracy,
                        "agreement_delta_vs_baseline_pp": delta_agreement,
                        "logit_mse_delta_vs_baseline": delta_mse,
                        "agreement_gap_closed": closed,
                        "qualification": QUALIFICATION,
                    })

    aggregate = []
    for dataset in DATASETS:
        for scope in ("full_validation", "screen_excluded"):
            for method, sites in CONDITIONS.items():
                for timestep in TIMESTEPS:
                    rows = [r for r in per_seed if r["dataset"] == dataset and r["scope"] == scope
                            and r["method"] == method and r["T"] == timestep]
                    row = {"dataset": dataset, "scope": scope, "method": method, "T": timestep,
                           "n_seeds": 3, "n_examples": rows[0]["n_examples"],
                           "ideal_site_count": len(sites), "qualification": QUALIFICATION}
                    columns = list(METRICS) + ["accuracy_delta_vs_baseline_pp",
                              "agreement_delta_vs_baseline_pp", "logit_mse_delta_vs_baseline"]
                    for metric in columns:
                        _, mean, sd, low, high = sample_summary([r[metric] for r in rows])
                        row[f"{metric}_mean"] = mean
                        row[f"{metric}_sample_sd"] = sd
                        row[f"{metric}_t95_low"] = low
                        row[f"{metric}_t95_high"] = high
                    count, mean, sd, low, high = sample_summary(
                        [r["agreement_gap_closed"] for r in rows], allow_undefined=True)
                    row["agreement_gap_closed_n_defined"] = count
                    row["agreement_gap_closed_mean"] = mean
                    row["agreement_gap_closed_sample_sd"] = sd
                    row["agreement_gap_closed_t95_low"] = low
                    row["agreement_gap_closed_t95_high"] = high
                    aggregate.append(row)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "aggregate.csv", aggregate)


if __name__ == "__main__":
    main()
