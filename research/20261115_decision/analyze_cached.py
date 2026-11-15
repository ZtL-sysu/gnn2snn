#!/usr/bin/env python
"""Read-only, CPU analysis of cached critical-site campaign outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CRIT = ROOT / "research" / "20261114_critical"
OUT = ROOT / "research" / "20261115_decision" / "analysis" / "cached"
BOOT_SEED = 20261115
BOOT_REPS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def align_indices(expected: np.ndarray, actual: np.ndarray, *, context: str = "") -> np.ndarray:
    """Return permutation mapping actual rows into expected order; reject any mismatch."""
    e, a = np.asarray(expected), np.asarray(actual)
    if e.ndim != 1 or a.ndim != 1 or len(e) != len(a):
        raise ValueError(f"{context}: index vectors must be 1-D with equal length")
    if len(np.unique(e)) != len(e) or len(np.unique(a)) != len(a):
        raise ValueError(f"{context}: duplicate graph indices")
    if not np.array_equal(np.sort(e), np.sort(a)):
        raise ValueError(f"{context}: graph index sets differ")
    pos = {int(v): i for i, v in enumerate(a)}
    return np.asarray([pos[int(v)] for v in e], dtype=np.int64)


def teacher_margins(teacher: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Teacher top-two gap, truth-vs-best-rival gap, and predicted-vs-truth gap."""
    z = np.asarray(teacher, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2 or y.shape != (len(z),) or np.any(y < 0) or np.any(y >= z.shape[1]):
        raise ValueError("invalid teacher logits/labels")
    order = np.argsort(z, axis=1)
    pred = order[:, -1]
    top = z[np.arange(len(z)), pred]
    second = z[np.arange(len(z)), order[:, -2]]
    truth = z[np.arange(len(z)), y]
    masked = z.copy()
    masked[np.arange(len(z)), y] = -np.inf
    best_other = masked.max(axis=1)
    return {
        "teacher_top2_margin": top - second,
        "teacher_truth_vs_competitor_margin": truth - best_other,
        "teacher_prediction_vs_truth_margin": top - truth,
        "teacher_prediction": pred,
    }


def accuracy_bootstrap(delta: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    d = np.asarray(delta, dtype=np.float64)
    draws = rng.integers(0, len(d), size=(BOOT_REPS, len(d)))
    means = d[draws].mean(axis=1) * 100.0
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean() * 100), float(lo), float(hi)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def _json_rows(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return obj["results"]
    return []


def audit_endpoints(hashes: dict[str, str]) -> dict[str, Any]:
    """Recompute each cached results.json endpoint when an unambiguous NPZ exists."""
    records: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for p in sorted((CRIT / "runs").rglob("results.json")):
        relative = p.relative_to(CRIT).as_posix()
        # Include every results summary in the raw provenance list, including
        # legacy percentile files whose schema cannot be paired with an NPZ.
        hashes["research/20261114_critical/" + relative] = sha256(p)
        run = p.parent
        for rownum, r in enumerate(_json_rows(p)):
            method, t = r.get("method"), r.get("T")
            # Historical percentile evaluators use a different result schema; retain
            # their rows as unevaluable instead of guessing an NPZ naming convention.
            candidate = run / f"{method}_T{t}.npz" if method is not None and t is not None else None
            rec = {"results_file": relative, "row": rownum, "method": method, "T": t,
                   "accuracy_recorded": r.get("accuracy_percent"),
                   "teacher_agreement_recorded": r.get("teacher_agreement_percent"),
                   "npz": candidate.relative_to(CRIT).as_posix() if candidate and candidate.exists() else "",
                   "status": "not_recomputable"}
            if candidate and candidate.is_file():
                a = _load_npz(candidate)
                if not {"logits", "teacher", "labels", "indices"}.issubset(a):
                    rec["status"] = "missing_required_fields"
                else:
                    pred = a["logits"].argmax(axis=1)
                    teacher_pred = a["teacher"].argmax(axis=1)
                    acc = float(np.mean(pred == a["labels"]) * 100)
                    agree = float(np.mean(pred == teacher_pred) * 100)
                    rec.update(accuracy_recomputed=acc, teacher_agreement_recomputed=agree,
                               graphs=len(pred), source_sha256=sha256(candidate))
                    da = abs(acc - float(r["accuracy_percent"]))
                    dg = abs(agree - float(r["teacher_agreement_percent"]))
                    rec["status"] = "pass" if da < 1e-7 and dg < 1e-7 else "mismatch"
                    if rec["status"] != "pass":
                        mismatches.append(rec.copy())
            records.append(rec)
    return {"results_json_rows": len(records),
            "recomputable_rows": sum(x["status"] in ("pass", "mismatch") for x in records),
            "verified_rows": sum(x["status"] == "pass" for x in records),
            "unevaluable_rows": sum(x["status"] == "not_recomputable" for x in records),
            "mismatch_count": len(mismatches), "mismatches": mismatches, "records": records}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    graph_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    missing: list[str] = []

    for ds in ("CIFAR10", "MNIST"):
        short = "CIFAR1041" if ds == "CIFAR10" else "MNIST41"
        alloc_dir = CRIT / "runs" / "allocation" / short
        diag_path = CRIT / "analysis" / "allocation_diagnostics" / f"{ds}_per_graph.npz"
        if not diag_path.exists():
            missing.append(str(diag_path)); continue
        diag = _load_npz(diag_path); hashes[diag_path.relative_to(ROOT).as_posix()] = sha256(diag_path)
        policies = [str(x) for x in diag["policy_names"]]
        if "local" not in policies or "message" not in policies:
            raise ValueError(f"{ds}: allocation diagnostics lack local/message policies")
        li, mi = policies.index("local"), policies.index("message")
        indices = diag["indices"]
        graph_features: dict[str, np.ndarray] = {"nodes": diag["graph_sizes"].astype(float)}

        alloc: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        for method in ("baseline", "local", "message", "random", "identity", "shuffled"):
            for t in (1, 2):
                p = alloc_dir / f"{method}_T{t}.npz"
                if not p.is_file(): missing.append(str(p)); continue
                a = _load_npz(p); perm = align_indices(indices, a["indices"], context=f"{ds} allocation {method} T{t}")
                a = {k: (v[perm] if v.ndim and len(v) == len(perm) else v) for k, v in a.items()}
                alloc[(method, t)] = a; hashes[p.relative_to(ROOT).as_posix()] = sha256(p)
                for feature in ("nodes", "mean_degree", "degree_variance"):
                    if feature in a and feature not in graph_features:
                        graph_features[feature] = a[feature].astype(float)

        if not all((m, t) in alloc for m in ("local", "message") for t in (1, 2)):
            continue
        local_mse = diag["after_message_actual"][:, li].astype(float)
        msg_mse = diag["after_message_actual"][:, mi].astype(float)
        local_improve = diag["before_message_actual"][:, li].astype(float) - local_mse
        msg_improve = diag["before_message_actual"][:, mi].astype(float) - msg_mse
        message_mse_better = msg_mse < local_mse

        for t in (1, 2):
            loc, msg = alloc[("local", t)], alloc[("message", t)]
            y = loc["labels"]
            if not np.array_equal(y, msg["labels"]) or not np.array_equal(loc["teacher"], msg["teacher"]):
                raise ValueError(f"{ds} allocation T{t}: teacher/labels mismatch")
            teacher = loc["teacher"]
            tm = teacher_margins(teacher, y)
            loc_pred, msg_pred = loc["logits"].argmax(axis=1), msg["logits"].argmax(axis=1)
            loc_runnerup = np.argsort(loc["logits"], axis=1)[:, -2]
            msg_runnerup = np.argsort(msg["logits"], axis=1)[:, -2]
            truth_teacher = tm["teacher_prediction"] == y
            loc_correct, msg_correct = loc_pred == y, msg_pred == y
            loc_teacher_correct, msg_teacher_correct = loc_pred == tm["teacher_prediction"], msg_pred == tm["teacher_prediction"]
            worsened = loc_correct & ~msg_correct
            improved = ~loc_correct & msg_correct
            paired = msg_correct.astype(float) - loc_correct.astype(float)
            acc, lo, hi = accuracy_bootstrap(paired, rng)
            switched_runner = loc_runnerup != msg_runnerup
            flip_from_local = loc_pred != msg_pred
            t1t2_flips = {}
            for method_name in ("baseline", "local", "message"):
                a1, a2 = alloc.get((method_name, 1)), alloc.get((method_name, 2))
                if a1 is not None and a2 is not None:
                    perm = align_indices(a1["indices"], a2["indices"], context=f"{ds} {method_name} T1/T2")
                    t1t2_flips[method_name] = a1["logits"].argmax(1) != a2["logits"][perm].argmax(1)
                else:
                    t1t2_flips[method_name] = np.zeros(len(y), dtype=bool)

            q = np.quantile(tm["teacher_top2_margin"], [0, .25, .5, .75, 1])
            for i, idx in enumerate(indices):
                row: dict[str, Any] = {
                    "dataset": ds, "seed": 41, "T": t, "index": int(idx),
                    "local_message_mse": float(local_mse[i]), "message_message_mse": float(msg_mse[i]),
                    "local_message_mse_improvement": float(local_improve[i]),
                    "message_message_mse_improvement": float(msg_improve[i]),
                    "message_mse_better_than_local": bool(message_mse_better[i]),
                    "local_correct": bool(loc_correct[i]), "message_correct": bool(msg_correct[i]),
                    "local_teacher_agreement": bool(loc_teacher_correct[i]), "message_teacher_agreement": bool(msg_teacher_correct[i]),
                    "teacher_correct": bool(truth_teacher[i]), "local_accuracy_worsened": bool(worsened[i]),
                    "local_accuracy_improved": bool(improved[i]), "runnerup_class_switched_local_vs_message": bool(switched_runner[i]),
                    "prediction_switched": bool(flip_from_local[i]), "teacher_prediction": int(tm["teacher_prediction"][i]),
                    "baseline_T1_to_T2_prediction_flip": bool(t1t2_flips["baseline"][i]),
                    "local_T1_to_T2_prediction_flip": bool(t1t2_flips["local"][i]),
                    "message_T1_to_T2_prediction_flip": bool(t1t2_flips["message"][i]),
                    "label": int(y[i]), "teacher_top2_margin": float(tm["teacher_top2_margin"][i]),
                    "teacher_truth_vs_competitor_margin": float(tm["teacher_truth_vs_competitor_margin"][i]),
                    "teacher_prediction_vs_truth_margin": float(tm["teacher_prediction_vs_truth_margin"][i]),
                    "teacher_truth_quadrant": "teacher_correct" if truth_teacher[i] else "teacher_wrong",
                    "nodes": int(graph_features["nodes"][i]) if "nodes" in graph_features else None,
                    "mean_degree": float(graph_features["mean_degree"][i]) if "mean_degree" in graph_features else None,
                    "degree_variance": float(graph_features["degree_variance"][i]) if "degree_variance" in graph_features else None,
                }
                row["teacher_margin_quartile"] = int(np.searchsorted(q[1:-1], row["teacher_top2_margin"], side="right") + 1)
                graph_rows.append(row)

            bins = np.searchsorted(q[1:-1], tm["teacher_top2_margin"], side="right") + 1
            strata = []
            for b in range(1, 5):
                mask = bins == b
                strata.append({"quartile": b, "n": int(mask.sum()),
                               "local_accuracy_percent": float(loc_correct[mask].mean() * 100),
                               "message_accuracy_percent": float(msg_correct[mask].mean() * 100),
                               "paired_delta_message_minus_local_pp": float(paired[mask].mean() * 100)})

            # Basic exploratory OLS for paired correctness difference, with available graph-size/degree covariates.
            cov_names = [k for k in ("nodes", "mean_degree", "degree_variance") if k in graph_features]
            X = np.column_stack([np.ones(len(paired))] + [graph_features[k] for k in cov_names])
            beta = np.linalg.lstsq(X, paired, rcond=None)[0]
            summaries.append({"dataset": ds, "seed": 41, "T": t, "n_graphs": len(y),
                "message_mse_better_count": int(message_mse_better.sum()),
                "message_mse_better_and_accuracy_worsened_count": int((message_mse_better & worsened).sum()),
                "message_mse_better_and_accuracy_improved_count": int((message_mse_better & improved).sum()),
                "message_mse_better_but_accuracy_same_count": int((message_mse_better & ~(worsened | improved)).sum()),
                "local_accuracy_worsened_count": int(worsened.sum()), "local_accuracy_improved_count": int(improved.sum()),
                "teacher_correct_count": int(truth_teacher.sum()), "teacher_wrong_count": int((~truth_teacher).sum()),
                "teacher_correct_and_conversion_correct_local": int((truth_teacher & loc_correct).sum()),
                "teacher_correct_and_conversion_wrong_local": int((truth_teacher & ~loc_correct).sum()),
                "teacher_correct_and_conversion_correct_message": int((truth_teacher & msg_correct).sum()),
                "teacher_correct_and_conversion_wrong_message": int((truth_teacher & ~msg_correct).sum()),
                "teacher_wrong_but_conversion_correct_local": int((~truth_teacher & loc_correct).sum()),
                "teacher_wrong_and_conversion_wrong_local": int((~truth_teacher & ~loc_correct).sum()),
                "teacher_wrong_but_conversion_correct_message": int((~truth_teacher & msg_correct).sum()),
                "teacher_wrong_and_conversion_wrong_message": int((~truth_teacher & ~msg_correct).sum()),
                "runnerup_class_switched_local_vs_message_count": int(switched_runner.sum()), "prediction_switched_count": int(flip_from_local.sum()),
                "baseline_T1_to_T2_prediction_flip_count": int(t1t2_flips["baseline"].sum()),
                "local_T1_to_T2_prediction_flip_count": int(t1t2_flips["local"].sum()),
                "message_T1_to_T2_prediction_flip_count": int(t1t2_flips["message"].sum()),
                "paired_accuracy_delta_message_minus_local_pp": acc, "paired_bootstrap95_low_pp": lo,
                "paired_bootstrap95_high_pp": hi, "bootstrap_seed": BOOT_SEED, "bootstrap_reps": BOOT_REPS,
                "teacher_top2_margin_mean": float(tm["teacher_top2_margin"].mean()),
                "teacher_truth_vs_competitor_margin_mean": float(tm["teacher_truth_vs_competitor_margin"].mean()),
                "teacher_prediction_vs_truth_margin_mean": float(tm["teacher_prediction_vs_truth_margin"].mean()),
                "margin_quartiles": strata,
                "exploratory_ols_covariates": cov_names,
                "exploratory_ols_coefficients_intercept_then_covariates": beta.tolist(),
                "exploratory_ols_note": "descriptive, non-causal, single teacher and selection-exposed validation subset"})

    # Critical endpoint CSV counts/hashes and original result records are separately retained by scope.
    endpoint_audit = audit_endpoints(hashes)
    all_local_scopes = []
    for rec in endpoint_audit["records"]:
        if rec.get("method") == "all_local" and rec.get("status") == "pass":
            path = rec["results_file"]
            scope = next((s for s in ("optimizer_check", "screen", "allocation", "book_screen", "full_book_screen") if f"/{s}/" in f"/{path}/"), "other")
            all_local_scopes.append({"scope": scope, "results_file": path, "T": rec.get("T"),
                                     "accuracy_percent": rec.get("accuracy_recomputed"),
                                     "teacher_agreement_percent": rec.get("teacher_agreement_recomputed"),
                                     "qualification": "scope-specific cached endpoint; do not pool across optimization protocols"})
    endpoint_csvs = {}
    raw_endpoint_audit: dict[str, Any] = {}
    for name in ("endpoints.csv", "rescue_endpoints.csv"):
        p = CRIT / "analysis" / name
        if p.exists():
            with p.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            endpoint_csvs[name] = {"rows": len(rows), "sha256": sha256(p), "columns": list(rows[0]) if rows else []}
            hashes[p.relative_to(ROOT).as_posix()] = sha256(p)
            checked = 0
            failures: list[dict[str, Any]] = []
            for i, row in enumerate(rows):
                source = row.get("source")
                expected = row.get("source_sha256") or row.get("sha256")
                if not source or not expected:
                    failures.append({"row": i, "reason": "missing source/hash"}); continue
                raw = CRIT / Path(source)
                if not raw.is_file():
                    failures.append({"row": i, "source": source, "reason": "source file missing"}); continue
                actual = sha256(raw)
                hashes[raw.relative_to(ROOT).as_posix()] = actual
                if actual.lower() != expected.lower():
                    failures.append({"row": i, "source": source, "reason": "sha256 mismatch",
                                     "expected": expected, "actual": actual}); continue
                try:
                    a = _load_npz(raw)
                    if {"logits", "teacher", "labels"}.issubset(a):
                        acc = float(np.mean(a["logits"].argmax(1) == a["labels"]) * 100)
                        agree = float(np.mean(a["logits"].argmax(1) == a["teacher"].argmax(1)) * 100)
                        if abs(acc - float(row["accuracy_percent"])) > 1e-7 or abs(agree - float(row["teacher_agreement_percent"])) > 1e-7:
                            failures.append({"row": i, "source": source, "reason": "metrics mismatch",
                                             "accuracy_recomputed": acc, "agreement_recomputed": agree})
                            continue
                    checked += 1
                except Exception as e:
                    failures.append({"row": i, "source": source, "reason": f"NPZ recomputation failed: {e}"})
            raw_endpoint_audit[name] = {"rows": len(rows), "source_hash_and_metrics_verified": checked,
                                        "failure_count": len(failures), "failures": failures}

    with (OUT / "per_graph_diagnostics.csv").open("w", encoding="utf-8", newline="") as f:
        if graph_rows:
            writer = csv.DictWriter(f, fieldnames=list(graph_rows[0]))
            writer.writeheader(); writer.writerows(graph_rows)
    with (OUT / "input_hashes.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "sha256"])
        for path, digest in sorted(hashes.items()): w.writerow([path, digest])
    with (OUT / "endpoint_recomputation.csv").open("w", encoding="utf-8", newline="") as f:
        rows = endpoint_audit["records"]
        if rows:
            fields = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    report = {"scope": "cached CPU-only diagnostics; teacher 41, fixed 512 validation indices; descriptive/selection-exposed",
              "bootstrap": {"seed": BOOT_SEED, "replicates": BOOT_REPS, "interval": "percentile 95% paired graph bootstrap"},
              "endpoint_audit": {k: v for k, v in endpoint_audit.items() if k != "records"},
              "critical_endpoint_csvs": endpoint_csvs, "raw_endpoint_audit": raw_endpoint_audit,
              "all_local_protocol_scopes": all_local_scopes, "diagnostic_summaries": summaries,
              "missing_inputs": missing, "input_hash_count": len(hashes),
              "limits": ["No teacher/checkpoint loading or GPU inference was performed.",
                         "Allocation message-MSE values are graphwise; no unrecorded message-error quantities were imputed.",
                         "Margin strata and OLS are exploratory, not causal; no independent validation/test claim."]}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "per_graph_rows": len(graph_rows),
                      "verified_result_rows": endpoint_audit["verified_rows"],
                      "unevaluable_result_rows": endpoint_audit["unevaluable_rows"],
                      "mismatches": endpoint_audit["mismatch_count"],
                      "endpoint_csvs": endpoint_csvs, "missing_inputs": missing}, indent=2, ensure_ascii=False))


def self_check() -> None:
    """Tiny assertions for index rejection and independent margin definitions."""
    p = align_indices(np.array([2, 1]), np.array([1, 2]))
    assert np.array_equal(p, [1, 0])
    try:
        align_indices(np.array([1, 2]), np.array([1, 3]), context="self-check")
    except ValueError:
        pass
    else:
        raise AssertionError("alignment accepted mismatched graph IDs")
    got = teacher_margins(np.array([[3., 2., 0.], [0., 4., 1.]]), np.array([1, 1]))
    assert np.allclose(got["teacher_top2_margin"], [1., 3.])
    assert np.allclose(got["teacher_truth_vs_competitor_margin"], [-1., 3.])
    assert np.allclose(got["teacher_prediction_vs_truth_margin"], [1., 0.])


if __name__ == "__main__":
    self_check()
    main()
