"""Aggregate the frozen E1 rows across seeds 41/42/43 (descriptive only).

Reads the two immutable seed-41 reports and the two new per-seed reports,
all on the full 5000-graph official validation split. Emits:

  * e1_multiseed_summary_v1.json: per (mode, T) per-seed accuracy, 3-seed
    mean and sample SD, same-teacher paired differences vs the strongest
    matched conversion control, and the .3pp criterion *status*.
  * METRICS.csv: fixed research-github-sync columns, one row per real metric.

This script NEVER claims E1 acceptance. The .3pp criterion is reported as
"accuracy_only_not_acceptable": protocol-v3 AND the head addendum require
BOTH the .3pp accuracy condition AND lower complete known-operation cost or
matched wall-clock at BOTH T4 and T8. Cost/latency evidence does not exist
yet, so the criterion stays NOT EVALUABLE. Missing required rows (ANN,
historical six-site DC-adapt, shared-theta) are listed explicitly.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(r"C:\code\gnn2snn-wb-resume")
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
STUDY = ROOT / "research" / "visual-graphs" / "continuation_20261113"
PROTOCOL_SHA = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
TASK, AGENT, RUN_ID = "visual-conversion", "20261113-wb-resume", "e1-multiseed-v1"

SOURCES = {
    41: [
        (STUDY / "results" / "e1-seed41-affine-v1" / "report.json",
         "research/visual-graphs/continuation_20261113/results/e1-seed41-affine-v1/report.json"),
        (RUNROOT / "results" / "e1-seed41-controls-v1" / "report.json",
         "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix/results/e1-seed41-controls-v1/report.json"),
    ],
    42: [(RUNROOT / "results" / "e1-seed42-multiseed-v1" / "report.json",
          "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix/results/e1-seed42-multiseed-v1/report.json")],
    43: [(RUNROOT / "results" / "e1-seed43-multiseed-v1" / "report.json",
          "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix/results/e1-seed43-multiseed-v1/report.json")],
}
TARGET_MODES = ("centered", "first_analog", "generic",
                "dc_adapt_analog_head", "per_consumer", "cse_only")
CONTROL_FOR_CANDIDATE = ("first_analog", "generic", "per_consumer", "cse_only",
                         "dc_adapt_analog_head")
MISSING_ROWS = ("ANN", "historical_six_site_DC_adapt", "shared_theta_controls",
                "complete_known_operation_cost", "matched_wall_clock_latency")


def sample_sd(values):
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def main() -> None:
    per_seed = {}
    evidence = {}
    teacher_acc = {}
    for seed, paths in SOURCES.items():
        rows = {}
        for path, rel in paths:
            doc = json.loads(path.read_text(encoding="utf-8"))
            assert doc.get("status") == "complete", f"incomplete report: {path}"
            assert doc.get("graphs") == 5000, f"not full validation: {path}"
            assert int(doc.get("seed", -1)) == seed, f"seed mismatch: {path}"
            teacher_acc[seed] = doc.get("teacher_accuracy")
            for r in doc["rows"]:
                key = (r["mode"], int(r["T"]))
                if key in rows:
                    raise RuntimeError(f"duplicate row {key} for seed {seed}")
                rows[key] = {"accuracy": float(r["accuracy"]),
                             "teacher_agreement": float(r["teacher_agreement"]),
                             "logit_mse": float(r["logit_mse"]),
                             "teacher_to_output_kl": float(r["teacher_to_output_kl"])}
                evidence[(seed, *key)] = rel
        per_seed[seed] = rows

    summary = {
        "status": "descriptive_only_not_E1_acceptance",
        "dataset": "CIFAR10", "split": "all official validation", "coverage": 5000,
        "seeds": [41, 42, 43], "threshold_scale": 1,
        "protocol_sha256": PROTOCOL_SHA,
        "teacher_accuracy_by_seed": teacher_acc,
        "rows": [], "criterion": {},
        "missing_required_rows": list(MISSING_ROWS),
        "cost_or_latency_evidence": False,
    }
    metric_rows = []
    for mode in TARGET_MODES:
        for T in (1, 2, 4, 8):
            key = (mode, T)
            vals = [per_seed[s][key]["accuracy"] for s in (41, 42, 43) if key in per_seed[s]]
            entry = {"mode": mode, "T": T, "n_seeds": len(vals),
                     "per_seed_accuracy": {str(s): (per_seed[s][key]["accuracy"] if key in per_seed[s] else None)
                                           for s in (41, 42, 43)},
                     "mean_accuracy": (sum(vals) / len(vals)) if vals else None,
                     "sample_sd_accuracy": sample_sd(vals) if len(vals) > 1 else None,
                     "per_seed_metrics": {str(s): per_seed[s][key] for s in (41, 42, 43) if key in per_seed[s]}}
            summary["rows"].append(entry)
            for s in (41, 42, 43):
                if key in per_seed[s]:
                    for metric, unit in (("accuracy", "fraction"), ("teacher_agreement", "fraction"),
                                         ("logit_mse", "mse"), ("teacher_to_output_kl", "nats")):
                        metric_rows.append({
                            "task_id": TASK, "agent_id": AGENT, "run_id": RUN_ID,
                            "dataset": "CIFAR10", "split": "val", "coverage": "5000_all_official_val",
                            "method": mode, "teacher_seed": s, "student_seed": "",
                            "T": T, "metric": metric, "value": per_seed[s][key][metric],
                            "unit": unit, "evidence": evidence[(s, *key)],
                            "protocol_sha256": PROTOCOL_SHA, "status": "measured" if s != 41 else "historical"})
            if len(vals) == 3:
                metric_rows.append({
                    "task_id": TASK, "agent_id": AGENT, "run_id": RUN_ID,
                    "dataset": "CIFAR10", "split": "val", "coverage": "5000_all_official_val",
                    "method": mode, "teacher_seed": "41,42,43", "student_seed": "",
                    "T": T, "metric": "accuracy_mean_3seed", "value": sum(vals) / 3,
                    "unit": "fraction", "evidence": "aggregated from per-seed reports listed above",
                    "protocol_sha256": PROTOCOL_SHA, "status": "measured"})
                sd = sample_sd(vals)
                metric_rows.append({
                    "task_id": TASK, "agent_id": AGENT, "run_id": RUN_ID,
                    "dataset": "CIFAR10", "split": "val", "coverage": "5000_all_official_val",
                    "method": mode, "teacher_seed": "41,42,43", "student_seed": "",
                    "T": T, "metric": "accuracy_sample_sd_3seed", "value": sd,
                    "unit": "fraction", "evidence": "aggregated from per-seed reports listed above",
                    "protocol_sha256": PROTOCOL_SHA, "status": "measured"})

    # .3pp accuracy condition at T4/T8, candidate centered vs strongest matched control.
    acc_check = {}
    for T in (4, 8):
        cand = None
        for e in summary["rows"]:
            if e["mode"] == "centered" and e["T"] == T and e["n_seeds"] == 3:
                cand = e["mean_accuracy"]
        controls = {e["mode"]: e["mean_accuracy"] for e in summary["rows"]
                    if e["mode"] in CONTROL_FOR_CANDIDATE and e["T"] == T and e["n_seeds"] == 3}
        best_name = max(controls, key=controls.get) if controls else None
        best_val = controls.get(best_name) if best_name else None
        gap = (best_val - cand) if (cand is not None and best_val is not None) else None
        acc_check[f"T{T}"] = {
            "centered_mean": cand, "strongest_matched_control": best_name,
            "strongest_control_mean": best_val, "gap_pp": (gap * 100) if gap is not None else None,
            "within_0.3pp_accuracy_only": (gap is not None and gap <= 0.003),
            "cost_or_latency_condition": "NOT EVALUABLE (no complete known-operation cost or matched wall-clock evidence)",
        }
    summary["criterion"] = {
        "protocol": "BOTH T4 and T8 three-seed mean accuracy within 0.3pp of strongest matched conversion control AND lower complete known-operation cost or matched latency",
        "accuracy_only": acc_check,
        "verdict": "NOT EVALUABLE / E1 NOT PASSED",
        "reason": "complete cost and matched wall-clock evidence are missing; ANN, historical six-site DC-adapt and shared-theta rows are missing; no three-seed significance or CI claim is made",
    }

    out = RUNROOT / "results" / "e1-multiseed-summary-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1_multiseed_summary_v1.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["task_id", "agent_id", "run_id", "dataset", "split", "coverage",
              "method", "teacher_seed", "student_seed", "T", "metric", "value",
              "unit", "evidence", "protocol_sha256", "status"]
    with (out / "METRICS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metric_rows)
    print(json.dumps({"status": summary["status"], "n_metric_rows": len(metric_rows),
                      "n_summary_rows": len(summary["rows"]),
                      "criterion": summary["criterion"]["verdict"],
                      "T4": acc_check["T4"], "T8": acc_check["T8"],
                      "missing_required_rows": list(MISSING_ROWS)}, indent=2))


if __name__ == "__main__":
    main()
