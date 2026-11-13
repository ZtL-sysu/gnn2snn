"""Aggregate the COMPLETE frozen E1 row set across seeds 41/42/43 (descriptive only).

Inputs (all already measured on the full 5000-graph official CIFAR10 val split):

  * seed41: immutable affine + matched-control reports (6 rows)
  * seed42/43: immutable six-row multiseed reports
  * seeds 41/42/43: new e1_residual_v1 reports (historical six-site DC-adapt,
    first_analog_shared_theta, centered_shared_theta)
  * ANN upper-reference row report (frozen teacher logits, 12 archives)

Outputs, under RUNROOT/results/e1-fullrow-summary-v1/:
  * e1_fullrow_summary_v1.json: per (mode, T) per-seed metrics, 3-seed mean and
    sample SD, the strongest matched conversion control, and the criterion STATUS.
  * METRICS.csv: fixed research-github-sync columns, one row per real metric.

This script NEVER claims E1 acceptance. Protocol-v3 + the head addendum require
BOTH the .3pp accuracy condition AND lower complete known-operation cost or
matched wall-clock latency at BOTH T4 and T8. Cost/latency evidence still does
not exist, so the criterion remains NOT EVALUABLE / E1 NOT PASSED.

ANN is an upper reference, NOT a conversion control. It is excluded from the
best-control set. cse_only / generic / per_consumer are known numerical aliases
and do not constitute independent method wins. No significance or CI is claimed.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = STUDY.parents[2]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
RR = "review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix"
SR = "research/visual-graphs/continuation_20261113"
PROTOCOL_SHA = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
ADDENDUM_SHA = "8bde5832d556c8c7b04ee630f2390703c33a26fa750f196f4dc1c94da61291f6"
TASK, AGENT, RUN_ID = "visual-conversion", "20261113-wb-resume", "e1-fullrows-v1"

SOURCES = {
    41: [
        (STUDY / "results" / "e1-seed41-affine-v1" / "report.json",
         f"{SR}/results/e1-seed41-affine-v1/report.json"),
        (RUNROOT / "results" / "e1-seed41-controls-v1" / "report.json",
         f"{RR}/results/e1-seed41-controls-v1/report.json"),
        (RUNROOT / "results" / "e1-residual-seed41-v1" / "report.json",
         f"{RR}/results/e1-residual-seed41-v1/report.json"),
    ],
    42: [
        (RUNROOT / "results" / "e1-seed42-multiseed-v1" / "report.json",
         f"{RR}/results/e1-seed42-multiseed-v1/report.json"),
        (RUNROOT / "results" / "e1-residual-seed42-v1" / "report.json",
         f"{RR}/results/e1-residual-seed42-v1/report.json"),
    ],
    43: [
        (RUNROOT / "results" / "e1-seed43-multiseed-v1" / "report.json",
         f"{RR}/results/e1-seed43-multiseed-v1/report.json"),
        (RUNROOT / "results" / "e1-residual-seed43-v1" / "report.json",
         f"{RR}/results/e1-residual-seed43-v1/report.json"),
    ],
}
ANN_REPORT = RUNROOT / "results" / "e1-ann-multiseed-v1" / "report.json"
ANN_REPORT_REL = f"{RR}/results/e1-ann-multiseed-v1/report.json"

LEGACY_MODES = ("centered", "first_analog", "generic",
                "dc_adapt_analog_head", "per_consumer", "cse_only")
RESIDUAL_MODES = ("dc_adapt_six_site", "first_analog_shared_theta",
                  "centered_shared_theta")
CONVERSION_CONTROLS = ("first_analog", "generic", "per_consumer", "cse_only",
                       "dc_adapt_analog_head", "dc_adapt_six_site",
                       "first_analog_shared_theta", "centered_shared_theta")
ALL_MODES = ("ANN",) + LEGACY_MODES + RESIDUAL_MODES
CANDIDATE = "centered"
MISSING_ROWS = ("complete_known_operation_cost", "matched_wall_clock_latency")

FIELDS = ("task_id", "agent_id", "run_id", "dataset", "split", "coverage",
          "method", "teacher_seed", "student_seed", "T", "metric", "value",
          "unit", "evidence", "protocol_sha256", "status")
METRICS = (("accuracy", "fraction"), ("teacher_agreement", "fraction"),
           ("logit_mse", "mse"), ("teacher_to_output_kl", "nats"))


def sample_sd(values):
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def main() -> None:
    per_seed, evidence, teacher_acc, npz_evidence = {}, {}, {}, {}
    for seed, paths in SOURCES.items():
        rows = {}
        for path, rel in paths:
            require(path.is_file(), f"missing input report: {rel}")
            doc = json.loads(path.read_text(encoding="utf-8"))
            require(doc.get("status") == "complete", f"incomplete report: {rel}")
            require(int(doc.get("graphs", -1)) == 5000, f"not full validation: {rel}")
            require(int(doc.get("seed", -1)) == seed, f"seed mismatch: {rel}")
            if doc.get("teacher_accuracy") is not None:
                teacher_acc[seed] = float(doc["teacher_accuracy"])
            for r in doc["rows"]:
                key = (r["mode"], int(r["T"]))
                require(key not in rows, f"duplicate row {key} for seed {seed}")
                rows[key] = {m: float(r[m]) for m, _ in METRICS}
                evidence[(seed, *key)] = rel
        for key in rows:
            mode, t = key
            if mode in RESIDUAL_MODES:
                npz_evidence[(seed, *key)] = (
                    f"{RR}/results/e1-residual-seed{seed}-v1/{mode}_T{t}.npz")
        per_seed[seed] = rows

    ann = json.loads(ANN_REPORT.read_text(encoding="utf-8"))
    require(ann.get("status") == "measured_upper_reference_row_only_not_E1_acceptance",
            "unexpected ANN row status")
    for r in ann["rows"]:
        seed, t = int(r["seed"]), int(r["T"])
        key = ("ANN", t)
        require(key not in per_seed[seed], f"duplicate ANN row for seed {seed}")
        per_seed[seed][key] = {m: float(r["metrics"][m]) for m, _ in METRICS}
        evidence[(seed, *key)] = ANN_REPORT_REL
        npz_evidence[(seed, *key)] = (
            f"{RR}/results/e1-ann-multiseed-v1/{r['npz']}")

    # Hard completeness gate: every (mode, T) on every seed must be present.
    expected = {(m, t) for m in ALL_MODES for t in (1, 2, 4, 8)}
    for seed in (41, 42, 43):
        missing = sorted(expected - set(per_seed[seed]))
        require(not missing, f"seed{seed} missing rows: {missing}")
    require(len(npz_evidence) == len(RESIDUAL_MODES) * 4 * 3 + 12,
            "unexpected residual/ANN archive count")

    summary = {
        "status": "descriptive_only_not_E1_acceptance",
        "dataset": "CIFAR10", "split": "all official validation", "coverage": 5000,
        "seeds": [41, 42, 43], "threshold_scale": 1,
        "protocol_sha256": PROTOCOL_SHA,
        "head_addendum_sha256": ADDENDUM_SHA,
        "teacher_accuracy_by_seed": teacher_acc,
        "row_families": {
            "ANN_upper_reference": ["ANN"],
            "legacy_conversion_rows": list(LEGACY_MODES),
            "new_residual_rows": list(RESIDUAL_MODES),
            "conversion_controls_for_reference": list(CONVERSION_CONTROLS),
            "candidate": CANDIDATE,
        },
        "per_sample_archives": 120,
        "rows": [], "criterion": {},
        "missing_required_rows": list(MISSING_ROWS),
        "cost_or_latency_evidence": False,
        "alias_warning": "cse_only/generic/per_consumer share a raw numerical trajectory and are not independent methods; ANN is an upper reference, not a conversion control; shared-theta rows are additional controls, not independent wins",
    }
    metric_rows = []
    for mode in ALL_MODES:
        for t in (1, 2, 4, 8):
            key = (mode, t)
            vals = [per_seed[s][key]["accuracy"] for s in (41, 42, 43)]
            entry = {
                "mode": mode, "T": t, "n_seeds": len(vals),
                "per_seed_accuracy": {str(s): per_seed[s][key]["accuracy"] for s in (41, 42, 43)},
                "mean_accuracy": sum(vals) / len(vals),
                "sample_sd_accuracy": sample_sd(vals),
                "per_seed_metrics": {str(s): per_seed[s][key] for s in (41, 42, 43)},
            }
            summary["rows"].append(entry)
            for s in (41, 42, 43):
                for metric, unit in METRICS:
                    metric_rows.append({
                        "task_id": TASK, "agent_id": AGENT, "run_id": RUN_ID,
                        "dataset": "CIFAR10", "split": "val",
                        "coverage": "5000_all_official_val", "method": mode,
                        "teacher_seed": s, "student_seed": "", "T": t,
                        "metric": metric, "value": per_seed[s][key][metric],
                        "unit": unit, "evidence": evidence[(s, *key)],
                        "protocol_sha256": PROTOCOL_SHA, "status": "measured"})
            for metric, value in (("accuracy_mean_3seed", sum(vals) / 3),
                                  ("accuracy_sample_sd_3seed", sample_sd(vals))):
                metric_rows.append({
                    "task_id": TASK, "agent_id": AGENT, "run_id": RUN_ID,
                    "dataset": "CIFAR10", "split": "val",
                    "coverage": "5000_all_official_val", "method": mode,
                    "teacher_seed": "41,42,43", "student_seed": "", "T": t,
                    "metric": metric, "value": value, "unit": "fraction",
                    "evidence": "aggregated from the per-seed reports listed above",
                    "protocol_sha256": PROTOCOL_SHA, "status": "derived"})

    acc_check = {}
    for t in (4, 8):
        by_mode = {e["mode"]: e["mean_accuracy"] for e in summary["rows"]
                   if e["T"] == t and e["n_seeds"] == 3}
        cand = by_mode[CANDIDATE]
        controls = {m: by_mode[m] for m in CONVERSION_CONTROLS}
        best_name = max(controls, key=controls.get)
        best_val = controls[best_name]
        gap = best_val - cand
        acc_check[f"T{t}"] = {
            "candidate": CANDIDATE, "centered_mean": cand,
            "all_conversion_controls_3seed_mean": controls,
            "strongest_matched_control": best_name,
            "strongest_control_mean": best_val,
            "gap_pp": gap * 100,
            "within_0.3pp_accuracy_only": gap <= 0.003,
            "cost_or_latency_condition": "NOT EVALUABLE (no complete known-operation cost or matched wall-clock evidence)",
        }
    summary["criterion"] = {
        "protocol": "BOTH T4 and T8 three-seed mean accuracy within 0.3pp of the strongest matched conversion control AND lower complete known-operation cost or matched wall-clock latency",
        "accuracy_only": acc_check,
        "verdict": "NOT EVALUABLE / E1 NOT PASSED",
        "reason": "ANN, historical six-site DC-adapt and shared-theta rows are now present, but complete known-operation cost and matched wall-clock latency evidence are still missing; the method-advantage criterion is a conjunction and cannot be evaluated; no three-seed significance or CI claim is made",
    }

    out = RUNROOT / "results" / "e1-fullrow-summary-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1_fullrow_summary_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    with (out / "METRICS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(metric_rows)
    print(json.dumps({
        "status": summary["status"],
        "n_metric_rows": len(metric_rows),
        "n_summary_rows": len(summary["rows"]),
        "per_sample_archives": summary["per_sample_archives"],
        "verdict": summary["criterion"]["verdict"],
        "T4": {k: acc_check["T4"][k] for k in ("centered_mean", "strongest_matched_control",
                                                "strongest_control_mean", "gap_pp",
                                                "within_0.3pp_accuracy_only")},
        "T8": {k: acc_check["T8"][k] for k in ("centered_mean", "strongest_matched_control",
                                                "strongest_control_mean", "gap_pp",
                                                "within_0.3pp_accuracy_only")},
        "missing_required_rows": list(MISSING_ROWS),
    }, indent=2))


if __name__ == "__main__":
    main()
