"""Build the missing E1 ANN row for seeds 41/42/43, T in {1,2,4,8}.

No GPU, no dataset pass, no teacher forward. The ANN row is the frozen teacher
ANN logits that are already stored as `teacher_logits` in every existing
per-sample E1 NPZ. This script only selects those arrays, checks their
consistency against the frozen canonical baseline and done.json, and writes
per-sample ANN evidence plus fixed-column metrics.

ANN is reported as an UPPER REFERENCE, never as a conversion control. The E1
accuracy reference remains the strongest matched conversion control.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = STUDY.parents[2]
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
SOURCE = ROOT / "research" / "visual-graphs" / "takeover_20261112"
OUT = RUNROOT / "results" / "e1-ann-multiseed-v1"
CANON = RUNROOT / "inputs" / "canonical-baseline-val.npz"
TEACHERS = RUNROOT / "clean-restore-v1" / "research" / "visual-graphs" / "takeover_20261112" / "full_teacher_runs"

EXPECTED_CANON = "7c0727a2902bb2c05a0ec4930734b0bfa05e0905c4b86a6c976155a0c702fcc3"
EXPECTED_PROTOCOL = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
EXPECTED_ACC = {41: 0.7066, 42: 0.703, 43: 0.7074}
TIMESTEPS = (1, 2, 4, 8)
METRIC_NAMES = ("accuracy", "teacher_agreement", "logit_mse", "teacher_to_output_kl")
METRIC_UNITS = {"accuracy": "fraction", "teacher_agreement": "fraction",
                "logit_mse": "mse", "teacher_to_output_kl": "kl"}
CSV_COLUMNS = ("task_id", "agent_id", "run_id", "dataset", "split", "coverage",
               "method", "teacher_seed", "student_seed", "T", "metric", "value",
               "unit", "evidence", "protocol_sha256", "status")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def load_frozen_metric_row():
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("frozen_affine_eval", CODE / "eval_affine_validation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["frozen_affine_eval"] = mod
    spec.loader.exec_module(mod)
    return mod.metric_row


def source_npz(seed: int) -> list[Path]:
    if seed == 41:
        return sorted((STUDY / "results" / "e1-seed41-affine-v1").glob("*.npz")) + \
               sorted((RUNROOT / "results" / "e1-seed41-controls-v1").glob("*.npz"))
    return sorted((RUNROOT / "results" / f"e1-seed{seed}-multiseed-v1").glob("*.npz"))


def main() -> None:
    require(not OUT.exists(), f"refusing existing output dir: {OUT}")
    require(sha256(CANON) == EXPECTED_CANON, "canonical baseline hash drift")
    require(sha256(STUDY / "PROTOCOL-v3.md") == EXPECTED_PROTOCOL, "protocol hash drift")
    metric_row = load_frozen_metric_row()

    with np.load(CANON, allow_pickle=False) as z:
        canon = {k: z[k].copy() for k in z.files}
    OUT.mkdir(parents=True)

    report = {
        "status": "measured_upper_reference_row_only_not_E1_acceptance",
        "method": "ANN",
        "role": "upper reference; NOT a conversion control and NOT the E1 accuracy comparator",
        "dataset": "CIFAR10",
        "split": "all official validation",
        "coverage": 5000,
        "seeds": [41, 42, 43],
        "timesteps": list(TIMESTEPS),
        "protocol_sha256": EXPECTED_PROTOCOL,
        "canonical_baseline_sha256": EXPECTED_CANON,
        "no_gpu_job": True,
        "no_new_teacher_forward": True,
        "per_seed": {},
        "rows": [],
    }
    metric_lines: list[dict] = []
    seed41_teacher = None

    for seed in (41, 42, 43):
        paths = source_npz(seed)
        require(len(paths) == 24, f"seed{seed}: expected 24 source NPZ, got {len(paths)}")
        with np.load(paths[0], allow_pickle=False) as z:
            ref = {k: z[k].copy() for k in ("indices", "labels", "teacher_logits")}
        for p in paths:
            with np.load(p, allow_pickle=False) as z:
                require(set(z.files) == {"indices", "labels", "teacher_logits", "logits"},
                        f"unexpected NPZ keys in {p.name}")
                require(np.array_equal(z["indices"], ref["indices"]), f"index drift in {p.name}")
                require(np.array_equal(z["labels"], ref["labels"]), f"label drift in {p.name}")
                require(np.array_equal(z["teacher_logits"], ref["teacher_logits"]),
                        f"teacher logit drift in {p.name}")
        require(np.array_equal(ref["indices"], canon["indices"]), f"seed{seed} canonical index drift")
        require(np.array_equal(ref["labels"], canon["labels"]), f"seed{seed} canonical label drift")

        done_p = TEACHERS / (f"gated_cifar10_seed{seed}_full_v1" if seed == 41
                             else f"gated_cifar10_seed{seed}_full_v2") / "done.json"
        done = json.loads(done_p.read_text(encoding="utf-8"))
        expected_acc = float(done["best_val_accuracy"])
        require(abs(expected_acc - EXPECTED_ACC[seed]) < 1e-12, f"seed{seed} done.json accuracy drift")
        acc = float(np.mean(ref["teacher_logits"].argmax(1) == ref["labels"]))
        require(abs(acc - expected_acc) < 1e-12, f"seed{seed} ANN accuracy != frozen done.json")

        if seed == 41:
            require(np.array_equal(ref["teacher_logits"], canon["teacher_logits"]),
                    "seed41 teacher logits differ from canonical")
            require(np.array_equal(ref["teacher_logits"], canon["ann_T4"]),
                    "seed41 teacher logits differ from canonical ann_T4")
            require(np.array_equal(ref["teacher_logits"], canon["ann_T8"]),
                    "seed41 teacher logits differ from canonical ann_T8")
            seed41_teacher = ref["teacher_logits"]
        else:
            require(not np.array_equal(ref["teacher_logits"], seed41_teacher),
                    f"seed{seed} teacher logits falsely identical to seed41")

        seed_rows = []
        for t in TIMESTEPS:
            npz = OUT / f"ann_seed{seed}_T{t}.npz"
            np.savez(npz, indices=ref["indices"], labels=ref["labels"],
                     teacher_logits=ref["teacher_logits"], logits=ref["teacher_logits"].copy())
            m = metric_row(ref["teacher_logits"], ref["teacher_logits"], ref["labels"])
            row = {"seed": seed, "T": t, "npz": npz.name, "npz_sha256": sha256(npz),
                   "metrics": {k: m[k] for k in METRIC_NAMES},
                   "logits_equal_teacher_logits": True,
                   "timestep_independent": True}
            seed_rows.append(row)
            report["rows"].append(row)
            for name in METRIC_NAMES:
                metric_lines.append({
                    "task_id": "visual-conversion", "agent_id": "20261113-wb-resume",
                    "run_id": "e1-ann-v1", "dataset": "CIFAR10", "split": "val",
                    "coverage": "5000_all_official_val", "method": "ANN",
                    "teacher_seed": str(seed), "student_seed": "", "T": str(t),
                    "metric": name, "value": repr(m[name]), "unit": METRIC_UNITS[name],
                    "evidence": str(npz.relative_to(ROOT)).replace("\\", "/"),
                    "protocol_sha256": EXPECTED_PROTOCOL,
                    "status": "measured_from_frozen_per_sample_teacher_logits",
                })
        report["per_seed"][str(seed)] = {
            "source_npz_count": len(paths),
            "source_npz_sha256": {p.name: sha256(p) for p in paths},
            "frozen_val_accuracy": expected_acc,
            "recomputed_accuracy": acc,
            "canonical_indices_labels_bitwise": True,
            "teacher_logits_identical_across_all_source_npz": True,
            "teacher_logits_differ_from_seed41": (seed != 41),
            "rows_written": len(seed_rows),
        }

    report["missing_required_rows_remaining"] = [
        "historical_six_site_DC_adapt", "shared_theta_controls",
        "complete_known_operation_cost", "matched_wall_clock_latency",
    ]
    report["criterion"] = {
        "verdict": "NOT EVALUABLE / E1 NOT PASSED",
        "reason": "ANN is an upper reference only; complete cost and matched wall-clock evidence still missing",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "METRICS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(metric_lines)

    print(json.dumps({
        "status": "PASS",
        "seeds": [41, 42, 43],
        "timesteps": list(TIMESTEPS),
        "npz_written": len(report["rows"]),
        "metric_rows": len(metric_lines),
        "frozen_teacher_accuracies": EXPECTED_ACC,
        "canonical_verified": True,
        "E1_verdict": report["criterion"]["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
