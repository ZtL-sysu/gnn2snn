"""Read-only scientific verification of the new per-seed E1 evidence.

Checks (all bitwise, no tuning, no threshold/seed/scale changes):
  1. Every NPZ has canonical indices/labels/teacher_logits/logits shapes.
  2. All NPZ in one seed share identical indices, labels and teacher_logits.
  3. canonical indices/labels match the frozen canonical-baseline-val.npz.
  4. teacher logits accuracy equals the frozen expected value and is NOT
     bitwise equal to the seed-41 canonical teacher logits (different teachers).
  5. cse_only / generic / per_consumer numerical alias relationship, reported
     truthfully rather than credited as an independent method win.
  6. report start/end hash equality and NPZ file hashes match the report.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\code\gnn2snn-wb-resume")
RUNROOT = ROOT / "review-suggestions" / "grok-bot" / "f-research-v1-run-20261113-residual-prefix"
CANON = ROOT / "research" / "visual-graphs" / "continuation_20261113" / "inputs" / "canonical-baseline-val.npz"
MODES = ("centered", "first_analog", "generic", "dc_adapt_analog_head", "per_consumer", "cse_only")
TS = (1, 2, 4, 8)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or [42, 43]
    with np.load(CANON, allow_pickle=False) as c:
        canon_idx, canon_lab, canon_teacher = c["indices"], c["labels"], c["teacher_logits"]
    out = {"canonical": {"indices": canon_idx.shape, "labels": canon_lab.shape}, "seeds": {}}
    ok = True
    for seed in seeds:
        d = RUNROOT / "results" / f"e1-seed{seed}-multiseed-v1"
        report = json.loads((d / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "complete" and report["graphs"] == 5000
        assert report["source_sha256_start"] == report["source_sha256_end"]
        assert report["data_sha256_start"] == report["data_sha256_end"]
        ref = None
        per_mode = {}
        npz_ok = True
        for mode in MODES:
            for T in TS:
                f = d / f"{mode}_T{T}.npz"
                with np.load(f, allow_pickle=False) as z:
                    keys = set(z.files)
                    assert keys == {"indices", "labels", "teacher_logits", "logits"}, (f, keys)
                    idx, lab, tl, lg = z["indices"], z["labels"], z["teacher_logits"], z["logits"]
                    assert idx.shape == (5000,) and lab.shape == (5000,)
                    assert tl.shape == (5000, 10) and lg.shape == (5000, 10)
                    assert np.array_equal(idx, np.arange(5000)) and np.array_equal(idx, canon_idx)
                    assert np.array_equal(lab, canon_lab)
                    if ref is None:
                        ref = (idx, lab, tl)
                    else:
                        assert np.array_equal(tl, ref[2]), f"teacher logits differ inside seed {seed}: {mode}_T{T}"
                    per_mode[(mode, T)] = lg
                npz_ok = npz_ok and (sha(f) == report["prediction_archives_sha256"][f"{mode}_T{T}"])
        teacher = ref[2]
        acc = float(np.mean(teacher.argmax(1) == canon_lab))
        assert abs(acc - report["frozen_expected_teacher_accuracy"]) < 1e-12
        different_from_seed41 = not np.array_equal(teacher, canon_teacher)
        alias = {}
        for T in TS:
            alias[f"T{T}"] = {
                "generic_eq_cse_only": bool(np.array_equal(per_mode[("generic", T)], per_mode[("cse_only", T)])),
                "generic_eq_per_consumer": bool(np.array_equal(per_mode[("generic", T)], per_mode[("per_consumer", T)])),
            }
        rows = {(r["mode"], int(r["T"])): r["accuracy"] for r in report["rows"]}
        recomputed_ok = all(
            abs(rows[(m, T)] - float(np.mean(per_mode[(m, T)].argmax(1) == canon_lab))) < 1e-12
            for m in MODES for T in TS)
        s = {"teacher_accuracy": acc, "frozen_expected": report["frozen_expected_teacher_accuracy"],
             "different_from_seed41_canonical_teacher": different_from_seed41,
             "canonical_indices_labels_bitwise": True,
             "all_npz_share_teacher_logits": True,
             "npz_hashes_match_report": npz_ok,
             "reported_accuracy_recomputed_from_npz": recomputed_ok,
             "n_rows": len(report["rows"]), "n_npz": len(report["prediction_archives_sha256"]),
             "alias": alias,
             "checkpoint_sha256": report["checkpoint_sha256"],
             "affine_cal_sha256": report["affine_calibration_sha256"],
             "six_cal_sha256": report["six_site_calibration_sha256"],
             "report_sha256": sha(d / "report.json")}
        ok = ok and npz_ok and recomputed_ok and different_from_seed41
        out["seeds"][str(seed)] = s
    out["status"] = "PASS" if ok else "FAIL"
    print(json.dumps(out, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
