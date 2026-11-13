# CIFAR10 GatedGCN: three-teacher-seed summary

Read-only CPU aggregation of saved logits for the full-data CIFAR10 GatedGCN adaptation. Accuracies and teacher agreement were recomputed from the per-example NPZs with `dyg-cold`; no official dataset was loaded. Each split has the complete ordered official evaluation set (5,000 validation or 10,000 test graphs). Every validation/test set and prediction archive has prior exposure; test results are historical/descriptive, not independent confirmation.

## Recomputed accuracy

Percentages; **teacher gap = ANN teacher accuracy − converted accuracy** in percentage points. A negative gap means the converted result is numerically higher on that split.

| Teacher seed | Split (N) | ANN teacher | Differential T4 | Gap T4 (pp) | Differential T8 | Gap T8 (pp) |
|---:|---|---:|---:|---:|---:|---:|
| 41 | Validation (5,000) | 70.66 | 70.72 | −0.06 | 70.66 | 0.00 |
| 42 | Validation (5,000) | 70.30 | 70.36 | −0.06 | 70.32 | −0.02 |
| 43 | Validation (5,000) | 70.74 | 70.68 | +0.06 | 70.84 | −0.10 |
| **Mean ± sample SD** | Validation | **70.57 ± 0.23** | **70.59 ± 0.20** | **−0.02 ± 0.07** | **70.61 ± 0.26** | **−0.04 ± 0.05** |
| 41 | Test (10,000) | 69.20 | 69.10 | +0.10 | 69.16 | +0.04 |
| 42 | Test (10,000) | 68.39 | 68.44 | −0.05 | 68.38 | +0.01 |
| 43 | Test (10,000) | 68.72 | 68.71 | +0.01 | 68.73 | −0.01 |
| **Mean ± sample SD** | Test | **68.77 ± 0.41** | **68.75 ± 0.33** | **+0.02 ± 0.08** | **68.76 ± 0.39** | **+0.01 ± 0.03** |

The descriptive accuracy means/sample SDs are across **three teacher seeds**, not confidence intervals. Per-seed differences are small and change sign; no method superiority or significance claim is warranted.

| Split | T4 agreement with own ANN teacher (seed 41 / 42 / 43) | T8 agreement with own ANN teacher (seed 41 / 42 / 43) | Mean ± sample SD, T4 / T8 |
|---|---|---|---|
| Validation | 99.62 / 99.52 / 99.42% | 99.84 / 99.88 / 99.74% | 99.52 ± 0.10 / 99.82 ± 0.07% |
| Test | 99.70 / 99.38 / 99.57% | 99.90 / 99.87 / 99.92% | 99.55 ± 0.16 / 99.90 ± 0.03% |

## Integrity and provenance checks

- For all six evaluation NPZs, per-example rows have the expected 5,000/10,000 counts and ten logits; `indices` equal the canonical ordered full split and labels/indices match across seeds within each split. JSON accuracies and teacher-agreement values agree with independent `argmax` recomputation from the NPZ logits.
- All six JSON `predictions_sha256` values match their NPZ files. I checked **54 recorded project-source hash entries** against the current local files; all match. The saved author-source provenance names pinned commit `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be`; the external `neurons/relu_neuron.py` SHA-256 matches `ae9018fb5d75d97d606f4f1e40075a5341fe166ab90772aecdebceb7dcc89851`.
- Each evaluation JSON checkpoint hash matches its seed’s `best.pt`; checkpoint hashes also match the teacher `done.json`. The seed-specific calibration hash matches the file: seed 41 `b28485dd…2cf8a9e`, seed 42 `b86ca53e…b8cb6f2b`, seed 43 `31471fc1…b8d582b8e`. Each calibration is complete, label-free, and records two passes over all 45,000 official training graphs with the corresponding teacher checkpoint.
- For each seed’s test split, the GatedGCN evaluator’s teacher archive hash matches the JSON; ordered indices, labels, teacher argmax predictions and teacher logits align with the independently saved full-test teacher archive. The three metadata files identify seeds 41, 42 and 43, respectively. Seed-41 result JSONs omit a direct `teacher_seed` field, so seed identity is tied through metadata, checkpoint and calibration provenance.
- Evidence paths: seed 41 `gd_val1.json/.npz`, `gd_test1.json/.npz`; seed 42 `g42val.json/.npz`, `g42test.json/.npz`, `g42cal.json`; seed 43 `g43val.json/.npz`, `g43test.json/.npz`, `g43cal.json`. Teacher archives/checkpoints are in `full_teacher_runs/gated_cifar10_seed{41,42,43}_full_v{1,2,2}/`.

## Interpretation and boundaries

This is a teacher-seed sensitivity summary, not a student-seed robustness study: the converted forward has no student-training phase, and the student RNG was not varied. Across these three frozen CIFAR10 GatedGCN teachers, the project differential adaptation stays within 0.10 pp of teacher accuracy at T4/T8 on the recorded full validation and historically exposed test sets. This supports narrow fidelity feasibility for these teachers/settings, not a new conversion-method advantage.

The converter adapts the established pinned MTH differential unit; its signed graded eight-level events are not binary spikes. The tested GatedGCN path recomputes dense analog gate/edge/controller work at each tick. These results do not show sparse execution, end-to-end T-step wall latency, energy reduction, operator-agnostic scope or novelty. Do not generalize a shared-dataset test result across seeds into independent test confirmation. The two-seed predecessor summary and broader claim boundary are in [gated-two-teacher-seed-summary-v1.md](gated-two-teacher-seed-summary-v1.md) and [gated-result-to-claim-v1.md](gated-result-to-claim-v1.md).
