# Gated CIFAR10 historical baseline audit

**Status: VERIFIED archived artifacts; no live reproduction or new evaluation.** Accuracy, teacher agreement, and MSE were recomputed from saved logits. All six validation/test sidecars and NPZ hashes, six checkpoint/calibration links, checkpoint completion hashes, 54 recorded source-file hashes, and three independent teacher test archives were checked.

The requested `gated-three-teacher-seed-summary-v1.json` is absent from both the worktree and source archive; its Markdown summary is available and included in the evidence manifest. Raw artifacts contain differential gated-hybrid outputs at T4/T8 only; T1/T2 outputs are not present.

## Recomputed endpoint accuracy / own-teacher agreement

| Split | Seed | ANN teacher accuracy | Differential T4 accuracy / agreement | Differential T8 accuracy / agreement |
|---|---:|---:|---:|---:|
| val | 41 | 70.66% | 70.72% / 99.62% | 70.66% / 99.84% |
| val | 42 | 70.30% | 70.36% / 99.52% | 70.32% / 99.88% |
| val | 43 | 70.74% | 70.68% / 99.42% | 70.84% / 99.74% |
| val | mean ± sample SD | 70.57% ± 0.23 | 70.59% ± 0.20% / 99.52% ± 0.10 | 70.61% ± 0.26% / 99.82% ± 0.07 |
| test | 41 | 69.20% | 69.10% / 99.70% | 69.16% / 99.90% |
| test | 42 | 68.39% | 68.44% / 99.38% | 68.38% / 99.87% |
| test | 43 | 68.72% | 68.71% / 99.57% | 68.73% / 99.92% |
| test | mean ± sample SD | 68.77% ± 0.41 | 68.75% ± 0.33% / 99.55% ± 0.16 | 68.76% ± 0.39% / 99.90% ± 0.03 |

## Provenance and interpretation

Checkpoint hashes match each seed’s teacher `done.json`; each calibration digest matches the archived calibration, which is recorded as label-free and fit over two passes of all 45,000 official training graphs. Saved test teacher logits, labels, and ordered indices exactly match the independent teacher prediction archive for each seed. The seed-41 evaluator JSON omits an explicit teacher-seed field; linkage is through run metadata/checkpoint/calibration.

The converted model is a dense analog gate/controller hybrid using the project MTH differential conversion for event-carrying paths. It is not an all-spiking network or a full reproduction of the authors’ conversion pipeline. All validation and test sets have prior project exposure. The test rows are historical descriptive endpoints, not independent confirmation; the three-seed spread is teacher-seed sensitivity, not student-training robustness. The small seed-wise gaps do not establish superiority, efficiency, energy reduction, novelty, or sparse execution.
