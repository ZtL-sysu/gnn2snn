# Two teacher-seed evidence summary (CIFAR10)

Read-only aggregation of the saved full validation and test prediction artifacts. Accuracies and teacher agreement below were recomputed on CPU in Conda `dyg-cold` from NPZ `labels`, `teacher_logits`, `dc_T4`, and `dc_T8` via argmax; they match the JSON summaries. Values are percentages. “Teacher − SNN” is the accuracy difference in percentage points (pp).

| Teacher seed | Split (N) | ANN teacher | DC T4 | Teacher − T4 | DC T8 | Teacher − T8 |
|---:|---|---:|---:|---:|---:|---:|
| 41 | Validation (5,000) | 70.66 | 70.72 | −0.06 | 70.66 | 0.00 |
| 41 | Test (10,000) | 69.20 | 69.10 | +0.10 | 69.16 | +0.04 |
| 42 | Validation (5,000) | 70.30 | 70.36 | −0.06 | 70.32 | −0.02 |
| 42 | Test (10,000) | 68.39 | 68.44 | −0.05 | 68.38 | +0.01 |

| Teacher seed | Split | DC T4 agreement with its ANN teacher | DC T8 agreement with its ANN teacher |
|---:|---|---:|---:|
| 41 | Validation | 99.62% | 99.84% |
| 41 | Test | 99.70% | 99.90% |
| 42 | Validation | 99.52% | 99.88% |
| 42 | Test | 99.38% | 99.87% |

## Integrity and provenance

- The seed-41 artifacts are `gd_val1.json/.npz` and `gd_test1.json/.npz`; JSON records teacher checkpoint SHA-256 `d26df76d724872edc91ea11055d114935ea173282a8ef049d1b85cc1f5598b40`, epoch 34, and calibration SHA-256 `b28485ddfa8ef259ca048f3272662fedc1e4f519b0324cfa514ccf82b2cf8a9e`. Their result JSON lacks a direct `teacher_seed` field; root verified seed 41 explicitly in the bound teacher metadata and calibration `teacher_metadata.seed`, with matching checkpoint hash.
- The seed-42 artifacts are `g42val.json/.npz`, `g42test.json/.npz`, and `g42cal.json`. The full-training run is `full_teacher_runs/gated_cifar10_seed42_full_v2/`; `metadata.json` says teacher initialization and train-order RNG seed 42, with student seed reference 41 explicitly separate and unchanged. `done.json` records finished, 98 epochs completed, 45,000 train / 5,000 validation, best validation accuracy 0.703, and best checkpoint SHA-256 `0ad3f3146f94bf73f5f398ef44a1fbc46f0c5924332c4d8fca3d711d0fbff1b6`. The on-disk `best.pt` hash matches. `g42cal.json` records complete calibration on 45,000 official training graphs, label-free, fixed order, and the same checkpoint hash.
- Seed-41 and seed-42 processed-data hashes in the evaluation JSONs agree for each split source: `train_data.pt` `b0755438ef38d1fe3f69d75ef61acc06d9334c2f35e490d4816886db163ee060`; `val_data.pt` `e5c407f4fa7f3530e04101a7832cf66687df7224a0dfbe792efb8ce693373dce`; `test_data.pt` `04d71270557759c939b783d33cf69303834388d287a92b17792d9da24f3725d9`. CPU comparison found identical ordered `indices` and `labels` across seeds for all 5,000 validation and 10,000 test examples.
- NPZ SHA-256: `gd_val1.npz` `473d88de22c46dce7fede61df1d6a8abdc7bdd682abd4a64704f978d28f5f177`; `gd_test1.npz` `19e039d0a42c2fb7949b572811ed59b3bfe075f616587a61aba77cbd5cc37ec1`; `g42val.npz` `b8e1c55f436c861cb10b2da78e1692c0edcfcf4c042f09981c8e4ec82a3dacfc`; `g42test.npz` `4e141c1222880449f6c358c7d9b8ed8e550eeaa0f5398175c7a7f6f6b4fe2e77`. The JSON `predictions_sha256` values match those file hashes. Both seeds use the same pinned `gated_differential_hybrid_v1.py` SHA-256 `66a8a3060a3888cc2ffe4e7daa452992758c9288f4463cd1fb28d5f0a14a7a1e` and same `dcgs_fixed_residual_gcn_event_forward_v1.py` SHA-256 `a031ebdbc816cbde28af3f919284b08e5ba087e271415b1bedec37306c9bc541`; their teacher checkpoint and calibration hashes differ as expected.

## Scope and interpretation

This is a two-teacher-seed robustness check, varying teacher parameter initialization and training order; these DC adaptations perform no student-training phase or student-seed sweep. The separate learned-initialization trials use student RNG 41. Two seeds do not support a significance claim. The converted path is a project adaptation with signed graded source events (eight levels), analog edge calculations, and dense analog controllers at each tick; it is not a binary-spike or network-efficiency result. The source notes no teacher hidden-state oracle. Validation exposure is recorded for both, and test artifacts explicitly record prior test exposure; treat test numbers as historical, exposed evidence rather than a fresh held-out estimate. These tables do not establish a final score or a new method claim. Raw JSON/NPZ/checkpoint/calibration artifacts were left unchanged.
