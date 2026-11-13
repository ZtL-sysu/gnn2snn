# Affine calibration and fidelity artifact audit v1

## Verdict

No numerical or provenance blocker found in the requested artifacts. The seed-41 affine calibration JSON is complete and internally consistent across all 14 sites and both passes. Independent CPU recomputation from the six fixed-GCN and three GatedGCN test NPZs reproduces the saved summary/figure accuracies and teacher-seed mean/sample-SD values. The plot does not display numerically inflated error bars, but its visible caption/legend omits that the official tests were previously exposed and that the SD is across three teacher seeds (not a confidence interval); add that qualifier when presenting the figure.

## Calibration artifact

`affine_cal1.json` reports CIFAR10 seed-41, label-free two-pass calibration over all 45,000 official training graphs: 352 batches per pass, final batch 72. Both passes have identical coverage totals: 5,294,119 node rows and 42,352,952 directed-edge rows. All 14 expected sites are present: `raw_node/raw_edge_{0..3}` and `branch_cumsum_node/branch_cumsum_edge_{1..3}`.

For every site, the stored per-channel histogram is 70×512, each channel sums to its recorded row count, and the global histogram has 512 bins summing to rows×70. I independently recomputed every per-channel threshold from the .999 quantile bin and recorded maximum using the 1e-6 floor; all matched. The global threshold similarly matches its global histogram and global maximum at all sites. Counts, maxima and thresholds are finite and nonnegative/positive as required. The two pass row maps agree.

The start/end input hash maps are identical (18 bound files); every bound file currently exists and hashes to its recorded SHA-256. This covers calibration dependencies/protocol, seed-41 teacher metadata/done/checkpoint and declared model sources, plus the processed CIFAR10 train inputs. The teacher checkpoint hash matches both `best.pt` and `done.json`; its recorded metadata matches the loaded run metadata for seed 41. The training split is intentionally hashed as provenance but was not loaded by this audit.

## Independent test-logit recomputation

The six fixed-GCN NPZs each contain 10,000 ordered test rows, labels, teacher logits and T1/T2/T4/T8 logits. The three seed archives within each dataset have identical indices and labels. Recomputed teacher and candidate top-1 rates match every displayed entry in `fixed-gcn-test-three-seed-summary-v1.json` exactly at stored precision; all checked logits are finite. Summary input hashes for all 12 source JSON/NPZ files match the current files.

| Dataset | Teacher seed | ANN teacher | T4 | T8 |
|---|---:|---:|---:|---:|
| MNIST | 41 | 90.94% | 90.94% | 90.93% |
| MNIST | 42 | 91.32% | 91.34% | 91.31% |
| MNIST | 43 | 90.81% | 90.86% | 90.81% |
| CIFAR10 | 41 | 54.87% | 54.88% | 54.93% |
| CIFAR10 | 42 | 54.94% | 54.86% | 54.95% |
| CIFAR10 | 43 | 55.01% | 55.01% | 55.02% |

The three GatedGCN CIFAR10 test NPZs also have 10,000 rows each, with identical ordered indices/labels across teachers and finite teacher/T4/T8 logits. Recomputed accuracies are:

| Teacher seed | ANN teacher | T4 | T8 |
|---:|---:|---:|---:|
| 41 | 69.20% | 69.10% | 69.16% |
| 42 | 68.39% | 68.44% | 68.38% |
| 43 | 68.72% | 68.71% | 68.73% |

All seven input hashes recorded in `figures_v5/teacher_fidelity_three_seeds_v1.json` (fixed-GCN summary plus three GatedGCN JSON/NPZ pairs) match current files. Its three panel per-teacher accuracy vectors, ANN values, means and `ddof=1` sample SDs reproduce exactly from the NPZs. These are same-test teacher-seed comparisons; the fixed-GCN summary labels the scope `descriptive_exposed_full_test`, and the GatedGCN result records prior test exposure.

## Figure review

Inspected `figures_v5/teacher_fidelity_three_seeds_v1.png`. Axes, timestep ticks, dataset/architecture panel labels, and accuracy units are legible. Black mean lines/error bars correspond to JSON conversion means and sample SDs; the three fixed-GCN per-teacher trajectories are shown separately, while the GatedGCN panel shows its two timestep points. The error bars are not mislabeled as confidence intervals in the figure (legend says “mean ± SD”) and are not numerically inflated.

The image itself does not state (1) these full official test sets were previously exposed, or (2) that SD is calculated across the three teacher seeds on one shared test split, not across data resamples and not a confidence interval. The JSON `qualification` records the test exposure and the intended descriptive interpretation, but a standalone figure can be separated from its JSON. Add a visible caption/footnote: “Historical, previously exposed official test sets; bars are sample SD across three teacher seeds (not CI); architectures are separate.” This is an annotation limitation, not a metrics blocker.

## Evidence files

- Calibration: `affine_cal1.json`.
- Fixed-GCN test aggregation: `fixed-gcn-test-three-seed-summary-v1.json`; test predictions `dc_mn_test1.npz`, `rep/M42/dc_test_v3.npz`, `rep/M43/dc_test_v3.npz`, `dc_cf_test1.npz`, `rep/C42/dc_test_v3.npz`, `rep/C43/dc_test_v3.npz`.
- GatedGCN predictions: `gd_test1.npz`, `g42test.npz`, `g43test.npz` and their corresponding JSON files.
- Figure data and render: `figures_v5/teacher_fidelity_three_seeds_v1.json` and `figures_v5/teacher_fidelity_three_seeds_v1.png`.

No official dataset was opened, no GPU used, no source/results edited, and the four remaining affine profile outputs are outside this audit.
