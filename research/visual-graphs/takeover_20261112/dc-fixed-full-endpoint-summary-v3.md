# Fixed-GCN DC endpoint summary (three teacher seeds)

## Scope and integrity

This CPU aggregation recomputes all accuracies, teacher-class agreement, logit MSE, and endpoint comparisons from saved prediction logits/labels/indices only. It loads no official dataset and runs no inference. This is the full-data four-layer width-146 fixed residual GCN project adaptation; it is separate from the historical narrow GCN and from the strong GatedGCN track.

The six new endpoint sidecars (M42/M43/C42/C43 full validation and seed-41 full test on MNIST/CIFAR10) all report `complete`, expected 5,000/10,000 graphs, matching input-hash start/end maps, and NPZ SHA-256 values matching the actual archive files. All share one source-v3/script/model/oracle/protocol hash set. The two original seed-41 full-validation archives also match their recorded prediction hashes. For each dataset, the three validation archives have identical indices 0–4,999 and labels; both test archives have unique ordered indices 0–9,999, with labels and indices exactly matching the stored teacher-reference archive. All arrays are finite and logits are 10-class.

On full test, recomputed seed-41 teacher logits match the independently stored teacher-reference logits within the recorded tolerance: MNIST max absolute difference 7.44e-5 (tolerance 4.97e-4); CIFAR10 6.20e-6 (tolerance 2.45e-4). Test reference accuracy recomputes to 90.94% MNIST and 54.87% CIFAR10. Both test sidecars explicitly mark the official test as historically exposed.

## Full validation: per-seed accuracy and teacher-relative gap

Accuracy is top-1 percent; each cell is `teacher / T1 / T2 / T4 / T8`, with candidate values recomputed from logits. Parentheses give candidate-minus-that-seed's teacher accuracy in percentage points.

| Dataset | Teacher seed | Teacher | T1 (gap) | T2 (gap) | T4 (gap) | T8 (gap) |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 41 | 92.38 | 75.00 (−17.38) | 91.32 (−1.06) | 92.36 (−0.02) | 92.40 (+0.02) |
| MNIST | 42 | 92.18 | 72.06 (−20.12) | 90.94 (−1.24) | 92.10 (−0.08) | 92.18 (0.00) |
| MNIST | 43 | 91.98 | 68.92 (−23.06) | 90.74 (−1.24) | 91.94 (−0.04) | 92.00 (+0.02) |
| CIFAR10 | 41 | 55.58 | 46.32 (−9.26) | 54.74 (−0.84) | 55.50 (−0.08) | 55.52 (−0.06) |
| CIFAR10 | 42 | 55.96 | 48.80 (−7.16) | 55.68 (−0.28) | 56.12 (+0.16) | 55.98 (+0.02) |
| CIFAR10 | 43 | 56.92 | 47.66 (−9.26) | 56.52 (−0.40) | 56.96 (+0.04) | 56.94 (+0.02) |

Descriptive seed mean ± sample SD (percentage points; only three teacher seeds):

| Dataset | Teacher | T1 | T2 | T4 | T8 |
|---|---:|---:|---:|---:|---:|
| MNIST accuracy | 92.18 ± 0.20 | 71.99 ± 3.04 | 91.00 ± 0.29 | 92.13 ± 0.21 | 92.19 ± 0.20 |
| MNIST candidate − teacher | — | −20.19 ± 2.84 | −1.18 ± 0.10 | −0.047 ± 0.031 | +0.013 ± 0.012 |
| CIFAR10 accuracy | 56.15 ± 0.69 | 47.59 ± 1.24 | 55.65 ± 0.89 | 56.19 ± 0.73 | 56.15 ± 0.72 |
| CIFAR10 candidate − teacher | — | −8.56 ± 1.21 | −0.507 ± 0.295 | +0.040 ± 0.120 | −0.007 ± 0.046 |

Teacher-class agreement across those same validation sets is 99.70–99.74% at T4 and 99.82–99.98% at T8 for both datasets. For context on the scale of logit convergence, candidate-to-teacher logit MSE at T4 is 0.00311/0.00360/0.00378 for MNIST seeds 41/42/43 and 0.000651/0.000720/0.002261 for CIFAR10; at T8 it is 0.000224/0.000224/0.000237 and 0.0000444/0.0000354/0.000420, respectively.

## Full test: seed 41 only

| Dataset | Teacher | T1 (gap; agreement) | T2 (gap; agreement) | T4 (gap; agreement) | T8 (gap; agreement) |
|---|---:|---:|---:|---:|---:|
| MNIST | 90.94 | 74.46 (−16.48; 76.75%) | 90.15 (−0.79; 96.38%) | 90.94 (0.00; 99.75%) | 90.93 (−0.01; 99.98%) |
| CIFAR10 | 54.87 | 45.50 (−9.37; 62.93%) | 53.87 (−1.00; 90.25%) | 54.88 (+0.01; 99.53%) | 54.93 (+0.06; 99.79%) |

No converted SNN full-test result is present for seeds 42 or 43. Their full-validation scores above are not new test evidence. The official validation and seed-41 test archives have prior exposure; treat all these metrics as descriptive and do not use them to retune or select settings.

## Method label and interpretation

These runs use the project’s fixed-GCN adapter with the pinned source `MTH_diff` ReLU-branch coding and project histogram-calibrated per-site thresholds. They do **not** run the authors’ full DCGS converter or its ReLU threshold-iteration procedure. This is a fidelity endpoint for that adapter: T4/T8 reproduce teacher accuracy closely across these seeds and test archives, while T1/T2 retain larger deficits. It says nothing by itself about energy, sparse-kernel speed, fivefold cost, novelty, or statistical significance. Three teacher seeds share the same datasets and a common source/protocol; mean ± sample SD is descriptive, not an independence or significance claim.
