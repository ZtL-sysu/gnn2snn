# Historical fixed-GCN baseline audit

**Status: VERIFIED archived historical audit; not a live reproduction.** Six saved prediction bundles were re-read and all aggregate metrics independently recomputed from labels/logits. Sidecar, NPZ, reference, checkpoint, and calibration hashes were checked where recorded. No GPU evaluation was run.

## Scope and caveat

These are project MTH-ReLU-branch graph adaptations using pinned source units and common full-train histogram calibration; not full DCGS reproduction or evidence of superiority. The complete official test split had prior project exposure. These are descriptive exposed-test fidelity endpoints, not untouched-test estimates. Seeds vary teacher training; each dataset reuses the same 10,000 examples.

## Recomputed mean ± sample SD across seeds

| Dataset | Metric | Mean | Sample SD |
|---|---|---:|---:|
| MNIST | teacher_accuracy_percent | 91.0233 | 0.265016 |
| MNIST | T1_accuracy_percent | 71.4333 | 2.8019 |
| MNIST | T1_teacher_agreement_percent | 73.4567 | 3.06414 |
| MNIST | T1_logit_mse | 11.5014 | 1.22106 |
| MNIST | T2_accuracy_percent | 90.1567 | 0.490034 |
| MNIST | T2_teacher_agreement_percent | 96.36 | 0.320468 |
| MNIST | T2_logit_mse | 0.766156 | 0.0713778 |
| MNIST | T4_accuracy_percent | 91.0467 | 0.257164 |
| MNIST | T4_teacher_agreement_percent | 99.7067 | 0.0585947 |
| MNIST | T4_logit_mse | 0.00351213 | 0.000329944 |
| MNIST | T8_accuracy_percent | 91.0167 | 0.261024 |
| MNIST | T8_teacher_agreement_percent | 99.9267 | 0.0472582 |
| MNIST | T8_logit_mse | 0.000228609 | 7.36312e-06 |
| CIFAR10 | teacher_accuracy_percent | 54.94 | 0.07 |
| CIFAR10 | T1_accuracy_percent | 46.8067 | 1.2178 |
| CIFAR10 | T1_teacher_agreement_percent | 64.79 | 1.70537 |
| CIFAR10 | T1_logit_mse | 1.56564 | 0.254052 |
| CIFAR10 | T2_accuracy_percent | 54.0633 | 0.238607 |
| CIFAR10 | T2_teacher_agreement_percent | 90.7633 | 0.584836 |
| CIFAR10 | T2_logit_mse | 0.102798 | 0.0182695 |
| CIFAR10 | T4_accuracy_percent | 54.9167 | 0.0814453 |
| CIFAR10 | T4_teacher_agreement_percent | 99.43 | 0.132288 |
| CIFAR10 | T4_logit_mse | 0.000741824 | 0.000254109 |
| CIFAR10 | T8_accuracy_percent | 54.9667 | 0.0472582 |
| CIFAR10 | T8_teacher_agreement_percent | 99.8467 | 0.051316 |
| CIFAR10 | T8_logit_mse | 4.82818e-05 | 1.29366e-05 |

## Per-seed summary

| Dataset | Seed | Teacher accuracy | T1 acc / agree / MSE | T2 acc / agree / MSE | T4 acc / agree / MSE | T8 acc / agree / MSE |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 41 | 90.94% | 74.46% / 76.75% / 10.2887 | 90.15% / 96.38% / 0.683929 | 90.94% / 99.75% / 0.00314174 | 90.93% / 99.98% / 0.000224189 |
| MNIST | 42 | 91.32% | 70.91% / 72.93% / 12.7307 | 90.65% / 96.67% / 0.802394 | 91.34% / 99.73% / 0.00362007 | 91.31% / 99.89% / 0.00022453 |
| MNIST | 43 | 90.81% | 68.93% / 70.69% / 11.4847 | 89.67% / 96.03% / 0.812145 | 90.86% / 99.64% / 0.00377459 | 90.81% / 99.91% / 0.000237109 |
| CIFAR10 | 41 | 54.87% | 45.50% / 62.93% / 1.58625 | 53.87% / 90.25% / 0.105973 | 54.88% / 99.53% / 0.000680248 | 54.93% / 99.79% / 4.60361e-05 |
| CIFAR10 | 42 | 54.94% | 47.91% / 66.28% / 1.30191 | 53.99% / 91.40% / 0.0831486 | 54.86% / 99.48% / 0.000524161 | 54.95% / 99.86% / 3.6615e-05 |
| CIFAR10 | 43 | 55.01% | 47.01% / 65.16% / 1.80876 | 54.33% / 90.64% / 0.119271 | 55.01% / 99.28% / 0.00102106 | 55.02% / 99.89% / 6.21942e-05 |

## Checkpoint provenance

| Dataset | Seed | Epoch | Checkpoint path | SHA-256 |
|---|---:|---:|---|---|
| MNIST | 41 | 79 | `C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\takeover_20261112\full_teacher_runs\MNIST_seed41_full\best.pt` | `b123168cbc8b8476d1fa2b65c934bb66ff4c67bb9a2cc33b1e5cd047176a9dd1` |
| MNIST | 42 | 96 | `C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\takeover_20261112\full_teacher_runs\MNIST_seed42_full\best.pt` | `3ed95ff125bdbc55e85c0f1f83c85d8d4af21b9072a444756a93fd88e53dfebe` |
| MNIST | 43 | 99 | `C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\takeover_20261112\full_teacher_runs\MNIST_seed43_full\best.pt` | `cf1649d8ce0371a1deced093c35a9d0cfa1a54a715b6633e4cb857e315fe28a5` |
| CIFAR10 | 41 | 113 | `C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\takeover_20261112\full_teacher_runs\CIFAR10_seed41_full\best.pt` | `bb35ddc12bc3251473cbe26768dcb51ae4e12e9f3cf814a3e0e94fbe2572f01e` |
| CIFAR10 | 42 | 79 | `C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\takeover_20261112\full_teacher_runs\CIFAR10_seed42_full\best.pt` | `12268d69b2ca86183b559c7eeebd7147d35f1c3cf6814eae405e61cfa8fc3d0c` |
| CIFAR10 | 43 | 135 | `C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\takeover_20261112\full_teacher_runs\CIFAR10_seed43_full\best.pt` | `b9f4f9e31f4f28cae61e9fdb77339e84dea17090c7c9640da4dc0eea6f7fd395` |

Sidecars report matching input hashes at run start and end. NPZ metrics match the committed six-row summary within strict floating-point tolerances. The manifest records expected and observed file hashes.

## Limits and unknowns

- This verifies archived artifacts and internal provenance records; it does not reproduce teacher training, conversion, or inference from source code.
- Dataset/runtime code was not re-executed. Sidecar semantics are accepted subject to file consistency checks.
- Archived qualification disclaims full DCGS reproduction and an untouched-test claim.
- Gated results are omitted; this audit only establishes the six fixed-GCN baseline rows.
