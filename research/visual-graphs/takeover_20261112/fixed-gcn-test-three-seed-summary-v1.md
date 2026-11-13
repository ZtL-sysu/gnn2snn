# Fixed-GCN full-test endpoints: three frozen teacher seeds

All official test graphs have prior exposure. These are descriptive fidelity endpoints of the project source-unit ReLU-branch adaptation with common histogram calibration, not full DCGS reproduction, a new method, superiority, efficiency, or an independent test claim. All runs use complete graphs and 10,000 examples, T1/2/4/8.

| Dataset | Teacher seed | ANN | T1 | T2 | T4 | T8 | T4 agreement | T8 agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 41 | 90.94 | 74.46 | 90.15 | 90.94 | 90.93 | 99.75% | 99.98% |
| MNIST | 42 | 91.32 | 70.91 | 90.65 | 91.34 | 91.31 | 99.73% | 99.89% |
| MNIST | 43 | 90.81 | 68.93 | 89.67 | 90.86 | 90.81 | 99.64% | 99.91% |
| MNIST | mean ± sample SD | 91.02 ± 0.27 | 71.43 ± 2.80 | 90.16 ± 0.49 | 91.05 ± 0.26 | 91.02 ± 0.26 | — | — |
| CIFAR10 | 41 | 54.87 | 45.50 | 53.87 | 54.88 | 54.93 | 99.53% | 99.79% |
| CIFAR10 | 42 | 54.94 | 47.91 | 53.99 | 54.86 | 54.95 | 99.48% | 99.86% |
| CIFAR10 | 43 | 55.01 | 47.01 | 54.33 | 55.01 | 55.02 | 99.28% | 99.89% |
| CIFAR10 | mean ± sample SD | 54.94 ± 0.07 | 46.81 ± 1.22 | 54.06 ± 0.24 | 54.92 ± 0.08 | 54.97 ± 0.05 | — | — |

Seeds vary teacher training only. The shared examples are not independent replications. No paired-significance or proposed-method gain follows from this summary. The earlier validation summary remains unchanged; this artifact fills its explicitly missing test-seed cells.

Source/protocol: dcgs_fixed_seed_eval_v3.py, dc-fixed-fulltest-protocol-v3.md, fixed-gcn-test-seed-completion-v1.md. Saved predictions, indices, labels, same-checkpoint teacher references and finite 10-class logits were checked directly.
