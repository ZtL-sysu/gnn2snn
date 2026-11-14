# Research decisions

## 2026-11-14: independent strong-teacher screen

Fetched origin/research/2026-11-14-graph-aware-conversion-plan and created independent branch research/20261114-graph-calibration. The supplied working directory is a historical material directory, not a Git checkout. Existing main has unrelated untracked work, left intact.

The previous graph-logit calibration on weaker teachers did not beat activation reconstruction (56.70 versus 56.78%, descriptive overlapping validation). Therefore no assumed method success, and no repeated broad tuning. Current test summaries suggest a substantial strong-teacher T1 deficit and a small T2 deficit. Audit raw archives independently before use.

Selected the smallest shared-parameter initial-membrane experiment. Exact hard-forward equivalence is required before optimizing. Direct graph-logit matching is a task-aware distillation objective; it alone does not establish graph-structure-specific novelty. Core causal and novelty claims require additional evidence.

Found another task's scheduled queue process on the local GPU host. It currently has no Python compute child. Do not stop or edit it; recheck pending queue and compute processes before launching. New campaign must avoid resource contention.

## Determinism gate and first screen

Non-deterministic real CUDA source/core comparisons failed at T4 with changing errors (0.05449 and 0.03070); deterministic algorithms and CUBLAS workspace configuration made all four time steps bitwise equal. The failed smoke produced no calibration endpoint. Preserve failure log and `cuda_trace_observations.json`. All subsequent comparisons use deterministic algorithms; this supports a CUDA numerical-order explanation, not proof of one unique kernel cause.

The frozen 512-train/512-val seed41 screen at Adam lr=.01 completed on both tasks. Graph-minus-node accuracy (pp) is MNIST T1 +0.1953, T2 -1.3672; CIFAR10 T1 -2.3438, T2 +0.1953. No graph-only setting passes the preregistered screen. The mixed objective improves CIFAR10, but loses to node calibration on MNIST; do not select a generic winner from this result.

Hard training-loss reevaluation identifies an optimization problem at T2: MNIST graph objective ends at raw logit MSE .9663 versus .6854 at zero offsets. This permits the one bounded optimization check in the protocol. Freeze a single ten-fold smaller learning rate (.001), same samples, update count, all five controls and T1/T2. No other hyperparameter search. If it does not produce robust evidence, stop method expansion; three-seed confirmation may still characterize the bounded negative result. This is a post-screen engineering correction and must be reported as such, not preregistered confirmation.

The original screen source is preserved in `source_snapshots/run_screen_initial.py`, SHA256 7749b81b91c81dc11b16300205e2ae4a63aa2249d3e1a6014ad2d56b74efd215, matching the stored run metadata. Subsequent runs store source snapshots and reject source changes during execution.
