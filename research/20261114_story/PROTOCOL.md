# Strong-teacher graph objective campaign — 2026-11-14

## Frozen first screen

Question: does graph-logit reconstruction outperform local activation reconstruction when calibrating exactly the same 693 initial-membrane channel offsets in a frozen, four-layer GCN?

Use official MNIST and CIFAR10 superpixels, existing teacher seed 41, fixed source-unit MTH differential conversion, T1/T2. Six sites: four graph-convolution ReLU branches and two MLP ReLUs. Initial membrane is theta times a dimensionless offset, initialized to zero and constrained to [-0.5,0.5]. Hard inference uses the pinned eight-level signed-event recurrence; optimization uses a ReLU straight-through surrogate. No teacher weights change. This is a project conversion adaptation, not the original full DCGS pipeline.

Calibration: fixed 512 official train indices sampled with RNG 20261114; batch size 32; Adam lr=0.01; 64 updates, same ordered batches for each objective. Fixed final iterate, no per-method early stopping. Node control minimizes the equal-site average of activation MSE divided by teacher site second moment (four node sites and two graph hidden sites). Graph objective minimizes graph logit MSE divided by teacher logit second moment. Mixed objective = 0.5 each. Output-only control fits 10 additive logit biases using graph loss and the identical sample/update budget (explicitly fewer parameters). Shuffled-target falsifier uses graph logits from a cyclic shift of the batch; it destroys target/sample correspondence, not graph topology. Its failure can support correspondence sensitivity, not a graph-structure-specific causal claim.

Validation: fixed 512 official val indices, RNG 20261115. Store all per-graph logits, graph sizes/degrees, local MSE by site, graph-logit MSE, flip, teacher margin and margin change. Main screening criterion: graph improves accuracy over node by >=0.5 pp at T1 or T2, with non-inferior teacher agreement. This single-seed small-sample screen is not confirmation. If promising, freeze this configuration for seeds 42/43 and full 5000 validation graphs, then two tasks. No new test access during selection. Historical tests are exposed; eventual frozen test endpoints remain descriptive.

## Gates and budget

1. Audit raw archives, checkpoint and data hashes; do not retrain teachers. All teacher assets must match before scientific calibration.
2. CPU exact source equivalence, BN folding, label independence, gradient and frozen-weight acceptance; then CUDA real-data equality versus existing baseline at all four T. A discrepancy blocks method results.
3. Two tasks × seed41 × T1/T2 small screen, <=2 GPU hours including profiling. Inspect one implementation/hyperparameter failure only; no open-ended search.
4. If screen passes, <=12 GPU hours reserved for three-seed/full-val controls and T4/T8. If it fails, diagnose and evaluate whether a bounded confirmation is justified; stop complexity expansion.
5. <=3 GPU hours cost measurements: fixed graph batches, warmup and synchronized repeats; report inference and calibration separately. Dense kernels do not skip zero events. No measured hardware energy claim.
6. Freeze evidence, select A/B/C/D, write an evidence-limited manuscript/report and publish this independent branch. Whole campaign <=48 wall-clock hours; estimates revised from observed speed.

## Measurements and provenance

Record command, code commit/hash, runtime, checkpoint/calibration/data hashes, actual train/val indices, update traces, parameters, GPU peak allocation, synchronized elapsed time, final offsets and predictions. Keep historical files unchanged. Pearson/Spearman associations are descriptive, not causal. Per-graph bootstrap does not establish teacher-seed robustness. Three-seed differences report sample SD and each paired value.

Initial risks: historically exposed validation/test; prior weak-teacher graph-MSE pilot failed to beat activation-MSE; basic calibration and potential initialization already have precedents; a STE may fail despite a useful hard objective. Scientific route remains undecided until current evidence exists.
