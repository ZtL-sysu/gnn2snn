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

## Freeze for descriptive three-teacher confirmation

At lr=.001, graph-minus-node seed41 screen accuracy is MNIST T1 +.78125 pp, T2 +.78125 pp; CIFAR10 T1 +.97656 pp, T2 +1.75781 pp. Teacher agreement is non-inferior for MNIST T1 and CIFAR10 T2, but is lower for the other two cells. This is sufficient to justify a bounded confirmation, not to accept H2. Freeze lr=.001, 512 training graphs, 64 updates, six original conditions and T1/2/4/8 on all three frozen teachers and 5000 validation graphs per task. No further tuning. Retain lr=.01 failures and all low-lr outcomes. Report full validation and separately the 4488 graphs outside the current 512-graph screening prefix; both remain historically exposed and teacher-selected validation data.

Expected measured runtime from screen is well within the reserved 12 GPU hours; initial estimate 20–45 minutes for six full-val runs. New teacher training and Gated method expansion remain excluded. Independent novel-graph-method and submission-readiness decisions await confirmation, mechanism and cost evidence.

Seed41 completed all 24 confirmation endpoints. Seed42 stopped before loading any scientific input because the historical calibration filename differs: replica seeds use `rep/M42/cal.json` (and analogous M43/C42/C43), whereas seed41 stores it beside the checkpoint. Resolve these two audited layouts and verify the same stored hashes. This is an asset-location repair with no changed objective, sample, optimizer or budget. Preserve the failed startup log, skip completed seed41, and resume remaining seeds.

## Complete validation and final endpoint freeze

All six runs completed 24 conditions each. Strict aggregation verified all saved source snapshots, common mathematical implementation, calibration budgets, full ordered validation coverage and graph-paired predictions. Full-validation graph-minus-local mean accuracy differences (T1/T2, pp): MNIST -.1000/-.1267; CIFAR10 -.2667/+.4333. The corresponding screen-excluded means are -.1263/-.1188 and -.4085/+.3268. All four three-seed t95 intervals on the 4488-graph slice cross zero. These observations fail the planned stable, practically meaningful cross-setting method gate; they do not prove all graph-aware objectives ineffective.

Baseline output-logit MSE is more correlated with prediction flip than the normalized four-site node-error aggregate in all 12 task/seed/low-T cells on the screen-excluded slice. This supports an association-level error hypothesis. It is not a readout/propagation intervention, and the final-output metric is closer to the decision by construction; graph-specific causal attribution remains unestablished.

Stop all algorithm and hyperparameter expansion. Bind every existing condition in TEST_FREEZE.json, then perform one forward-only full official test pass for all frozen conditions at T1/2/4/8 and all teachers. The test has historical exposure, so no untouched-test claim follows. Complete matched costs and training-loss reevaluations without additional updates. The validation evidence points to route C; final framing will consider all frozen endpoints and will not use test outcomes to change any parameter or comparison.

## Bounded mechanism intervention, before observing its outputs

The negative objective comparison and positive error association leave the location of the quantization bottleneck unidentified. Add one validation-only diagnostic, without modifying any fitted condition or test comparison. At a chosen converted ReLU site, replace the quantizer with exact causal differential ReLU: accumulate decoded incoming current u_t=u_(t-1)+c_t/t and emit t[ReLU(u_t)-ReLU(u_(t-1))]. This preserves incoming converted errors while removing the local quantization discrepancy. It uses no teacher hidden states and has continuous-valued outputs; it is an ideal diagnostic, not a deployable SNN candidate.

Fixed masks: none; each of six individual sites; four node sites together; two readout sites together; all six. Run all full official validation graphs, T1/T2 and all six teachers. Empty-mask output must match the frozen baseline; all-six output must recover the folded teacher within floating-point tolerance at every checked T. CPU synthetic acceptance precedes execution. Compare conditional effects, not additive contributions; the grouped masks restore different numbers of nonlinearities. The question is whether readout quantization or upstream node quantization accounts for more of the observed low-T deficit. No answer is assumed. This targets RQ1 and route-C interpretation and opens no new hyperparameter search. Keep calibration_core.py and TEST_FREEZE inputs immutable; execute in a separate process after the frozen endpoint queue to avoid GPU contention.

## Final scientific decision

Frozen test graph-minus-local differences at T1/T2 are +.2033/+.1767 pp on MNIST and -.1767/+.1933 pp on CIFAR10. The fixed mixture also fails to improve consistently across tasks and splits. The larger T1 gains belong to ordinary local calibration, not a unique graph objective. The six ideal-site runs passed exact empty-mask replay and all-six teacher recovery; the first node site is the strongest individual T1 accuracy intervention. Its mean teacher-disagreement gap closure is 54.1%/23.5%, while restoring the two readout sites closes 8.8%/14.9%. This supports a conditional upstream quantization effect, not a dominant readout-only or topology-specific explanation.

All six matched runtime profiles are slower than the ANN in the measured diagnostic implementation, and graph/local timings nearly coincide. No physical energy evidence exists. Select route C and end this bounded campaign without more tuning or teacher training. Deliver the full technical report, complete manuscript, raw and derived records and reproduction instructions; submission readiness remains false because novelty, full source-faithful strong controls and independent broader confirmation remain open. A read-only numeric review checked the final prose against CSVs and all 142 TEST_FREEZE hashes remained unchanged.
