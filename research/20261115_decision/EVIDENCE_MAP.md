# Evidence map and paper roles

|Evidence|Status / location|Paper role|Boundary|
|---|---|---|---|
|Three GCN/Gated teachers and historical conversion/local baselines|Recomputed `analysis/historical/audit.json`|Setup / benchmark / motivation|Already-exposed archived test logits; no new independent test inference; Gated keeps analog gates|
|142 critical endpoints|142 raw-array/hash checks, `analysis/cached/summary.json`|Negative objective/capacity ablations|Historical methods failed under their own budgets; not universal impossibility|
|High T fidelity|Historical audit plus new shared-parameter T4/8|Baseline, functional regression|DCGS-style recurrence is prior work|
|Ideal restoration 54.1%/23.5%|Historical evidence retained unchanged|Conditional motivation|Teacher-disagreement gap closure, not recovered accuracy gap or deployable gain|
|Teacher-conditioned class/label margins and correctness quadrants|`analysis/cached/per_graph_diagnostics.csv`|Decision mismatch diagnosis|512 exposed validation graphs, teacher41|
|Actual graph message error decreases but hard decision worsens|`runs/intervention_v2/*/graphs.csv`, `analysis/phase3/intervention_cluster_ci.csv`|Controlled mechanism evidence|One first-site nonzero amplitude changed; downstream totals differ; not deployable encoder|
|Graph degree/neighborhood controls|`analysis/cached/structure_features.csv`, incremental OLS in summary|Structure motivation / limitation|Constant outdegree; weak in-sample increment, no independent predictive benefit|
|Matched five-arm decision objective screen|`runs/objectives/*`, auto `analysis/phase3/main_table.md`|Strong ordinary objective baselines|No novel method; 693 offsets, same training/index/update budget and shared prefixes|
|Ordinary logits accuracy/event tradeoff|T1 +1.171875pp both tasks; `paired.csv`, `event_accuracy.svg`|Exploratory objective ablation / Pareto|CIs cross zero; CIFAR agreement −2.1484pp; not fixed exact event equality, not graph innovation|
|Ranking loss decreases while third competitors win|`analysis/optimization_audit/results.json`|Objective limitation / next mechanism|Training-only, one bounded audit; not all-class ranking failure|
|Events, amplitude proxies, storage and timing|`analysis/phase3/main_table.csv`, raw NPZ|Cost transparency|Dense backend; event proxy excludes residual/readout arithmetic; no chip energy or deployment latency claims|
|Graph-conditioned finite-event reassignment and6 same-capacity controls|Implemented,28 val endpoints,method gate FAIL,`analysis/learned_swap/*`|Encoding mechanism failure boundary|Single pass has no teacher/lookahead,but accuracy/structure/cost gates fail;not a successful deployable method|
|Fixed-support same-channel event reassignment capacity|`analysis/swap_oracle/summary.md`, two training128 candidate banks|Finite-code realizability diagnosis|Oracle sees teacher and full future cost offline; strict equal-events joint feasibility0; no deployable selector or CIFAR accuracy gain established|
|One feature/target/solver/hard-path audit|`analysis/learned_swap/alignment_audit.json`,unit tests|Optimization versus realizability limit|Correct target alignment/solved ridge does not imply reliable sign/joint-feasibility prediction;no stronger-predictor search|
|Source-linked TI graph adaptation|16 raw-derived/source-checked endpoints, `analysis/source_ti/summary.md`|Stronger source-algorithm control|Source TI/fallback unchanged; eight-level channel GCN adaptation, not literal ImageNet reproduction or new method|
|New multi-teacher/external task confirmation|Not launched|Explicit missing evidence|Old seed results cannot validate new objective/encoder|

The prior route-D manuscript is unchanged. No new manuscript/PDF is generated: the evidence supports diagnostics and ordinary/source controls, not an algorithm-type method paper. The missing DCGS threshold-iteration component now executes faithfully in the graph adaptation; faithful One-Spike comparison, new-encoder graph independence and external confirmation remain explicit gaps.
