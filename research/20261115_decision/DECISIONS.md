# Phase 3 decisions — 2026-11-15

Base: `cb742f2bd30b2408cab5d09a7d9af8ff4d7c67c8`. Independent branch:
`research/20261115-decision-aware`. Historical directories are read-only.
The attached research brief defines desired gates; its provisional algorithm name is not an established contribution.

**Current gate outcome:** Gate0/1/2 executed; Gate2 admission FAIL. Gate3 and Gate4/5 are deliberately **not admitted**, not silently omitted or claimed complete. The bounded candidate family is stopped; the long-term algorithm-discovery objective remains unmet.

## Execution plan and design

- [x] Locate original repository, frozen evidence and conda `dyg-cold`; inspect GPU (RTX3060, no scientific compute process).
- [x] Gate0: reconstruct 142 endpoints from raw arrays and verify source/data/teacher hashes; review closest published mechanisms and source-code fidelity.
- [x] Gate1: `analyze_cached.py` joins graph IDs, teacher/class margins, correctness quadrants and actual message errors. Bootstrap paired graph differences; treat all validation as selection-exposed.
- [x] Gate1: `run_interventions.py` makes bounded hard first-site amplitude interventions. Same existing nonzero coordinate, adjacent original geometric amplitude, no new first-site event. Update membrane/cumulative/exp_out consistently. Count all downstream nonzero events rather than claiming total cost equality. Compare message improvement against teacher-class margin and classify prefix effects. No teacher retraining, no official test loading.
- [x] Gate2 conditional: fixed 512 training/512 validation graphs, teacher41 on both datasets, 693 initial-state parameters, Adam64 lr.01. At most local/logits-MSE/KL/ranking and one joint-prefix control (no hyperparameter sweep). Use identical data/batches/steps; offline teachers only. T1/T2 evaluated with the *same* fitted parameters. Historical separately T2-fitted parameters are not a prefix-consistency baseline.
- [ ] Gate3 conditional: only after meaningful decision-objective evidence; choose at most two genuinely different encoding mechanisms. Four-arm parameter/supervision-matched static/task/structure/full controls, actual event and deployment cost accounting. No extra-event slot equivalence.
- [ ] Gate4/5 conditional: expand teachers/external task and manuscript only after both tasks exceed strong local by >=0.5pp, T2 degrades <=0.5pp, and independent structure/cost controls pass. Otherwise retain failed gates and identify the missing evidence.
- [x] Verify artifacts, update evidence/index, commit bounded reproducible work.

Initial compute cap: 30 GPU minutes total for Gates1–2, serial, existing teachers only. No uncontrolled retries. Gate3 requires a written evidence-dependent design before execution. Every attempted validation endpoint is retained. No new PDF unless method gates pass.

## Falsifiable initial hypothesis

Message MSE is an unweighted norm before downstream nonlinearities/readout. A lower norm may move error into a teacher-versus-competitor direction and reduce the minimum teacher-class margin. Predict: same-graph hard perturbations that lower true message MSE can reduce final teacher-class margin, with larger disagreement risk near small teacher margins. This is not established until measured. Teacher top2 margin alone may miss competitor switching, so all-class competitor margin is also recorded.

Alternatives considered: repeat MSE-weight searches (excluded by existing failures); directly develop graph-dependent thresholds (premature without mechanism); bounded decision diagnostics and matched task objectives (chosen for information gain).

## Gate1 intervention outcome and Gate2 pre-run decision

Two tasks, 512 graphs each, four interventions/graph/prefix. Message MSE decreased in 811 MNIST / 928 CIFAR10 interventions; T1 teacher-class margin worsened in 332 / 296 of these, teacher agreement lost in 16 / 36 and gained in 16 / 30. T2 margin worsened in 395 / 409. Thus a lower message norm demonstrably does not imply decision preservation, even for actual hard forward interventions. Counts are clustered within graphs, not independent trials. First-site nonzero support is fixed, downstream events change. No fixed-total-cost claim.

Proceed to the bounded Gate2 controls: local_t1, local_joint, normalized logits MSE, temperature KL, teacher top2 ranking. The last four average T1/T2; all five execute both prefixes and use identical 693 offsets, Adam64 lr.01, training512, fixed batches. KL temperature uses per-graph teacher logit standard deviation clamped >=1, no temperature tuning. Ranking uses softplus(1 - converted teacher-top2 gap / same scale). No labels in fitting. Ordinary objectives are explicitly controls, not algorithm novelty. All fitted rules checked at T1/2/4/8 using a shared parameter vector. Primary admission: a decision objective beats both matched local controls by >=0.5pp T1 on both datasets, T2 loses <=0.5pp against local_t1, with nonworse teacher agreement and transparent actual event Pareto. One optimization/capacity audit allowed upon failure; no broader loss tuning.

## Post-run decision (no threshold change)

Forty endpoints complete. Ordinary logits has T1 +1.171875pp in both tasks with fewer actual events; MNIST satisfies the exploratory admission criteria, CIFAR10 does not because teacher agreement falls 2.1484375pp. Paired graph intervals for both accuracy gains cross zero. KL and top2 ranking fail. One full-training hard-objective interpolation/capacity audit confirms objective decreases and zero saturated offsets; top2 optimization increases predictions outside the teacher top2. This is evidence of an incomplete pairwise objective, not proof that every task objective or structure mechanism fails.

Do not override the registered agreement criterion after seeing the attractive accuracy result. Preserve the exploratory Pareto result as an ordinary strong baseline, stop this candidate expansion, and leave graph-independent coding novelty unclaimed. The next different mechanism requires fixed true event/amplitude/delivery-budget oracle evidence, full-competitor margin and prefix constraints, followed by a same-capacity structure falsifier. Source threshold iteration is also a baseline gap. These are not already successful algorithms. No new teacher training, external task expansion or paper PDF was launched.

Validation/quality corrections: intervention v1's `random` label was a deterministic coordinate rule; v2 names it correctly and records coordinates/actual event changes. Public numeric columns reproduce exactly; no additional distinct hypothesis endpoint. Exact per-graph event counts are reduced in int64 for canonical tables rather than float32 totals. Independent Luna audit found no material state/forward/leakage defect, and prompted the trace correction and explicit independent-T2-fit distinction. Primary acceptance verifies all raw endpoints, source identities, historical immutability and shared-prefix baselines. All 21 targeted current/historical tests pass; library deprecation warnings retained.

## Source-TI baseline follow-up (fixed protocol, no reopened candidate family)

Executed SOURCE_TI_PROTOCOL after explicit direct-source conformance. Six per-channel pre-ReLU moment streams match the frozen teacher site path bitwise; original100-step TI, NaN→1 fallback and converter negative→0 semantics execute unchanged. Adapter records source example's four-level/default-layer differences; graph controls deliberately match eight channelwise levels. Source algorithm reused, no new contribution.

Both source horizons H1/H2 evaluated T1/2/4/8 without threshold recalibration;16 endpoints, fixed teacher41/train512/val512. At source intended H=T: MNIST T1 71.484375%,T2 89.453125%; CIFAR T1 45.3125%,T2 52.1484375%. All four source-forward equality checks per horizon pass. NaN fallback35 MNIST channels at each horizon,20/19 CIFAR; no external epsilon or numerical rescue. Low-T conversion still trails strong local in relevant points, while highT source fidelity is strong; do not generalize to all source settings or published datasets. This closes the previously missing source-TI component in our fixed-GCN adaptation, not faithful One-Spike or original ImageNet reproduction. Gate2 remains FAIL; ordinary objective search stays stopped.

Fresh combined verification:26 targeted tests pass (previous21 plus3 source-TI conformance and2 source preactivation/zero-channel tests). Final raw/source acceptance includes all16 added endpoints and recoverable source snapshots. Branch deliveries are pushed and remote revision is verified; method discovery remains unachieved.
