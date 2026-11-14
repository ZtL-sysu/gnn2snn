# P9 Full Manuscript Review

## 1. Report metadata

- **Date:** 2026-11-14
- **Target venue:** Unspecified; assessed as a general ML/graph empirical manuscript. No venue fit or formatting verdict is possible.
- **Title:** *Critical-Site Sensitivity and Realizable Calibration in GCN-to-SNN Conversion*
- **Materials:** Supplied complete manuscript, captions and tables; `analysis/rescue_paired.csv`, `analysis/allocation_mechanism_cost.csv`, `RESEARCH_SUMMARY.md`, `paper/references.bib`, and `audits/P5A_scientific_review.md` as context. No code execution, raw checkpoint inspection, or rendered figures/layout.
- **Search basis:** Public-safe spot searches for Differential Coding, Adaptive Calibration and A2Q; primary PMLR/AAAI/OpenReview records. No systematic novelty search.
- **Mode:** Full scientific, writing and integrity review. Reviewer perspectives below are role-specific, not separate human reviews.

## 2. Desk assessment

- Length/format: uncertain; venue, compiled manuscript and layout were not supplied.
- Topic compatibility: pass for graph learning and ANN-to-SNN conversion; venue unspecified.
- Minimum quality: pass as a carefully bounded exploratory empirical report; not established as a new-method contribution.
- Anonymity/policy: uncertain without venue package. No hidden instructions detected in manuscript prose.
- Ethics/reviewability: no human-subject or sensitive-data concern apparent. Reviewability is limited by one-teacher method screens, prior validation exposure, and no source/package inspection here.

## 3. Summary and contribution map

The manuscript asks whether graph-conditioned sensitivity can guide low-tick GCN-to-SNN calibration and residual-event allocation. It presents a continuous ideal-recovery diagnostic, local versus next-message objectives, finite-alphabet and offset probes, and a fixed-opportunity residual selector with graph/operator controls. Historical three-seed results contextualize baseline fidelity; new method screens use teacher 41 and 512 previously exposed validation graphs per task. The supported result is negative and bounded: actual-operator allocation improves immediate message MSE but does not establish robust classification advantage across controls, datasets and ticks; measured diagnostic runtime is higher. Limitations around teacher count, validation exposure, opportunity versus energy, and implementation timing are stated clearly.

The contribution is a scoped negative result and a separation of mechanism diagnostics from finite-event/classification outcomes. The principal submission risk is whether that result is novel and broadly informative enough for a methods venue given two related tasks, one new-method teacher, reused validation, and no independent confirmation. This is a readiness limitation, not evidence that the bounded findings are false.

## 4. Search and related-work basis

- **Queries:** Public searches for “Differential Coding for Training-Free ANN-to-SNN Conversion”, “Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks”, and “A2Q: Aggregation-Aware Quantization for Graph Neural Networks”.
- **Sources checked:** PMLR ICML 2025, AAAI 2025, OpenReview ICLR 2023. Existence/metadata for these entries corroborated; not a full bibliography audit.
- **Closest cited work:** Huang et al. differential coding; Wang et al. adaptive calibration; graph quantization A2Q and Degree-Quant. The manuscript fairly says its source-percentile transfer is not a full reproduction and avoids treating graph quantization as a direct baseline.
- **Unverified novelty risk:** No systematic search for graph-specific ANN-to-SNN conversion, graph event allocation or residual correction. Add a primary-source comparison on objective, graph operator, event budget and task setting. No particular omitted paper is asserted here.

## 5. Expected review outcome

- **Outcome:** Borderline/reject for a methods-contribution track as framed; potentially suitable for a venue that values bounded negative/diagnostic empirical results, after positioning and auditability work.
- **Accept signal:** Explicit controls, clear separation of message fidelity and classification, and unusually candid disclosure of scope and cost boundaries.
- **Reject signal:** No successful algorithmic advance; actual graph is not consistently better than controls; new methods are single-teacher and validation-exposed; closest prior work is not systematically compared.
- **Confidence:** Medium for supplied claim/numeric review; low-to-medium for novelty, implementation correctness, full citation audit and venue stance.

## 6. Strengths and weaknesses

### Strengths

- Abstract, discussion and conclusion maintain a bounded negative conclusion; message-MSE improvement is not recast as classification improvement.
- The message objective is scoped to immediate linear propagation, with residual identity, later nonlinearities and downstream effects acknowledged.
- Residual formula, top-quarter selection, event-slot cap, hard-forward/surrogate distinction and unequal realized events are specified.
- Historical results are distinguished from new method selection; interval and validation exposure limitations are stated.
- Runtime is labeled synchronized implementation timing, not energy or deployment efficiency.

### Weaknesses

1. **Novelty unresolved (major).** Related work does not establish whether graph-specific ANN-to-SNN calibration/allocation has already addressed this mechanism. Search and compare closest work; if no algorithmic novelty survives, frame the contribution as an empirical audit.
2. **Selection reuse (major).** The same fixed 512-graph validation subset supports exploratory comparisons and the practical expansion decision, amid many candidate families/settings. Explicitly call endpoints selection-exposed and descriptive; do not use intervals as confirmatory evidence. Independent data are needed only if claiming confirmatory performance.
3. **One-teacher/two-related-task scope (major for broad claims).** New results do not establish seed or task transfer. Restrict conclusions to teacher 41, these superpixel benchmarks, T=1/2 and tested policies.
4. **Comparator roles (moderate).** Local, feature-only, permuted and actual selector controls share a base/opportunity budget; the stronger all-position local Adam accuracy reference does not use extra events. A compact comparator map would prevent readers from treating all as matched.
5. **Reproducibility (moderate).** Provide a safe immutable manifest with source commit, subset indices, configs/thresholds, paired-stat and timing scripts, and table-to-result identities. Define within-graph conjugation/permutation scope. Source-level event-state and gain checks were beyond this review.
6. **Clarity/layout (minor/moderate).** Section 3 packs multiple diagnostics and unequal fitting budgets. Add a table mapping each family to changed variables, fitting budget, subset and claim role; inspect final rendered figures/tables once available.
7. **Several method/statistic definitions remain underspecified (moderate).** Define the all-message calibration loss algebraically and distinguish it from first-site message calibration; specify (u_0=0) for the stated telescoping recovery identity; give numerator and denominator for the “fraction of teacher-prediction disagreement closed”; and state how the per-graph paired correctness differences produce the descriptive 95% normal intervals. These are fixable reporting gaps, not evidence of incorrect results.
8. **Cost boundary omits offline fitting/selection work (moderate).** The per-graph timing describes the policy path but does not report the wall-clock cost of calibration/optimization/selection loops. State whether these are excluded and, if so, report them separately or label the timing as inference-policy-only. Do not combine one-time preprocessing with per-graph timing without an amortization rule.

## 7. Potentially missing related work

- Graph-specific ANN-to-SNN conversion / graph event allocation: **unverified**; this is the key novelty gap. Search primary literature and compare mechanism/budget/task. No specific omitted work is asserted.
- Differential Coding, Huang et al. ICML 2025: **searched and cited**; close coding foundation. Preserve precise attribution and distinguish the present graph question.
- Adaptive Calibration, Wang et al. AAAI 2025: **searched and cited**; relevant sensitivity/adaptive firing context. Clarify overlap with fixed-opportunity graph-operator intervention.
- A2Q / Degree-Quant: **searched and cited**; relevant graph-aware quantization context, not direct SNN baselines. Current distinction is appropriate.

## 8. Claim-evidence audit

| Claim | Location | Evidence | Strength | Action |
|---|---|---|---|---|
| Historical three-teacher low-T gap and longer-prefix fidelity | Abstract, §4.1, historical table | Official test; three seeds, mean and sample SD | Adequate historical context | Retain exposure disclosure; not new-method confirmation. |
| Continuous recovery closes 54.08%/23.54% of teacher-prediction disagreement | Abstract, §3.3, caption | Full historically exposed validation; three teachers; conditional nonadditive intervention | Adequate diagnostic if denominator retained | Do not call this accuracy-gap recovery or deployable conversion. |
| T1 actual-message accuracy 79.49%/49.22% vs strong local references 78.13%/49.41% | Abstract, §4.3, tables | Teacher 41, 512 graphs; +1.367/−0.195 pp; intervals cross zero | Weak/descriptive | Keep task-wise result and distinguish strong reference from same-base selector controls. |
| T2 actual-message is 0.586 pp below same fitted base on both tasks | Abstract, §4.3, table | 88.67 vs 89.26 MNIST; 53.32 vs 53.91 CIFAR10; intervals cross zero | Adequate descriptive | Correctly reported; do not imply reliable regression. |
| Actual-message allocation reduces immediate normalized message MSE vs local allocation | §4.3/4.4 and cost table | .002422→.001280 vs .001378 MNIST; .004127→.001717 vs .002018 CIFAR10 | Strong for endpoint | Preserve separation from classification and validation exposure. |
| Allocation timing is higher for actual-message selector than local selector | §4.4 and cost table | 0.8343 vs .6678 ms MNIST; .9914 vs .8063 CIFAR10; 20 repeats | Adequate for stated software boundary | No energy, optimized deployment, or general penalty claim. |
| Tested candidates do not establish robust graph-specific classification advantage | Abstract, conclusion | Cross-task controls and failed joint gate | Adequate, bounded | Retain “do not establish”; avoid universal null claim. |

## 9. Experiments, benchmark and reproducibility audit

- **Baselines/controls:** Local fitting, feature identity, graph conjugation/permutation, random selection, base and stronger full-position local reference are useful. Strong local is an accuracy reference without extra events; selector controls share the fitted base/opportunity cap. Keep these roles visible.
- **Data:** Official splits are stated. New work uses 512 train calibration graphs and a fixed-index 512 validation subset; no test loading for method selection is claimed. Prior validation/test exposure is disclosed. These are exploratory, not independent confirmation.
- **Metrics:** Accuracy, agreement/logit MSE, normalized immediate message MSE and host time have distinct meanings. Clarify in final figure labels that all policies' MSE is measured under actual A.
- **Statistics:** Historical SD is across three seeds; new paired intervals are for graph sampling for one teacher. Intervals do not cover multiple candidate selection, seed variation or external generalization. Avoid significance or confirmatory language.
- **Robustness:** Several controls, coding families, two datasets and T=1/2 support a bounded gate-failure conclusion, not cross-teacher/task robustness.
- **Methods:** Formulas and hard event behavior are detailed. Unverified audit targets: negative-event state updates, graph conjugation semantics, single-node gain implementation, and calibration graph provenance.
- **Cost:** Slots are not equal nonzero events or energy. Timing includes policy instrumentation and excludes teacher forward and data transfer. No energy/deployment claim is supported.
- **Artifacts:** Need safe scripts/configs/provenance and source/result identity mapping. This review did not inspect raw bundle hashes or compile the paper.

## 10. Reviewer panel (lenses, not independent people)

| Lens | Score / confidence | Positive signal | Main negative signal / condition to raise |
|---|---|---|---|
| Mechanism/soundness | 3/5, medium | Immediate message identity and hard-forward scope are clear. | Implementation not source-checked; verify event/gain identities and operator control. |
| Evidence/experiment | 2/5, medium | Useful controls and fixed opportunity cap. | One teacher, reused subset, many screens; new independent evidence needed for performance claims. |
| Novelty/positioning | 2/5, low-medium | Specific graph-operator question and candid outcome. | Closest graph-SNN work not searched systematically; compare primary sources. |
| Writing/clarity | 3/5, medium | Headline is recoverable and limitations are repeated. | Dense method and pending layout; add policy/budget map and rendered check. |
| Reproducibility | 3/5, medium-low | Many parameters and timings stated. | Source/index/control identity and package not inspected; provide manifest and scripts. |
| Domain application | 3/5, medium | Low-timestep conversion is a relevant setting. | Two superpixel tasks; no deployment evidence. |
| Novice advocate | 3/5, medium | Main lesson—sensitivity/MSE do not ensure task gains—is understandable. | A short family map would prevent comparison confusion. |
| AC | Borderline/reject for method track, low-medium | Bounded negative evidence is internally coherent. | Novelty/evidence package does not support successful method contribution. |

**Panel synthesis:** Agreement: no successful conversion method is demonstrated; mechanism and classification are separated; limitations are candid. No disagreement is forced. Positive axis: transparent controls and scope. Negative axis: novelty and independent evidence. Unresolved: closest work, implementation checks, compiled presentation. AC stance: bounded empirical draft, not submission-ready method paper.

## 11. Concerns table

| ID | Severity | Concern | Evidence | Criterion | Fix class | Required action | Owner | Score-change condition |
|---|---|---|---|---|---|---|---|---|
| P9-1 | Major | Novelty boundary unverified. | Related work; limited public search. | Originality/significance | related-work | Compare closest primary graph-SNN conversion/allocation work by objective, operator, budget, task; narrow claim if needed. | ccf-literature-searcher / reviewer | Raise if distinct knowledge contribution is documented; lower if prior work subsumes it. |
| P9-2 | Major | Validation slice is previously exposed and used for exploratory comparisons. | §3.5, §4.3; one teacher, many conditions. | Evidence/soundness | experiment/limitations | Label results selection-exposed/descriptive; require independent evidence for confirmation. | paper-writer | Raise for performance claims only with independent seed/data evidence. |
| P9-3 | Major | New-method scope is one teacher and two related datasets. | Methods and result tables. | Evidence/significance | limitations | Restrict scope; no transfer/robustness claims. | paper-writer | Broaden only with matching new evidence. |
| P9-4 | Moderate | Strong local reference may be confused with opportunity-matched selectors. | Abstract/table/captions. | Clarity/soundness | writing | Label strong full-position local as no-extra-event accuracy reference; identify common selector base/budget. | paper-writer | Raise if matching is clear at a glance. |
| P9-5 | Moderate | Provenance/control implementation not independently auditable here. | No source package inspected; permutation scope incomplete. | Reproducibility | reproducibility | Provide source/config/index hashes, define permutation, link checks and analysis scripts. | integrity-auditor | Raise after identity and implementation checks. |
| P9-6 | Minor | Dense method and unavailable rendered layout. | Multiple budgets and figure placeholders. | Clarity | writing | Add policy/fitting-budget map; inspect final rendered paper. | paper-writer / visual-composer | Raise after final presentation check. |
| P9-7 | Moderate | Objective, intervention and interval definitions are incomplete for independent reconstruction. | “All-message,” (u_0), disagreement closure, and normal interval construction. | Soundness/reproducibility | method/soundness | Give algebraic loss, set (u_0=0), define closure numerator/denominator, and specify paired unit/SE/interval formula. | paper-writer / integrity-auditor | Raise once each quantity is reproducible from the paper. |
| P9-8 | Moderate | Reported per-graph runtime does not account for offline fitting/selection loop cost. | §4.4 timing boundary; fitting diagnostics use differing optimization budgets. | Reproducibility/significance | reproducibility | Separate offline calibration/selection time from timed policy path and state exclusions/amortization. | paper-writer / integrity-auditor | Raise once total-cost boundary is explicit. |

## 12. AC / meta-review

The role-specific reviews converge on a coherent, carefully qualified negative finding, but this does not itself establish novelty or fit for a method-contribution venue. I would not recommend acceptance as a successful graph-aware conversion method. A bounded empirical submission could be defensible if a targeted literature comparison establishes its contribution, selection exposure is explicit, and auditability improves. No consensus among independent human reviewers is claimed. Discussion risk: emphasizing the MNIST T1 difference despite the nonmatched strong-local comparator, validation selection exposure and lack of consistent graph-control advantage; current prose largely avoids this.

## 13. Quantitative scores

General 1–5 criteria and 1–10 overall tendency; venue unknown.

| Dimension | Score | Confidence | Deduction | Repair condition |
|---|---:|---|---|---|
| Quality | 3 | Medium | Coherent bounded study, no demonstrated new method. | Establish contribution and finish audit. |
| Clarity | 3 | Medium | Understandable but dense; layout unavailable. | Comparator/budget map and rendered read-through. |
| Significance | 2 | Low-medium | Narrow tasks and uncertain audience value. | State audience and value beyond failed candidates. |
| Originality | 2 | Low-medium | Closest graph-SNN work not systematically checked. | Primary-source comparison. |
| Soundness | 3 | Medium | Scope stated; code not independently checked. | Verify implementation identities and controls. |
| Evidence | 2 | Medium | One teacher, selected validation, many screens. | Keep bounded claim or add independent evidence. |
| Reproducibility | 3 | Medium-low | Many details stated; provenance/package not inspected. | Frozen manifest and derived scripts. |
| Ethics/limitations | 4 | Medium | Exposure, cost and scope candidly disclosed. | Preserve disclosures. |
| **Overall** | **4/10, borderline/reject for methods track** | **Medium-low** | Novelty/evidence scope is decisive; no numeric contradiction identified. | Prior-work and artifact audit; venue-specific framing. |

## 14. Questions for authors

1. Which prior graph-specific ANN-to-SNN calibration/allocation methods are closest, and what mechanism distinction remains?
2. Were candidates/conditions selected using the same 512 validation graphs? Can all statistics be labeled selection-exposed and exploratory?
3. Can exact subset indices, permutation construction, source commit and analysis scripts be provided in a safe reproducibility package?

## 15. Score revision criteria

- **Raise:** Sourced comparison establishes distinct contribution; controls/event implementation verified; final presentation exposes evaluation and event-count differences. Independent evidence is needed for general performance claims.
- **Lower:** A close prior method subsumes the question, implementation differs from stated formula, or undisclosed validation/test selection is found.
- **Unlikely to change:** Lack of multi-seed algorithmic gain cannot be repaired by prose. Keep a bounded negative report or leave method-paper novelty unresolved.

## 16. Action plan and handoffs

1. **Priority 1:** Targeted public literature comparison for graph-aware SNN conversion/allocation. **Owner:** ccf-literature-searcher + ccf-paper-reviewer. **Output:** closest-work matrix and narrowed claim. **Handoff:** yes.
2. **Priority 2:** Freeze safe source/config/index/result manifest and verify event/gain implementation. **Owner:** ccf-integrity-auditor. **Output:** provenance and claim-evidence audit. **Handoff:** yes.
3. **Priority 3:** Add comparator/resource map; inspect rendered figures/tables. **Owner:** ccf-paper-writer / ccf-visual-composer. **Output:** clarity/layout review preserving data and claim scope. **Handoff:** yes.

**Checks run:** Supplied manuscript/tables/captions read; headline accuracy deltas, intervals, MSE and timing cross-checked against supplied CSVs; sampled primary-source records for Huang 2025, Wang 2025 and A2Q; definitions of recovery, all-message calibration and offline cost checked for reproducibility gaps.

**Checks skipped:** Full bibliography metadata/context audit; systematic novelty search; raw-source execution and checkpoint/result-hash audit; compiled layout/LaTeX; venue-specific fit.

**Unresolved risks:** External validity, teacher robustness, selection exposure, closest graph-SNN work, implementation/provenance and final layout.
