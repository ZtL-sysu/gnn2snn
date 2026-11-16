# Module Quality Review (S6)

## Inputs and comparison basis

This review scores the frozen baseline in `sources/main.tex` against the ten comparators frozen in `comparator-literature-table.md` and the narrative patterns summarized in `comparator-section-patterns.md`. It is a presentation review, not a new scientific review. All numbers, conditions, equations, citation keys, the LLM Usage Statement, and the scope of the claims are treated as protected facts from `execution-contract-v1.md` and `claim-disposition.md`.

The comparator set consists of recent CVPR/ICCV/ECCV/ICLR/ICML papers, so the journal impact component of the original f-submit-v1 rubric was replaced by verifiable top-venue status during S4. Where only abstracts and section-level records were accessible, the module scores below are explicit editorial estimates for presentation comparison. They are not calibrated probabilities, not scientific quality rankings, and not a claim that every comparator was read at identical depth. Consistent with the rubric, the operational pass threshold is the fourth-highest comparator score for each module.

## Comparator module score matrix (0–100)

Comparator IDs follow `comparator-literature-table.md`.

| Module | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | Range | 4th-highest threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Title | 92 | 91 | 90 | 94 | 93 | 92 | 90 | 92 | 91 | 89 | 89–94 | 92 |
| Abstract | 92 | 90 | 91 | 91 | 92 | 90 | 92 | 90 | 89 | 89 | 89–92 | 91 |
| Introduction | 93 | 91 | 90 | 90 | 92 | 90 | 93 | 89 | 89 | 89 | 89–93 | 91 |
| Related Work | 88 | 87 | 89 | 88 | 89 | 90 | 89 | 87 | 88 | 87 | 87–90 | 89 |
| Method / Analysis | 92 | 91 | 94 | 91 | 90 | 92 | 91 | 92 | 91 | 91 | 90–94 | 92 |
| Results | 94 | 93 | 93 | 92 | 92 | 92 | 94 | 92 | 93 | 91 | 91–94 | 93 |
| Discussion | 90 | 89 | 89 | 88 | 92 | 88 | 92 | 88 | 89 | 88 | 88–92 | 89 |
| Limitations | 88 | 90 | 87 | 87 | 91 | 86 | 89 | 87 | 90 | 86 | 86–91 | 89 |
| Conclusion | 91 | 90 | 90 | 89 | 92 | 89 | 92 | 89 | 90 | 88 | 88–92 | 90 |
| Figures / Tables | 92 | 91 | 90 | 91 | 90 | 90 | 92 | 91 | 92 | 88 | 88–92 | 91 |
| Citation Integration | 91 | 90 | 91 | 89 | 90 | 92 | 91 | 89 | 90 | 89 | 89–92 | 91 |

Keywords are not applicable: the CVPR template and submission profile do not define a separate keyword module. The supplementary material is audited separately below because the frozen comparator matrix concerns main-conference presentation modules.

## Module decisions

Allowed statuses: `passed`, `failed-needs-revision`, `blocked-needs-author-input`, `not-applicable-with-reason`.

| Module | User score | Threshold | Status | Dimension-level reason | Named defect / required S7 action |
|---:|---:|---:|---|---|---|
| Title | 93 | 92 | `passed` | Specific object, audited property, and scope; non-promotional. | no concrete defect |
| Abstract | 89 | 91 | `failed-needs-revision` | Complete chain and concrete evidence, but first-result sentence carries several stacked qualifications and the ending compresses three reporting actions. | **Clause-stacked endpoint inventory and compressed pacing.** Split the conditional-rate statement from its scope clause where possible; make the final implication read as one clean reporting consequence. Preserve every rate, denominator, scope phrase, and the two-invariant structure. |
| Introduction | 88 | 91 | `failed-needs-revision` | Strong claim spine, but the opening starts with generic GNN background and the last two paragraphs restate the contribution/checklist after an already explicit central finding. | **Slow opening hook plus repeated contribution announcement.** Start with the finite-event failure mode, compress conversion background, and merge the contribution framing with the central finding. Do not remove exposed-data, conditional-rate, or method-identity qualifications. |
| Related Work | 88 | 89 | `failed-needs-revision` | Accurate placement and precise non-identities, but the first paragraph is a long method catalogue and the final question arrives after a dense citation run. | **Mechanical citation inventory.** Add capability-led topic sentences and tighten the transition to the graph-specific question. Keep PD-Quant and One-Spike distinctions verbatim in substance. |
| Method / Analysis | 92 | 92 | `passed` | Definitions, event recurrence, margin condition, exact invariants, and intervention rules are reproducible; algebra is correctly bounded. | no concrete defect |
| Results | 89 | 93 | `failed-needs-revision` | The evidence is strong and the tables carry the decisive numbers, but several subsections open by naming a table or restating an activity before giving the finding, and one paragraph stacks many coordinate counts and event totals. | **Table-dependent topic sentences and clause-stacked numeric inventory.** Lead result blocks with the answered question or finding; let Table 2 and Fig. 5 carry dense cell values. Keep all rates, intervals, counts, split labels, and immediate factual qualifications. |
| Discussion | 89 | 89 | `passed` | Interprets both information losses, separates diagnostic and constructive controls, and states material boundaries. | no concrete defect |
| Limitations | 90 | 89 | `passed` | Limitations are distributed with affected claims and concentrated again in the final discussion paragraph. | no concrete defect |
| Conclusion | 88 | 90 | `failed-needs-revision` | Correct boundary and callback, but the first sentence largely repeats the Results summary and the final sentence uses a defensive negation. | **Repetitive summary plus defensive negation.** Rebuild the conclusion around the two reported quantities and the evaluation consequence; replace “without claiming...” with a direct scope statement. No conclusion-only claim may be added. |
| Figures / Tables | 89 | 91 | `failed-needs-revision` | Captions are self-contained and scopes are accurate, but several are overloaded with caveat clauses and one says what the table does not establish. | **Overloaded caption syntax and defensive caption wording.** Shorten captions by moving nonessential repetition into the body, while retaining split/teacher/independence caveats needed to interpret the plotted quantity. |
| Citation Integration | 92 | 91 | `passed` | Citations support distinct roles; no new bibliography facts or unsupported novelty claims. | no concrete defect |

## Supplement audit

The supplement is not scored against the main-paper comparator matrix. It is reviewed for reproducibility and internal consistency:

- Evidence families, split roles, and exposure caveats are explicit in Table 1 and surrounding text.
- The directional derivation, executable temporal counterexample, and gate caveat are mathematically bounded and do not claim a new theorem.
- Historical node-classification results remain separate from visual graph adapters; provenance limitations are stated.
- Centering cost proxies, timing protocols, and unresolved accounting items remain separate and are not promoted to an efficiency claim.
- No supplementary sentence in the reviewed baseline requests a new experiment or changes a main-paper numerical result.

Supplement disposition: `passed` for this prose/layout pass; no supplement edit is required unless a main-text edit creates a cross-reference or terminology inconsistency.

## Per-module comparator notes

| Module | Top comparator patterns to learn | Patterns to avoid | Revision instruction |
|---|---|---|---|
| Abstract | Observation-first opening; one result per sentence; implication in final sentence. | Packing denominator, scope, and interpretation into one clause. | Re-pace the existing chain without adding or deleting a finding. |
| Introduction | Concrete phenomenon first; one problem-to-gap path; evidence preview before a single contribution statement. | Generic field opening and repeated contribution/checklist announcements. | Front-load the finite-event failure and consolidate the closing claim. |
| Related Work | Capability grouping; explicit distinction between precedent and this study. | Author-by-author inventory without a capability topic. | Keep citations, regroup sentences around the capability they address. |
| Method / Analysis | Definition before claim; bounded algebra; reproducible intervention eligibility. | Inflating identities into theory or hiding assumptions. | No substantive edit required. |
| Results | Question-led subsection openings; decisive table plus local interpretation; uncertainty beside the statistic. | Beginning with “Table X shows” or transcribing every cell into prose. | Strengthen topic sentences and reduce numeric repetition while preserving all protected values. |
| Discussion / Limitations | Interpret evidence, then state consequence and scope. | Defensive repetition or hiding threats in a final disclaimer. | No substantive edit required. |
| Conclusion | Return to named quantities and evidence boundary; no new result. | Rewriting the Results section and ending with meta-level hedging. | Recast as a concise callback to decision, prefix, and activity. |
| Figures / Tables | Self-contained caption; plot-specific scope; keys explained in the figure where possible. | Captions that repeat the whole result section or state what the table is not. | Tighten caption syntax and preserve necessary scope labels. |

## Narrative closure summary

Core claims require full closure. Supporting and diagnostic findings require only a local interpretation.

| Selling point | Gap / claim location | Method / design embodiment | Supporting result | Discussion interpretation | Conclusion callback | Status |
|---|---|---|---|---|---|---|
| Message fidelity omits signed decision direction | Abstract, Introduction, Analysis §3.2 | Exact $\Delta M=AEW^\top$, margin condition, legal one-event changes | Table 2, Fig. 4, single-graph example, stratification | Discussion ¶1 names the omitted direction | Final paragraph calls back the decision quantity | supported and closed |
| Local terminal equality omits consumed temporal history | Abstract, Introduction, Analysis §3.3 | Source-verified codeword pairs and exact local decoded-tensor equality | Fig. 5, event totals, executable recurrence | Discussion ¶1 names the omitted trajectory | Final paragraph calls back required prefixes and activity | supported and closed |
| Calibration and centering do not remove the need for downstream validation | Introduction boundary, Results §§4.3 and 4.5 | Matched objectives, shared-prefix controls, centering comparisons | Table 3 and centering/cost results | Discussion ¶2 treats them as boundary evidence | Conclusion frames the reporting consequence, not a new method | supported and bounded |

No orphan contribution, conclusion-only claim, terminology drift, or changed causal strength was found. The paper consistently uses `audit`, `test`, `show`, and `control`; `propose` is limited to the evaluation checklist and does not introduce a converter.

## Limitation-placement ledger

| Passage / location | Classification | Why it is needed | Final location / action | Status |
|---|---|---|---|---|
| Abstract: rates are conditional on message-improving interventions | Result uncertainty | Prevents a universal-frequency reading | Keep beside the two rates; do not remove | correct |
| Introduction and Results: one teacher, exposed validation/training graphs | Material limitation | Determines evidence scope | Keep in design and Discussion; do not claim blind confirmation | correct |
| Analysis and Results: first-prefix profile and amplitude are uncontrolled in temporal exchange | Material limitation | Prevents an event-timing isolation claim | Keep beside temporal result and in Discussion | correct |
| Event semantics: whole-network activity is nonzero scalar events | Necessary factual definition | Distinguishes the quantity from latency, energy, and spike count | Keep at first definition | correct |
| Hard-intervention intervals: graph-cluster bootstrap | Result uncertainty | Repeated interventions per graph are not independent | Keep with Table 2 and stratified figure | correct |
| Centering: partial projection proxy and separate cost regimes | Material limitation | Prevents a complete-efficiency claim | Keep with centering result and supplement | correct |
| Discussion final paragraph: fixed residual GCNs, one teacher, open comparisons | Material limitation | States consequence for generalization | Keep in Discussion; do not move to abstract | correct |
| Caption wording “not superiority of a new method” and conclusion “without claiming...” | Defensive rhetoric | Not needed once the audit framing is explicit | Rewrite as direct scope statements in S7 | needs revision |

## Selective humanizer scan

| Module | Scan result | Exact passage / defect if flagged | Humanizer action | Technical-anchor and register recheck |
|---|---|---|---|---|
| Title | `no concrete defect` | — | none | Not applicable |
| Abstract | flagged | First result sentence and final checklist sentence: clause-stacked endpoint inventory and compressed pacing | Use academic profile with the named defect in S7 | Recheck all rates, percentages, denominators, invariant wording, and scope |
| Introduction | flagged | Opening paragraph and final two paragraphs: slow opening hook and repeated contribution announcement | Use academic profile with the named defect in S7 | Recheck 332/811, 296/928, recovery percentages, and checklist scope |
| Related Work | flagged | Conversion/calibration paragraph: mechanical citation inventory | Use academic profile with the named defect in S7 | Recheck all citation keys and PD-Quant / One-Spike distinctions |
| Method / Analysis | `no concrete defect` | — | none | Not applicable |
| Results | flagged | Subsection openings and temporal eligible-pool sentence: table-dependent topic sentences and clause-stacked numeric inventory | Use academic profile with the named defect in S7 | Recheck all counts, intervals, task order, split labels, and caveats |
| Discussion | `no concrete defect` | — | none | Not applicable |
| Limitations | `no concrete defect` | — | none | Not applicable |
| Conclusion | flagged | First and last sentences: repetitive summary and defensive negation | Use academic profile with the named defect in S7 | Recheck audit framing, one-teacher/exposed-data boundary, and absence of new claims |
| Figures / Tables | flagged | Table 1 and Fig. 5 captions: overloaded caption syntax and defensive caption wording | Use academic profile with the named defect in S7 | Recheck teacher count, split labels, local equality, amplitude caveat, and training-only status |
| Citation Integration | `no concrete defect` | — | none | Not applicable |

## Professionalism override check

No manuscript-production narration, journal-facing language, rhetorical self-questioning, unsupported novelty, or deployment claim was found in the scientific body. The LLM Usage Statement is a required declaration and is preserved under `declaration_mode: preserve`. The recurring “does not / not” constructions were reviewed individually; most are immediate factual qualifications required to keep a bounded claim true and are not flagged as defensive prose. Only the two passages listed as defensive rhetoric in the limitation ledger require S7 attention.

## S7 worklist (priority order)

1. Re-pace the abstract while preserving its complete evidence chain.
2. Rewrite the Introduction opening and consolidate the duplicated contribution/checklist framing.
3. Add capability-led topic sentences to Related Work without changing citation roles.
4. Convert Results topic sentences from table references to findings and reduce dense numeric repetition.
5. Tighten captions, especially Table 1 and Fig. 5.
6. Rebuild the Conclusion callback and remove the final defensive negation.
7. Recompile and re-run all hash-bound checks after any change.

No module is `blocked-needs-author-input`. The failed modules are prose/layout defects with sufficient evidence already present in the manuscript and run root.
