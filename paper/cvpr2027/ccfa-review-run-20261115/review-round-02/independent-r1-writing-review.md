# Independent R1 Writing and Positioning Review

## 1. Report Metadata

- Review date: 2026-11-16
- Target venue/year/track: CVPR 2027 main conference; paper type is an empirical/analysis paper
- Paper title: **A Finite-Event Audit of Decision and Temporal Fidelity in Graph-to-Spike Conversion**
- Input materials reviewed: frozen R1 `main.pdf` and `supplement.pdf`; both TeX sources; `revision-log.md`; `audit/verification.json`; all three derived CSVs; `references.bib`; and `../input-manifest-r1.json`
- Search basis: public title/abstract checks for PD-Quant, temporal misalignment, residual membrane potential/uneven spike arrivals, and One-Spike SNN. No private manuscript text was sent to a search service.
- Reviewer mode: fresh-context CCFA `full`, independently weighted toward writing, novelty positioning, and reviewer-facing presentation
- Scope: independent writing/positioning review with scientific consistency checks; no manuscript or package files were edited

The R1 freeze is intact. All 12 manifest entries match both byte counts and SHA-256 hashes. Re-extracting layout text from the PDFs produces files byte-for-byte identical to `audit/main.txt` and `audit/supplement.txt`. All eight main-paper pages and all four supplement pages were rendered and visually inspected.

## 2. Desk Rejection Assessment

| Check | Status | Evidence | Consequence / required action |
| --- | --- | --- | --- |
| Paper length | Pass under the stated CVPR assumption | Main PDF has seven content pages followed by the LLM statement and references; supplement has four pages | No apparent over-length risk. Recheck the official CVPR 2027 policy immediately before submission. |
| Topic compatibility | Pass | Frozen graph classifiers, ANN-to-SNN/graph-to-spike conversion, decision fidelity, temporal coding, and activity accounting are relevant to CVPR's analysis and empirical-method audience | No venue-mismatch action. |
| Minimum quality | Pass | Complete title, abstract, introduction, related work, method/analysis, results, discussion, supplementary evidence, references, limitations, and LLM statement are present | No desk-level completeness issue. |
| Policy, anonymity, and compliance | Pass on supplied artifacts | Anonymous author block; no author-identifying links in the reviewed TeX; LLM Usage Statement appears after the body and before references | Recheck official 2027 policy at submission time; no current-package repair identified. |
| Prompt injection and hidden manipulation | Pass | No reviewer-directed or model-directed instruction found in TeX, PDF text, captions, or source comments | No action. |
| Ethics and reviewability | Pass | No human-subject, privacy, or sensitive-data claim; limitations and absence of energy measurements are explicit | No ethics repair. |

Desk rejection risk: **none identified in the frozen package**.

## 3. Paper Summary and Contribution Map

The paper argues that reconstruction-oriented metrics are incomplete when a frozen graph classifier is converted to a finite-event spiking computation. It reports two controlled mismatches. First, interventions that reduce a real graph-message MSE can worsen the teacher-class margin. Second, legal two-tick codewords that give the same local decoded tensor and have the same local event count can change downstream predictions and total whole-network events. Supporting analyses localize the largest one-tick sensitivity to the first node site, compare calibration objectives, test residual centering, and separate local event sparsity from end-to-end cost.

The claimed gap is that local reconstruction fidelity does not determine downstream decision fidelity or the complete temporal and activity behavior of a graph converter. The contribution is positioned as a controlled audit plus an evaluation recommendation, not a new converter or a conversion-wide frequency estimate. That scope is the right contribution type for the evidence available.

The core evidence package is differentiated:

- 2,048 hard-event attempts per visual task, with pooled and rule/direction-stratified results;
- exact source-replay codeword exchanges on 128 fixed training graphs per task;
- three-teacher position and calibration controls;
- residual-centering comparisons and separate timing/activity accounting;
- explicit exposed-split, single-teacher, and training-only limitations.

## 4. Search and Related-Work Basis

Public searches used only paper titles and public concepts.

| Closest work | Verified public positioning | Current manuscript treatment | Residual reviewer risk |
| --- | --- | --- | --- |
| PD-Quant | Proposes post-training quantization and selects quantization parameters from prediction differences rather than layerwise feature reconstruction. See [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.html). | Related Work correctly says it is prediction-driven, but calls it “Pred-Quant” and does not state the stronger delta: PD-Quant changes quantization selection, whereas this audit changes neither converter nor calibration rule and tests when reconstruction conditions fail to determine behavior. | Moderate positioning risk, fully addressable in prose. |
| Temporal Misalignment | Identifies random spike rearrangement across layers and proposes probabilistic neurons. See [PMLR](https://proceedings.mlr.press/v267/bojkovic25a.html). | The manuscript explicitly says event order affects conversion and distinguishes its own exact graph-local endpoint/count study. | The distinction is credible, but “temporal-prefix history” is slightly stronger than the graph pair alone because the codewords also differ in absolute amplitude. |
| Residual membrane / uneven arrivals | Prior work analyzes uneven conversion error and uses residual membrane information to reduce it; see the publicly described AAAI 2023 work summarized in the citation record. | The manuscript treats this as a precedent and confines its node-classification experiment to historical context. | Low risk. The paper does not claim this mechanism as a new contribution. |
| One-Spike SNN | Proposes single-spike phase coding with base manipulation, reports GCN conversion, and reports large energy-efficiency advantages. See the [DGIST record](https://scholar.dgist.ac.kr/handle/20.500.11750/57057?mode=simple). | It is cited only as demonstrating GCN conversion; the conclusion admits that direct comparison is open. | Moderate-to-major framing risk: the title's “finite-event” language and the one-event interventions can look adjacent to a one-spike conversion method unless the diagnostic-versus-converter distinction is stated explicitly. |
| DCGS | Supplies the public differential event unit and threshold-iteration procedure used as the basis for the graph adaptation. | The adapter, source-replay constraints, and deviations are explained in the supplement. | Low risk; scope is already explicit. |

Search screening: the four closest works are user-provided citations and their public abstracts were checked. No missing decisive related work was established. The remaining issue is the precision of the comparison, not citation absence.

## 5. Expected Review Outcome

**Expected outcome:** borderline positive, **6/10**, with **confidence 4/5**.

**Main accept signal:** the paper has a legible analysis contribution, controlled counterexamples, unusually careful negative results, honest split provenance, and clear separation of teacher agreement, ground-truth accuracy, local events, and complete cost.

**Main reject signal:** the underlying phenomena are adjacent to known prediction-aware quantization and temporal-misalignment work, while the core graph evidence is limited to one teacher and exposed or training-only data. The paper must be judged as a scoped audit, not as a new conversion algorithm or a general frequency claim.

## 6. Strengths and Weaknesses

### Strengths

- The title accurately says “audit” and “fidelity”; it does not promise a new converter.
- The abstract opens with the proxy/decision gap, scopes the study to frozen superpixel classifiers, reports two findings, and warns that the pooled rates are intervention-specific.
- The introduction contains a clear problem -> two information losses -> controlled evidence -> bounded contribution progression. The central-finding paragraph is strong and avoids a method-first jump.
- Figure and table narration is generally effective. Every figure/table is introduced, captioned with data scope, and followed or preceded by the takeaway. Table 2, Figure 4, Figure 5, and Figure 6 are particularly careful about dependent attempts, local versus global cost, and proxy limitations.
- The discussion closes on the measured distinctions and limitations. It does not introduce a new claim or present the residuals as a SOTA conversion method.
- Quantitative claims directly checkable from the frozen CSVs match the manuscript: 332/811 = 40.94%, 296/928 = 31.90%, 395/811 = 48.71%, and 409/928 = 44.07%; temporal prediction changes and event deltas also reproduce.

### Weaknesses

**Weakness 1: closest-work deltas are present but unequal in precision.**

- Evidence basis: main PDF p.2, lines 101-115 groups One-Spike, temporal misalignment, “Pred-Quant,” and closed-loop alignment, then states only that these works establish the general concern.
- Reviewer deduction: a skeptical reviewer can accept the citations yet still ask whether the paper contributes a graph-specific mechanism, a counterexample, or merely another demonstration of known prediction/timing sensitivity.
- Required fix: give one explicit sentence for PD-Quant and one for One-Spike. The current paper already has enough material to make both distinctions without a new experiment.

**Weakness 2: the graph temporal experiment supports “terminal equality is insufficient,” but does not isolate temporal history from amplitude.**

- Evidence basis: main PDF p.3, lines 225-238 and p.6, lines 419-449 disclose that the codewords differ in first-prefix values and absolute amplitude. The conclusion at p.7, lines 519-521 says terminal equality “leaves downstream temporal ambiguity.”
- Reviewer deduction: a decision change can arise from the unequal first-prefix amplitude as well as event timing. The scoped existence claim is valid, but the causal label “temporal ambiguity” is stronger than the exchanged pair alone.
- Required fix: consistently describe what is controlled—terminal decoded tensor, event count, source legality, and task graph—and what is not—absolute amplitude. A small equal-amplitude codeword search would strengthen a specifically temporal claim, but is not required if the prose is narrowed.

**Weakness 3: the frozen package does not contain the raw rows and generators that the supplement says are available.**

- Evidence basis: supplement p.1, lines 010-029 say the anonymous audit package contains raw hard-intervention rows, per-graph logits, figure generators, and reproduction registers. The frozen R1 `data/` directory contains only three aggregate CSVs.
- Reviewer deduction: headline arithmetic is auditable, but cluster-bootstrap intervals, Table 1/Table 3 inputs, Figure 3/Figure 6 data, and the stated generator-level reproduction are not independently reconstructible from this freeze.
- Required fix: include the promised raw CSVs, cluster identifiers, figure generators, and hashes, or revise the supplement to distinguish the private audit store from the submitted review package.

**Weakness 4: a small but conspicuous close-work naming error remains.**

- Evidence basis: main PDF p.2, line 107 says “Pred-Quant”; the cited paper and reference entry are PD-Quant.
- Reviewer deduction: this is not a scientific error, but it weakens confidence in the otherwise careful related-work pass and is especially visible because the paper intentionally positions against PD-Quant.
- Required fix: replace “Pred-Quant” with “PD-Quant” and add the explicit audit-versus-optimization distinction.

## 7. Potentially Missing Related Work

No decisive missing citation was established from the supplied scope and public checks.

| Work/status | Why relevant | Overlap | Needed response |
| --- | --- | --- | --- |
| PD-Quant / searched and user-provided | Prediction-based evaluation is the nearest precedent for rejecting a pure reconstruction proxy | Overlaps in using downstream prediction rather than local reconstruction | Explain that this paper audits two invariants in a frozen graph event computation and does not optimize quantization parameters. |
| Temporal misalignment / searched and user-provided | Directly concerns spike ordering and low-latency conversion | Strong conceptual overlap with intermediate-prefix effects | State the unique unit of evidence: source-verified graph-local exchange with exact terminal tensor and count equality. |
| Residual membrane and uneven arrivals / user-provided | Directly concerns information omitted by output spike counts | The manuscript uses this precedent and historical node context | Keep it framed as inherited context; do not present it as a new readout mechanism. |
| One-Spike SNN / searched and user-provided | Closest one-event graph conversion method | “Finite-event” and one-event interventions invite comparison | State explicitly that this is an audit of DCGS-derived graph adapters, not a one-spike coding proposal or matched comparison. |

A multi-teacher replication or matched One-Spike implementation is not a missing citation issue. It would support a broader comparative or generalization claim that the manuscript currently declines to make.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Message-MSE improvement can worsen teacher-class margin: 40.9% MNIST and 31.9% CIFAR10 | Abstract lines 003-010; main p.5, lines 312-331; Table 2 | `hard_intervention_stratified.csv`; stratum counts; graph-level rates | Strong for the stated intervention distributions | Arithmetic reproduces exactly. The claim is correctly conditional, but intervals cannot be recomputed without graph IDs. | Keep wording; add raw/cluster rows if the audit-package claim is retained. |
| The adverse effect appears in every rule/direction stratum | Main p.5, lines 333-343; Figure 4 | `hard_intervention_stratified.csv` | Strong | All eight task/readout strata reproduce; Figure 4 does not imply independence. | No scientific edit required. |
| Exact local two-tick equality plus equal local event count can change predictions and activity | Abstract; main p.6, lines 412-449; Figure 5 | `temporal_decision_effects.csv`; source-replay construction; zero local decode error | Strong for the scoped existence claim; partial for causal “temporal” interpretation | Counts and event deltas reproduce, but unequal absolute amplitude remains an alternate explanation for the downstream changes. | Narrow the causal wording or add an optional small equal-amplitude reanalysis. |
| The first node site has the largest single-site one-tick recovery | Main p.4, lines 292-300; Figure 3 | Three-teacher position analysis and verification record | Adequate | Directionally clear; underlying per-site rows are not in the freeze. | No main-text wording change; include source rows for full auditability. |
| No tested calibration objective has a consistent accuracy advantage | Abstract; main p.5, lines 358-381; Table 3 | Matched three-teacher calibration; audit verification | Adequate but descriptive | “No consistent advantage among the tested settings” is appropriately narrower than superiority or optimality. | No overclaim fix. |
| Residual centering improves a training-split fidelity/projection proxy but not complete efficiency | Main p.6-7, lines 463-490; Figure 6; supplement lines 197-255 | Training-split ratios, validation campaign, timing limitations | Adequate | Every relevant caveat is visible: one teacher, proxy arithmetic, unresolved cost items, no hardware energy. | No overclaim fix. |
| The work is a precedent-aware audit rather than a new converter | Intro contribution lines 076-087; conclusion lines 512-546 | Citations, comparisons, explicit limitations | Adequate | Scientifically honest, but PD-Quant and One-Spike distinctions need sharper wording to make the novelty boundary unmistakable. | Add the two exact deltas in Related Work. |
| The frozen package reproduces the claimed aggregate tables | Supplement lines 010-029 | Manifest, aggregate CSVs, verification JSON | Partial | Headline numbers and strata reproduce; the described raw rows, generators, and several table/CI inputs are absent from the freeze. | Complete the package or revise the claim. |

Overall claim-evidence consistency: **pass with a reproducibility caveat**. No numerical contradiction was found. The main writing risk is semantic: several claims are valid when read with the carefully stated scope, but a few short sentences such as “temporal ambiguity” can be read as stronger than the evidence.

## 9. Experiment, Benchmark, and Reproducibility Audit

- **Baselines and controls:** appropriate for this audit. The paper does not claim a new method, so a SOTA converter baseline is not automatically required. Rule/direction strata, per-graph prevalence, both exchange directions, objective controls, topology controls, and proxy-versus-timing comparisons address the stated questions.
- **Datasets and splits:** scope is unusually transparent. Exposed official test/validation data are labeled descriptive, and core exact-equality tests are explicitly training-only.
- **Statistical rigor:** descriptive SDs and graph-cluster bootstrap are appropriate for the two headline rates. The frozen aggregate CSV is insufficient to recompute the intervals because graph identifiers and intervention rows are absent.
- **Robustness and failure analysis:** strong at the level of the audit. The paper reports adverse individual decisions, opposite-direction agreement changes, alternative calibration families, failed full-cost criteria, and explicit assumptions.
- **Reproducibility:** the main/supplement TeX, bibliography, compiled PDFs, hash manifest, three aggregate CSVs, and verification record are present. Raw hard-intervention rows, temporal per-graph logits, checkpoints, dataset installation, and figure generators are not present in the freeze.
- **Limitations:** strong and specific. One teacher, exposed data, training-only exact-equality studies, no direct One-Spike comparison, no topology changes, no complete cost account, and no chip energy measurement are all stated.

The text was not independently recompiled because the frozen `sources/` directory contains TeX and bibliography sources but not the CVPR style file or all figure-generation inputs. The supplied PDFs are readable, visually coherent, and exact text matches to their audit files.

## 10. Multi-Reviewer Panel

These are independent role lenses applied to the same frozen evidence.

| Reviewer lens | Likely score/tendency | Confidence | Main positive signal | Main negative signal | Score-change condition |
| --- | --- | --- | --- | --- | --- |
| Best-justified reviewer | 7 / weak accept | 4 | Controlled graph-specific mismatches that force separate decision and activity reporting | The contribution is an audit rather than a new capability | Sharper close-work framing and corrected package claims could support the lean-accept band. |
| Critical reviewer | 5 / borderline negative | 4 | The paper knows and admits its narrow scope | Known phenomena, one-teacher core tests, and unequal amplitude in the temporal exchange | State the novelty as evidence about a frozen DCGS graph adapter and narrow temporal causality. |
| Method/soundness | 4/5 axis | 4 | Exact forward pass, source replays, directionally correct margin condition, and explicit event semantics | “Temporal prefix” and absolute amplitude are not independently varied | A wording fix is sufficient for the current scoped claim; an equal-amplitude family is the optional small strengthening. |
| Evidence/experiment | 3/5 axis | 4 | Headline counts, strata, event deltas, and several caveats | No second teacher/task for the core counterexamples; package lacks raw rows | Add raw rows now; broader replication would require a large experiment and a broader claim. |
| Novelty/positioning | 3/5 axis | 5 | PD-Quant, temporal misalignment, residual membrane, and One-Spike are cited; the paper explicitly declines a new-converter claim | Related-work prose does not make the exact delta from PD-Quant and One-Spike vivid enough | A manuscript-only positioning pass could raise this axis to a weak 4 without changing the science. |
| Writing/clarity | 4/5 axis | 5 | Clear logical progression, self-contained figure/table captions, scoped abstract, strong limitations | “Pred-Quant,” undefined first use of “activity,” and temporal terminology | Closed by four local manuscript edits. |
| Reproducibility | 3/5 axis | 4 | Hashes, source TeX, aggregate CSVs, exact PDF-audit match, and clear split labels | Supplement promises raw rows/generators not present in the frozen package | Include the promised package files or narrow the claim. |
| AC/meta-reviewer | 6 / borderline positive | 4 | Honest, inspectable audit with no fatal claim failure | Novelty depends on exact positioning and careful reading of scope | Manuscript/package cleanup can move the decision to 7; broader claims require evidence beyond this paper. |

**Agreement:** all lenses see a coherent, scoped audit with no discovered numerical contradiction.

**Disagreement:** clarity and novelty diverge. The writing is good enough to make the contribution understandable, while the contribution itself is still vulnerable to “known phenomenon, small controlled demonstration” criticism.

**Decisive positive axis:** controlled, graph-specific evidence of two distinct fidelity failures under explicitly bounded assumptions.

**Decisive negative axis:** proximity to prediction-aware and temporal-misalignment work without maximally explicit deltas.

**Unresolved evidence:** raw per-graph/figure rows; independent multi-teacher or multi-task replication; amplitude-matched temporal pairs.

**AC stance:** **borderline positive, 6/10**, with manuscript-only fixes to 7 and no justification for large experiments under the current scope.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 | Major | PD-Quant and One-Spike are named but their exact technical deltas from this audit are not explicit enough. | Main p.2, lines 101-115; conclusion lines 535-540 | Novelty, positioning, reviewer-facing risk | Writing / related-work | Replace “Pred-Quant” with “PD-Quant”; add one sentence comparing its parameter-selection goal with this audit’s frozen-counterexample goal; add one sentence separating DCGS-adapter diagnostics from single-spike phase coding. | `ccf-paper-writer` | Closes the highest-impact novelty-communication risk and supports 6 -> 7 without new evidence. |
| W2 | Major | The graph codeword exchange is described as temporal evidence although the two codewords also differ in absolute amplitude. | Main p.3, lines 222-238; p.6, lines 412-449; p.7, lines 519-521; supplement lines 130-143 | Soundness, claim-evidence alignment | Writing / claim qualification; optionally small reanalysis | Use “terminal equality plus fixed event count does not determine downstream trajectory” as the claim. State that first-tick profile and amplitude both differ. Only use “temporal-only” if an amplitude-matched family is run. | `ccf-paper-writer`; optional experiment owner | Closes the main semantic overclaim. Optional small result would support a stronger mechanism claim but is not needed for the current existence claim. |
| W3 | Moderate | The supplement promises raw audit data and generators that are absent from the frozen R1 package. | Supplement p.1, lines 010-029; frozen `data/` contains only three aggregate CSVs | Reproducibility, trustworthiness | Reproducibility / package cleanup | Include the promised rows, cluster identifiers, figure generators, and hashes, or revise the supplement to distinguish private records from submitted records. | `ccf-paper-writer` + experiment owner | Removes a direct package-to-text mismatch; no new experiment required if the files already exist. |
| W4 | Moderate | The reviewed term “activity” is not defined at first use and can be confused with latency, energy, or delivery traffic. | Abstract lines 011-021; event semantics begin at lines 141-155; supplement defines nonzero scalar events in Table 2 | Terminology consistency | Writing | After defining signed scalar events, add: “We call the total number of nonzero scalar events whole-network activity.” Use the same phrase throughout. | `ccf-paper-writer` | Small terminology-risk reduction. |
| W5 | Minor | The revision log points to `audit/verification-r1.json`, while the frozen file is `audit/verification.json`. | `revision-log.md`, Verification paragraph | Package auditability | Package metadata | Correct the stale filename in the review log. | Review-package owner | No score effect; prevents avoidable provenance confusion. |
| W6 | Minor | “`This is not confined to teachers with a small top-two margin`” is supported by one example and reads as a plural empirical generalization. | Main p.5, line 327 | Prose discipline | Writing | Rewrite as “The mismatch also occurs in one teacher example with a large top-two margin,” or equivalent singular wording. | `ccf-paper-writer` | No substantive score effect; improves precision. |

## 12. AC / Meta-Review

- **Reviewer consensus:** the R1 text is substantially clearer than a baseline version would be. The title, abstract, and contribution paragraph now state that the work is an audit, not a converter. Claims about exposed data and intervention frequency are bounded.
- **Reviewer disagreement:** a clarity reviewer can recommend weak accept, while a novelty reviewer can prefer borderline because PD-Quant and temporal misalignment already establish prediction-aware and order-sensitive concerns.
- **Decisive acceptance axis:** whether readers accept the graph-specific finite-event mismatch as useful, inspectable evidence rather than an obvious restatement.
- **Decisive rejection axis:** whether the paper appears to claim novelty for the general observation that reconstruction, decisions, and temporal trajectories differ.
- **AC stance:** **6/10 borderline positive**. The strongest unresolved concern is positioning, not correctness. W1 and W2 can be repaired without new large experiments.
- **Discussion risk:** the authors should not answer novelty concerns by broadly claiming generality. The safer route is to emphasize controlled invariant failures in a frozen DCGS-derived graph computation.

## 13. Quantitative Scores

### Scientific/dimension scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
| --- | :---: | :---: | --- | --- |
| Novelty | 3 | 5 | PD-Quant, temporal misalignment, residual-membrane, One-Spike, and DCGS are cited; the paper identifies graph-specific controlled evidence | General phenomena are known. A clear audit-versus-optimizer and DCGS-versus-One-Spike delta could support a weak 4, but would not create a new-method contribution. |
| Soundness | 4 | 4 | Actual hard forward passes, source replay, exact margin condition, and explicit event semantics | The temporal mechanism label exceeds the amplitude-controlled variation. Narrowing the label closes the main issue. |
| Evidence | 3 | 4 | Headline rows reproduce; core graph tests use one teacher and exposed/training-only data | Existence claims are adequately supported, but generalization and package-level reconstruction are incomplete. New raw rows are a cheap fix; new teachers/tasks are a large experiment. |
| Significance | 3 | 4 | The audit exposes a practical distinction between proxy fidelity, decisions, and complete cost | The lesson is useful but narrow and overlaps known prediction-aware evaluation. A broader converter comparison would be a larger claim. |
| Clarity | 4 | 5 | Strong storyline, self-contained captions, scoped conclusion, effective tables/figures | Four local fixes remain: PD-Quant name, One-Spike delta, temporal/amplitude wording, and activity definition. |
| Reproducibility | 3 | 4 | Source TeX, PDFs, three CSVs, manifest, and verification JSON are present | Raw rows, figure generators, checkpoints, and several inputs are absent. Package completion could raise this dimension. |
| Ethics / Limitations | 5 | 5 | Exposed data, one teacher, proxy cost, training-only tests, absent energy measure, and direct One-Spike comparison are disclosed | No deduction. |

**Overall:** **6/10**  
**Scholarly confidence:** **4/5**  
**Recommendation:** **borderline positive**

### Writing scorecard

| Dimension | Weight | Score (1-5) | Confidence | Concrete repair |
| --- | ---: | ---: | ---: | --- |
| Storyline and motivation | 10 | 4 | 5 | Preserve the current problem -> two losses -> audit progression. |
| Contribution display | 10 | 4 | 5 | Make the comparison with PD-Quant and One-Spike explicit. |
| Paragraph logic | 10 | 4 | 5 | Split the one-sentence comparison cluster in Related Work. |
| Claim-evidence alignment | 12 | 4 | 4 | Narrow “temporal ambiguity” to the exact controlled and uncontrolled variables. |
| Method readability | 9 | 4 | 5 | No material rewrite; terminology is mostly stable. |
| Experiment narration | 9 | 4 | 5 | Figures, tables, and captions already state scope and takeaway. |
| Related-work positioning | 8 | 3 | 5 | Correct PD-Quant and state exact deltas on method, comparison, and evidence. |
| Terminology and notation | 8 | 3 | 5 | Fix PD-Quant; define activity at first use; separate prefix, amplitude, and trajectory. |
| Prose discipline and voice | 10 | 4 | 5 | Replace the one plural overgeneralization and retain the bounded language. |
| LaTeX and format discipline | 6 | 5 | 5 | No visible LaTeX/layout defect in the compiled artifacts. |
| Reviewer-facing risk | 8 | 3 | 5 | Close W1-W3 to remove the avoidable “incremental and overbroad” reading. |

**Writing score:** **3.82/5**  
**Writing risk band:** **moderate**  
**Writing confidence:** **5/5**

## 14. Questions for Authors

1. Does “temporal-prefix history” deliberately denote both the first-tick event profile and its unequal absolute amplitude, or only event order? The answer should appear once in the results before the term is reused.
2. If the anonymous package is expected to reproduce the supplement's raw-row and figure claims, can the missing per-graph CSVs, cluster identifiers, generators, and hashes be added to the frozen package?
3. Is any novelty claim intended against One-Spike SNN as a converter baseline? If not, state that explicitly. If yes, the current evidence does not support that comparison.

## 15. Score Revision Criteria

**Raising the score would require:** completing W1-W4. The likely movement is from 6 to 7 by making the audit contribution and controlled variables unmistakable. An optional amplitude-matched codeword family could strengthen method confidence but is unlikely to move the paper alone.

**Lowering the score would be triggered by:** retaining a conversion-wide or temporal-only causal interpretation despite the disclosed amplitude difference, or presenting the missing raw package as already reproducible.

**Concerns unlikely to change before submission without a broader project:** multi-teacher/multi-task replication, a matched One-Spike implementation, feature-dependent gate perturbation analysis, topology changes, and hardware energy experiments.

## 16. Action Plan and CCFA Handoffs

| Priority | Action | Fix class | Input needed | Expected output | Handoff |
| --- | --- | --- | --- | --- | --- |
| P1 | Correct “Pred-Quant” to “PD-Quant” and state the optimizer-versus-audit distinction. | Manuscript-only | Existing manuscript and cited abstract | One or two precise sentences in Related Work | `ccf-paper-writer` |
| P1 | State the One-Spike phase-coding/converter distinction and clarify that one-event modifications are diagnostic interventions inside DCGS-derived graph adapters. | Manuscript-only | Existing manuscript and cited abstract | Explicit novelty boundary; no new comparison implied | `ccf-paper-writer` |
| P1 | Rewrite the graph temporal conclusion around what is controlled: source-legality, exact terminal tensor, local event count, and task graph. Explicitly include unequal first-prefix amplitude as an uncontrolled variable. | Manuscript-only | Main lines 222-238 and 412-449 | Claim that matches the existing pair | `ccf-paper-writer` |
| P2 | Define whole-network activity as the total nonzero scalar event count at first use. | Manuscript-only | Existing event definition | Stable terminology across abstract, results, figures, and conclusion | `ccf-paper-writer` |
| P2 | Include the promised raw rows, figure generators, cluster identifiers, and hashes, or revise the supplement's package statement. | Package-only if artifacts exist; otherwise manuscript-only | Existing audit store | Text and frozen package agree | Reproduction owner |
| P3 | Optional: enumerate source-verified, equal-absolute-amplitude codeword families on the existing graphs and replay them. | Small reanalysis | Existing checkpoints and registered graph set | Stronger evidence for a temporal-only interpretation | Only if a stronger mechanism claim is desired |
| Not recommended now | Multi-teacher/multi-task replication or a matched One-Spike implementation. | Large experiment | Fresh inference/training and a broader claim | Comparative/generalization evidence | Outside the no-large-experiment constraint |

## Diminishing-Returns Verdict

**Reached for the stated no-large-experiment constraint.** The must-fix items are manuscript- or package-only. One small equal-amplitude reanalysis is optional and should be run only if the authors want to retain a stronger temporal-only mechanism claim. Multi-teacher replication, a matched One-Spike baseline, and new task families would support broader claims but are not justified by the paper's current scoped existence statements. More experiments are therefore unlikely to improve the writing review materially before the pending prose and package defects are fixed.

## Checks Run, Skipped, and Unresolved

- **Checks run:** manifest byte/SHA-256 verification (12/12); exact PDF-audit text comparison; visual review of all 12 PDF pages; TeX, bibliography, revision log, verification JSON, and CSV inspection; independent recomputation of headline rates, per-graph counts, and event deltas; section/label/reference scans; public related-work abstract checks; claim-evidence and figure/table narration review.
- **Checks skipped:** independent LaTeX recompilation because the frozen source package lacks the style file and all figure-generation inputs; execution of network inference because it is outside the no-large-experiment constraint and the checkpoints/data are absent.
- **Unresolved:** raw per-graph rows and generator inputs; cluster IDs needed for exact interval reproduction; amplitude-matched temporal pairs; current official CVPR 2027 policy should be rechecked by the submission owner.

## Output Self-Check

- Mode and target venue are explicit.
- Scores and confidence are separated.
- Every dimension score of 3 or below includes a deduction and repair condition.
- Claim-evidence rows cite exact PDF or supplement locations.
- Concerns identify severity, fix class, owner, and closure condition.
- Figure/table narration, terminology, and prose overclaims are explicitly assessed.
- The manuscript-only, small-reanalysis, and large-experiment boundary is explicit.
- The diminishing-returns verdict is consistent with the existing scoped claims.
