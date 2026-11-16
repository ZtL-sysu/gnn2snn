## 1. Report Metadata

Review date: 2026-11-16  
Target venue/year/track: CVPR 2027, main conference; full research paper assumed  
Paper title: Beyond Reconstruction: Decision and Temporal Fidelity in Graph-to-Spike Conversion  
Input materials reviewed: `sources/main.pdf`, `sources/supplement.pdf`, `sources/main.tex`, `sources/supplement.tex`, `sources/CLAIMS.md`, `audit/main.txt`, `audit/supplement.txt`, and `audit/verification.json`. The frozen `input-manifest.json` was found one level above `frozen/`; all 12 listed files match their recorded byte counts and SHA-256 hashes.  
Search basis: Public-safe generic queries only; no manuscript sentences, private numerical results, or identifying text were sent to search. Public proceedings, official CVPR pages, and publisher/index pages were used.  
Report file: `review-round-01/independent-scientific-review.md`  
Reviewer mode: Fresh-context independent CCFA scientific reviewer; `full`; no manuscript edits; no sub-agents.

## 2. Desk Rejection Assessment

- **Paper length — pass.** The rendered main PDF has eight pages: seven body pages and one page containing the LLM statement and references. The supplement has four pages. The supplied verification reports 7 body pages, 8 total pages, and no overflow. This is compatible with the official CVPR 2027 eight-page limit excluding references and the LLM statement.
- **Topic compatibility — pass with significance risk.** Visual superpixel graph conversion and event-based inference are relevant to CVPR. The main risk is whether a diagnostic analysis with two small graph-classification tasks and no new converter is substantial enough for the venue.
- **Minimum quality — pass.** The paper has a coherent question, explicit recurrence and graph computation, real hard-event interventions, matched controls, limitations, and readable figures. Scientific novelty and evidence breadth remain review-level concerns.
- **Policy/anonymity/compliance — pass for visible PDF content; administrative items unverified.** Main and supplement use anonymous author blocks and review-mode CVPR formatting. No author-identifying acknowledgment or external project link is visible. Author enrollment, reciprocal-reviewer eligibility, conflicts, and dual submission cannot be assessed from the frozen files.
- **Prompt injection and hidden manipulation detection — pass on inspected text/source and rendered pages.** No reviewer-directed hidden instruction was found in the TeX, extracted text, or visual inspection. The LLM statement is disclosure, not an instruction to a reviewer.
- **Ethics and reviewability — pass with artifact caveat.** Standard public image/graph benchmarks are used; no human-subjects or sensitive-data issue is apparent. The extensive LLM disclosure is present in the required location. The manuscript's scientific results are not independently reproduced here; `verification.json` checks source/derived-asset integrity rather than experimental correctness.

Desk rejection risk: **low on visible formatting and policy; scientific rejection risk remains high.**  
Reason: The official CVPR 2027 author guidelines require an LLM statement and permit eight pages plus references/statement; both are present. Administrative eligibility is unknown.  
Can be fixed before review? The visible package is reviewable; author-side administrative checks remain external.

## 3. Paper Summary And Contribution Map

The paper studies when activation/message reconstruction is insufficient to preserve decisions in low-timestep graph-to-spike conversion. On residual GCN superpixel classifiers, legal one-level event changes that reduce immediate graph-message MSE nevertheless worsen the teacher-class margin in 332/811 MNIST and 296/928 CIFAR10 selected interventions. A separate temporal construction exchanges legal two-tick codewords with exactly equal local decoded tensors and local nonzero-event counts, then observes changed downstream predictions and whole-network activity. Three-teacher ideal-ReLU recovery localizes substantial one-tick sensitivity to the first node site. Calibration, graph-permutation, residual-centering, and historical node-classification experiments provide controls and context, not a new state-of-the-art converter.

- **Claimed problem:** Existing conversion evaluation over-relies on activation/message reconstruction and local activity proxies.
- **Claimed gap:** Actual graph-message error omits decision direction; exact terminal local reconstruction omits the temporal trajectory seen by downstream units.
- **Method/contribution map:** (i) linear-algebraic message/logit and class-margin analysis; (ii) legal hard-event interventions through the actual nonlinear converter; (iii) exact two-tick codeword exchanges with source replay; (iv) three-teacher ideal position recovery; (v) matched calibration and centering controls that delimit what proxies establish.
- **Evidence package:** Three-teacher exposed-test fidelity; three-teacher exposed-validation position recovery; one-teacher exposed-validation hard interventions; one-teacher training-only temporal exchanges; one-teacher GatedGCN centering; historical node results; graph-cluster bootstrap intervals for selected hard-event rates.
- **Stated limitations:** Validation/test exposure, one-teacher hard/temporal tests, no independent task replication or direct One-Spike comparison, hybrid rather than fully spiking GatedGCN computation, incomplete cost accounting, no chip energy, and no claim of complete runtime advantage.

## 4. Search And Related-Work Basis

Queries used: Generic combinations of `ANN SNN conversion temporal misalignment spike timing reconstruction error decision accuracy`, `ANN SNN conversion output logit loss task aware calibration decision boundary`, `graph neural network spiking conversion quantization aggregation`, `spiking graph neural networks superpixel MNIST CIFAR conversion`, and exact titles of public papers surfaced by those queries. Official `CVPR 2027 Author Guidelines` and `Reviewer Guidelines` were searched separately for current policy.

Sources searched: PMLR ICML 2025/2026, AAAI Proceedings, CVF Open Access, IJCV/DBLP metadata, OpenReview/arXiv result pages where surfaced, and the official CVPR 2027 website. Search status: **completed but not exhaustive**. No private manuscript wording was used.

Closest works found:

- [Bojkovic et al., ICML 2025, Temporal Misalignment](https://proceedings.mlr.press/v267/bojkovic25a.html): directly establishes that spike rearrangement changes low-latency conversion behavior. The manuscript cites it, but must make the exact-terminal-equality/count-controlled graph delta more prominent.
- [Hao et al., AAAI 2023, Residual Membrane Potential](https://ojs.aaai.org/index.php/AAAI/article/view/25071): explicitly analyzes uneven spike-arrival sequences and short-timestep conversion error. It is absent from the bibliography and materially affects temporal novelty framing.
- [Li et al., IJCV 2024, Error-Aware Conversion](https://doi.org/10.1007/s11263-024-02046-2): analyzes propagation of local conversion error; relevant to the claimed spatial/error-propagation framing.
- [Li et al., ICML 2026, Closed-Loop CKA Distillation](https://proceedings.mlr.press/v306/li26db.html): combines local feature and global output alignment and reports horizon-dependent roles. It is contemporaneous under the official CVPR 2027 date rule, so it should not be treated as proof of novelty collapse, but it is important positioning for the calibration discussion.
- [Bu and Yu, CVPR 2025, Inference-Scale Complexity](https://openaccess.thecvf.com/content/CVPR2025/html/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.html): relevant to the paper's insistence that local event counts are not complete efficiency measures.

Unverified related-work risks: The search was not a systematic survey of every graph-SNN, temporal-code, adversarial-margin, or decision-aware quantization paper. No claim of first-ever novelty is warranted.  
Source-quality screening status: Proceedings and official publisher/conference pages were prioritized; aggregators were not used as the sole basis for a scientific overlap conclusion.

## 5. Expected Review Outcome

Expected outcome: **Weak reject, overall 4/10.**  
Main accept signal: Both central existence claims are supported within the explicitly stated networks and codes by realizable interventions rather than correlation alone. The temporal local-equality control is unusually clean, and the manuscript is candid about exposure and incomplete efficiency evidence.  
Main reject signal: The contribution currently reads as a careful but narrow confirmation/extension of known temporal-misalignment and decision-boundary intuitions. Headline spatial frequencies describe an arbitrary intervention distribution on one exposed-split teacher, while the broader contribution and CVPR-level significance remain insufficiently differentiated from close work.  
Confidence: **4/5** on the frozen manuscript and internal logic; **3/5** on novelty because the public search was targeted rather than exhaustive. Raw experimental artifacts and independent reruns were outside the permitted frozen review inputs.

## 6. Strengths And Weaknesses

Strengths:

- The spatial argument uses actual legal events and the full nonlinear forward pass, avoiding a claim based only on a linearized suffix or an MSE–accuracy correlation (Sections 3.2 and 4.2; Fig. 4; Table 2).
- The temporal claim has an exact invariant: bitwise-equal local decoded tensors and equal local nonzero-event counts, with source replay and unchanged downstream execution (Sections 3.3 and 4.4; Fig. 5).
- The margin definition correctly allows the competing class to change, and teacher agreement is distinguished from ground-truth accuracy (Section 3.2).
- Three-teacher ideal position recovery is a useful sensitivity diagnostic, and the paper does not confuse continuous recovery with a realizable finite-event correction (Section 4.1; Fig. 3).
- The actual-versus-conjugated/feature-only controls and negative graph-structure result are reported honestly rather than used to imply an adjacency advantage (Section 4.3).
- The paper separates local events, scalar amplitude, graph deliveries, partial additions, latency, and energy; centering is not promoted to an unsupported end-to-end efficiency win (Section 4.6; Supplement Section 5).

Major weaknesses:

1. **Weakness: Novelty is not yet demonstrated strongly enough for CVPR.**  
   Evidence basis: Only nine references are included. Temporal misalignment and uneven spike arrival are already central in Bojkovic 2025 and Hao 2023; the margin derivation is explicitly elementary linear algebra. The new controlled graph experiments are useful, but the introduction does not state a sharp technical delta from those precedents.  
   Reviewer deduction: A skeptical reviewer can summarize the paper as “norms omit direction and spike timing matters,” both already known, with limited graph-specific empirical scope.  
   Required fix: Add a closest-work comparison by invariant, task operator, intervention type, and conclusion. Position the exact terminal/count-controlled graph exchange as the specific delta, cite Hao 2023 and error-propagation/output-alignment work, and narrow universal wording.

2. **Weakness: The spatial headline rates lack an intervention-distribution interpretation and null comparison.**  
   Evidence basis: Four attempts per graph arise from two coordinate rules and two directions, but Table 2 pools them. It reports adverse rates only conditional on message improvement, with no per-rule/per-direction breakdown, random-legal-move reference, effect sizes, or per-graph prevalence.  
   Reviewer deduction: The 40.94% and 31.90% values can be read as properties of graph conversion generally, although they depend on the selected coordinate/eligibility distribution and one teacher. The existence claim survives; the frequency interpretation is exposed.  
   Required fix: Reanalyze existing rows by rule, direction, task, graph, baseline margin, and intervention size; show all-attempt denominators and a predeclared random-legal or residual-agnostic reference. Keep the rates explicitly descriptive.

3. **Weakness: Exposure and teacher scope prevent confirmatory generalization.**  
   Evidence basis: All visual validation and test splits were previously used in development. Hard and temporal tests use teacher 41 only; temporal tests use the first 128 training graphs; graph-cluster intervals quantify graph resampling, not teacher or pipeline variability.  
   Reviewer deduction: The disclosure is excellent, but it does not create independent confirmation. Three-seed SDs on shared exposed examples are not evidence of unseen-task robustness.  
   Required fix: Freeze the rules and evaluate untouched data across all teachers and at least one additional graph setting; otherwise keep every numerical conclusion at the current descriptive, one-teacher scope.

4. **Weakness: The central empirical package is fragmented by secondary studies.**  
   Evidence basis: Residual centering, source threshold iteration, historical node classification, and gate bounds occupy substantial space but do not independently validate the two central claims. The main novelty claim appears only after several different adapters and evidence levels.  
   Reviewer deduction: The breadth makes the paper look like several negative diagnostics rather than one decisive graph-conversion result.  
   Required fix: Reorganize around the two central claims, move historical node context and nonessential centering detail to supplement, and state what each retained control rules out.

5. **Weakness: Reproduction of headline metrics is not fully specified in the manuscript.**  
   Evidence basis: “Deterministic coordinate” selection is not defined precisely enough to reimplement; frozen inputs provide hashes and derived-asset provenance but not the raw experimental rows, scripts, teacher checkpoints, or independent rerun logs. `audit/verification.json` verifies manuscript assets, not numerical recomputation.  
   Reviewer deduction: The reported evidence is auditable in principle but not independently verifiable from this package.  
   Required fix: Specify exact selection/eligibility algorithms and seed/teacher IDs, provide the raw rows and deterministic analysis code, and distinguish table/figure reconstruction from network-level reproduction.

## 7. Potentially Missing Related Work

1. **Work:** Hao et al., *Reducing ANN-SNN Conversion Error through Residual Membrane Potential*, AAAI 2023.  
   **Status:** Searched; official AAAI page inspected.  
   **Why relevant:** It analyzes unevenness from different temporal spike-arrival sequences under short horizons.  
   **Overlap:** Close conceptual precedent for the temporal-history claim.  
   **Needed comparison:** Explain that the present experiment holds terminal decoded activation and local event count exactly fixed inside a graph adapter, whereas prior work analyzes/mitigates unevenness; do not imply the broader phenomenon is new.

2. **Work:** Li et al., *Error-Aware Conversion from ANN to SNN via Post-training Parameter Calibration*, IJCV 2024.  
   **Status:** Searched; DOI and bibliographic metadata verified, abstract inspected through indexed results.  
   **Why relevant:** Local conversion-error decomposition and propagation are close to the spatial motivation.  
   **Overlap:** Error propagation and calibration rather than generic activation mismatch.  
   **Needed comparison:** Distinguish signed margin ordering under actual graph propagation and hard legal events from second-order/local calibration analysis.

3. **Work:** Li et al., *High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation*, ICML 2026.  
   **Status:** Searched; PMLR page inspected; contemporaneous under CVPR 2027 rules.  
   **Why relevant:** It directly compares local and global output alignment across horizons.  
   **Overlap:** Section 4.3's local/output/mixed calibration conclusion.  
   **Needed comparison:** Cite and contrast optimizer, supervision, horizon, and task scope; no novelty-collapse claim is made from contemporaneous work.

4. **Work:** Bu and Yu, *Inference-Scale Complexity in ANN-SNN Conversion*, CVPR 2025.  
   **Status:** Searched; CVF page inspected.  
   **Why relevant:** Complete inference complexity and low-power evaluation bear on local-event/proxy conclusions.  
   **Overlap:** Section 4.6's separation of sparsity, operations, and runtime.  
   **Needed comparison:** Use it to position the cost-accounting checklist; no new efficiency baseline is required unless an efficiency advantage is claimed.

5. **Work:** Direct One-Spike GCN comparison.  
   **Status:** User-provided citation in the manuscript; direct new comparison not run.  
   **Why relevant:** It is the closest cited graph converter with a different temporal code.  
   **Overlap:** Graph conversion rather than the diagnostic invariant itself.  
   **Needed comparison:** A direct benchmark is optional for the narrow existence claims but necessary if the paper implies broad graph-converter ranking or practical superiority.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| **Central spatial claim:** improving the actual graph-message norm need not improve/preserve the decision | Abstract; Intro; Sections 3.2, 4.2; Fig. 4; Table 2 | Legal first-site event changes; exact message perturbation; full nonlinear replay; 332/811 and 296/928 adverse cases; example flip | **Adequate for existence in teacher 41 and these two tasks; weak for general frequency** | This is the strongest spatial evidence. It does not show that the rates transfer across coordinate rules, teachers, converters, or tasks | Retain the existence claim; provide breakdowns/null reference and independent replication before generalizing rates |
| **Central temporal claim:** equal local terminal decoding and event count need not preserve downstream predictions/activity | Abstract; Intro; Sections 3.3, 4.4; Fig. 5; Supplement 3.2 | Source-recurrence counterexample; legal codeword pairs; bitwise local equality; equal local count; real-network global changes | **Strong for existence in the tested adapter; adequate internally** | The invariant is clean and the causal path is credible. Equal count does not equalize amplitude cost; the paper discloses this. Generality remains untested | Keep exact scope; report eligible-set sizes and paired effect sizes; replicate across teachers/tasks and optionally add equal-amplitude controls |
| “Reconstruction equivalence is weaker than decision equivalence in both space and time” | Introduction | Two different experiments and standard linear/state arguments | **Adequate as scoped informal statement** | “Equivalence” is broader than the tested metrics: spatial evidence concerns MSE improvement, not every reconstruction equivalence | Define precisely which reconstruction function/equality is meant or change to “improving local reconstruction need not preserve decisions” |
| First node activation is a major low-timestep bottleneck | Abstract; Section 4.1; Fig. 3 | Three-teacher full-validation ideal-ReLU recovery; 54.08%/23.54% first-site gap closure | **Adequate for conditional sensitivity** | Nonadditive ideal recovery does not identify a realizable correction or an additive causal contribution | Say “largest single-site ideal recovery in these teachers”; avoid mechanism-independent causal wording |
| High-prefix fidelity coexists with severe first-prefix loss | Table 1; Sections 4.1, 4.3; Supplement Table 3 | Three teachers on exposed test; source-linked threshold controls on exposed validation | **Adequate descriptively** | Prior exposure prevents blind confirmation, but the coexistence is plausible and directly measured | Preserve exposure labels; add fresh evaluation if used as confirmatory evidence |
| Output-aware calibration is not consistently superior | Table 3; Section 4.3; Supplement Table 2 | Matched parameters/training graphs/updates; small mixed-sign differences; third-class failure of top-two objective | **Adequate for the tested optimizer/settings** | Does not establish that output-aware objectives are generally ineffective; search shows closely related local/global alignment work | Keep optimizer-specific wording and position against IJCV 2024/ICML 2026 |
| Actual graph structure gives a special aggregate advantage | Section 4.3 | Actual, feature-only, conjugated controls; small incremental in-sample R² | **Negative/mixed evidence** | The paper appropriately weakens rather than supports this attribution | Retain the negative result; do not use “graph-aware” as an established advantage |
| Residual centering improves representation and a partial projection proxy | Section 4.5; Fig. 6; Supplement 5 | Shared thresholds; 45k training graphs; three-teacher exposed-validation fidelity; separate timings | **Adequate for partial proxy; absent for complete efficiency** | Latency and unresolved cost items contradict an end-to-end efficiency claim | Retain only the scoped representation/proxy conclusion or cut to focus |
| “Establish practical conditions” for finite-event evaluation | Abstract; Discussion | Checklist derived from counterexamples and cost controls | **Heuristic, not sufficient/necessary conditions** | “Establish” sounds more formal/general than the evidence | Call it an evaluation checklist or empirical criteria in the tested setting |

**Two-central-claim verdict:** Both are supported as carefully scoped existence/mechanism claims. Neither supports a universal frequency, a general impossibility theorem, or superiority of a new conversion algorithm. That distinction is decisive to the score.

## 9. Experiment / Benchmark / Reproducibility Audit

- **Baselines:** Appropriate for diagnostic controls: local/output/mixed/KL/ranking calibration, actual/feature/conjugated graph operators, source threshold iteration, and raw/centered coding. A direct One-Spike comparison is not essential to prove the two counterexamples, but broader graph-conversion claims would require it. No new SOTA baseline should be demanded merely to compensate for weak novelty framing.
- **Ablations:** Ideal site/group recovery and objective controls are useful. Missing decisive ablations are per-rule/per-direction hard-event analysis, teacher replication for hard/temporal tests, and decomposition of residual versus event-boundary effects. Centering does not isolate all downstream cost components.
- **Datasets/benchmarks:** MNIST and CIFAR10 superpixels are standard and suitable for controlled graph classification. Two related tasks, one fixed residual architecture for central interventions, and training/exposed splits limit breadth. Historical node datasets are contextual, not independent validation of the visual claims.
- **Metrics:** Teacher agreement, ground-truth accuracy, teacher-class margin against the current strongest competitor, immediate graph-message MSE, local decoded equality, global activity, and latency are appropriately distinguished. Margin change should be accompanied by effect-size and flip summaries, not only conditional adverse fractions.
- **Statistical rigor:** Graph-cluster bootstrap correctly keeps four interventions from a graph together, and three-teacher sample SDs are correctly described. The intervals do not cover teacher variability, intervention-rule selection, or prior split exposure. Conditioning on message improvement makes the denominator selection-dependent. No significance claim is necessary for the existence result.
- **Robustness/failure cases:** The paper reports negative graph-operator results, ranking-objective failure, centering latency regressions, and incomplete cost items. These are strengths. Missing robustness is cross-teacher/cross-architecture replication and an untouched evaluation split.
- **Implementation details:** Recurrence, thresholds, model width, graph normalization, fitting resources, and timing conventions are unusually explicit. Exact deterministic-coordinate selection and complete eligibility details remain insufficient for clean reimplementation from the manuscript alone.
- **Artifacts and reproducibility:** Frozen hashes all pass, and the claim map identifies raw source paths and qualifications. Those source files were outside this review's permitted inputs, so no raw numerical recomputation was performed. `verification.json` confirms derived publication assets and compilation checks; it is not independent experimental verification. A reproducible package should include raw rows, analysis scripts, checkpoint/seed identifiers, and a distinction between aggregate replotting and full inference.
- **Limitations:** Excellent disclosure of exposure, nonadditivity, hybrid GatedGCN computation, unequal amplitude cost, and incomplete energy/runtime accounting. The main needed change is to align abstract/introduction emphasis with those limitations rather than add more generic caveats.

## 10. Multi-Reviewer Panel

Reviewer: Best-justified reviewer  
Expertise: Scientific analysis papers and conversion diagnostics  
Likely score: 6/10 borderline positive  
Confidence: 4/5  
Main positive signal: Two clean, realizable counterexamples with explicit invariants and unusually honest negative controls.  
Main negative signal: Limited breadth and uncertain novelty premium.  
Evidence basis: Sections 3.2–4.4; Tables 2–3; Figs. 4–5.  
Fatal concern if any: None to the scoped existence claims.  
Score-change condition: Sharpen closest-work delta and replicate the central interventions across teachers on untouched data.

Reviewer: Critical reviewer  
Expertise: CVPR empirical methods  
Likely score: 3/10 reject  
Confidence: 4/5  
Main positive signal: Careful experimental hygiene.  
Main negative signal: Known spatial/temporal intuitions repackaged as a contribution without a method or broad decisive evaluation.  
Evidence basis: Nine-item bibliography; elementary identities; two superpixel tasks; one-teacher central tests.  
Fatal concern if any: Contribution/novelty may be below the CVPR threshold even if all reported numbers are correct.  
Score-change condition: Demonstrate a graph-specific insight that changes practice beyond “evaluate decisions and trajectories.”

Reviewer: Method / soundness reviewer  
Expertise: ANN-to-SNN recurrence and mathematical analysis  
Likely score: 5/10 borderline negative  
Confidence: 4/5  
Main positive signal: The margin derivation is valid under its stated linear-region assumptions; hard replay avoids relying on those assumptions; the scalar recurrence and codeword arithmetic are internally consistent.  
Main negative signal: “Reconstruction equivalence” and “practical conditions” are broader than the tested formal statements.  
Evidence basis: Equations 2–9; Supplement Sections 3 and 5.  
Fatal concern if any: No central mathematical invalidity found in the supplied text.  
Score-change condition: Tighten claim language and specify the intervention distribution; no new theorem is required.

Reviewer: Evidence / experiment reviewer  
Expertise: Graph learning evaluation  
Likely score: 4/10 weak reject  
Confidence: 4/5  
Main positive signal: Real interventions, graph-cluster resampling, three-teacher position analysis, and negative controls.  
Main negative signal: Pooled selected-event rates and exposed/single-teacher evidence.  
Evidence basis: Section 3.4; Tables 1–3; Supplement Table 1.  
Fatal concern if any: Headline generalization would be unsupported if read beyond descriptive one-teacher evidence.  
Score-change condition: Existing-data breakdowns plus frozen-rule replication on fresh splits and all teachers.

Reviewer: Novelty / positioning reviewer  
Expertise: Conversion, temporal coding, and graph quantization  
Likely score: 3/10 reject  
Confidence: 3/5  
Main positive signal: Exact local terminal/count equality in a graph adapter is a specific, testable delta.  
Main negative signal: Temporal unevenness/misalignment and local/global conversion-error analysis have close precedents; the missing Hao 2023 discussion is material.  
Evidence basis: Public PMLR/AAAI/IJCV sources and manuscript Related Work.  
Fatal concern if any: Novelty collapse remains possible until the precise delta is established; the targeted search does not prove collapse.  
Score-change condition: Add a claim-level closest-work matrix and remove novelty implications for the general phenomena.

Reviewer: Writing / clarity reviewer  
Expertise: Scientific presentation  
Likely score: 5/10 borderline negative  
Confidence: 5/5  
Main positive signal: Clear notation, self-contained captions, explicit scope, and readable figures.  
Main negative signal: Too many secondary studies dilute the two central contributions; the abstract lists centering before the core claims are conceptually settled.  
Evidence basis: Abstract, Sections 4.3–4.6, Discussion, Supplement Sections 4–6.  
Fatal concern if any: None.  
Score-change condition: Restructure around two claims and move historical/centering detail behind them.

Reviewer: Ethics / reproducibility reviewer  
Expertise: Research artifacts and responsible disclosure  
Likely score: 5/10 borderline negative  
Confidence: 4/5  
Main positive signal: Excellent exposure and LLM disclosure; no apparent ethical hazard; exact frozen hashes pass.  
Main negative signal: Manuscript-only reproduction of selection and full-network results remains incomplete; raw verification was not available in this review.  
Evidence basis: Supplement Sections 1, 5, 6; LLM statement; `CLAIMS.md`; `verification.json`.  
Fatal concern if any: No visible ethics desk issue.  
Score-change condition: Supply deterministic selection code, raw rows, checkpoints, and independent regeneration logs.

Reviewer: Domain application reviewer  
Expertise: Event-based visual graph inference  
Likely score: 4/10 weak reject; practical/domain validity 3/5  
Confidence: 4/5  
Main positive signal: Correctly rejects simplistic event-count efficiency claims and measures dense latency.  
Main negative signal: Small superpixel tasks, continuous residual/readout arithmetic, and hybrid GatedGCN computation narrow deployment relevance.  
Evidence basis: Sections 3.1, 4.5–4.6; Supplement 5.  
Fatal concern if any: Any implication of practical neuromorphic efficiency would be unsupported.  
Score-change condition: Keep efficiency scoped or evaluate a fully specified event-driven backend; this is optional for the diagnostic claims.

Reviewer: Evidence / ablation reviewer  
Expertise: Controlled causal interventions  
Likely score: 4/10 weak reject; experimental rigor 3/5  
Confidence: 4/5  
Main positive signal: Hard interventions and exact codeword exchanges directly test mechanisms.  
Main negative signal: Nonadditive position recovery cannot rank causal contributions cleanly; pooled intervention rules obscure what drives adverse rates.  
Evidence basis: Figs. 3–5 and Section 3.4.  
Fatal concern if any: None to existence; causal/bottleneck wording would be overstated if generalized.  
Score-change condition: Stratified intervention analysis and factorial site/interaction controls.

Reviewer: Reproducibility reviewer  
Expertise: Numerical audit and replay  
Likely score: 4/10 weak reject; reproducibility completeness 3/5  
Confidence: 4/5  
Main positive signal: Source-indexed hashes, exact equations, split sizes, and confidence-interval conventions are documented.  
Main negative signal: Frozen package verifies documents, not experimental outputs; no independent rerun was possible under the input boundary.  
Evidence basis: `input-manifest.json`, `source_sha_manifest.json`, and Supplement Section 1.  
Fatal concern if any: Not fatal for review, but artifact confidence is limited.  
Score-change condition: Independent raw-data recomputation and network replay.

Reviewer: Novice-advocate reviewer  
Expertise: First-pass readability/accessibility  
Likely score: 5/10 borderline negative; accessibility 4/5  
Confidence: 5/5  
Main positive signal: Figure 1 explains the intuition and Figure 5 makes the temporal distinction visible.  
Main negative signal: Multiple adapters and evidence levels are difficult to keep separate despite the scope table.  
Evidence basis: Main Figs. 1–6 and Supplement Table 1.  
Fatal concern if any: None.  
Score-change condition: Introduce a two-row central evidence roadmap and remove nonessential branches.

Reviewer: AC / meta-reviewer  
Expertise: CVPR decision synthesis  
Likely score: 4/10 weak reject  
Confidence: 4/5  
Main positive signal: Technically credible and honest analysis.  
Main negative signal: Contribution strength and empirical breadth do not yet compensate for close prior art.  
Evidence basis: All frozen manuscript materials and targeted public search.  
Fatal concern if any: Current novelty/significance threshold, not demonstrated scientific invalidity.  
Score-change condition: Decisive replication and sharper graph-specific conceptual contribution.

Agreement: Both central claims are valid within scope; presentation and exposure disclosure are strong.  
Disagreement: Whether clean counterexamples alone merit a borderline-accept analysis paper or remain too incremental for CVPR.  
Decisive positive axis: Exact controlled interventions make the limitations of local reconstruction inspectable and actionable.  
Decisive negative axis: Known underlying phenomena plus one-teacher/exposed-split evidence yield limited new knowledge.  
Unresolved evidence: Cross-teacher robustness, fresh-split results, selection-rule effects, and exhaustive closest-work comparison.  
AC stance: Weak reject; do not average away the novelty/significance concern.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | fatal | Novelty/significance delta from temporal unevenness, misalignment, and error propagation is too small as presented | Related Work; nine references; Bojkovic 2025; Hao 2023; IJCV 2024 | Originality, significance | related-work | Add closest-work matrix; cite Hao 2023 and error/output alignment work; state the exact controlled graph delta; remove broad novelty implications | ccf-literature-searcher + ccf-paper-writer | A defensible nonincremental delta can raise originality from 2 to 3; without it, overall remains 4 or lower |
| C2 | major | Pooled adverse rates depend on intervention selection and lack a null reference | Section 3.4; Table 2 | Evidence, soundness | experiment | Reanalyze existing rows by rule/direction/graph, report all denominators and effect sizes, and add a predeclared residual-agnostic/random-legal comparison | ccf-experiment-designer + analysis owner | If the effect persists and interpretation is scoped, evidence can rise from 3 to 4 |
| C3 | major | Exposed validation/test and one-teacher central tests block confirmatory generalization | Section 3.4; Supplement Table 1 | Evidence | experiment | Freeze rules; replicate hard and temporal tests across teachers on untouched splits; retain descriptive wording meanwhile | ccf-experiment-designer | Successful independent replication could add +1 overall; failed replication would lower soundness/evidence |
| C4 | major | Temporal generality and eligible-set/effect-size reporting are limited | Section 4.4; Fig. 5; Supplement 3.2 | Evidence | experiment | Report eligibility and paired prediction/activity effects; replicate across seeds/models; optionally control amplitude as well as count | ccf-experiment-designer | Robust replication raises evidence; absence keeps the claim one-adapter-only |
| C5 | major | Deterministic selection and full-network replay are not independently reproducible from the manuscript/package | Section 3.4; Supplement 1; verification files | Reproducibility | reproducibility | Specify exact algorithm; release raw rows, analysis code, checkpoints/seeds, and regeneration instructions | ccf-submission-checker | Independent recomputation can raise reproducibility from 3 to 4 |
| C6 | moderate | “Major bottleneck” and nonadditive ideal recovery can be overread causally | Fig. 3; Section 4.1 | Soundness | writing | Use “largest single-site ideal recovery”; add interactions/factorial recovery only if making a causal decomposition claim | ccf-paper-writer | Qualification removes a likely reviewer objection; dimension-only change |
| C7 | moderate | Contribution is diluted by centering, node history, and multiple adapters | Sections 4.3–4.6; Supplement 4–6 | Clarity, significance | compression | Reorganize around two central claims; move nonessential context to supplement; label the role of every control | ccf-paper-writer | Clearer contribution can raise clarity/significance but cannot fix novelty alone |
| C8 | moderate | “Reconstruction equivalence” and “practical conditions” exceed the precise tested statements | Abstract; Introduction; Discussion | Soundness, clarity | writing | Replace with metric-specific improvement/equality language and “evaluation checklist” | ccf-paper-writer | Qualification prevents overclaim deduction; no new result needed |
| C9 | moderate | Temporal count equality does not equalize amplitude/energy | Supplement 3.2; Section 4.4 | Evidence, limitations | ethics/limitations | Keep amplitude caveat adjacent to the main temporal result; add equal-amplitude test only if making energy claims | ccf-paper-writer | Adequate adjacency resolves presentation risk; energy claims still require new evidence |
| C10 | minor | Current-year compliance depends on administrative items outside the PDF | Official CVPR 2027 guidelines; frozen source | Ethics/compliance | reproducibility | Verify author enrollment, reviewer eligibility, conflicts, dual submission, and final 2027 author-kit version | ccf-submission-checker | Administrative confirmation prevents desk risk but does not change scientific score |

## 12. AC / Meta-Review

Reviewer consensus: The paper is careful, readable, and technically credible. Its two central claims hold as scoped existence results. The spatial norm analysis and exact temporal invariants are not invalidated by the exposed splits; exposure instead restricts generalization and confirmatory weight.

Reviewer disagreement: A sympathetic reviewer could accept the work as a useful analysis/checklist paper with real interventions. A stricter CVPR reviewer sees limited conceptual advance over temporal misalignment/unevenness and standard decision-margin geometry, with no new algorithm and narrow empirical coverage.

Decisive acceptance axis: The exact local terminal/count-controlled temporal experiment and hard spatial interventions provide direct, mechanistically interpretable evidence that goes beyond reporting a weak metric correlation.

Decisive rejection axis: The paper does not yet turn those observations into a sufficiently novel, general, or practice-changing result. The most memorable numbers are selected conditional rates from one teacher and previously exposed data.

AC stance: **Weak reject (4/10).** This is not a finding of fabricated evidence or mathematical invalidity. It is a contribution-threshold decision driven by novelty, evidence scope, and interpretation.

Discussion risks: The best-justified reviewer may argue that analysis papers should not be forced to provide a new converter. The critical response is that the analysis still needs a sharper nonobvious insight and independent robustness. Adding more calibration tables or a SOTA baseline alone would not resolve that disagreement.

## 13. Quantitative Scores

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
| --- | ---: | ---: | --- | --- |
| Quality | 3 | 4 | Real controlled interventions and candid controls, but narrow and fragmented contribution | Stronger independent evidence and clearer central contribution required for 4 |
| Clarity | 4 | 5 | Figures 1–6, explicit equations and scope table; multiple secondary branches dilute focus | Reorganization around two claims could reach 5 |
| Significance | 3 | 4 | Useful evaluation caution for graph conversion; practical consequence not yet demonstrated | Show the checklist changes design/evaluation on fresh tasks to reach 4 |
| Originality | 2 | 3 | Close temporal-misalignment/unevenness and error-propagation precedents; specific exact-control delta is underdeveloped | Closest-work matrix plus defensible graph-specific delta required for 3; broader validated insight required for 4 |
| Soundness | 4 | 4 | Valid stated derivations and actual nonlinear replay; claims occasionally broader than formal/test scope | Metric-specific wording and selection analysis would remove remaining concern |
| Evidence | 3 | 4 | Two central existence claims supported; one-teacher/exposed-split limits and no null reference for rates | Independent replication and stratified/null analyses required for 4 |
| Reproducibility | 3 | 4 | Hashes/provenance and detailed protocols; deterministic selection and full raw replay not independently verified here | Release exact code/raw data/checkpoints and independently regenerate headline tables |
| Ethics / Limitations | 4 | 5 | Strong exposure, LLM, cost, energy, and limitation disclosure; no apparent sensitive-data risk | Maintain disclosure and verify administrative compliance |

**Overall: 4/10 (weak reject).**  
**Scholarly confidence: 4/5.**  
**Recommendation:** Weak reject; potentially viable after major revision, but not a borderline accept on current novelty and evidence scope.  
**Verdict:** Both central claims are supported only in their scoped existential form; a strong score requires more than copyediting or another aggregate table.

Writing score: **3.48/5**, writing risk **moderate-high for reviewer contribution perception**. This measures communication, not acceptance probability.

| Writing dimension | Weight | Score (1-5) | Confidence (1-5) | Evidence basis | Concrete repair |
| --- | ---: | ---: | ---: | --- | --- |
| Storyline and motivation | 10 | 3 | 5 | Clear two-gap motivation, then several secondary studies | Keep two-gap spine throughout |
| Contribution display | 10 | 3 | 5 | Abstract includes many findings but not the sharpest prior-work delta | State exact controlled-experiment novelty early |
| Paragraph logic | 10 | 4 | 5 | Most paragraphs have one role and explicit scope | Split overfull calibration/centering paragraphs |
| Claim-evidence alignment | 12 | 4 | 5 | Strong claim map and limitations; a few broad phrases remain | Qualify “equivalence” and “practical conditions” |
| Method readability | 9 | 4 | 5 | Recurrence and graph computation are explicit | Add exact deterministic selection pseudocode |
| Experiment narration | 9 | 4 | 5 | Captions state split and metric; pooled rates need interpretation | Add denominators/strata before interpretation |
| Related-work positioning | 8 | 2 | 4 | Nine references; Hao 2023 and error/output alignment missing | Compare closest work by invariant and conclusion |
| Terminology and notation | 8 | 4 | 5 | Generally stable; “reconstruction equivalence” is imprecise | Define each reconstruction metric exactly |
| Prose discipline and voice | 10 | 4 | 5 | Direct and restrained, with limited defensive repetition | Reduce repeated “not X” caveats by using a single scope statement |
| LaTeX and format discipline | 6 | 4 | 5 | Review mode, eight pages, readable figures; source inspected, supplied build verified | Verify final official 2027 kit; no scientific change needed |
| Reviewer-facing risk | 8 | 2 | 5 | Central novelty and selected-rate interpretation are easy to attack | Lead with delta and robust controlled evidence |

High-impact paragraph review:

| ID | Current role | Reviewer takeaway | Main problem | Concrete edit | Severity |
| --- | --- | --- | --- | --- | --- |
| Abstract-P1 | Summarize all findings | Two local fidelity gaps exist | Too many secondary results; novelty delta invisible | Prioritize two claims and scope; compress centering/calibration to one sentence | High |
| Intro-P5 | State central finding | Reconstruction is weaker than decision equivalence | “Space/time equivalence” is broader than tested metrics | Use metric-specific language and name hard/temporal invariants | High |
| Intro-P6 | State contributions | Empirical and analytical account | No explicit closest-work distinction | Add one sentence identifying what Bojkovic/Hao do not control | High |
| Related-P1 | Position conversion/readout | Temporal misalignment is known | Delta remains a narrow final sentence without Hao 2023 | Add direct comparison of sequence, terminal equality, count, graph setting | High |
| Method-Interventions-P1 | Define evidence levels | Splits and exposure are explicit | Deterministic coordinate is not reproducible | Specify exact selection and eligibility algorithm | High |
| Results-Hard-P1 | Headline adverse rates | Message improvement often hurts margin | Pooling and conditioning obscure intervention distribution | Add rule/direction/null table and descriptive-scope sentence | High |
| Results-Temporal-P1/P2 | Establish exact invariant and consequence | Terminal equality does not preserve trajectory | Eligibility/effect-size detail is not prominent | Add eligible counts, paired outcomes, and amplitude caveat | High |
| Results-Centering-P1–P3 | Constructive representation example | Proxy gains do not imply efficiency | Diverts from central contribution | Compress and move full campaign details to supplement | Medium |
| Conclusion-P1–P3 | Synthesize and bound | Claims are intentionally scoped | Still says practical conditions are established | Rename as checklist; restate two-claim verdict and one-teacher scope | High |

Format verdict: **Pass visible scientific formatting; no blocking PDF issue found.**  
Blocking compliance issues: None visible; administrative eligibility and final author-kit provenance unverified.  
LaTeX/source issues: `\paperID{-----}` is a normal pre-submission placeholder but must be replaced through the official process. The forced `\clearpage` before the LLM statement is acceptable in the present layout; no style-breaking margin/font overrides found.  
Figure/table/caption issues: Figures are readable and captions state crucial split/scope conditions. The main need is additional scientific stratification, not visual decoration.  
Citation/reference issues: Bibliography resolves in supplied PDF; coverage is too sparse for close-work positioning.  
Notation/equation issues: Core notation is defined. Use more precise wording around reconstruction “equivalence.”  
Suggested checks or build command: No rebuild was needed for this frozen review; supplied verification reports compiled/resolved/no-overflow. PDF visual inspection covered all 8 main and 4 supplement pages.

| Change | Condition | Likely affected dimensions | Expected movement |
| --- | --- | --- | --- |
| Raise score | Sharp closest-work delta plus successful frozen-rule replication across teachers on fresh splits | Originality, evidence, significance | +1 overall plausible; not guaranteed |
| Raise score | Existing-data stratified/null analysis supports robust interpretation | Evidence, soundness | Dimension-only improvement without broader novelty |
| Lower score | Adverse effects disappear under prespecified controls or independent replay | Evidence, soundness | -1 overall or rejection of central empirical claim |
| Lower score | Closer prior work already demonstrates the same exact controlled graph invariants | Originality | Fatal novelty collapse |
| No quick change | CVPR-level significance and independent breadth | Originality, significance, evidence | Unlikely to change through prose alone |

## 14. Questions For Authors

1. What is the precise new scientific fact relative to Hao 2023's unevenness analysis and Bojkovic 2025's temporal misalignment, beyond using a graph architecture?
2. How do the 40.94%/31.90% rates decompose by deterministic versus largest-residual coordinate, upward versus downward move, and graph? What happens under a residual-agnostic legal-move reference?
3. How many graphs have at least one adverse message-improving intervention, and what are the median/quantile margin changes and actual prediction-flip rates, rather than only intervention-level conditional fractions?
4. For the temporal exchange, how many codewords/graphs are eligible in each direction, and what are the paired effect sizes for agreement, accuracy, and whole-network events?
5. Were any coordinate rules, thresholds, codeword pairs, or reported endpoints selected after inspecting the same validation/test outcomes? Can the selection history be frozen and audited?
6. Can all three teachers reproduce both central effects on untouched graphs without changing the intervention rules?
7. Which exact artifact lets an independent reviewer regenerate Table 2 and Fig. 5 from raw rows, and which artifact reruns network inference with teacher 41?
8. Which claim would be falsified if actual-graph, conjugated-graph, and feature-only controls remain equivalent on fresh tasks?

## 15. Score Revision Criteria

Raising the score would require:

- A defensible, explicit novelty delta from temporal unevenness/misalignment and error-propagation work, centered on what the exact controlled graph experiments newly reveal.
- Existing-data analyses showing that the spatial result is not an artifact of coordinate/direction pooling, with per-graph and effect-size summaries and a meaningful reference distribution.
- Frozen-rule replication of both central interventions across all teachers on untouched data; broader architecture/task replication for a score above borderline.
- Exact reproduction of headline tables/figures from released raw artifacts, with deterministic selection fully specified.

Lowering the score would be triggered by:

- Evidence that the selected adverse rates collapse under sensible rule/direction controls or are driven by post-outcome selection.
- Failure to replicate the spatial or temporal effect across teachers after freezing the rule.
- Discovery of prior work with the same exact terminal/count-controlled graph intervention and conclusion.
- Any inconsistency between raw rows and the reported denominators, margins, or local equality checks.

Concerns unlikely to change before submission:

- Prior development exposure cannot be erased by disclosure or bootstrapping; only a new untouched evaluation changes its evidentiary status.
- One-teacher central experiments cannot acquire teacher-level generality through additional graph bootstrap samples.
- Residual centering's nine unresolved cost items and hybrid/dense computation prevent an end-to-end efficiency claim without a new implementation/evaluation.
- The paper cannot become a strong algorithmic contribution through related-work additions alone; it must stand as a rigorous analysis contribution.

## 16. Action Plan And CCFA Handoffs

### No-new-experiment fixes

Priority: P0  
Action: Build a closest-work matrix for Bojkovic 2025, Hao 2023, IJCV 2024, ICML 2026, One-Spike, and graph quantization. State the exact new invariant and delete broad novelty implications.  
Owner skill: `ccf-literature-searcher`, then `ccf-paper-writer`  
Input needed: Public papers and the two central claim statements  
Expected output: Revised Related Work/Introduction with a defensible graph-specific delta  
Handoff required: yes

Priority: P0  
Action: Reanalyze existing hard-intervention rows by rule, direction, graph, baseline margin, and event size; report all-attempt rates, per-graph prevalence, flip rates, margin effect sizes, and a predeclared reference computed without selecting on outcomes.  
Owner skill: Experiment/data analysis owner; `ccf-experiment-designer` for protocol review  
Input needed: Existing `graphs.csv` rows and deterministic selection definitions  
Expected output: Stratified Table 2 replacement and interpretation limited to the intervention distribution  
Handoff required: yes

Priority: P0  
Action: Align abstract, introduction, and conclusion with the two scoped existence claims. Replace “reconstruction equivalence” and “establish practical conditions” with precise metric-specific language and “evaluation checklist.”  
Owner skill: `ccf-paper-writer`  
Input needed: Claim-evidence audit in Section 8  
Expected output: Claims that do not exceed the evidence  
Handoff required: yes

Priority: P1  
Action: Compress centering and historical node material; foreground exact temporal eligibility/effect sizes and the negative actual-graph controls.  
Owner skill: `ccf-paper-writer`  
Input needed: Existing main/supplement and raw summaries  
Expected output: Two-claim narrative with a clearer role for each retained control  
Handoff required: yes

Priority: P1  
Action: Specify deterministic coordinate selection and eligibility exactly; add an artifact guide linking every headline number to raw rows, code, teacher/seed, and replay command.  
Owner skill: `ccf-submission-checker` / reproducibility owner  
Input needed: Analysis code, raw rows, checkpoint manifest  
Expected output: Independently auditable reproduction path  
Handoff required: yes

Priority: P1  
Action: Independently recompute Table 2, Fig. 5, and the scalar recurrence from raw artifacts; reconcile all denominators and local equality assertions.  
Owner skill: `ccf-integrity-auditor` / experiment-claim audit  
Input needed: Raw artifacts outside this frozen review set  
Expected output: Numerical verification report, distinct from PDF/source verification  
Handoff required: yes

### Large-experiment suggestions

Priority: P0 for confirmatory submission strength  
Action: Freeze intervention rules and run hard-event and exact temporal experiments across all teacher seeds on untouched graphs. Keep teacher, graph, and intervention variability separate in the analysis.  
Owner skill: `ccf-experiment-designer`  
Input needed: Checkpoints, fresh split, frozen code/protocol  
Expected output: Independent replication of both central claims  
Handoff required: yes

Priority: P1 for broader CVPR significance  
Action: Add at least one genuinely different graph operator/task, including the promised feature-dependent GatedGCN perturbation analysis. Test whether the first-site recovery ordering and adverse mechanisms persist.  
Owner skill: `ccf-experiment-designer`  
Input needed: Additional trained teachers and frozen evaluation protocol  
Expected output: Cross-architecture evidence rather than historical context  
Handoff required: yes

Priority: P1 for temporal robustness  
Action: Add alternative legal codeword families and an equal-amplitude/equal-count subset where feasible; report eligibility and failure cases rather than selecting only favorable exchanges.  
Owner skill: `ccf-experiment-designer`  
Input needed: Source unit and legal alphabet enumerator  
Expected output: Stronger temporal invariants and bounded generality  
Handoff required: yes

Priority: P2 only if broader converter claims are desired  
Action: Run a direct One-Spike GCN comparison under a common graph/task protocol and separately evaluate event-driven backend costs.  
Owner skill: `ccf-experiment-designer`  
Input needed: One-Spike implementation and matched graph teachers  
Expected output: Converter positioning or complete-cost evidence; not required for the narrow counterexamples  
Handoff required: yes

Checks run: Exact SHA-256 and byte verification for all 12 manifest entries; full main/supplement TeX and extracted-text reading; visual inspection of every rendered PDF page; central recurrence/margin/gate-derivation inspection; exposure/statistics/artifact audit; targeted public related-work search; official CVPR 2027 policy check; fixed-format and score-consistency self-check.  
Checks skipped: Raw CSV/JSON recomputation, source-code execution, network inference, checkpoint verification, exhaustive systematic literature review, bibliography-by-bibliography external verification, and administrative OpenReview checks. These were outside the frozen inputs or not necessary for the scoped scientific judgment.  
Unresolved risks: Exact novelty relative to all prior art; intervention-selection sensitivity; independent cross-teacher/fresh-split robustness; raw numerical correctness; final administrative compliance.
