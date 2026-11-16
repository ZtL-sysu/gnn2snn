# Independent Frozen-R1 Review

## 1. Report Metadata

Review date: 2026-11-16  
Target venue/year/track: CVPR 2027, main conference  
Paper title: A Finite-Event Audit of Decision and Temporal Fidelity in Graph-to-Spike Conversion  
Input materials reviewed: all 12 manifest-listed files in `review-round-01/frozen-r1`; main and supplementary PDFs and TeX, extracted PDF text, revision log, verification record, all three CSVs, and `../input-manifest-r1.json`.  
Search basis: public-safe generic queries for temporal misalignment, graph SNN conversion, prediction-difference quantization, and CVPR 2027 policy; no private manuscript text or prior-round report used as evidence.  
Report file: `review-round-02/independent-r1-review.md`  
Reviewer mode: full, fresh-context scientific, evidence, writing, and format review.  

## 2. Desk Rejection Assessment

- Paper length: **Pass.** The main PDF has eight pages: seven body pages, then the LLM statement and references on page 8.
- Topic compatibility: **Pass, with narrow scope.** Frozen superpixel graph conversion and event fidelity are relevant to efficient visual inference.
- Minimum quality: **Pass.** The paper has a coherent mechanism, reproducible-looking protocols, and explicit limitations; its CVPR-level significance remains uncertain.
- Policy/anonymity/compliance: **Pass for inspected materials.** Anonymous author block, no author-identifying link found, and a detailed LLM statement before references. Official submission eligibility and administrative checks are outside this package.
- Prompt injection and hidden manipulation detection: **Pass.** No embedded instruction or hidden manipulation found in the reviewed text.
- Ethics and reviewability: **Pass.** Exposed splits, historical aggregates, absent hardware-energy evidence, and the different converters are disclosed.

All 12 manifest entries match their recorded byte counts and SHA-256 hashes. The supplied `verification.json` also agrees with the source/PDF/table hashes. The revision log calls this file `verification-r1.json`, although the frozen filename is `audit/verification.json`.

## 3. Paper Summary And Contribution Map

The paper argues that local reconstruction metrics do not determine downstream decisions or temporal computation in graph-to-spike conversion. A directional derivation links graph-message perturbations to class margins; controlled one-event interventions show that reduced next-message MSE can worsen the converted teacher-class margin. A second experiment exchanges source-eligible two-tick codewords with identical local terminal decoding and equal local event counts, yet observes downstream prediction and whole-network event changes. Position recovery, calibration controls, GatedGCN residual centering, and historical node-classification results provide context rather than a new state-of-the-art converter.

- Claimed problem: evaluating low-timestep graph conversion by reconstruction alone.
- Claimed gap: graph-specific controlled interventions that explicitly match decision-relevant or temporal invariants.
- Contribution map: exact linear-suffix margin identity and sufficient norm bound; hard-event message interventions; executable recurrence and local codeword equivalence test; evaluation checklist covering decisions, prefixes, and global activity.
- Evidence package: three-teacher historical test and validation results; one-teacher hard interventions on 512 exposed validation graphs per task; temporal exchanges on 128 training graphs per task; one-teacher GatedGCN training-split analysis; contextual historical node results.
- Stated limitations: no conversion-wide frequency, no new converter or efficiency guarantee, no blind confirmation, no equal-amplitude codeword control, no direct matched One-Spike comparison, and no feature-dependent GatedGCN perturbation study.

## 4. Search And Related-Work Basis

Queries used: generic terms combining graph SNN conversion, One-Spike, temporal misalignment, prediction-difference quantization, and CVPR 2027 LLM policy.  
Sources searched: PMLR, arXiv/public proceedings results, and official CVPR 2027 author/call pages.  
Closest works found: Bojkovic et al. (ICML 2025), *Temporal Misalignment in ANN-SNN Conversion and its Mitigation via Probabilistic Spiking Neurons*; Hwang and Kung (2025), *One-Spike SNN*; Liu et al. (CVPR 2023), *PD-Quant*. All three are already discussed in the manuscript.  
Unverified related-work risks: the search was not an exhaustive graph-conversion literature review; the manuscript's 2026 CKA citation was not independently verified.  
Source-quality screening status: primary/public scholarly sources used; no secondary-source claims used for scoring.

## 5. Expected Review Outcome

Expected outcome: **borderline, 5/10.**  
Main accept signal: a now-coherent pair of controlled counterexamples, with exact eligibility rules and honest claim scoping.  
Main reject signal: the novelty and evidence remain those of a small diagnostic study, while substantial parts of the manuscript concern historical or secondary analyses rather than the central contribution.  
Confidence: **4/5** for manuscript-internal claims and supplied aggregates; lower for unprovided raw registers and source-linked inference replication.

**Direct R1 assessment:** R1 materially improves on a 4/10 baseline through manuscript-only/small-reanalysis changes. The reframed novelty and scoped claims are defensible as existence claims, not as evidence of a general conversion failure rate or a superior method. The stratification and temporal decision figures are valid against the supplied CSVs and appropriately qualified. Remaining high-value work is mostly artifact correction, provenance tightening, and visual emphasis. Under the prohibition on large experiments, the paper has reached diminishing returns after those small fixes.

## 6. Strengths And Weaknesses

Strengths:

- The main paper explicitly separates teacher agreement, ground-truth accuracy, local event count, whole-network activity, and runtime.
- Equation (5) is an exact decision condition; the authors correctly reject interpreting the norm bound as a universal time budget.
- Figure 4 shows a nonzero adverse-margin rate in all eight dataset/rule/direction strata, preventing a single pooled selection rule from carrying the result.
- The temporal intervention controls local terminal decoding and event count while changing the first-prefix signal; Figure 5 exposes opposing agreement transitions hidden by aggregate accuracy.
- The supplement clearly distinguishes separately calibrated historical horizons from shared-prefix controls, and identifies exposed rather than blind evaluations.

Weakness:

Evidence basis: the central empirical claim still rests on one teacher, two tasks, exposed validation graphs, and a deliberately chosen intervention distribution (Sections 3.4, 4.2, 4.4).  
Reviewer deduction: this limits generalization and CVPR-level significance, but does not invalidate the scoped existence result.  
Required fix: retain the current scope; only fresh multi-teacher/untouched-data evaluation could materially remove it.

Weakness:

Evidence basis: the supplement says the anonymous audit package contains source-indexed registers and deterministic figure generators reproducing Tables 1--3 and Figures 3--5, but none is in the frozen package. Only three small aggregate CSVs are supplied. The revision log also misnames the verification file.  
Reviewer deduction: headline aggregates are auditable, but bitwise tensor equality, source replay, graph-cluster intervals, and most historical results cannot be independently reconstructed from this package.  
Required fix: correct the artifact inventory and filename, or include the stated registers/generators and recompute the checks from them.

Weakness:

Evidence basis: main Figure 4 overlays the pooled count on the final residual bar, and its rule/direction labels are very small; Figure 5 has small, overlapping legend/axis annotations. Both are readable when enlarged but weak at normal two-column viewing size.  
Reviewer deduction: the new evidence is scientifically useful but not presented with enough visual priority.  
Required fix: remove the overlay, enlarge labels/legend, and use the freed secondary-material space for the two core figures.

## 7. Potentially Missing Related Work

Work: temporal misalignment in ANN-SNN conversion.  
Status: searched; already cited and discussed.  
Why relevant: directly establishes the importance of event order.  
Overlap: conceptual overlap is substantial.  
Needed comparison: the manuscript correctly narrows its difference to exact terminal decoding and equal local event count within graph adaptation; avoid implying discovery of temporal sensitivity itself.

Work: One-Spike SNN and PD-Quant.  
Status: searched; already cited and discussed.  
Why relevant: graph conversion and prediction-aware approximation, respectively.  
Overlap: neither negates the scoped counterexample as presented.  
Needed comparison: a matched One-Spike experiment remains an empirical extension, not a requirement for the existence claim; maintain the present distinction.

No additional missing work was established by the limited public-safe search.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Message-MSE improvements need not preserve decision direction | Abstract; Introduction; Section 4.2 | 332/811 MNIST and 296/928 CIFAR10 adverse cases; exact margin condition; all rule/direction strata | Adequate for scoped existence claim | Pooled and graph-level counts agree with supplied aggregates; not a conversion-wide rate | Keep denominator and eligibility qualifications adjacent to every headline rate |
| Stratification supports robustness to selection rule | Section 4.2; Figure 4 | Eight strata, 20.4--49.0% adverse rates; graph-cluster interpretation disclosed | Adequate | Direction is stable; magnitude is highly rule-dependent, as acknowledged | Enlarge figure labels and remove pooled-count overlay |
| Local exact terminal equality need not preserve downstream computation | Sections 3.3, 4.4; Figure 5 | Executable scalar recurrence; eligible codewords; 2--16 changed predictions per 128; changed whole-network events | Adequate for existence, limited for prevalence | Decision and activity arithmetic reproduces; actual bitwise/source-replay registers absent | Supply/check raw local-equality registers or narrow artifact claim |
| Equal local events can coexist with increased global events | Section 4.4 | Halving increases events 3.41%/4.99%; doubling increases MNIST 0.81% and lowers CIFAR10 1.33% | Strong for reported settings | CSV exactly reproduces rounded percentages; direction varies by task/policy | Report both directions in the figure/caption or table |
| No tested calibration objective is consistently superior | Abstract; Section 4.3 | Three-teacher matched fits, shared-prefix single-teacher controls, disagreement intervals | Adequate | Appropriately avoids an optimality claim | None beyond keeping test-exposure caveat visible |
| First-node sensitivity is concentrated | Introduction; Section 4.1 | Three-teacher full-validation continuous recovery | Adequate as diagnostic | Continuous recovery is not a finite-event intervention and is not additive | Preserve existing qualification; avoid using it as independent causal validation |
| Reproduction package enables audit | Supplement Section 1; revision log | Three aggregate CSVs, source hashes, verification flags | Weak | Most stated registers, generators, and inference prerequisites are not frozen here | Correct inventory and include promised small artifacts where available |

## 9. Experiment / Benchmark / Reproducibility Audit

- Baselines: local/output/mixed calibration, KL/ranking, feature-only/conjugated graph objectives, source threshold iteration, and residual-centering controls are useful; direct One-Spike parity is not established.
- Ablations: position recovery, six activation sites, prefixes, threshold scales, intervention direction, and selection rule address mechanism rather than leaderboard performance.
- Datasets/benchmarks: MNIST/CIFAR10 superpixels are appropriate for the scoped graph diagnostic; classical node results are correctly labeled contextual and not pooled.
- Metrics: message MSE, teacher-class margin, prediction agreement, accuracy, decoded endpoint, event count, and complete activity are clearly distinguished.
- Statistical rigor: graph-cluster bootstrap is appropriate in principle for four interventions per graph; three-seed SDs are descriptive. The temporal sample is only 128 training graphs and no prevalence inference is warranted. Raw bootstrap outputs are not supplied here.
- Robustness/failure cases: stratification strengthens the spatial result; unequal codeword amplitudes and non-monotonic temporal outcomes are explicit. The tiny example MSE improvement does not undermine the existence counterexample but should not be used as an effect-size demonstration.
- Implementation details: recurrence, eligibility, and calibration are unusually specific. Shared versus separately fitted prefixes are clearly separated.
- Artifacts: all frozen hashes pass, but the package verifies internal consistency rather than full result regeneration. The revision-log filename and supplement artifact inventory need correction.
- Limitations: exposed data, one-teacher core interventions, historical aggregate node results, continuous gates, unequal amplitudes, and missing hardware-energy measurement are responsibly stated.

## 10. Multi-Reviewer Panel

Reviewer: Best-justified / domain application  
Expertise: efficient visual graph inference  
Likely score: 6/10  
Confidence: 4/5  
Main positive signal: realizable interventions and exact local controls make a useful evaluation argument.  
Main negative signal: limited task breadth and no new conversion method.  
Evidence basis: Sections 3--4 and supplement evidence matrix.  
Score-change condition: stronger visual emphasis and a complete small audit bundle.

Reviewer: Critical / novelty-positioning  
Expertise: ANN-SNN conversion and quantization  
Likely score: 4/10  
Confidence: 4/5  
Main positive signal: exact endpoint plus local-count matching is a clean controlled distinction.  
Main negative signal: signed direction and temporal history are conceptually anticipated by PD-Quant and temporal-misalignment work.  
Evidence basis: Related Work and Sections 3.2--3.3.  
Score-change condition: cannot be resolved by rhetoric; broader graph-specific empirical significance would require new experiments.

Reviewer: Method / soundness  
Expertise: graph computation and event dynamics  
Likely score: 5/10  
Confidence: 4/5  
Main positive signal: exact decision condition and properly constrained temporal induction argument.  
Main negative signal: actual hard-network behavior depends on source-verified registers absent from this package.  
Evidence basis: Equations (2)--(8), supplementary derivations, temporal CSV.  
Score-change condition: independently reproduce source replay and bitwise equality.

Reviewer: Evidence / ablation  
Expertise: empirical ML evaluation  
Likely score: 5/10  
Confidence: 4/5  
Main positive signal: rule/direction strata and paired agreement transitions add exactly the needed small analyses.  
Main negative signal: one-teacher exposed-data core evidence and descriptive-only temporal sample.  
Evidence basis: Sections 3.4, 4.2, 4.4 and all CSVs.  
Score-change condition: no small reanalysis supplies teacher/task replication.

Reviewer: Writing / clarity  
Expertise: scientific presentation  
Likely score: 5/10  
Confidence: 5/5  
Main positive signal: scoped, technically precise, unusually candid about evidence roles.  
Main negative signal: many secondary controls dilute the two central figures; Figure 4/5 annotations are too small.  
Evidence basis: main PDF pages 5--7 and supplement pages 1--4.  
Score-change condition: concise compression and readable central figures.

Reviewer: Ethics / reproducibility  
Expertise: artifact and policy review  
Likely score: 5/10  
Confidence: 4/5  
Main positive signal: all supplied hashes pass; no anonymity or LLM-statement defect found.  
Main negative signal: artifact claims exceed the frozen contents; verification filename mismatch.  
Evidence basis: manifest, supplement Section 1, revision log.  
Score-change condition: fix inventory and reconstruct plots/checks from included artifacts.

Reviewer: AC  
Expertise: CVPR meta-review  
Likely score: 5/10  
Confidence: 4/5  
Main positive signal: coherent scoped paper with defensible controlled evidence.  
Main negative signal: borderline significance and weak independent reproducibility of core interventions.  
Evidence basis: full package.  
Score-change condition: small fixes improve credibility, not the underlying evidence ceiling.

Agreement: the R1 scoping and new figures are meaningful improvements.  
Disagreement: whether the controlled graph diagnostic is sufficiently novel/significant for CVPR.  
Decisive positive axis: exact, realizable counterexamples with transparent boundaries.  
Decisive negative axis: limited external validity and artifact-level verification.  
Unresolved evidence: raw source replay, bitwise equality, graph-cluster intervals, and historical tables.  
AC stance: borderline; no fatal technical contradiction identified.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | major | Core intervention evidence is one-teacher, exposed-data, and narrow | Sections 3.4, 4.2, 4.4 | Evidence, significance | experiment | Retain scope; future multi-teacher untouched replication only | ccf-experiment-designer | Would raise evidence/significance, but prohibited here |
| C2 | major | Artifact inventory exceeds supplied package; verification filename mismatch | Supplement Section 1; revision-log.md; manifest | Reproducibility | reproducibility | Correct path/inventory or include registers/generators and rerun checks | ccf-integrity-auditor | Could improve reproducibility to 4/5 |
| C3 | moderate | Conceptual novelty remains incremental despite clean graph controls | Related Work; searched temporal-misalignment and PD-Quant | Originality | related-work | Keep narrow differential statement; do not broaden novelty | ccf-paper-writer | No quick score increase expected |
| C4 | moderate | New figure text/overlays impair readability | Main PDF Figures 4--5 | Clarity | writing | Enlarge annotations, simplify legend, remove pooled overlay | ccf-visual-composer | Improves reviewability, unlikely alone to change overall |
| C5 | moderate | Temporal design is small and amplitude-unmatched | Section 4.4; temporal CSV; supplement Section 3.2 | Evidence | ethics/limitations | Keep training-only/descriptive and amplitude caveats next to the figure | ccf-paper-writer | Already mostly satisfied; prevents overclaim |
| C6 | minor | Secondary centering/node material dilutes central result | Sections 4.1, 4.5; supplement Sections 4--6 | Clarity, significance | compression | Compress secondary narrative and foreground Figures 4--5 | ccf-paper-writer | Dimension-only clarity gain |

## 12. AC / Meta-Review

The reviewer consensus is that R1 resolves the principal overclaiming problem. The paper no longer depends on an unsupported universal frequency, optimal calibration objective, or efficiency claim. Its two central existence claims are supported by logically appropriate controls and internally consistent aggregates. Disagreement concerns value: a skeptical reviewer can reasonably view the work as a careful diagnostic whose main conceptual lessons are anticipated. The decisive acceptance axis is the exact graph-specific intervention design; the decisive rejection axis is narrow external validity and modest novelty. The AC stance is borderline rather than weak reject after R1. Discussion should focus on whether the clean audit is sufficient contribution, not on demanding unsupported SOTA claims.

## 13. Quantitative Scores

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 3 | 4 | Related Work; controlled graph-specific invariants | Defensible incremental framing; broader significance needs new evidence |
| Soundness | 4 | 4 | Equations (2)--(8); supplementary derivations | No central contradiction found; raw replay remains unverified |
| Evidence | 3 | 4 | Tables 1--3; Figures 3--5; CSVs | Narrow/exposed core evidence; independent replication would be needed for 4 |
| Significance | 3 | 4 | Evaluation checklist and two counterexamples | Useful diagnostic, uncertain field-wide impact |
| Clarity | 4 | 5 | Main/supp TeX and rendered PDFs | Clear claims and limitations; dense secondary material and small figures |
| Reproducibility | 3 | 4 | Manifest, verification, supplement artifact claims | Aggregates reproduce; promised registers/generators absent |
| Ethics / Limitations | 4 | 5 | Sections 3.4, 5; supplement Sections 1, 6 | Strong disclosure; maintain temporal amplitude and exposure caveats |

Quality: 3/5  
Clarity: 4/5  
Significance: 3/5  
Originality: 3/5  
Soundness: 4/5  
Evidence: 3/5  
Reproducibility: 3/5  
Ethics / Limitations: 4/5  
Overall: **5/10**  
Confidence: **4/5**

| Change | Condition | Likely affected dimensions | Expected movement |
| --- | --- | --- | --- |
| Raise score | Complete/correct the small artifact bundle and independently regenerate new analyses | Reproducibility, soundness | Dimension gains; at most borderline-positive overall |
| Lower score | Show that local endpoint equality, event-count matching, eligibility, or CSV provenance is invalid | Soundness, evidence | -1 to fatal depending on scope |
| No quick change | Multi-teacher untouched replication or matched One-Spike/feature-dependent graph study | Evidence, significance | Material increase, but requires prohibited experiments |

## 14. Questions For Authors

1. Can the anonymous submission include the exact source-replay registers and deterministic generators promised in supplement Section 1, including bitwise local decoded-tensor checks and graph-cluster bootstrap outputs?
2. Can Figures 4 and 5 be regenerated at readable two-column scale without overlaying pooled counts or legend entries on the bars?
3. Can the revision/artifact manifest use the actual `audit/verification.json` filename and distinguish verification of supplied aggregates from full inference replication?

## 15. Score Revision Criteria

Raising the score would require: verifiable source-linked intervention artifacts; materially higher significance would require multi-teacher/untouched-data evidence beyond the current constraint.  
Lowering the score would be triggered by: invalid eligibility/source replay, non-identical local decoded tensors, mismatched graph-level denominators, or an unsupported broader novelty claim.  
Concerns unlikely to change before submission: one-teacher core evidence, exposed validation endpoints, no matched One-Spike adaptation, no equal-amplitude temporal family, and no complete hardware-energy result.

## 16. Action Plan And CCFA Handoffs

Priority: P0  
Action: reconcile `audit/verification.json` naming and the supplement's artifact inventory; include small raw registers/generators if available and regenerate hashes/checks.  
Owner skill: ccf-integrity-auditor  
Input needed: frozen manifest, actual audit package contents  
Expected output: truthful, reproducible evidence inventory  
Handoff required: yes

Priority: P1  
Action: enlarge and simplify Figures 4--5; show both temporal directions and keep training-only/amplitude caveats visible.  
Owner skill: ccf-visual-composer  
Input needed: supplied stratified and temporal CSVs  
Expected output: legible central evidence figures  
Handoff required: yes

Priority: P2  
Action: compress secondary centering/node discussion and move saved emphasis to the two core interventions.  
Owner skill: ccf-paper-writer  
Input needed: current main and supplement  
Expected output: more focused eight-page narrative  
Handoff required: yes

Priority: P3  
Action: stop expanding scientific claims; retain explicit one-teacher, exposed-data, unequal-amplitude, and no-efficiency-guarantee boundaries.  
Owner skill: ccf-paper-reviewer  
Input needed: current conclusions and captions  
Expected output: stable scoped submission  
Handoff required: no

Checks run: independent byte/SHA-256 verification of all 12 manifest entries; full main/supplement TeX and extracted-text reading; rendered main pages 5--6 and all supplement pages; recomputation of pooled/stratified counts, event deltas, and agreement transitions; public-safe related-work/policy search.  
Checks skipped: checkpoint inference, source-code execution, full raw-register reconstruction, bootstrap reproduction, exhaustive citation audit, and administrative submission checks; required inputs are absent from the frozen package.  
Unresolved risks: raw bitwise/source-replay verification and broader teacher/task generalization. These do not negate the supplied existence aggregates but bound confidence and score.

**Diminishing returns:** Yes, under the no-large-experiments constraint. Complete P0--P2 because they are inexpensive credibility/readability fixes; further manuscript expansion or additional small secondary analyses is unlikely to change the central borderline assessment.
