# Independent CCFA Writing Review

## 1. Report Metadata

| Field | Value |
| --- | --- |
| Review date | 2026-11-16 |
| Target venue/year/track | CVPR 2027, Main Track review; Findings Track considered as a fallback position |
| Paper title | *Beyond Reconstruction: Decision and Temporal Fidelity in Graph-to-Spike Conversion* |
| Input materials reviewed | `sources/main.pdf`, `sources/supplement.pdf`, `sources/main.tex`, `sources/supplement.tex`, `sources/CLAIMS.md`, `audit/main.txt`, `audit/supplement.txt`, `sources/references.bib`, `audit/final-audit.md`, `audit/verification.json`, and `input-manifest.json` |
| Frozen-package integrity | Pass. All 12 manifest entries matched their recorded byte counts and SHA-256 hashes. `main.pdf` is 8 pages; `supplement.pdf` is 4 pages. |
| Search basis | Public-safe generic queries only. No private manuscript wording was submitted to a search service. Sources were screened through official CVPR 2027, CVF, PMLR, and arXiv pages. |
| Report file | `paper/cvpr2027/ccfa-review-run-20261115/review-round-01/independent-writing-review.md` |
| Reviewer mode | CCFA `full`: scientific, novelty/positioning, evidence, writing, format, and reviewer-facing presentation |
| Manuscript edits | None |

## 2. Desk Rejection Assessment

- **Paper length: Pass.** The scientific body ends on page 7. Page 8 contains the LLM Usage Statement and references, which are excluded from the eight-page limit under the current [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines).
- **Topic compatibility: Pass, with track uncertainty.** The paper studies low-timestep conversion for visual superpixel graph classifiers, so it fits CVPR's efficient-vision area. Its analysis-first contribution is a more natural Main Track stretch and a plausible Findings Track fit if the empirical validation is judged sound.
- **Minimum quality: Pass.** The manuscript has a recoverable research question, a defined frozen computation, analytical relations, experiments, evidence boundaries, related work, and limitations.
- **Policy, anonymity, and compliance: Pass.** The paper is anonymous, uses the CVPR review format, contains no external content-expanding links, and includes the required LLM Usage Statement after the main text and before references.
- **Prompt injection and hidden manipulation detection: Pass.** Source, PDF text, captions, and supplementary text contained no reviewer-directed or automated-tool-directed instructions.
- **Ethics and reviewability: Pass.** The work uses established image classification benchmarks and does not involve human subjects. Efficiency limits, missing energy measurements, exposed evaluation data, and absent direct One-Spike comparisons are disclosed.

```text
Desk rejection risk: none
Reason: The package is anonymous, readable, correctly bounded to eight scientific pages, and contains the required LLM statement.
Can be fixed before review? Not applicable.
```

## 3. Paper Summary And Contribution Map

The paper asks what activation- or message-reconstruction metrics must preserve for a finite-event graph classifier to retain its teacher decision. It studies frozen residual GCN and GatedGCN conversions on MNIST and CIFAR10 superpixel graphs. Two failure modes are developed: a smaller immediate graph-message error may move the class margin in an adverse direction, and two legal local temporal codes with the same decoded endpoint and event count may expose different first-prefix histories to later event layers. The evidence combines exact identities, hard-event interventions, continuous position interventions, calibration controls, source-linked threshold iteration, temporal codeword exchanges, residual-centering diagnostics, timing, and historical node-classification context.

| Element | Manuscript claim | Evidence package | Novelty status |
| --- | --- | --- | --- |
| Problem | Reconstruction does not ensure decision preservation in low-timestep graph conversion. | Abstract and Introduction, p. 1, lines 001-088. | The statement is intuitive and has close analogues outside graph SNNs; the graph-event instantiation is the defensible delta. |
| Spatial mismatch | Improving immediate message MSE can worsen the teacher-class margin. | Main p. 5, §4.2, lines 312-340; Table 2; Fig. 4. | Direct hard-event evidence is the strongest contribution, but it uses one teacher and previously exposed graphs. |
| Position sensitivity | First node activation closes the largest single-site disagreement gap. | Main p. 4, lines 291-299; Fig. 3. | Three-teacher descriptive localization; intervention is continuous and non-realizable, as the paper states. |
| Temporal mismatch | Exact local terminal equality and local event-count equality do not preserve downstream behavior. | Main p. 3, lines 205-236; p. 6, §4.4, lines 395-419; Eqs. 8-9; Fig. 2 and Fig. 5. | The local-forwarding result is useful; the general temporal-misalignment principle is already established. |
| Controls | Calibration objectives, graph operators, and local work proxies do not deliver a consistent advantage. | Main p. 5-7, lines 341-460; Supplement §2 and §5. | Controls are honest and unusually careful, but several conclusions are descriptive. |
| Constructive diagnostic | Residual centering lowers a local projection proxy without resolving full-system decision and cost questions. | Main p. 6-7, §4.5, lines 420-460; Fig. 6; Supplement §5. | Useful representation observation, not a new conversion algorithm or complete efficiency claim. |
| Practical rule | Evaluate decisions, every required prefix, and whole-network activity after a realizable intervention. | Main p. 7, lines 492-515. | This is a sensible evaluation heuristic, but it has not yet been validated as a reusable protocol across teachers, tasks, or backends. |

**Claimed limitations.** The paper states that official validation and test splits were exposed during development; the hard-event and temporal tests use one teacher; temporal evidence uses training graphs; the centered full-validation campaign does not establish a complete efficiency advantage; GatedGCN gate perturbation and topology changes are outside scope; and a direct One-Spike comparison remains an extension. These limitations appear concretely in main p. 3, lines 237-249 and p. 7, lines 502-515, and supplement p. 4, lines 248-256.

## 4. Search And Related-Work Basis

```text
Queries used:
- graph neural network ANN to SNN conversion decision fidelity
- spiking graph neural network conversion temporal coding reconstruction decision
- One-Spike SNN graph convolution conversion
- ANN SNN conversion output logit decision agreement calibration
- quantization reconstruction error class margin decision boundary neural network
- CVPR 2027 author guidelines page limit double blind

Sources searched:
- Official CVPR 2027 Author Guidelines
- CVF Open Access
- PMLR proceedings
- arXiv records for key cited work

Closest works found:
- One-Spike SNN, already cited, explicitly reports successful GCN conversion with a 0.90% average accuracy loss.
- Temporal Misalignment in ANN-SNN Conversion, already cited, establishes temporal misalignment as a prior problem.
- PD-Quant, CVPR 2023, uses a prediction-difference metric because local reconstruction can differ from the behavior that matters.
- High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation, PMLR 2026, studies low-latency local/global alignment.
- Dissecting Quantization Error, PMLR 2026, relates quantization behavior to concentration and alignment directions.

Unverified related-work risks:
- Direct numerical comparability with One-Spike's reported GCN result was not checked because the tasks, architecture, splits, and adaptation protocol are not supplied.
- Related-work completeness beyond the searched axes cannot be claimed as exhaustive.

Source-quality screening status: Public, venue-hosted, or primary preprint pages only; no social-media, unverifiable aggregation, or citation-count evidence used.
```

Search changed the novelty judgment. The paper correctly acknowledges two of the closest works, but the broad proposition that reconstruction fidelity differs from decision fidelity is not itself new. [PD-Quant](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.html) had already used prediction differences instead of local feature reconstruction to select quantization parameters. The defensible contribution here is the graph-specific, exact finite-event evidence showing which spatial and temporal information the reconstruction scalar omits. The manuscript should claim that narrower contribution.

## 5. Expected Review Outcome

```text
Expected outcome: Weak reject on the Main Track, overall 4/10.
Main accept signal: The paper performs real hard-event interventions, proves two clean information losses, uses unusually careful controls, and reports its negative and limited findings honestly.
Main reject signal: The central distinction is close to established quantization and temporal-alignment knowledge, while the strongest new evidence is narrow: one teacher, previously exposed graphs, and no blind or independent task confirmation.
Confidence: 4/5.
```

This is a calibrated diagnostic score, not an acceptance probability. The paper is scientifically readable and technically plausible, but it currently reads below the CVPR Main Track bar for a new general insight or conversion capability. A tight analysis-paper reframing plus existing-data reanalyses could plausibly move it to 5-6. A direct One-Spike adaptation or independent task/teacher confirmation could support 7, but that is a large pre-submission effort rather than a rebuttal requirement.

## 6. Strengths And Weaknesses

### Top strengths

- **The hard-event experiment tests a decision-relevant question directly.** The paper modifies actual legal first-site events, reruns the hard network, and measures the full teacher-class margin. This is stronger than correlating a layerwise MSE with final accuracy. Evidence: main p. 4-5, lines 257-266 and 312-340.
- **The temporal construction is exact within its stated local claim.** Source replay verifies eligible codewords, local decoded tensors are bitwise equal, local event counts match, and downstream states are not reset. Evidence: main p. 3, lines 220-236; p. 6, lines 395-401.
- **The controls and boundaries distinguish unlike quantities.** Teacher agreement is separated from ground-truth accuracy, local events from whole-network activity, event opportunities from realized activity, and projected additions from latency. Evidence: main p. 3, lines 183-191; p. 7, lines 462-480; Supplement p. 3, lines 195-207.
- **The analysis is mathematically conservative.** The linear identity and sufficient condition are presented as elementary identities; nonlinear and gate-dependent deviations are acknowledged. Evidence: main p. 2-3, lines 156-202; Supplement §3.
- **The paper is unusually candid about exposed data and absent comparisons.** It does not promote negative or inconclusive controls as state-of-the-art. Evidence: main p. 3, lines 237-249; p. 7, lines 502-515; Supplement p. 4, lines 248-256.
- **The package is presentationally disciplined.** Captions usually state the split, teacher count, metric, and evidence class; formatting, anonymity, references, and the required LLM statement are in order.

### Major weaknesses

**W1. Novelty framing is broader than the defensible delta.**

- **Weakness:** The title and conclusion suggest a general theory of “decision and temporal fidelity,” but several components are already established: graph ANN-to-SNN conversion by One-Spike, temporal misalignment, and prediction-driven rather than reconstruction-driven quantization.
- **Evidence basis:** Main p. 2, lines 105-112 explicitly says graph conversion and an alternative temporal code do not establish novelty. Main p. 7, lines 507-508 lists direct One-Spike comparisons as an extension. Absent from the related work are prediction-difference and later low-latency alignment precedents.
- **Reviewer deduction:** The reviewer can classify the work as an elementary application of known principles to a small graph converter rather than a new CVPR-level result.
- **Required fix:** Reframe the contribution as a counterexample-driven audit of frozen graph-event conversions. State the exact new object: graph-specific hard-event evidence of adverse margin changes and an exact local-prefix construction that survives terminal decoding while changing downstream computation. Add PD-Quant and the 2026 alignment work as positioning, not as unrelated citation volume.

**W2. The strongest new claims lack breadth and confirmation.**

- **Weakness:** The main mismatch tests use one teacher and previously exposed validation graphs; the temporal test uses one teacher and 128 training graphs per task.
- **Evidence basis:** Main p. 3, lines 237-249; p. 5, lines 312-320; p. 6, lines 395-410; p. 7, lines 504-509. Supplement p. 1, lines 022-032 calls the training profiles descriptive.
- **Reviewer deduction:** The reviewer cannot tell whether the reported rates are stable properties of graph conversion or artifacts of this teacher, event adapter, and development history.
- **Required fix:** Reuse existing frozen checkpoints and stored predictions to repeat the hard and temporal diagnostics on all three teachers, with teacher-level variation and paired graph-level intervals. If an untouched confirmation split cannot be created, keep the claim explicitly descriptive. An independent task or a direct One-Spike run is a larger experiment and should not be framed as a rebuttal-only obligation.

**W3. The temporal decision evidence is under-rendered.**

- **Weakness:** The abstract and introduction claim changed downstream predictions, but Fig. 5 displays only whole-network event counts; the decision evidence appears as aggregate accuracy and teacher-agreement changes in text.
- **Evidence basis:** Claim at main p. 1, lines 010-011 and lines 071-073. Figure 5 and §4.4, p. 6, lines 395-419, report event totals and aggregate accuracies but not the number of changed graph predictions or a paired interval.
- **Reviewer deduction:** A reader can accept that the codeword exchange changes global activity while remaining uncertain whether the decision effect is broad, localized, or driven by a small subset.
- **Required fix:** From retained per-graph predictions, report the number of changed predictions, teacher-agreement transitions, class-margin deltas, and a paired graph-bootstrap interval for both exchange directions. Make this a second panel in Fig. 5. Qualify the abstract as “same local endpoint and local event count, with a different first-prefix history and absolute amplitude.”

**W4. Related-work positioning omits conceptual precedents.**

- **Weakness:** The paper cites only nine works and does not relate its central principle to prediction-difference quantization or later low-latency ANN-SNN fidelity work.
- **Evidence basis:** Related Work, main p. 2, lines 089-125; bibliography p. 8, lines 537-577.
- **Reviewer deduction:** The apparent novelty can collapse because the general “local reconstruction does not equal prediction behavior” lesson is already known outside graph SNNs.
- **Required fix:** Add one compact positioning paragraph: acknowledge prediction-difference quantization as a conceptual precedent, distinguish static convolution/quantization from graph-dependent hard-event propagation, and explain how the temporal result adds first-prefix history rather than repeating temporal-misalignment in general. No large experiment is required to cite these works accurately.

**W5. Several calibration statements exceed their uncertainty.**

- **Weakness:** Small differences of 0.1-0.4 percentage points are used to argue that a metric “does not select a superior target,” while most do not have paired intervals.
- **Evidence basis:** Abstract p. 1, lines 016-017; §4.3 and Table 3, p. 5, lines 341-365. Supplementary §Statistics, p. 1, lines 022-032, does not claim inference for these comparisons.
- **Reviewer deduction:** The result is consistent with no stable advantage, but it can also arise from insufficient power or seed variation.
- **Required fix:** Either report paired graph-level intervals for all calibration contrasts using retained predictions or narrow the claim to “the tested objectives show no consistent advantage.” The latter is a manuscript-only fix.

### Moderate weaknesses

**W6. The paper carries too many evidence families for its central story.**

- **Weakness:** Six main-result subsections move among fidelity, positions, hard interventions, calibration, graph-operator controls, temporal codewords, centering, timing, and historical node classification.
- **Evidence basis:** Main §4.1-§4.6, p. 4-7, lines 279-480; Supplement Table 1, p. 2, summarizes seven families.
- **Reviewer deduction:** The strongest spatial and temporal counterexamples compete for attention with secondary controls and historical context.
- **Required fix:** Keep the hard-event result and temporal-prefix result as the two pillars. Compress calibration into one paragraph/table panel, move source-threshold iteration detail and historical node context to the supplement, and reduce timing to the one caveat needed to prevent an efficiency overread.

**W7. Figure and terminology support unevenly carries the argument.**

- **Weakness:** Fig. 4 relies on the caption to explain quadrants; Fig. 5 shows activity rather than the decision change announced in the abstract; Fig. 6 uses “ADDs,” “additions,” “arithmetic proxy,” and “projected additions” for closely related concepts. “Low-timestep,” “short-prefix,” “active tick,” “prefix,” and “horizon” alternate.
- **Evidence basis:** Fig. 4, p. 5; Fig. 5, p. 6; Fig. 6, p. 7; main p. 1, lines 001-023 and p. 3-4, lines 237-283; Supplement p. 3, lines 185-207.
- **Reviewer deduction:** The reader must reconstruct what is plotted and whether two quantities are identical, which weakens a visually driven CVPR submission.
- **Required fix:** Choose canonical terms once: “active ticks” for simulation depth, “reconstruction metric” for local error, “teacher agreement” for decision matching, and “projected additions” for the proxy. Add quadrant labels and effect summaries to Fig. 4; add decision-flip evidence to Fig. 5; rename the Fig. 6 axis consistently.

**W8. Artifact availability is asserted but not exposed in the reviewed package.**

- **Weakness:** Supplement p. 1, lines 011-021 says a data package contains CSVs, hashes, and deterministic generators, but the frozen manuscript package contains no artifact pointer or manifest intended for reviewers.
- **Reviewer deduction:** The reproducibility claim is not independently verifiable from the supplied materials, and rerunning the teacher network still requires checkpoints.
- **Required fix:** In the submitted supplementary material, include or clearly package the anonymous data/code manifest described in the text; state exactly which tables/figures can be regenerated without checkpoints and which experiments require model artifacts.

## 7. Potentially Missing Related Work

### One-Spike SNN

```text
Work: Hwang and Kung, “One-Spike SNN,” IEEE TETC 2025, arXiv:2403.08786.
Status: searched
Why relevant: It is the closest low-spike graph conversion work and reports GCN conversion with an average 0.90% accuracy loss.
Overlap: The manuscript cites it and explicitly declines to claim novelty for graph conversion itself.
Needed comparison: A matched One-Spike adaptation would test whether the T=1 bottleneck persists under the closest competing low-timestep code. This is a large experiment; the current analysis paper should not be rejected solely for omitting it, but its absence limits any converter-level superiority claim.
```

### Temporal Misalignment in ANN-SNN Conversion

```text
Work: Bojkovic, Wu, and Gu, “Temporal Misalignment in ANN-SNN Conversion and its Mitigation via Probabilistic Spiking Neurons,” ICML/PMLR 2025.
Status: searched
Why relevant: It establishes that temporal treatment is a conversion problem rather than a new observation.
Overlap: The manuscript makes a narrower exact-local-equivalence claim.
Needed comparison: Cite the distinction explicitly in the title/contribution list: prior work motivates temporal misalignment; this work isolates a first-prefix ambiguity that survives exact terminal local decoding and local count equality in a graph adapter. No direct numerical comparison is necessary for this narrower claim.
```

### PD-Quant

```text
Work: Liu et al., “PD-Quant: Post-Training Quantization Based on Prediction Difference Metric,” CVPR 2023.
Status: searched
Why relevant: It already demonstrates that local reconstruction can be a poor objective relative to prediction behavior.
Overlap: Conceptual precedent for the spatial thesis; no graph-event or temporal-prefix overlap.
Needed comparison: Cite and delimit the delta. The paper should not present “reconstruction is weaker than decision fidelity” as a new general principle; it should present graph-dependent signed-event evidence as the new domain-specific result.
```

### Closed-loop low-latency ANN-SNN alignment

```text
Work: Li et al., “High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation,” PMLR 306, 2026.
Status: searched
Why relevant: It studies the low-latency regime and jointly uses output and feature alignment.
Overlap: Different model family and no graph-specific terminal-prefix counterexample.
Needed comparison: Cite as current competing framing for low-latency fidelity. Explain why a training-free graph adapter requires a different analysis. Because it became public near the contemporaneous-work window, cite and discuss if it informed the work; do not claim exhaustive coverage.
```

### Concentration–alignment quantization analysis

```text
Work: Federici et al., “Dissecting Quantization Error: A Concentration-Alignment Perspective,” PMLR 306, 2026.
Status: searched
Why relevant: It analyzes why error direction/alignment matters rather than only magnitude.
Overlap: General conceptual overlap with the direction-versus-norm explanation.
Needed comparison: A citation-level distinction is appropriate. Its linear quantization analysis does not test graph-dependent hard events or downstream temporal states.
```

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Message-improving hard events worsen the teacher-class margin in 40.94% of MNIST and 31.90% of CIFAR10 cases. | Abstract p. 1, lines 005-009; §4.2 p. 5, lines 312-320 | 512 exposed validation graphs per task; actual hard forward pass; graph-cluster 95% bootstrap intervals; Fig. 4 and Table 2 | Adequate within sample; weak for breadth | The factual rate is supported for one teacher, but not established as a general property of graph conversion. | Add three-teacher replication from frozen checkpoints and report between-teacher variation. |
| The first node site is the largest single-site low-timestep recovery point. | Intro p. 1, lines 073-077; §4.1 p. 4, lines 291-299; Fig. 3 | Full exposed validation; continuous causal ReLU; three teachers; nonadditive groups | Adequate | Supported descriptively for the tested fixed GCNs; “bottleneck” may imply an actionable event-level result although the intervention is continuous. | Rename as “largest continuous-recovery site” and state the finite-event separation in the same sentence. |
| Exact local terminal decoding and event count can change downstream predictions and activity. | Abstract p. 1, lines 009-011 and 071-073; §4.4 p. 6, lines 395-419 | Bitwise local equality; source replay; 128 training graphs; teacher 41; aggregate event and accuracy changes | Adequate for existence; weak for magnitude and generality | The local construction is valid, but per-graph decision flips, intervals, and teacher variation are absent. Fig. 5 shows only activity. | Add per-graph decision flips and paired intervals from retained outputs; add decision evidence to Fig. 5. |
| Tested calibration objectives do not identify a consistently superior correction target. | Abstract p. 1, lines 016-017; §4.3 p. 5, lines 341-365; Table 3 | Matched resources; three-teacher historical tests; single-teacher shared-prefix controls with some paired intervals | Moderate | The direction of the evidence is supportive, but small differences and missing intervals for most contrasts prevent a strong inference. | Replace “distinguish” with bounded wording and add paired intervals where retained predictions permit. |
| Residual centering lowers projection activity without resolving the decision problem. | Abstract p. 1, lines 019-021; §4.5 p. 6-7, lines 420-460; Fig. 6 | Shared thresholds; full training split; one teacher; separate three-teacher validation; complete timing caveats | Adequate, strongly qualified | The reduction in the declared proxy is supported; the wording correctly avoids claiming a complete efficiency advantage. | Define the proxy consistently across text, figure, and supplement and foreground that it excludes the dominant continuous work. |
| Assessing a decision requires every consumed prefix and whole-network activity after a realizable intervention. | Conclusion p. 7, lines 492-515 | Derived from the two failure modes and several controls | Weak as a validated protocol; adequate as a recommendation | The rule follows logically from the examples but has not been evaluated as a decision procedure across models or backends. | Frame it as a diagnostic protocol motivated by the results, not a fully established condition. |
| Graph-specific reconstruction evidence identifies information discarded by message norms and terminal reconstruction. | Abstract p. 1, lines 021-023; Conclusion p. 7, lines 481-515 | Analytical identities, hard interventions, exact codewords, controls | Adequate for the stated networks | This is the strongest bounded claim and the appropriate center of the paper. | Put this claim in the opening contribution paragraph and make every result point to one of its two parts. |

## 9. Experiment / Benchmark / Reproducibility Audit

### Baselines

- The source-linked DCGS threshold iteration, histogram calibration, feature-only allocation, conjugated-operator allocation, and actual-message allocation are reasonable controls for the paper's diagnostic questions. Evidence: main p. 3-6, lines 267-392; supplement §2.
- The closest low-timestep graph converter, One-Spike, has no matched experiment. The manuscript correctly avoids converter-level superiority claims and names the gap at main p. 7, lines 507-508. Treat this as a score-lifting long-term experiment, not a prerequisite that invalidates the local analysis.
- No external graph model or independent vision task confirms that the two failure modes recur beyond the frozen four-layer adapters.

### Ablations and controls

- The position analysis includes empty and all-site interventions and states that group effects are nonadditive. Evidence: main p. 4, lines 250-256.
- The hard-event analysis includes upward/downward legal moves, two label-free coordinate rules, cluster-preserving intervals, and a concrete high-margin flip example. Evidence: main p. 5, lines 312-327.
- The graph-operator controls weaken attribution of aggregate gains to the correct adjacency, which is a positive scientific choice. Evidence: main p. 6, lines 378-392.
- The centering experiment controls thresholds and first-layer exactness, but studies one GatedGCN teacher in the training-split profile. Evidence: main p. 6-7, lines 420-460.

### Datasets and benchmarks

- MNIST and CIFAR10 superpixel classification are established but relatively simple for a CVPR analysis of visual graph conversion. The historical Cora/CiteSeer/PubMed/Chameleon/Squirrel/ogbn-arxiv table is context and does not validate the hard-event result. Evidence: supplement §2 and Table 4, p. 2-4.
- All visual official validation and test splits were exposed during development. This is honestly disclosed but prevents treating reported effects as blind confirmation.

### Metrics

- Teacher agreement, ground-truth accuracy, full teacher-class margin, immediate message MSE, local and global event counts, projected additions, and latency are mostly defined and kept distinct.
- Decision-flip counts are missing for the exact temporal exchange even though stored per-graph predictions are said to exist. This is the most valuable small reanalysis.
- Timing is appropriately scoped to dense software execution and excludes energy. The paper should continue to avoid any chip-level efficiency implication.

### Statistical rigor

- Hard-event intervals use graph-cluster bootstrap and preserve repeated interventions. This is appropriate for the intervention design.
- Three-teacher tables report sample SD and correctly describe variability across training seeds on one shared evaluation set.
- Calibration comparisons use resource matching but largely lack paired intervals for small differences.
- Temporal code exchanges report no uncertainty intervals, teacher variation, or graph-flip counts.

### Robustness and failure cases

- The manuscript is unusually explicit about failures of proposed policies and avoids converting unsuccessful prototypes into claims. Evidence: supplement p. 4, lines 234-256.
- The main missing robustness analysis is teacher heterogeneity and a confirmation subset that was not used for prior development. A new dataset is a larger experiment and should remain an accepted limitation if unavailable.

### Implementation and artifacts

- Model width, graph normalization, event levels, event timing, calibration resources, timing protocol, and array provenance are described in more detail than most conference papers.
- The frozen review folder does not expose the data package named in supplement p. 1, lines 011-021. The package hash manifest itself is strong evidence of provenance, but reviewer auditability depends on the package being submitted or referenced through an anonymous mechanism that CVPR policy permits.
- Restoring teacher checkpoints is necessary for full inference reruns. The paper says this clearly, but the review package does not establish checkpoint availability.

### Limitations

- Main p. 7, lines 502-515 and supplement p. 4, lines 234-256 give concrete boundaries for teacher count, exposed data, topology changes, event amplitude, full cost, and One-Spike comparison. This section is a genuine strength.

## 10. Multi-Reviewer Panel

```text
Reviewer: Best-justified / sympathetic
Expertise: graph-efficient vision and ANN-SNN conversion
Likely score: 6 / borderline positive
Confidence: 4
Main positive signal: The hard-event intervention directly connects reconstruction error to the class margin, and the temporal construction is exact rather than correlational.
Main negative signal: The result is not yet shown to survive teacher changes or blind confirmation.
Evidence basis: Main p. 3-7, Eqs. 5-9, Tables 2-3, Figs. 3-6.
Score-change condition: Existing-data teacher replication plus decision-flip intervals would make the analysis defensible at 6.
```

```text
Reviewer: Critical
Expertise: novelty and scientific contribution
Likely score: 3 / reject tendency
Confidence: 4
Main positive signal: The paper does not overclaim a new converter and reports failures honestly.
Main negative signal: The central principle is familiar from prediction-aware quantization and temporal-misalignment work; the new evidence is narrow and retrospective.
Evidence basis: Main p. 2, lines 105-112; p. 7, lines 504-508; searched PD-Quant and temporal-misalignment records.
Score-change condition: A precise domain-specific delta and stronger cross-teacher evidence would raise this reviewer's tendency.
```

```text
Reviewer: Method / soundness
Expertise: spiking event dynamics and graph propagation
Likely score: 5 / borderline
Confidence: 5
Main positive signal: The margin identity is correct, the hard forward pass is rerun, and temporal equality is verified by source replay and actual forwarding.
Main negative signal: The linear sufficient condition is elementary, while the central graph claim relies on an intervention that is not a realizable encoder.
Evidence basis: Main p. 2-3, lines 156-236; supplement §3.
Score-change condition: A realizable policy or a finite-event selector that improves decision/prefix outcomes without increasing global cost would move soundness and significance upward.
```

```text
Reviewer: Evidence / experiment
Expertise: empirical evaluation and statistics
Likely score: 3 / reject tendency
Confidence: 4
Main positive signal: Hard-event bootstrap intervals and explicit descriptive-data boundaries are appropriate.
Main negative signal: Main effects use one teacher on exposed data; temporal decision changes lack paired intervals and graph-flip counts.
Evidence basis: Main p. 3, lines 237-249; p. 5-6, lines 312-419; supplement p. 1, lines 022-032.
Score-change condition: Three-teacher existing-data replication, paired decision intervals, and a fixed confirmation subset would materially improve the evidence score.
```

```text
Reviewer: Novelty / positioning
Expertise: quantization, conversion, and related-work coverage
Likely score: 3 / reject tendency
Confidence: 4
Main positive signal: Exact graph-event counterexamples are more specific than a generic restatement of reconstruction-versus-accuracy.
Main negative signal: Prediction-aware quantization and temporal misalignment already occupy much of the conceptual space; One-Spike already converts GCNs.
Evidence basis: Main p. 2, lines 089-125; public PD-Quant, One-Spike, and 2025-2026 conversion records.
Score-change condition: Sharper scoped novelty plus a direct discussion of these precedents would raise this reviewer's tendency without requiring a new method.
```

```text
Reviewer: Writing / clarity and novice advocate
Expertise: reviewer-facing exposition
Likely score: 5 / borderline
Confidence: 5
Main positive signal: The question is understandable, captions report evidence scope, and the prose avoids defensive or hype-heavy claims.
Main negative signal: Seven evidence families compete, “tick/prefix/horizon” drift, and the key temporal decision evidence is in text rather than the figure.
Evidence basis: Abstract p. 1; §4.1-§4.6 p. 4-7; Figs. 4-6.
Score-change condition: A two-pillar story, canonical terminology, and decision-based Fig. 5 would make the paper substantially easier to champion.
```

```text
Reviewer: Domain application
Expertise: visual superpixel graphs and conversion practice
Likely score: 5 / borderline
Confidence: 4
Main positive signal: The experiments respect graph normalization, continuous residual work, event amplitude, and local-versus-global cost.
Main negative signal: MNIST/CIFAR10 superpixels are a limited domain for a general claim about visual graph conversion, and no deployment or harder visual graph is tested.
Evidence basis: Main p. 2, lines 126-153; p. 7, lines 462-480 and 502-515.
Score-change condition: A hard visual graph task or an external teacher family would improve practical scope; this is a large extension.
```

```text
Reviewer: Ethics / reproducibility
Expertise: packaging and research integrity
Likely score: 6 / borderline positive
Confidence: 4
Main positive signal: Anonymous format, LLM disclosure, public benchmarks, event/cost limitations, and frozen-file hashes are well handled.
Main negative signal: The claimed data package and checkpoint availability are not visible in the frozen review materials.
Evidence basis: Main p. 8, lines 516-535; supplement p. 1, lines 011-021 and p. 3, lines 195-232; `input-manifest.json`.
Score-change condition: Package the anonymous audit artifact and state precisely which reported results can be regenerated without teacher checkpoints.
```

```text
Reviewer: AC / meta-reviewer
Expertise: calibration across the panel
Likely score: 4 / weak reject
Confidence: 4
Main positive signal: A sympathetic expert can defend the exact local counterexample and the careful controls.
Main negative signal: The strongest evidence is one-teacher and retrospective, and the novelty delta needs a tighter domain-specific claim.
Evidence basis: Full main paper and supplement.
Score-change condition: Manuscript reframing and existing-data reanalysis could produce a credible 5-6 discussion; a new direct comparator or independent confirmation could support 7.
```

```text
Agreement: The analysis is careful, the evidence is honestly bounded, and the current story is too broad for its support.
Disagreement: The best-justified reviewer sees a useful diagnostic analysis; the critical and evidence reviewers see a retrospective observation paper below the Main Track bar.
Decisive positive axis: Hard-event and exact-prefix counterexamples are concrete and inspectable.
Decisive negative axis: Novelty and external validity are not yet sufficient for a general CVPR claim.
Unresolved evidence: Between-teacher stability, temporal decision-flip frequency, confirmation beyond exposed development data, and numerical One-Spike comparability.
AC stance: Weak reject at 4/10. Reward a reframed and strengthened version; do not average away the novelty and validation risks.
```

## 11. Concerns Table

Execution classes are marked in the action text: `[manuscript only]`, `[small reanalysis]`, or `[large experiment]`.

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | major | General novelty claim outruns the graph-specific delta. | Main p. 1, lines 001-023 and 078-088; p. 2, lines 105-112; p. 7, lines 507-515 | Originality, significance | writing | `[manuscript only]` Reframe around a frozen graph-event audit with two counterexamples; state that the general reconstruction-versus-decision principle has precedents. | `ccf-paper-writer` | Clear scoped contribution plus honest prior-work delta: +0.5 to +1 overall. |
| C2 | major | Main spatial and temporal findings use one teacher and exposed/training data. | Main p. 3, lines 237-249; p. 5, lines 312-320; p. 6, lines 395-419; p. 7, lines 504-509 | Evidence, soundness, significance | experiment | `[small reanalysis]` Reuse frozen three teachers and retained outputs; report teacher-level effects and paired graph intervals. `[large experiment]` Add an independent task only if feasible. | `ccf-experiment-designer` / `run-experiment` | Stable same-direction effects across three teachers and a fixed confirmation subset: +1 overall; independent task adds further support. |
| C3 | major | Temporal prediction claim is stronger and less visible than the evidence. | Abstract p. 1, lines 009-011 and 071-073; Fig. 5 and §4.4 p. 6, lines 395-419 | Claim-evidence alignment, figure narration | writing, figure/table | `[small reanalysis]` Add per-graph prediction flips, margin deltas, and paired intervals from retained outputs; `[manuscript only]` add caveat on absolute amplitude. | `paper-figure` / `ccf-paper-writer` | Decision evidence becomes explicit and stable: +0.5 to +1. |
| C4 | major | Related work omits prediction-aware quantization and 2026 low-latency alignment. | Main p. 2, lines 089-125; bibliography p. 8, lines 537-577; searched public records | Positioning, originality | related-work | `[manuscript only]` Add a concise positioning paragraph distinguishing conceptual precedents from graph-dependent hard-event and temporal-prefix evidence. | `ccf-paper-writer` | Reviewer can name the exact novelty delta: +0.5 overall. |
| C5 | moderate | Calibration conclusion lacks uncertainty for small differences. | Abstract p. 1, lines 016-017; Table 3 and §4.3 p. 5, lines 341-365 | Soundness, evidence | experiment, writing | `[small reanalysis]` Add paired intervals from retained predictions; `[manuscript only]` otherwise replace “distinguish” with “no consistent advantage in the tested objectives.” | `ccf-experiment-designer` / `ccf-paper-writer` | Intervals preserve the direction or wording is properly narrowed: dimension-only improvement. |
| C6 | moderate | Six result subsections dilute the two-pillar story. | Main §4.1-§4.6 p. 4-7, lines 279-480; Supplement Table 1 p. 2 | Clarity, contribution display | compression, writing | `[manuscript only]` Foreground hard events and temporal prefix; merge calibration/control paragraphs and move secondary context to supplement. | `ccf-paper-writer` | A reader can state both contributions and their evidence after one page: +0.5 overall. |
| C7 | moderate | Figures and terminology are not fully self-explanatory. | Fig. 4 p. 5; Fig. 5 p. 6; Fig. 6 p. 7; main p. 3-4, lines 237-283; supplement p. 3, lines 185-207 | Clarity, visual evidence | figure/table, writing | `[manuscript only]` Canonicalize terms and redesign Figs. 4-5 around effect direction and decision flips; harmonize Fig. 6 labels. | `ccf-visual-composer` / `ccf-paper-writer` | Captions and figures alone communicate the evidence: +0.25 to +0.5 overall. |
| C8 | moderate | Reviewer-visible artifact package is not demonstrated. | Supplement p. 1, lines 011-021; p. 3, lines 195-232 | Reproducibility | reproducibility | `[manuscript only]` Include or clearly enumerate the anonymous audit package and state checkpoint requirements and regeneration limits. | `ccf-submission-checker` | Reviewers can identify exactly which plots/tables they can regenerate: +0.25 to +0.5 overall. |
| C9 | moderate | Direct One-Spike comparison is absent. | Related Work p. 2, lines 105-108; Discussion p. 7, lines 507-508 | Positioning, evidence | experiment | `[large experiment]` Add a matched One-Spike graph adaptation before submission or leave the paper scoped as diagnostic analysis. Do not request it as a rebuttal-only contribution. | `run-experiment` | A fair matched baseline would materially strengthen a converter-level claim; the current paper should avoid that claim. |
| C10 | minor | “Bottleneck” can be read as a realizable event-level correction. | Abstract p. 1, lines 014-016; §4.1 p. 4, lines 291-299 | Terminology, claim scope | writing | `[manuscript only]` Rename it as the largest continuous-recovery site and repeat the finite-event caveat at first use. | `ccf-paper-writer` | Removes a foreseeable overinterpretation: dimension-only improvement. |

## 12. AC / Meta-Review

- **Reviewer consensus:** The paper is scientifically careful, readable, and honest. Its strongest contribution is the exact event-level demonstration that local reconstruction and local terminal equality are weaker than the downstream decision computation.
- **Reviewer disagreement:** A sympathetic reviewer can defend an empirical/analytical paper. A critical reviewer can reject it as a narrow, retrospective transfer of familiar concepts with insufficient teacher or task breadth.
- **Decisive acceptance axis:** Whether the paper can establish a clear graph-specific novelty delta and show that the two counterexamples are stable across existing teachers or a fixed confirmation subset.
- **Decisive rejection axis:** The current combination of one-teacher main evidence, exposed data, absent per-graph temporal decision statistics, and no close conceptual positioning.
- **AC stance:** Weak reject at 4/10. The technical care is real, but the contribution is not yet positioned or validated broadly enough for the Main Track.
- **Discussion risks:** Authors may try to solve the issue by adding more controls. The more effective move is a smaller, sharper paper centered on the two failure modes with explicit novelty and stronger use of retained evidence.

## 13. Quantitative Scores

### Review scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
| --- | ---: | ---: | --- | --- |
| Quality | 3 | 4 | Whole manuscript and supplement | The technical execution and honesty are above average, but the contribution is analysis-only and evidence breadth is limited. Raise with focused reanalysis and compression. |
| Clarity | 4 | 5 | Abstract p. 1; analysis p. 2-4; results p. 4-7 | The core question and event semantics are clear. Dense secondary controls and terminology drift keep it below 5. |
| Significance | 3 | 4 | Main p. 7, lines 481-515; Supplement p. 4, lines 234-256 | Useful for graph conversion and evaluation practice, but the paper does not establish a reusable method or broad new capability. |
| Originality | 2 | 4 | Main p. 2, lines 105-112; searched PD-Quant, One-Spike, temporal-misalignment, and low-latency alignment work | The general principle is familiar; exact graph-event and first-prefix counterexamples are the narrower novel component. |
| Soundness | 4 | 5 | Main p. 2-3, lines 156-236; Supplement §3 | The identities, hard forwarding, and event semantics are credible. The claim boundary remains more important than a correctness defect. |
| Evidence | 2 | 4 | Main p. 3-6, lines 237-419; Tables 2-3; Fig. 3-5 | One-teacher retrospective evidence, absent temporal uncertainty, and missing confirmation prevent a stronger score. |
| Reproducibility | 3 | 4 | Supplement p. 1, lines 011-021; p. 3, lines 195-232; verification package | Provenance and timing details are strong; the described artifact package and checkpoints are not visible in the frozen review input. |
| Ethics / Limitations | 4 | 4 | Main p. 7, lines 502-515; p. 8, lines 516-535; Supplement p. 4, lines 234-256 | Limitations and LLM disclosure are strong. No material safety or ethics issue was found. |

```text
Overall: 4/10
Scholarly confidence: 4/5
Recommendation: weak reject on the CVPR 2027 Main Track / borderline for Findings Track if validation is judged adequate
```

All dimensions and the overall score are internally consistent with the concerns table: technical soundness is good, but originality and evidence breadth are the decision-limiting weaknesses.

### Writing-review scorecard

| Dimension | Weight | Score (1-5) | Confidence (1-5) | Evidence basis | Concrete repair |
| --- | ---: | ---: | ---: | --- | --- |
| Storyline and motivation | 10 | 3 | 4 | Abstract and Introduction p. 1 | State one graph-event audit thesis before the list of evidence families. |
| Contribution display | 10 | 3 | 4 | Main p. 1, lines 078-088; p. 7, lines 492-515 | Separate the two counterexamples from global controls and recommendations. |
| Paragraph logic | 10 | 4 | 5 | Main body | Paragraphs generally carry one role; §4.3-§4.5 mix several controls. |
| Claim-evidence alignment | 12 | 3 | 4 | Abstract; Tables 2-3; Fig. 5 | Add temporal decision intervals and narrow calibration wording. |
| Method readability | 9 | 4 | 5 | §3 p. 2-3 | Explain the event update before the margin analysis and move some node context later. |
| Experiment narration | 9 | 3 | 4 | §4 p. 4-7 | Begin result paragraphs with the question and end with the bounded conclusion; make figures carry decision evidence. |
| Related-work positioning | 8 | 2 | 4 | main p. 2 | Add prediction-difference and recent low-latency conversion positioning. |
| Terminology and notation consistency | 8 | 3 | 5 | Abstract; §4.1-§4.6; Fig. 6 | Canonicalize tick/prefix/horizon and additions/proxy terminology. |
| Prose discipline and voice | 10 | 4 | 5 | Whole paper | Prose avoids hype and is mostly direct; remove residual overclaims such as “establish.” |
| LaTeX and format discipline | 6 | 5 | 5 | PDF and source | Format, captions, labels, anonymity, and page allocation pass. |
| Reviewer-facing risk | 8 | 3 | 4 | Whole paper | The main risks are hidden novelty delta and diffuse evidence, not language quality. |

```text
Weighted writing score: 3.33/5
Writing risk band: moderate
Interpretation: The manuscript is readable, but it does not yet make the strongest defensible contribution easy to champion.
```

## 14. Questions For Authors

1. How many individual graph predictions change under each exact codeword-exchange direction, and what are the paired graph-level intervals and teacher-agreement transitions from the retained raw predictions?
2. Do the adverse-margin rates remain directionally stable across all three frozen teachers, or is the effect concentrated in teacher 41?
3. Which calibration-objective contrasts remain non-null only because intervals were not reported, and which survive paired graph-resampling intervals?
4. What exact domain-specific claim distinguishes this work from PD-Quant's prediction-difference principle and the established temporal-misalignment result?
5. Which tables and figures can a reviewer regenerate from the submitted anonymous package without teacher checkpoints?

## 15. Score Revision Criteria

```text
Raising the score would require:
- A manuscript reframe to a frozen graph-event audit rather than a general decision-fidelity theory.
- Existing-data three-teacher repetition of the hard-event and temporal diagnostics.
- Per-graph temporal decision-flip counts and paired intervals, represented visually.
- Paired uncertainty for calibration contrasts or correspondingly weaker wording.
- Accurate citation and positioning against prediction-aware quantization and 2026 low-latency conversion work.
- A focused two-pillar layout that moves secondary controls to the supplement.

Expected movement:
- Manuscript-only framing and compression: +0.5 overall.
- Those changes plus existing-data reanalysis: +1 to +1.5 overall, with 6/10 plausible.

Lowering the score would be triggered by:
- Evidence that the temporal prediction changes occur only when event amplitude or unaccounted continuous work changes in a way that invalidates the stated interpretation.
- Failure to reproduce the adverse-margin effect across existing teachers.
- Evidence that the claimed data package cannot regenerate the reported tables and figures.
- Continued presenter-level novelty claims after the closest public work is acknowledged.

Concerns unlikely to change before submission:
- The absence of a new realizable conversion algorithm.
- The absence of blind, independent visual task confirmation.
- The absence of hardware energy measurements and a complete cost model.
These are acceptable only as explicit limitations and must not be converted into claims.
```

The largest feasible score lift now is a smaller, sharper manuscript with stronger use of data that already exists. A new One-Spike adaptation or an independent task would improve the ceiling, but the paper should not depend on a large experiment for its basic coherence.

## 16. Action Plan And CCFA Handoffs

```text
Priority: P0
Action: Reframe the Abstract, Introduction, title, and Conclusion around one contribution: a graph-specific finite-event audit showing two reconstruction-to-decision information losses.
Owner skill: ccf-paper-writer
Input needed: Frozen main/supplement sources, CLAIMS.md, this report.
Expected output: One-page contribution map plus exact edit list for Abstract/Intro/Conclusion; no new claims.
Handoff required: yes

Priority: P0
Action: Add per-graph temporal decision flips, margin changes, teacher-agreement transitions, and paired intervals from retained outputs; redesign Fig. 5 with an activity panel and a decision panel.
Owner skill: ccf-experiment-designer, then ccf-visual-composer
Input needed: Raw per-graph temporal predictions and exchange labels already described in Supplement p. 1.
Expected output: Deterministic CSV plus figure-ready data and caption.
Handoff required: yes

Priority: P0
Action: Replace strong calibration inference with paired intervals or bounded descriptive wording.
Owner skill: ccf-experiment-designer and ccf-paper-writer
Input needed: Retained per-graph calibration predictions and objective labels.
Expected output: Interval table or revised paragraph stating “no consistent tested advantage.”
Handoff required: yes

Priority: P1
Action: Repeat hard-event and temporal diagnostics across the three frozen visual teachers using existing checkpoints and fixed rules.
Owner skill: run-experiment
Input needed: Frozen teacher checkpoints, deterministic intervention/exchange code, and fixed split manifest.
Expected output: Teacher-level effect table, paired intervals, and a pass/fail stability statement.
Handoff required: yes

Priority: P1
Action: Add a compact positioning paragraph for PD-Quant, One-Spike, temporal misalignment, and the 2026 low-latency alignment work.
Owner skill: ccf-paper-writer
Input needed: Verified public bibliographic entries and this report's overlap distinctions.
Expected output: One related-work paragraph and an updated contribution sentence with no first-ever claim.
Handoff required: yes

Priority: P1
Action: Compress §4 into two principal result sections and move secondary calibration, source-TI, node-context, and diagnostic-proxy detail to the supplement.
Owner skill: ccf-paper-writer
Input needed: Current eight-page source and result-to-claim map.
Expected output: Reordered main body with a one-sentence evidence map at the start of Results.
Handoff required: yes

Priority: P1
Action: Make terminology and figures self-contained.
Owner skill: ccf-visual-composer and ccf-paper-writer
Input needed: Result CSVs, current figure generators, and canonical glossary.
Expected output: “Active ticks,” “teacher agreement,” “reconstruction metric,” and “projected additions” used consistently; labels and captions readable without body context.
Handoff required: yes

Priority: P2
Action: Expose the anonymous data/code audit package and checkpoint requirements.
Owner skill: ccf-submission-checker
Input needed: Data package, hashes, deterministic generators, checkpoint inventory, and anonymous packaging policy.
Expected output: Reviewer-visible artifact manifest and exact table/figure regeneration matrix.
Handoff required: yes

Priority: P2
Action: Optional large extension: add a matched One-Spike graph adaptation or independent task confirmation before submission.
Owner skill: run-experiment
Input needed: Public implementation, matched protocol, compute budget, and architecture adaptations.
Expected output: Fair baseline or independent confirmation that supports a converter-level claim.
Handoff required: no for the current submission; do not request it as a rebuttal-only experiment.
```

```text
Recommended next CCFA owner: ccf-paper-writer, with ccf-experiment-designer and ccf-visual-composer for the existing-data reanalysis and Fig. 5.

Checks run:
- Read the CCFA reviewer skill, fixed output format, calibration rubric, CVPR venue guide, writing-review references, shared output standard, privacy/evidence policy, and prose guardrails.
- Verified all 12 `input-manifest.json` entries by path, byte count, and SHA-256; all matched.
- Inspected both PDFs, both TeX sources, CLAIMS.md, audit extracts, bibliography, final audit, and verification manifest.
- Rendered and visually inspected all 8 main pages and all 4 supplement pages.
- Checked page count against the current official CVPR 2027 Author Guidelines.
- Checked anonymity, LLM Usage Statement placement, hidden manipulation phrases, placeholders, captions, tables, references, and resolved compilation claims.
- Ran public-safe related-work searches through official CVF, PMLR, arXiv, and CVPR 2027 sources.

Checks skipped:
- No network training or inference was rerun.
- No access to the referenced experiment data package, teacher checkpoints, or OpenReview enrollment state was available.
- The related-work search was targeted rather than exhaustive.

Unresolved risks:
- Cross-teacher stability of the hard-event and temporal results.
- Per-graph frequency and uncertainty of temporal decision changes.
- Fair numerical One-Spike comparability.
- Reviewer-visible artifact availability.
- Whether a Main Track analysis contribution of this scope is better positioned for Findings.

Output self-check:
- Fixed section order is present.
- Every material concern has an exact main/supplement location or a public-search basis.
- Scorecard dimensions and overall score are consistent.
- Concern severities use the required vocabulary.
- No result or citation was invented.
- Manuscript-only, small-reanalysis, and large-experiment actions are explicitly separated.
```
