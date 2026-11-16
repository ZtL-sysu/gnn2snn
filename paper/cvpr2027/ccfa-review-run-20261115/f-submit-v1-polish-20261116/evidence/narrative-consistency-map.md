# Narrative Consistency Map (S1)

## Central question

Inside a frozen graph-to-spike computation, do the local reconstruction quantities commonly used to judge conversion fidelity determine the quantities a classifier actually needs: the downstream decision and the whole-network event trajectory?

## Core claim chain

1. **Operating regime.** Four active ticks approximately recover teacher accuracy for the visual graph adapters, while one tick exposes a large gap. This establishes that the audit concerns a short-prefix regime in which fidelity metrics are practically useful but incomplete.
2. **Localization.** Conditional continuous interventions locate the largest single-site `T=1` sensitivity at the first node activation. This motivates studying finite changes at that site.
3. **Spatial information loss.** The message identity $\Delta M=AEW^\top$ uses the true graph operator, but its norm discards signed direction relative to class margins. Legal one-event changes that reduce actual message MSE can worsen the converted teacher-class margin.
4. **Temporal information loss.** The DCGS recurrence stores membrane, cumulative output, and integrated states beyond the decoded endpoint. Two legal codewords can have identical local two-tick decoded tensors and equal local event counts while presenting different first-prefix values and amplitudes to downstream consumers; the downstream decisions and whole-network activity can change.
5. **Boundary and implication.** Centered coding and matched calibration controls show that representation and objective choices matter, yet neither removes the need for separate decision and complete-cost validation. The paper therefore proposes a reporting checklist rather than a new conversion algorithm or a universal error-frequency law.

## Module-to-claim alignment

| Module | Claim served | Narrative function | Consistency requirement |
|---|---|---|---|
| Abstract | Whole chain | Compressed problem → two findings → mechanism → scope → implication. | Conditional rates must keep their denominators and task order; temporal equality must include both decoded tensor and local count. |
| Introduction | Central question + claim chain | Creates the spatial/temporal tension before introducing interventions. | Must not imply the two effects form a general conversion theorem or that One-Spike/PD-Quant are baselines. |
| Concept figure | Both information losses | Gives an immediate visual intuition before technical definitions. | Left panel is a schematic; right panel is a fully specified recurrence example, not a dataset estimate. |
| Related Work | Novelty positioning | Groups prior methods by the fidelity concern they address and isolates the graph-specific question. | Existing mechanisms keep their published roles; the paper's evidence remains diagnostic interventions. |
| Analysis | Claims 3-4 | Defines event semantics, margins, and the exact invariants of each test. | Eq. (5) is an exact decision condition; Eq. (6) is only sufficient; terminal equality is weaker than per-tick equality. |
| Interventions / design | Claims 3-4 | Explains how the hard and codeword tests are made legal and how data exposure is handled. | Teacher labels/logits are used only for measurement after replay; exposed splits are descriptive. |
| Results 1 | Claim 1-2 | Establishes the regime and locates the bottleneck. | High-prefix GCN and GatedGCN results come from separate adapters and stay in their own table. |
| Results 2 | Claim 3 | Delivers the primary spatial counterexample with stratified and single-graph support. | Conditional adverse rates are intervention-specific; accuracy improvement is not conversion fidelity. |
| Results 3 | Diagnostic boundary | Shows why more sensitive-looking objectives do not settle the decision question. | Historical per-horizon and shared-prefix fits remain separate; no optimality claim. |
| Results 4 | Claim 4 | Delivers the temporal counterexample and global-activity consequence. | First-prefix profile and absolute amplitude are not controlled; no timing isolation claim. |
| Results 5-6 | Boundary / applicability | Shows a constructive representation change and the gap between local work proxies and complete cost. | Centering is one teacher / training split in its focused comparison; dense timing and prospective additions are different measures. |
| Discussion / Conclusion | Whole chain | Converts the findings into a scoped evaluation checklist. | Must retain the one-teacher, exposed-data, fixed-GCN, and open-comparison boundaries. |
| Supplement | Auditability | Records evidence families, derivations, historical context, and cost accounting caveats. | Historical node experiments and adapter families are never pooled with the visual hard-event experiments. |

## Abstract, contribution, and experiment alignment

- The abstract's first finding corresponds exactly to Table 2 and Fig. 4. Its denominator is message-improving interventions, not all attempts and not all conversions.
- The abstract's second finding corresponds exactly to Fig. 5 and the source verification around Eq. (4). The matched quantities are the local decoded tensor and local nonzero event count.
- The introduction's numerical summary repeats the same Table 2 and position-recovery values; no result is introduced there that the results section does not support.
- The stated contribution is a controlled audit plus an evaluation checklist. No sentence may promise a converter, a universal rate, a hardware-efficiency gain, or a new training objective.
- Ground-truth accuracy and teacher agreement are reported separately wherever they could diverge.
- Whole-network activity is described as nonzero scalar events; local event count, delivered weight, latency, and energy are not substituted for it.

## Narrative progression for the prose pass

1. Begin with the specific failure mode: a smaller graph-message error or a matched local endpoint can still leave the downstream decision unprotected.
2. Compress background conversion methods into what they establish and what remains untested in a frozen graph adapter.
3. State the two information losses in parallel, then define exact invariants before showing counterexamples.
4. Let tables carry numbers; use prose to answer the question each result block tests and to explain the mechanism.
5. Place centering and calibration after the two core tests, as boundary evidence showing why proxies require downstream validation.
6. Close by returning to the evaluation checklist and the exact scope of the counterexamples.

## Consistency checks to repeat after any edit

- [ ] Title, abstract, introduction, and conclusion all use "audit" framing.
- [ ] 40.9% / 31.9%, 332/811, 296/928, and all confidence intervals agree with Table 2.
- [ ] Temporal prediction changes, total event counts, and percentage changes agree across text and captions.
- [ ] CIFAR10 doubling activity change remains 1.33% with counts 6,876,233 → 6,785,108 where counts are shown.
- [ ] Exposed-data caveat appears wherever full validation/test evidence is interpreted.
- [ ] First-prefix/amplitude and timing caveats remain in the temporal-results paragraph.
- [ ] Citations remain keyed to their established claims; bibliography facts are untouched.
- [ ] LLM Usage Statement is unchanged and remains in the anonymous submission.
- [ ] Figure and table labels, captions, and in-text references remain one-to-one.
