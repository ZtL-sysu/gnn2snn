# P9 merged scientific actions

Review execution: fresh independent native Luna reviewer; two ARC perspectives in one response plus a separate CCFA full report, not two independent human reviews. Primary checked implementation and raw-array integrity separately. Actual ARC18 completed. The bounded empirical manuscript may proceed to targeted repair; no publication-readiness or method-success acceptance is implied.

| Concern | Proposition/action | Location | Evidence/status and check |
|---|---|---|---|
| A1 novelty | NARROW: known calibration, graph quantization and GCN conversion are precedents; present a conditional empirical test, with no first-method claim | Related Work, Discussion | Existing LITERATURE_DEEP_REVIEW plus public-safe graph/SNN conversion queries on2026-11-14. One-Spike arXiv2403.08786 explicitly includes GCN conversion; IJCAI2022 SpikingGCN is directly trained rather than conversion. Search is not exhaustive; novelty and direct complete-source comparisons remain publication limitations. |
| B1 selection exposure | REWRITE: many candidate families reuse the fixed subset; all new endpoint intervals are descriptive and selection-exposed | Evaluation scope | Source metadata,142 exploratory endpoints, existing disclosures. No independent set or seed claim permitted. |
| A2 exact objectives | REWRITE: all-message=0.5*(six-site local mean+first-site normalized next-message error); first-message uses first-local in place of mean | Method3.2 | objective.py loss_terms; primary code inspected. |
| A3 causal/gap definitions | REWRITE u0=0; closure=1-d_intervention/d_base per teacher before averaging | Method3.3 | ideal diagnostic source; current converted input, no teacher-hidden overwrite. |
| B2 intervals | REWRITE paired correctness variable, sample SD, 1.96 multiplier, n512, percent conversion | Method3.5 | summarize_rescue.py lines38–42; all raw pairing IDs checked. |
| B3 costs | REWRITE offline cached fitting loop1.1214–1.7101s for listed T1 fits; separate from online timing and setup | Results4.4 | *_T1_training.json and timed source loop; single-run observations, no speed ranking. |
| B4 comparator roles | KEEP scientifically qualified strong reference; REWRITE compact family/budget prose and explicit visual references | Method3.4, Results | Same-base selector controls event-opportunity matched; stronger local is accuracy-only reference. |
| A4 implementation/provenance | KEEP precise equations; define permutation fixed seed and source/subset identity in external reproduction index | Method3.2, delivery | make_permutation seed20261116, per-graph CPU generator with identity permutation rolled for n>1. State transitions and exact single-node gain tested previously; no new execution claimed by reviewer. |
| Layout | REWRITE placement anchors/references, inspect compiled pages | Results, P15/P18 | No fabricated layout verdict; later PDF required. |

Primary resolution: no fatal integrity defect in current finite evidence. Definition, cost and exposure gaps require P10 edits before scientific writing acceptance. Research breadth, closest complete converter baseline and independent confirmation remain explicit limitations; they bar a successful-method/submission-ready claim, not an honest completed empirical draft. No new GPU experiments are justified by this manuscript repair.

Primary sources checked for graph-specific positioning: https://arxiv.org/abs/2403.08786 ; https://www.ijcai.org/proceedings/2022/0338.pdf . The latter is a context screen, not an additional uncited quantitative baseline or a novelty-absence proof.
