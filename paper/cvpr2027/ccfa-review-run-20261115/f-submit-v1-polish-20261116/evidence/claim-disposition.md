# Prior-Review Claim Disposition

## Scope

This disposition covers the five archived independent review reports, the R2 revision log, the R3 inherited fast scientific recheck, and the current R2 source snapshot under `sources/`. The current f-submit-v1 pass is structural, prose, layout, and package polish. It introduces no new experiment, no new numerical result, no changed eligibility rule, and no broader scientific claim.

## Round 0 reports

| ID | Concern | Support status | Action in R1–R2 and current pass | Evidence link |
| --- | --- | --- | --- | --- |
| R0-S1 | Novelty framing was broader than the defensible graph-specific delta; close temporal-misalignment, uneven-arrival, error-propagation, and prediction-aware precedents were under-positioned. | Supported at R0; substantially closed by R2 prose. | The manuscript is explicitly an audit of frozen DCGS-derived graph adapters. PD-Quant, One-Spike, temporal misalignment, and related alignment work are positioned by what they optimize or control. Current pass preserves this boundary. | `prior-reviews/r0-independent-scientific-review.md`, `prior-reviews/r0-independent-writing-review.md`; current `sources/main.tex` Related Work and Discussion. |
| R0-S2 | Pooled adverse rates lacked rule/direction strata and a clear intervention-distribution interpretation. | Supported at R0; closed by R1 stratified analysis. | Rule/direction strata and graph-level counts are retained. Numbers remain descriptive of the tested intervention distribution and one teacher. | `prior-reviews/r0-independent-scientific-review.md`; `sources/data/hard_intervention_stratified.csv`, `sources/data/hard_intervention_per_graph.csv`. |
| R0-S3 | Exposure and teacher scope prevent confirmatory generalization. | Supported and still materially open. | Kept explicit in Abstract, Methods, Results, Discussion, and supplement. No fresh-data or multi-teacher claim was added. | `prior-reviews/r0-independent-scientific-review.md`; current `sources/main.tex` and `sources/supplement.tex`. |
| R0-S4 | Secondary studies fragmented the two central counterexamples. | Supported at R0; partially closed by R1–R2 compression. | Current pass keeps the two counterexamples as the reading spine and treats position, calibration, centering, historical, and cost evidence as motivation, controls, or boundaries. | `prior-reviews/r0-independent-scientific-review.md`; current `sources/main.tex`. |
| R0-S5 | Deterministic selection and artifact path were insufficiently specified. | Supported at R0; substantially closed by R2 artifact inventory. | Exact eligibility rules, source scripts, raw rows, endpoint arrays, and artifact hashes are retained; replay prerequisites are stated. Current pass preserves the package. | `prior-reviews/r0-independent-scientific-review.md`; `sources/artifact/**`, `sources/supplement.tex`, `prior-reviews/r2-artifact-manifest.json`. |
| R0-W1 | Temporal decision evidence was under-rendered relative to the abstract claim. | Supported at R0; closed by R1 figure and retained CSV. | Figure 5 reports paired decision changes and agreement transitions. Current pass may improve caption clarity only; no numerical change. | `prior-reviews/r0-independent-writing-review.md`; `sources/data/temporal_decision_effects.csv`, `sources/figures/temporal_decision_effects.pdf`. |
| R0-W2 | Terminology drift across ticks, prefixes, activity, and projected additions. | Supported at R0; mostly closed by R2. | Current pass canonicalizes existing terminology where possible without changing definitions. | `prior-reviews/r0-independent-writing-review.md`; current source terminology. |

## Round 1 reports

| ID | Concern | Support status | Action in R2 and current pass | Evidence link |
| --- | --- | --- | --- | --- |
| R1-S1 | Core intervention evidence remains one-teacher, exposed-data, and narrow. | Supported; unavoidable under the no-new-experiment constraint. | Scope is preserved verbatim in substance. Current pass must not strengthen frequency, generality, or converter-ranking language. | `prior-reviews/r1-independent-scientific-review.md`. |
| R1-S2 | Artifact inventory exceeded the supplied package and the verification filename was inconsistent. | Supported at R1; closed in R2. | R2 package contains the promised small raw and endpoint artifacts; supplement distinguishes replay requirements. Current pass preserves truthful inventory. | `prior-reviews/r1-independent-scientific-review.md`; `prior-reviews/r2-revision-log.md`, `prior-reviews/r2-artifact-manifest.json`, `prior-reviews/r2-verification.json`. |
| R1-S3 | Figures 4–5 had small labels and a pooled-count overlay. | Supported at R1; closed after R3 synchronization. | Regenerated figure files are in `sources/figures/` and the rebuilt main PDF embeds them. | `prior-reviews/r1-independent-scientific-review.md`; `prior-reviews/r3-inherited-fast-scientific-recheck.md`; `sources/figures/**`. |
| R1-W1 | PD-Quant and One-Spike deltas were not explicit; stale `Pred-Quant` spelling remained. | Supported at R1; closed in R2. | Correct spelling and explicit optimizer/converter versus audit distinctions are retained. | `prior-reviews/r1-independent-writing-review.md`; current `sources/main.tex`. |
| R1-W2 | Temporal exchange supports terminal-equality insufficiency but does not isolate timing from absolute amplitude. | Supported; closed by claim narrowing. | The manuscript names controlled variables and explicitly leaves first-prefix profile and absolute amplitude uncontrolled. Current pass preserves this caveat. | `prior-reviews/r1-independent-writing-review.md`; current `sources/main.tex` §§3.3, 4.4, and Discussion. |
| R1-W3 | Whole-network activity needed a first-use definition. | Supported at R1; closed in R2. | Defined as total nonzero scalar events and used consistently. | `prior-reviews/r1-independent-writing-review.md`; current `sources/main.tex` §3.1. |
| R1-W4 | Calibration conclusions could overstate small differences. | Supported at R1; closed by descriptive wording. | Current wording is limited to no consistent advantage among tested objectives. | `prior-reviews/r1-independent-writing-review.md`; current `sources/main.tex`. |

## Round 2 and inherited R3 reports

| ID | Concern | Support status | Action in current pass | Evidence link |
| --- | --- | --- | --- | --- |
| R2-1 | Pure prose had reached diminishing returns. | Accepted as a prior assessment. | Current pass is a bounded presentation refinement under the user's explicit request; it does not add scientific claims or experiments. | `prior-reviews/r2-independent-writing-review.md`. |
| R2-2 | Figure synchronization defect in the first R2 compilation. | Closed before this run. | R3 recheck and R2 verification record document synchronized figures and rebuilt PDF. Current pass rechecks figure placement before release. | `prior-reviews/r2-independent-writing-review.md`, `prior-reviews/r2-revision-log.md`, `prior-reviews/r3-inherited-fast-scientific-recheck.md`. |
| R2-3 | CIFAR10 doubling activity percentage mismatch, 1.31% versus 1.33%. | Closed before this run. | `sources/main.tex` and the R2 verification record use 1.33%, with counts 6,876,233 → 6,785,108. This number must remain unchanged. | `prior-reviews/r3-inherited-fast-scientific-recheck.md`; `sources/main.tex`; `prior-reviews/r2-verification.json`. |
| R2-4 | Seven-line conclusion spill onto page 8 is a layout polish issue. | Supported; optional. | Current pass may tighten prose or figure placement if it improves layout, with no scientific compression that changes the claim chain. | `prior-reviews/r2-independent-writing-review.md`. |
| R3-1 | No new scientific must-fix. | Accepted. | Current pass preserves scientific content and scopes all prose changes to presentation. | `prior-reviews/r3-inherited-fast-scientific-recheck.md`. |

## Standing non-negotiable boundaries

- No new experiment, retraining, or new comparison is authorized in this pass.
- Hard-event rates remain descriptive within the tested rule/direction distribution and teacher.
- Temporal exchange remains an exact local endpoint/count counterexample; timing and amplitude are not separated.
- Historical node-classification and exposed-split results remain contextual rather than blind confirmation.
- Centered residual evidence supports a representation/proxy improvement, not a complete efficiency or energy claim.
- PD-Quant is a quantization parameter optimization method; One-Spike is a single-spike phase-coding converter. Neither is implemented or matched here.
- `whole-network activity` means the total number of nonzero scalar events.

## Overall disposition

All high-impact wording defects from R0 and R1 were closed by R1/R2 before this run. The remaining open concerns are external validity, evidence breadth, and contribution significance. They cannot be resolved by prose and are explicitly out of scope for this pass. The current run is therefore limited to presentation refinement, package verification, and faithful preservation of the established claim-evidence contract.
