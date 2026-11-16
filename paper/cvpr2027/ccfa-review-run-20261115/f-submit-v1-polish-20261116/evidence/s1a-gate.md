# S1A Gate: Prior Review Evidence

## Purpose

S1A records the prior independent review evidence and dispositions that constrain the current f-submit-v1 polish pass. The gate checks that earlier scientific and writing concerns are represented, that each concern has an explicit disposition, and that no new experiment is introduced.

## Archived evidence

- R0 independent scientific review: `prior-reviews/r0-independent-scientific-review.md`
- R0 independent writing review: `prior-reviews/r0-independent-writing-review.md`
- R1 independent scientific review: `prior-reviews/r1-independent-scientific-review.md`
- R1 independent writing review: `prior-reviews/r1-independent-writing-review.md`
- R2 independent writing review: `prior-reviews/r2-independent-writing-review.md`
- R2 revision log: `prior-reviews/r2-revision-log.md`
- Inherited R3 fast scientific recheck: `prior-reviews/r3-inherited-fast-scientific-recheck.md`
- R2 verification and artifact manifests: `prior-reviews/r2-verification.json`, `prior-reviews/r2-artifact-manifest.json`

The R3 fast scientific recheck was delivered as a conversation message rather than a standalone repository report. It is preserved with explicit provenance and is not represented as an independently regenerated review file.

## Disposition coverage

`claim-disposition.md` covers all five standalone review reports, the R2 package/figure corrections, and the inherited R3 outcome. It records support status, action, and an evidence link for each material concern. It also fixes the standing boundaries for the current pass.

## No-new-experiment check

The current execution contract permits structural, prose, layout, figure-placement, and metadata package updates only. The disposition document explicitly prohibits new experiments, new comparisons, changed eligibility rules, and broader scientific claims. All remaining open concerns are classified as external-validity or significance limits that prose cannot resolve.

## Closed package corrections

- The CIFAR10 amplitude-increasing temporal activity change is 1.33%, matching 6,876,233 → 6,785,108 in the bundled temporal data.
- Figures 4 and 5 were regenerated, synchronized into the source figure directory, and embedded in the rebuilt main PDF before this run.
- The supplement's artifact inventory distinguishes included raw/endpoint artifacts from results requiring checkpoints, datasets, or historical records.

## Verdict

Pass. Prior evidence is archived inside the run root, every material concern is dispositioned, and the current pass is explicitly bounded to presentation work.
