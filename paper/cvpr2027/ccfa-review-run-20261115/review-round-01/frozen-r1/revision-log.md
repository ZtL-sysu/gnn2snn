# CCFA R1 revision

This revision is based on the two independent CCFA reports in `../../review-round-01/`.

## Accepted actions

- Reframed the abstract, introduction, title, and conclusion around a finite-event audit with two scoped counterexamples.
- Added verified positioning against PD-Quant, residual-membrane/uneven-arrival analysis, error-aware conversion, temporal misalignment, One-Spike, low-latency CKA alignment, and inference-scale complexity.
- Added a rule/direction stratification of the hard interventions and graph-level prevalence. The pooled adverse rates remain unchanged and are explicitly described as dependent on the intervention distribution.
- Added paired per-graph decision changes, teacher-agreement transitions, exact eligibility counts, and global event changes for the temporal codeword exchange.
- Replaced the unqualified calibration conclusion with "no consistent advantage among the tested objectives."
- Specified the exact deterministic coordinate, residual, amplitude-direction, and temporal eligibility rules in the supplement.
- Added an artifact/reproduction matrix and separated table reconstruction from checkpoint-dependent inference replication.
- Clarified quadrant interpretation in the message-versus-margin figure and scoped all conclusion wording to the tested metrics.

## Deferred large experiments

- Repeating both interventions across all three teachers on untouched graphs: requires new inference on frozen checkpoints and a fresh split. The existing endpoints are exposed and cannot be converted into blind confirmation.
- A matched One-Spike graph adaptation: requires implementing and validating a different published converter under a common protocol.
- A genuinely independent graph task or feature-dependent GatedGCN perturbation study: requires new training or a substantial new inference pipeline.
- Equal-amplitude temporal codeword families and hardware-energy measurement: outside the available artifacts and not needed for the scoped existence claims.

## Verification

`audit/verification-r1.json` records all package hashes and automated checks. Main: 8 pages, 7 body pages plus statement/references. Supplement: 4 pages. No unresolved citations/references, overflow, author-identifying links, or missing embedded fonts. Raw rows reproduce the headline counts and the new strata/effect tables.
