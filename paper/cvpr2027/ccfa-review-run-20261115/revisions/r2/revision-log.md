# CCFA R2 revision

This revision incorporates the two independent frozen-R1 reviews in `../../review-round-02/`. It preserves all R1 numerical results and the R1 frozen package.

## Accepted actions

- Corrected the close-work name to PD-Quant and stated the exact delta: PD-Quant optimizes quantization parameters from prediction differences, whereas this paper audits invariants of a frozen graph event computation.
- Stated explicitly that One-Spike SNN is a single-spike phase-coding converter. The one-event modifications here are diagnostic interventions in a DCGS-derived adapter, not a One-Spike implementation or matched comparison.
- Narrowed every temporal claim to what the codeword exchange controls: source legality, the task graph, exact local terminal decoding, and local event count. The first-prefix profile and absolute amplitude are explicitly uncontrolled.
- Defined whole-network activity as the total number of nonzero scalar events and used the term consistently.
- Replaced the supplement's overbroad audit-package claim with an exact inventory. The anonymous package now contains the hard-intervention rows, temporal endpoint arrays, source scripts, derived tables, figures, manifests, and verification record. Results requiring checkpoints or historical aggregates are explicitly marked as outside this package.
- Corrected the LLM Usage Statement to name Kimi K3.1 Pro and DeepSeek V4.1 Flash as the actual fresh-context reviewers and removed the inaccurate GPT-6-Luna claim.
- Regenerated Figures 4 and 5 with larger labels, removed the pooled-count overlay from Figure 4, and moved Figure 5's legend outside the bars.
- Compressed the conclusion while retaining the evidence boundaries and evaluation checklist.

## Deferred large experiments

- Multi-teacher and untouched-data replication, a matched One-Spike graph adaptation, a new task or topology, feature-dependent gate perturbations, equal-amplitude temporal families, and hardware-energy measurement. Both R2 reviewers judged further science to be outside the no-large-experiment constraint and at diminishing returns.

## Verification

`audit/verification.json` records all package hashes and automated checks. Main: 9 PDF pages, with 7 body pages, a short continuation page, the LLM statement, and references. Supplement: 4 pages. No unresolved citations/references, overflow, author-identifying links, or missing embedded fonts. All headline counts, strata, temporal endpoint invariants, positioning strings, LLM disclosure strings, and artifact hashes pass.
