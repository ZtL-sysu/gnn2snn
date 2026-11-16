# Inherited R3 Fast Scientific Recheck Record

## Provenance

This record preserves the outcome of the R3 fast scientific recheck delivered in the inherited task context. The recheck was a conversation message rather than a standalone file in the CCFA run directory. It is recorded here as an inherited-context summary so that the current f-submit-v1 pipeline has an auditable disposition; it is not presented as an independently regenerated review report.

## Recheck outcome

The R3 fast scientific recheck reported no new scientific must-fix items for the R2 manuscript.

It identified two non-scientific/package issues:

1. The CIFAR10 amplitude-increasing temporal exchange was reported as 1.31% in an earlier PDF layer while the bundled temporal CSV and revised source give 1.33% (6,876,233 to 6,785,108).
2. The regenerated Figures 4 and 5 had not been synchronized into the compiled main PDF at the time of the recheck.

## Disposition in the current source set

Both issues are closed in `sources/`:

- `sources/main.tex` contains the corrected 1.33% statement.
- `sources/main.pdf` is the R2 rebuilt PDF whose revision log records synchronized Figures 4 and 5.
- `sources/figures/hard_intervention_stratified.pdf` and `sources/figures/temporal_decision_effects.pdf` are the regenerated figure files copied into the figure directory used by the compiled source.
- The copied R2 revision log and verification record in this directory document the resynchronization and package-wide hash refresh.

No experimental value, eligibility rule, claim scope, or citation was changed by that recheck.
