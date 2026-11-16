# CCFA independent review and optimization run

This run freezes the manuscript and evidence map used for the review.

- Frozen input: `frozen/`
- Hash manifest: `input-manifest.json`
- Review reports: `review-round-*/`
- Revision records: `revisions/`
- Final audit: `final-audit/`

The manuscript must be reviewed from the frozen inputs. Any revision creates a new run subdirectory and never overwrites an earlier review report.
