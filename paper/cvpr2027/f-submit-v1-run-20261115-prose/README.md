# Wording revision

This is the separate wording revision requested on 2026-11-15. The original paper and source package one directory above remain unchanged.

- `revised/main.pdf`: revised manuscript, seven body pages plus disclosure/references.
- `revised/supplement.pdf`: revised four-page supplement.
- `polished-source.zip`: self-contained revised LaTeX source and the six used figures.
- `revision-log.json`, `audit/*.diff`: passage-level before/after changes.
- `audit/prose-review.md`: bounded independent language/evidence review and accepted correction.
- `audit/verification.json`: protected-content, original-preservation, PDF and font checks.
- `writer-request.json`, `writer-response-final.json`: actual input adapter and separated native output records; building an input is not claimed as a model execution.

Applied the f-submit-v1 scientific-writing contract, narrative/prose criteria, explicit-context adapter and final output checks together with the WorkBuddy scientific-paper-writing argument and language guides. This request concerns wording, not a new journal submission-readiness evaluation. The full ten-comparator SCI percentile pipeline is not certified; its canonical ledger is retained pending. No comparator scores, new research results or submission status were invented.

The revision reduces rhetorical questions, repeated contribution announcements, negative-parallel disclaimers and report-style exposition. All equations, numbers, citations, tables, figures and the complete LLM Usage Statement are preserved. Scientific risks and evidence boundaries are unchanged. No experiments or conference submission were performed.

To compile, enter `revised/` and run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`, then the same command for `supplement.tex`. Parsing and verification scripts were run in the local Conda `dyg-cold` environment.

