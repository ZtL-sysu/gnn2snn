# Beyond Reconstruction

English mechanism-study manuscript prepared in the official CVPR 2027 review template. Main text: seven pages; LLM disclosure and references begin on a separate page. Separate supplementary material contains detailed protocols, source-linked controls, temporal derivations, classical node results and cost boundaries. This is a completed manuscript package, not a claim of acceptance, registered submission, or new successful conversion algorithm.

- `main.pdf`, `main.tex`: manuscript.
- `supplement.pdf`, `supplement.tex`: supplementary material.
- `figures/`: vector PDF figures and PNG previews; main uses concept, temporal_mechanism, position_recovery, message_decision, temporal_equivalence and centered_tradeoff.
- `data/`: exact derived numerical tables; `early/`: historical six-dataset sources, table and audit trail.
- `source_sha_manifest.json`: primary figure sources and hashes. `CLAIMS.md` maps manuscript claims to repository evidence.
- `中文说明.md`: story selection, actual contributions and substantive outstanding concerns.
- `audit/`: compile logs, verification, policy and scientific review notes.
- `paper_source.zip`: self-contained LaTeX compilation package. `evidence_inputs.zip`: retained inputs for the data figures (extract at repository root).

Build in this directory with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`, then the same command for `supplement.tex`. Standard TeX Live packages are required. The official `cvpr.sty` and `ieeenat_fullname.bst` are included unchanged. Their original template archive is retained under `audit/cvpr2027-template.zip`.

Rebuild data/figures from repository root using the local Conda environment:

```
conda run -n dyg-cold python paper/cvpr2027/prepare_paper.py
conda run -n dyg-cold python paper/cvpr2027/refine_assets.py
```

`finalize.py` renders the temporal-mechanism figure and makes idempotent final manuscript adjustments; `enrich_main.py` was a one-time authoring transformation and should not be rerun on the finished manuscript. Historical raw results are never overwritten. Data-figure rebuilding requires the retained repository evidence inputs; LaTeX compilation requires only the contents of the source ZIP.

No conference registration confirmation was located. Author identities are intentionally absent. Before any actual submission, the human authors must confirm an existing timely registration, enrollment and reciprocal-reviewer compliance, the final author list, disclosure completeness and approval. No OpenReview submission was performed. The public GitHub archive is not an anonymous supplemental attachment; use the anonymous PDFs and inspect any code/data attachment independently.
