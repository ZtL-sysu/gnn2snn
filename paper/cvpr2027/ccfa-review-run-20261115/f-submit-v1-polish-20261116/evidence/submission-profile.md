# Submission Profile (S2)

## Target

- Venue mode: CVPR-style top-conference submission, as explicitly requested by the user for this polish run.
- Conference label in source: `CVPR`.
- Year label in source: `2027`.
- Paper category: main conference paper.
- Presentation: double-blind review mode.

## Source and package facts

- Main source: `sources/main.tex`.
- Supplement source: `sources/supplement.tex`.
- Style: local `sources/cvpr.sty`, loaded with `\usepackage[review]{cvpr}`.
- Bibliography style: `ieeenat_fullname.bst`; bibliography database: `references.bib`.
- Document class options: `10pt,twocolumn,letterpaper`.
- Author block: `Anonymous authors`.
- Paper identifier: `-----`.
- Line numbers: enabled by `\switchlinenumbers*` in the review build.

## Page and content rules applied to this run

- The main paper is prepared against the CVPR eight-page body limit.
- References and the required LLM Usage Statement are outside the substantive eight-page content allowance; the current source places the statement on a separate `\clearpage` before the bibliography.
- Supplemental material is a separate PDF and source file. It is not counted as part of the main eight pages.
- Review mode requires no author names, affiliations, acknowledgements, or identifying metadata.
- Figures and tables must remain readable at final two-column size; captions must be self-contained enough to state setting and scope.
- Final deliverables must include both `main.pdf` and `supplement.pdf`, compiled from the final working sources.

## Current manuscript inventory

- Main PDF baseline: `sources/main.pdf`.
- Supplement PDF baseline: `sources/supplement.pdf`.
- Main text sections: Introduction; Related Work; Analysis and Experimental Design; Results; Discussion and Conclusion; LLM Usage Statement.
- Main figures: six.
- Main tables: three.
- Supplement sections: Evidence and Reproducibility; Graph Models and Calibration; Directional and Temporal Derivations; Classical Node Classification Context; Residual Centering and Complete Costs; Failure Boundaries and Research Scope.
- Supplement tables: four.

## Declaration and metadata policy

- Declaration mode from the execution contract: `preserve`.
- The existing LLM Usage Statement must be carried into the final working source verbatim.
- Anonymous metadata must remain unchanged.
- Venue/year text is already present in the source and must remain consistent across main and supplement.
- No author-identifying material may be added during prose or layout edits.

## Local verification envelope

- TeX toolchain: `latexmk` / `pdflatex` with the local `cvpr.sty` in the same directory as the compiled source.
- Python environment for hashes and evidence checks: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe`.
- Compilation must happen inside the run root's `working/` directory so repository-root files are not polluted by auxiliary output.
- Final checks: page count, undefined references/citations, overfull boxes, figure rendering, anonymity, bibliography, LLM statement, and source/PDF hash manifest.

## Venue-mode note

The f-submit-v1 skill's default journal comparator policy is intentionally over-ridden here by the user's explicit instruction to learn from recent top-conference writing and narrative practice. Comparator work therefore targets CVPR/ICCV/ECCV/NeurIPS/ICML. This changes stylistic references only; it does not change the scientific claims, venue metadata already present in the manuscript, or the no-new-experiment rule.
