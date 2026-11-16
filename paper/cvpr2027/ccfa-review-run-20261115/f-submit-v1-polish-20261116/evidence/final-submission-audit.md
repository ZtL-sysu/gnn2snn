# Final Submission Audit (S9)

## Gate result

PASS with one scoped, non-blocking limitation. There is no unresolved major package issue.

Current audited artifacts:

- `working/main.tex` — SHA-256 `6b160fa447e2c1f453d9d7f91f57d9535671cb98e2b06a6f982cfbd451f5cf25`
- `working/main.pdf` — SHA-256 `23e1f84488c656832fc3d6839fcda07b122759f7311c1cf8b0a039049fe4d7bb`
- `working/supplement.tex` — SHA-256 `435aedb434ba8e5c61a2f49dbdd6004afed4d9064552aee84c28a02b0d1b925b`
- `working/supplement.pdf` — SHA-256 `81af234f5b6e1d17f2a14e62aaf7e2e0873ba19234d7111df647a535298768e4`
- `delivery/gnn2snn-cvpr2027-source.zip` — clean-extract compile verified

## A. Requirement authority and metadata

`submission-constraints.md` records the CVPR-style requirements applied to this run. `manuscript-metadata.md` records the canonical title, anonymous author block, venue/year, abstract, declaration state, and the absence of undisclosed author metadata. Source, PDF, and supplement metadata are consistent.

The run did not have access to a live CVPR 2027 submission portal or registration. The package is therefore verified against the stated CVPR-style constraints, not against a portal-specific form. This is recorded as `L-S9-1`, a non-blocking scope limitation.

## B. Page count, format, and anonymity

- Main PDF: 8 pages, letter size; references and the LLM Usage Statement are on the final page after the substantive body.
- Supplement PDF: 4 pages, separate source and separate PDF.
- Both sources use `\usepackage[review]{cvpr}` and `\switchlinenumbers*`.
- Main PDF shows `CVPR 2027 Submission. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.` and `Anonymous CVPR submission`.
- Supplement shows `Anonymous authors`.
- No university, institute, department, correspondence, email address, or @-address appears in either PDF author/title area. Name-like strings in the main PDF were checked and occur in bibliography entries, not in author metadata.
- No author-identifying material was added by this polish run.

## C. Scientific claim support and scope

The claim spine remains the two scoped information losses established in prior review:

1. A message-norm decrease can worsen the teacher-class margin under legal one-event changes. The main evidence remains 332/811 (40.94%) on MNIST and 296/928 (31.90%) on CIFAR10, with graph-cluster intervals and an explicit intervention-distribution boundary.
2. Exact local two-tick decoded-tensor equality and equal local event counts can coexist with changed downstream predictions and whole-network activity. The temporal result explicitly retains the first-prefix and amplitude confound.

The manuscript still separates teacher agreement from ground-truth accuracy, local event counts from whole-network activity, projected additions from runtime, and exposed historical results from blind confirmation. Centering remains a representation/proxy result without a complete efficiency claim. PD-Quant remains a quantization-parameter optimization method, and One-Spike remains a separate single-spike phase-coding converter. No claim was strengthened during the polish run.

## D. Structure, paragraph flow, and professionalism

S8 round 2 accepted all previously failed modules: Abstract, Introduction, Related Work, Results, Conclusion, and Figures/Tables. The second pass changed only Results topic sentences and captions. The reverse-outline and limitation-placement checks remain current: material boundaries are retained beside affected claims and consolidated in Discussion; no reviewer-facing, journal-facing, packaging, or audit-process narration was introduced into the scientific body. The four lint candidates are necessary scope or operational statements and were accepted with reasons in round 1 and round 2.

## E. Figures, tables, citations, and disclosure

- Main PDF contains six numbered figures and three numbered tables; all captions identify the plotted quantity and essential scope.
- Supplement contains four numbered tables and preserves the evidence/reproducibility, historical, cost, and failure-boundary sections.
- `main.log` reports zero undefined citations, zero undefined references, zero overfull boxes, zero underfull boxes, zero multiply-defined labels, and zero LaTeX errors.
- The generated `.bbl` contains 14 bibliography entries and no undefined citation remains.
- The LLM Usage Statement appears once in the main PDF before references. Extracted text confirms the named tools, Kimi/DeepSeek independent reviews, WorkBuddy/Claude/Grok earlier-stage assistance, the model-version limitation, the evidence-versus-language-model distinction, the figure-rendering disclosure, and the human-author responsibility sentence. Its source text is unchanged from the protected baseline.

## F. Package integrity and clean extraction

`delivery/gnn2snn-cvpr2027-source.zip` was created from the accepted `working/` tree. Clean extraction was verified:

- `main.tex`, `supplement.tex`, `references.bib`, `cvpr.sty`, and `ieeenat_fullname.bst` hashes match the working files.
- All figure PDFs, data files, and artifact files match their working counterparts.
- A clean `latexmk -pdf -halt-on-error main.tex` build produced 8 pages, with zero undefined citations, undefined references, overfull boxes, and LaTeX errors.
- A clean supplement build produced 4 pages.
- Normalized extracted text from the clean main PDF is byte-equivalent after whitespace normalization to the accepted working PDF.

## G. Declaration and upload readiness

Declaration mode is `preserve`. The existing LLM Usage Statement is retained on every submission-facing surface prepared in this run. No declaration was removed, rewritten, or routed to a standalone file. `upload-map.md` maps the final main PDF, supplement PDF, source ZIP, and accepted source directories to prepared portal designations; it is not an upload authorization.

## H. Final determination

The current package is internally consistent, compiles from a clean source archive, respects the CVPR-style page and anonymity constraints used in this run, preserves all protected scientific facts and disclosure text, and has no unresolved major issue. The only scoped limitation is `L-S9-1`: a live CVPR 2027 portal was unavailable, so portal-specific fields and registration-dependent claims are outside this audit.
