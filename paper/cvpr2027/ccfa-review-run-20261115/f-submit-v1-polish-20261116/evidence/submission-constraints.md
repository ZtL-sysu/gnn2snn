# Submission Constraints (S9)

| Field or file | Requirement applied in this run | Hard / recommended | Authority and source | Final value / count | Status |
|---|---|---|---|---|---|
| Article type | CVPR main-conference paper | Hard for this prepared package | `submission-profile.md`, user-requested CVPR-style target | Main conference | Set |
| Venue metadata | CVPR 2027 | Hard in current source | `working/main.tex`, `working/supplement.tex` | `CVPR`, `2027` | Verified |
| Template | CVPR two-column proceedings style | Hard for format consistency | Local `cvpr.sty`, document class `10pt,twocolumn,letterpaper` | Compiles with local TeX Live 2026 | Verified |
| Review mode | Anonymous double-blind review | Hard | `\usepackage[review]{cvpr}`; `Anonymous authors`; PDF watermark | Anonymous author block only | Verified |
| Main-body length | Eight pages | Hard in this run | `submission-profile.md`, CVPR 8-page body limit | `pdfinfo`: 8 pages | Verified |
| References and LLM statement | Outside substantive eight-page allowance | Run convention | `submission-profile.md`; source places statement on `\clearpage` before bibliography | 1 statement; 14 bibliography entries | Verified |
| Line numbers | Review build | Required in review mode | `\switchlinenumbers*` in both sources | Present in PDFs | Verified |
| Supplement | Separate PDF, not counted in main pages | Hard in this run | `submission-profile.md` | `supplement.pdf`: 4 pages | Verified |
| Figures and tables | Readable at two-column size; captions state setting and scope | Recommended | `submission-profile.md`; S8 caption rescore | 6 main figures; 3 main tables; 4 supplement tables | Pass |
| Declarations | Preserve existing declaration; no author-managed changes | Hard from run contract | `execution-contract-v1.md`: `declaration_mode: preserve` | LLM Usage Statement unchanged | Verified |
| Author metadata | No real names, affiliations, emails, or acknowledgements in review files | Hard | Review-mode profile and PDF inspection | `Anonymous authors` / `Anonymous CVPR submission` | Verified |
| Source package | Minimal compilable source plus figures/data/artifacts | Hard for reproducibility | `delivery/gnn2snn-cvpr2027-source.zip` | Clean extract compiles to 8 + 4 pages | Verified |

## Scope note

This is a CVPR-style package prepared from the existing manuscript and source tree. No live CVPR 2027 submission portal, registration, or special-issue form was available to this run; no assumption about portal eligibility or successful registration is made. The audit checks the current local package against the stated CVPR-style requirements.
