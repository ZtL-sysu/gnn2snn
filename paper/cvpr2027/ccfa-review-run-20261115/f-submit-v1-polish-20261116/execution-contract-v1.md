# Execution Contract v1

## Goal and authorization
- User goal: use f-submit-v1 and scientific-paper-writing to improve layout and wording, and learn writing/narrative techniques from recent top-conference papers.
- Authorized scope: structural, prose-only, layout/figure placement, and metadata/package updates in an independent run root. No new experiments.
- Venue mode deviation: target is CVPR-style top-conference paper, so comparator selection uses CVPR/ICCV/ECCV/NeurIPS/ICML papers instead of f-submit-v1 default SCI journals. Recorded here under user-explicit instruction.
- Edit-review budget: 3 S7/S8 passes. Stop after two consecutive passes without fewer named defects.
- New experiments: false.
- Declaration mode: preserve.
- Protected artifacts: all source data, CSV/NPZ registers, source scripts, figures, bibliography facts, numerical results, equations, citations, and the LLM usage statement. Scientific claims may be clarified but not strengthened beyond evidence.
- Original manuscript: immutable baseline at `sources/` in this run root and original revision directory.

## Environment
- Python: C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe
- TeX: latexmk/pdflatex with local cvpr.sty
- Network: web search for comparator and narrative writing patterns
- Actual executor/model: current primary Codex agent; exact model version not exposed to this run.

## Quality requirements
- Preserve all scientific numbers, conditions, citations, figure/table references, and claim scope.
- Improve claim spine, paragraph function, narrative pacing, academic register, and two-column presentation.
- Compile final main and supplement PDFs; verify page limits, anonymity, references, figures, and package hashes.
