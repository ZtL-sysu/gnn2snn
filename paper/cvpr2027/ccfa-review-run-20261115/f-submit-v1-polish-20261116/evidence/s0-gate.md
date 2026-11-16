# S0 gate: manuscript load and immutable baseline

## Gate result

PASS.

## Evidence

- The submitted manuscript and companion source package are present under `sources/`.
- The immutable baseline includes `main.tex`, `supplement.tex`, `references.bib`, `cvpr.sty`, `ieeenat_fullname.bst`, both compiled PDFs, figures, data, and analysis artifacts.
- The run root is located under the directory containing the manuscript source package.
- The original upstream revision directory was not modified. All revision work will occur in this run root.
- The execution contract is versioned as `execution-contract-v1.md` and hashed in `pipeline-state.json`.

## Scope statement

This gate establishes file preservation and run-root isolation only. It makes no claim about scientific or visual correctness.
