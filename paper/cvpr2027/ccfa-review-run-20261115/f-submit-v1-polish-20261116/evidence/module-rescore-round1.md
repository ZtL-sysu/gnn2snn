# S8 Module Rescore: Round 1

## Review basis

This is an independent same-agent-role re-evaluation of the six modules that failed S6. It compares `sources/main.tex` with `working/main.tex` against the frozen comparator thresholds in `module-quality-review.md`, the section patterns in `comparator-section-patterns.md`, and the standing claim boundaries in `claim-disposition.md`. Scores are editorial presentation scores, not calibrated probabilities and not new scientific judgments.

## Protection and integrity checks

- The source baseline hash remains `efa5dba3efa114f48c9f42bab5a870dbede6f37c7cf526818a8dfd99d642738a`.
- Citation keys are unchanged as a set; no bibliography entry, equation, label, data file, or figure file was edited.
- The LLM Usage Statement is byte-identical from its section marker onward.
- Checked numeric tokens have identical source/working counts: `332/811`, `296/928`, `40.9`, `31.9`, `37.28,44.57`, `28.66,35.31`, `48.71`, `44.07`, `6/128`, `16/128`, `2/128`, `10/128`, `3.41`, `4.99`, `1.33`, `54.1`, `23.5`, `8.8`, `14.9`, `84,905`, `132,899`, `55,510`, `63,500`.
- `latexmk` completed with 8 main pages; `main.log` has 0 undefined citations, 0 undefined references, 0 overfull boxes, and 0 LaTeX errors.
- `manuscript_lint.py` returned `semantic-review-required; candidates=4` with no integrity errors. The four candidates are necessary boundary or operational statements, not unresolved prose defects: the One-Spike non-identity, the continuous diagnostic definition, the whole-network activity mismatch, and the first-prefix/amplitude limitation.

## Rescored modules

| Module | S6 score | S8 score | Threshold | Status | Named-defect disposition |
|---|---:|---:|---:|---|---|
| Abstract | 89 | 92 | 91 | PASS | Resolved. The finding now moves cleanly from proxy gap to first result, second invariant, mechanism, and reporting consequence. Conditional scope remains beside the first rate, and the temporal amplitude/prefix boundary is explicit. |
| Introduction | 88 | 92 | 91 | PASS | Resolved. The opening starts with the finite-event failure mode; conversion background is compressed; the duplicated contribution/checklist announcement is merged into one consequence. Numeric previews and split/teacher scope remain. |
| Related Work | 88 | 90 | 89 | PASS | Resolved. Calibration/readout, event coding/timing, and decision-aware objectives now have capability-led topic sentences. PD-Quant and One-Spike distinctions are preserved in substance, and the graph-specific question remains narrow. |
| Results | 89 | 91 | 93 | FAIL | Partial. Four subsection openings now lead with findings, and the temporal paragraph starts with the central invariant. However, not every results subsection was reorganized; several table-dependent openings and dense numeric inventories remain. All numbers and qualifications are intact. |
| Conclusion | 88 | 92 | 90 | PASS | Resolved. The conclusion no longer replays the Results section or uses defensive negation. It calls back decision, prefix, and activity as the quantities made visible by the audit and keeps the one-teacher/exposed-data boundary. |
| Figures / Tables | 89 | 90 | 91 | FAIL | Partial. Table 1 and Fig. 5 captions were tightened where possible, but the table captions still retain multiple scope clauses because those conditions are needed for interpretation. The defensive phrase in Table 1 was removed. No figure or table content changed. |

## Decision

Four of six previously failed modules now pass. Two remain below their comparator thresholds. The remaining defects are presentation compression issues, not evidence blockers, and no scientific claim needs to be strengthened or withdrawn. Because the edit-review contract allows up to three S7/S8 passes, the next action is a narrow second pass on Results topic sentences and table captions only. Already passing modules should remain unchanged.

## Scope statement

This rescore does not introduce new experiments, citations, or claims. It does not convert the mechanism audit into a method paper. It accepts the retained boundaries that hard-intervention rates are conditional, temporal exchanges do not isolate timing from amplitude, exposed splits are descriptive, and whole-network activity counts nonzero scalar events.
