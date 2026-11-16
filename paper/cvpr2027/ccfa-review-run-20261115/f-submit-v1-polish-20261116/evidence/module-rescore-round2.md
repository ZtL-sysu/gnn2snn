# S8 Module Rescore: Round 2

## Review basis

This is an independent same-agent-role rescore of the second bounded revision pass. It compares the immutable baseline `sources/main.tex` with the current `working/main.tex`, using the frozen comparator thresholds in `module-quality-review.md`, the round-1 dispositions in `module-rescore-round1.md`, the section patterns in `comparator-section-patterns.md`, and the standing boundaries in `claim-disposition.md`. Scores are editorial presentation scores, not calibrated probabilities or new scientific judgments.

## Scope of the second pass

The second pass was limited to Results topic sentences and table/figure captions. Abstract, Introduction, Related Work, Discussion, Conclusion, equations, labels, citation keys, figures, data, and the LLM Usage Statement were checked for unintended changes.

## Protection and integrity checks

- Baseline hash remains `efa5dba3efa114f48c9f42bab5a870dbede6f37c7cf526818a8dfd99d642738a`; current working hash is `6b160fa447e2c1f453d9d7f91f57d9535671cb98e2b06a6f982cfbd451f5cf25`.
- The source-versus-working counts of all checked numeric tokens are identical, including `332/811`, `296/928`, `40.94`, `31.90`, `37.28,44.57`, `28.66,35.31`, `48.71`, `44.07`, `6/128`, `16/128`, `2/128`, `10/128`, `3.41`, `4.99`, `1.33`, `54.1`, `23.5`, `8.8`, `14.9`, `84,905`, `132,899`, `55,510`, `63,500`, and all timing values.
- The label set and citation-key set are unchanged (16 labels and 14 citation keys).
- The LLM Usage Statement section and all following bytes are byte-identical to the baseline.
- `main.log` reports 8 pages, 0 undefined citations, 0 undefined references, 0 overfull boxes, and 0 LaTeX errors.
- `writing/lint-round2.json` returns `semantic-review-required; candidates=4` with no integrity errors. The same four candidates were accepted in round 1 as necessary boundary or operational statements.
- Figure files were not edited.

## Rescored modules

| Module | S6 | S8 round 1 | S8 round 2 | Threshold | Status | Named-defect disposition |
|---|---:|---:|---:|---:|---|---|
| Results | 89 | 91 | 93 | 93 | PASS | Resolved. Figure-led openings were replaced by finding-led topic sentences: one-tick recovery is localized to the first node site, selection rule controls the adverse-rate magnitude, graph-operator controls do not yield a unique aggregate signal, and dense event kernels explain why low counts do not imply low simulation time. Tables retain dense cell values; all rates, intervals, split labels, and scope clauses remain. |
| Figures / Tables | 89 | 90 | 92 | 91 | PASS | Resolved. Captions now begin with the plotted finding and move essential scope into compact trailing clauses. The defensive Table 1 wording was removed; exposed tests, graph-cluster intervals, repeated-intervention independence, local endpoint equality, amplitude confound, and training-only status remain where needed to interpret each panel. |

## Confirmation of previously passing modules

- Abstract: unchanged from accepted round-1 text; remains PASS (92 vs. 91).
- Introduction: unchanged from accepted round-1 text; remains PASS (92 vs. 91).
- Related Work: unchanged from accepted round-1 text; remains PASS (90 vs. 89).
- Conclusion: unchanged from accepted round-1 text; remains PASS (92 vs. 90).

No previously passing module was weakened, and no new conclusion-only claim was introduced.

## Decision

Both modules that failed the first S8 gate now meet their frozen comparator thresholds. The second pass stayed within its authorized scope and introduced no evidence or claim changes. The manuscript is accepted at S8 for progression to the final submission-package audit.

## Scope statement

This rescore does not introduce new experiments, citations, or claims. It accepts the retained boundaries that hard-intervention rates are conditional, temporal exchanges do not isolate timing from amplitude, exposed splits are descriptive, whole-network activity counts nonzero scalar events, and the study audits a frozen graph converter rather than introducing a new one.
