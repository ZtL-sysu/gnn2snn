# Citation Verification Note

## Scope and threshold

The user explicitly required more than 20 citations and a manuscript of at most nine pages including references. The final manuscript cites 24 distinct sources and compiles to nine pages. The local policy therefore uses `min_unique = 21`.

The installed `f-research-v1/scripts/citation_requirements.py` hard-codes `min_unique >= 36` and rejects any lower policy before reading the project files. Its PASS status cannot be claimed for this manuscript without contradicting the explicit user threshold. The checks below follow the skill's substantive requirements: authenticity, relevance, concrete claim support, recency, identifier checks, duplicate detection, and frozen inputs.

## Verification summary

- Distinct cited sources: 24; every key appears in both `references.bib` and the citation inventory.
- Inventory status: all 24 records marked `verified` and `relevant`; every record has a concrete section-level support location derived from `main.tex`.
- Recency: 17 of 24 sources are from 2022--2026, satisfying the strict recent-majority rule for the frozen five-year window. Older sources are retained for foundational ANN--SNN conversion work and are genuinely required by the claims they support.
- Source verification: Crossref, OpenAlex, arXiv API, PMLR, JMLR, AAAI, CVF, IEEE, Springer, and NeurIPS records were consulted on 2026-11-17. Official venue pages were preferred over mirrors.
- Relevance: each cited source is attached to a specific claim about conversion fidelity, decision-aware quantization, graph quantization, temporal coding, graph classifiers, or inference cost. No citation was added solely to increase a count.
- Deduplication: see `reference_dedup_audit.md`. No duplicate DOI, arXiv id, normalized title, or conference/preprint pair remains.

## Corrections made during verification

The first inventory contained several machine-extracted identifiers that did not match the cited papers. These were corrected before final compilation:

1. `li2021` had DOI `10.1145/3446776`, which Crossref resolves to an unrelated CACM paper. The DOI was removed; arXiv `2106.06984` and the official PMLR page were recorded.
2. `liu2023pdquant` had an off-by-one CVPR DOI. It was corrected to `10.1109/CVPR52729.2023.02340`.
3. `degreequant2021` and `a2q2023` had DOI-shaped arXiv identifiers that Crossref does not resolve. They were replaced by arXiv ids `2008.05000` and `2302.00193`.
4. `han2022vig` received the Crossref-verified DOI `10.52202/068431-0603`.
5. `bojkovic2025` received the arXiv id `2502.14487` corresponding to the PMLR paper.
6. `jiang2025fission` was reconciled with the NeurIPS 2025 official page and Crossref; the authoritative DOI `10.52202/085713-5157` was added to the bibliography and inventory.

## Frozen audit inputs

`citation_gate_report.json` records the SHA-256 hashes of `main.tex`, `references.bib`, `citation_inventory.json`, and `citation_policy.json`. Recompute those hashes after any future change to the manuscript or bibliography.
