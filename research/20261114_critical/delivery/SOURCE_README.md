# Accepted empirical paper source

This bundle contains the accepted twelve-page anonymous research draft, TeX, bibliography, four tables and four figure assets. It reports a bounded negative empirical study; it does not establish a successful new graph-aware converter or method-paper submission readiness.

From this bundle's root, with TeX Live available on PATH:

```text
conda run -n dyg-cold latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

The existing project Conda environment is used for local Python work. TeX Live supplies the document classes/fonts/packages. `paper/main.tex` uses only relative asset paths. `paper/main.pdf` is the accepted compiled copy; the bibliography contains eleven distinct cited works. No author identities, venue constraints or declarations were invented.

`paper/render_tex.py` regenerates the TeX from the accepted Markdown and captions. Table/figure regeneration scripts require the full research repository's raw result CSV/JSON inputs; this compact paper bundle already includes their final source-bound outputs. Full executors, source snapshots, seeds, subset identities, raw arrays, controls, failures, statistical scripts and evidence audits are in the [research branch](https://github.com/ZtL-sysu/gnn2snn/tree/research/20261114-critical-site/research/20261114_critical). Datasets and checkpoints are external dependencies with recorded hashes and are not bundled as unnecessary large copies.

Scientific limits: new candidates use one teacher and selection-exposed validation; historical three-teacher results are context. Opportunity caps do not equalize realized event counts or energy. Timing includes instrumented policy operations. Primary-source literature verification overrides five API title-search false flags, transparently retained in the repository. Independent confirmation, complete source-converter matching and publishable method novelty remain unresolved.
