# Early node-classification evidence (historical context)

This folder preserves a minimal evidence package for the early six-dataset GNN-to-SNN node-classification line. It is historical context for the CVPR 2027 paper; it does not validate the later visual-graph method or establish a new method claim. Source copies below are byte-for-byte copies from the local archive under `C:\code\phd-snn\auto-paper\20261028-171500\Spiking-GNN-Conversion\f-research-v1-run-20261028-174053-gnn-snn-conv\`.

## Rebuilding the table

From the repository root, run `conda run -n base python paper/cvpr2027/early/build_early_table.py`. This parses only the archived JSON summaries, writes `six_dataset_summary.csv`, and refreshes `sha256_manifest.json`. It does not run neural experiments.

The table uses analytic-bias FTBC (`beta*=theta/(2T)`) membrane accuracy and the same JSON's `acc_count_mean` as the integer spike-count readout comparator at T=8 and T=16. For Cora, CiteSeer, and PubMed, the script recomputes ANN means from the retained five ANN seed records (`ann_per_seed`). SNN per-seed records are not retained in these chosen analytic summaries, so FTBC means and standard deviations cannot be independently recomputed here. Chameleon, Squirrel, and ogbn-arxiv summaries have aggregate-only ANN and SNN results. These record levels are marked in the CSV. No seed-level uncertainty should be inferred from aggregate-only rows.

## Source inventory

- `raw/results__cora_analytic.json`, `...citeseer...`, `...pubmed...`: five-seed ANN rows plus aggregate conversion means; analytic beta, T grid, membrane/count accuracies and spike totals.
- `raw/results__strengthen_extended_hetero.json`: aggregate Chameleon and Squirrel results, including N, hidden width, class count, homophily and FTBC membrane/count readouts.
- `raw/results__strengthen_extended_arxiv.json`: aggregate ogbn-arxiv results and metadata.
- `raw/protocols__snn_gnn_experiment.py`, `graph_data.py`, `strengthen.py`, `arxiv_strong.py`: model/conversion/data and extension protocol source snapshots.
- `raw/audits__d1-theory-derivation.md`, `protocols__theory_verify.py`: derivation and numerical check for the conversion error identity, IF floor quantization, analytic bias and graph/margin/time bound.
- `raw/audits__d1-experiment-audit.md`, `protocols__consolidate_strengthen.py`: historical audit and consolidation code. The audit is retained as source evidence, not treated as a substitute for raw seed artifacts.

## Audit boundaries and exclusions

The reusable claims are limited to this old two-layer GCN setup: a layer-1 floor-quantization mechanism, the closed-form mean-bias correction under the stated quantizer assumptions, membrane versus integer-count readout behavior, and an error/margin sufficient-condition analysis checked against this implementation. Do not present these as novel components or as validation of the visual-graph experiments.

The archived audit documents repairs to early Planetoid test-index misalignment, incorrect global lambda scaling, layer-2 constant-current accumulation, duplicate-edge degree changes, and an invalid `calibrated` upper-bound label. Earlier values affected by these defects are excluded. It also records that test-node activation statistics once entered bias tuning; the corrected protocol restricted tuning to training nodes. For the theory, symmetric normalization is not row-stochastic, the global `|E1|<=1/T` statement needs saturated units handled separately, a single scalar systematic-error mean can get the sign wrong, and the predicted failure of sum aggregation did not occur. The Chameleon hidden-width-16 loss was a width/input-dimension artifact; the retained extended result uses hidden width 64. These are reasons to cite only the corrected source summaries and keep all claims narrowly scoped.

The table contains model accuracy, not a controlled baseline comparison or hardware energy measurement. Reported spike totals in source summaries are not converted here into joules, latency, or energy advantage. The public repository snapshot does not contain the full early dataset cache or all per-seed SNN outputs; the retained protocol files and summaries support provenance and table reconstruction, not a fresh end-to-end rerun.

## Integrity

`sha256_manifest.json` records sizes and SHA-256 hashes for this folder's files other than the manifest itself. Source archive locations and exact filenames are recorded above; originals were not modified.
