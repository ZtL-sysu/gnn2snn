# Graph-objective strong-teacher campaign

Work from the repository root using the local Conda environment `dyg-cold`. See `PROTOCOL.md` for the frozen selection rules and `DECISIONS.md` for changes. Historical tests and validation are exposed; all new calibration selection uses only train and validation splits.

1. `python research/20261114_story/audit_baselines.py` verifies historical archives and checkpoints at the original local paths.
2. Clone `https://github.com/h-z-h-cell/ANN-to-SNN-DCGS.git`, checkout `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be`, pass its directory via `--author-repo`. The CPU check currently defaults to `C:/code/gnn2snn-dcgs-reference`.
3. `python research/20261114_story/test_calibration_core.py` checks the hard recurrence and gradient availability on synthetic data.
4. `python research/20261114_story/run_screen.py --dataset MNIST --seed 41 --output research/20261114_story/runs/screen/MNIST41` runs the frozen screen; substitute CIFAR10. Use `--assets` to select the historical run root containing `data/pyg_bench` and `takeover_20261112/full_teacher_runs`. No teacher retraining is required. Existing outputs are never overwritten.
5. `python research/20261114_story/analyze_screen.py --runs research/20261114_story/runs/screen --output research/20261114_story/analysis/screen` recomputes summaries from saved graph predictions.
6. `python research/20261114_story/benchmark_cost.py --run research/20261114_story/runs/screen/MNIST41 --output research/20261114_story/cost_MNIST41.json` measures a fixed cached batch with synchronization and reports a partial operation ledger. It does not estimate physical energy.

The default historical asset root is the local `C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision` directory. The repository's existing `artifacts/verified-evidence.zip` bundles all nine teacher `best.pt` checkpoints and all six fixed-GCN calibration records. Run `python tools/restore_evidence.py` to verify and expand that bundle safely; then `--assets <checkout>/research/visual-graphs` selects the restored layout. Place the official PyG data under that directory's `data/pyg_bench`. Results retain checkpoint and data hashes for identifying matching assets. No newly trained teacher weights or datasets are added by this campaign.

Historical seed41 stores calibration beside its teacher; seed42/43 records are under `takeover_20261112/rep/M42|M43|C42|C43/cal.json`. The resolver supports both audited layouts. Calibration JSON contains original absolute provenance paths, while the new executor verifies actual relocated checkpoint, code and data hashes. The archive audit scripts retain local historical paths; raw new result summaries are reproducible from published prediction arrays without loading weights or datasets.

`runs/smoke*` are engineering checks, never scientific samples. Failed runs retain a failure record and raw log. Use separate parent directories for each calibration budget and sample-count protocol; do not mix a smoke run with the scientific screen or aggregate different configurations.
