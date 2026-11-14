# Graph-objective strong-teacher campaign

Work from the repository root using the local Conda environment `dyg-cold`. See `PROTOCOL.md` for the frozen selection rules and `DECISIONS.md` for changes. Historical tests and validation are exposed; all new calibration selection uses only train and validation splits.

1. `python research/20261114_story/audit_baselines.py` verifies historical archives and checkpoints at the original local paths.
2. Clone `https://github.com/h-z-h-cell/ANN-to-SNN-DCGS.git`, checkout `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be`, pass its directory via `--author-repo`. The CPU check currently defaults to `C:/code/gnn2snn-dcgs-reference`.
3. `python research/20261114_story/test_calibration_core.py` checks the hard recurrence and gradient availability on synthetic data.
4. `python research/20261114_story/run_screen.py --dataset MNIST --seed 41 --output research/20261114_story/runs/screen/MNIST41` runs the frozen screen; substitute CIFAR10. Use `--assets` to select the historical run root containing `data/pyg_bench` and `takeover_20261112/full_teacher_runs`. No teacher retraining is required. Existing outputs are never overwritten.
5. `python research/20261114_story/analyze_screen.py --runs research/20261114_story/runs/screen --output research/20261114_story/analysis/screen` recomputes summaries from saved graph predictions.
6. `python research/20261114_story/benchmark_cost.py --run research/20261114_story/runs/screen/MNIST41 --output research/20261114_story/cost_MNIST41.json` measures a fixed cached batch with synchronization and reports a partial operation ledger. It does not estimate physical energy.

The default historical asset root is the local `C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision` directory. Large datasets and teacher weights are local-only. Results retain checkpoint and data hashes for identifying matching assets. Official datasets can be obtained through PyG `GNNBenchmarkDataset`; full pretrained weights are needed to reproduce these exact endpoints.

`runs/smoke*` are engineering checks, never scientific samples. Failed runs retain a failure record and raw log. Use separate parent directories for each calibration budget and sample-count protocol; do not mix a smoke run with the scientific screen or aggregate different configurations.
