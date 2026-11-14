# DCGS percentile baseline source audit

## What the pinned source does

The pinned checkout is `C:/code/gnn2snn-dcgs-reference`. Its threshold collector places hooks on nonlinear layers, runs a model in evaluation mode over a calibration dataloader, and stores activation statistics ([`converter/threshold_getter.py`, lines 25–38 and 50–57](C:/code/gnn2snn-dcgs-reference/converter/threshold_getter.py)). For percentile mode, channel granularity reduces every dimension other than channel and applies `torch.quantile` ([same file, lines 237–266](C:/code/gnn2snn-dcgs-reference/converter/threshold_getter.py)). The ResNet34 ImageNet example selects `99.9%`, channel level, and calibration batch size 200 ([`resnet34_config_get_thre_channel_999.yaml`, lines 11–30](C:/code/gnn2snn-dcgs-reference/configs/resnet/resnet34_config_get_thre_channel_999.yaml)).

The MTH_diff source defines fractional thresholds `theta / 2**i`, uses a `0.75` firing boundary, subtracts the first input as a bias in later ticks, and maintains cumulative and exponentially accumulated input/output state ([`neurons/relu_neuron.py`, lines 483–540](C:/code/gnn2snn-dcgs-reference/neurons/relu_neuron.py)). The checked-in ResNet34 MTH_diff example loads channel thresholds, uses threshold scale 4, sets `num_thresholds: 4`, and evaluates T=32 ([`resnet34_config_test_channel_mth_diff_999.yaml`, lines 27–38 and 57–59](C:/code/gnn2snn-dcgs-reference/configs/resnet/resnet34_config_test_channel_mth_diff_999.yaml)). Thus the author's general neuron default of eight levels is not the level count in this example.

## Scope of this GCN runner

[`run_percentile.py`](run_percentile.py) is an explicit GCN adaptation, not a literal reproduction of the ImageNet/ResNet experiment. It freezes the existing teacher, collects the six post-ReLU graph/MLP site activations over the same deterministic 512-graph train subset used by the critical-site screens, computes exact per-channel 0.999 quantiles over concatenated observations, and floors thresholds at `1e-6`. No labels enter threshold estimation, no teacher is retrained, and no parameter or scale is optimized. The selected 512 validation graphs are used only for reporting. No test split is loaded.

Two preregistered conditions are emitted: `author_cfg` uses quantile × 4 with four threshold levels, following the cited author configuration; `matched8` uses quantile × 1 with eight levels, to match the prior screen's level count and avoid mixing in threshold rescaling. Both use T=1,2 by default. These GNN settings remain adaptations: GNNBenchmark data, six graph/MLP sites, and T=1/2 differ from the cited ImageNet config's ResNet and T=32.

The MTH_diff recurrence is source-matched at the step level for both four and eight levels in [`test_percentile.py`](test_percentile.py); the eight-level network forward is additionally checked bitwise against the imported author evaluator in the run. Tests are CPU-only. Run with the project Conda interpreter:

```powershell
& 'C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe' -X utf8 -m pytest research/20261114_critical/test_percentile.py -q
```

Outputs record checkpoint, historical calibration, processed data, dependency, and author-source identities; save all six quantile threshold vectors, validation predictions, source snapshots, and `parameter_updates: 0`. Calibration cost is one frozen-teacher pass over the selected train graphs; reported wall time is stored in `DONE.json` and `metadata.json`.
