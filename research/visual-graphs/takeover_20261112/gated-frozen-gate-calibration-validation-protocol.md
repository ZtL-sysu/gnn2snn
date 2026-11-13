# Frozen-gate calibration and validation protocol v1

## Frozen scope

This protocol consumes only the completed CIFAR10 `GatedTeacher` run at
`full_teacher_runs/gated_cifar10_seed41_full_v1`. It requires `done.json`,
`metadata.json`, and `best.pt`, checks terminal status, exact official split
counts (45,000 train / 5,000 validation), the checkpoint hash recorded in
`done.json`, and every source hash recorded by the training metadata. The
checkpoint metadata must match the run metadata. The source/data adaptation is
the recorded PyG teacher, not an exact reproduction of the original DGL input
pipeline.

## Threshold calibration

Run `gated_frozen_gate_calibrate.py` only after teacher completion. It loads
the complete official CIFAR10 train split in deterministic dataset order and
wraps it in `LabelFreeGraphView`, which constructs fresh `Data` objects from
`x`, optional `pos`, `edge_index`, and `edge_attr` only. No target/label field
is placed in a calibration batch or read by the model.

Two full passes use the same 45,000 graphs and fixed order. Pass one records
per-channel activation maxima for the four post-node-BN/ReLU residual branch
sites (`gated_0` ... `gated_3`) and the two MLP ReLU sites (`mlp_0`, `mlp_1`).
Pass two updates the existing `ActivationHistogram` with 512 bins. For each
site/channel, the threshold is the upper edge of the first cumulative bin
reaching `ceil(N * 0.999)`, where `N` is that site's activation-row count
(nodes for the four graph sites; graphs for each MLP site). This matches the
existing full-GCN calibration semantics. Inactive channels are floored at
`1e-6`. The record includes graph and activation-row counts, maxima, channel
histograms, thresholds, processed-data hashes, teacher checkpoint/run
metadata, and source hashes. It is written with exclusive creation; an
existing output is an error.

Example invocation after a completed run:

```powershell
conda run -n dyg-cold python gated_frozen_gate_calibrate.py `
  --teacher-run full_teacher_runs/gated_cifar10_seed41_full_v1 `
  --output full_teacher_runs/gated_cifar10_seed41_full_v1/gcal_v1.json `
  --device cuda
```

## Full validation evaluation

Only after a complete calibration, run `gated_frozen_gate_validate.py`. It
verifies calibration status, all-train graph count, source hashes, checkpoint
hash, and teacher-run identity before loading the 5,000 validation graphs in
deterministic order. The teacher's label-free logits and exact frozen-gate
oracle replay must match within `rtol=1e-5, atol=1e-6`. It reports the
teacher-conditioned exact oracle and staged channel-threshold results at
`T={8,16,32}`. Validation labels are used only after the logits are computed,
for final accuracy metrics. The script stores each graph's label and logits
for the teacher, exact oracle, and all three staged settings in a compressed
NPZ; the JSON summary records teacher accuracy, each method's accuracy,
teacher agreement, logit MSE/MAE, per-site and total spike counts, hashes,
elapsed time, and exact-path maximum logit difference. Both output paths reject
existing files.

Example invocation after calibration:

```powershell
conda run -n dyg-cold python gated_frozen_gate_validate.py `
  --teacher-run full_teacher_runs/gated_cifar10_seed41_full_v1 `
  --calibration full_teacher_runs/gated_cifar10_seed41_full_v1/gcal_v1.json `
  --summary full_teacher_runs/gated_cifar10_seed41_full_v1/gval_v1.json `
  --predictions full_teacher_runs/gated_cifar10_seed41_full_v1/gval_v1_pred.npz `
  --device cuda
```

## Interpretation boundary

Every graph's gates come from a complete ANN gated-convolution forward. The
exact oracle is therefore a teacher-conditioned hybrid control, and its ANN
controller cost is not counted in spikes. Node activations are quantized in
six ordered stages; `T` is a count resolution per site, not a measured
end-to-end latency or six-layer causal simulation horizon. Dense graph
operations and gate generation are not event-driven energy measurements.
Validation was used in checkpoint selection by the teacher training protocol,
so these 5,000 examples are complete validation evidence but not independent
test evidence. This protocol never loads the test split.

## CPU synthetic check

Run `conda run -n dyg-cold python -m pytest -q
test_gated_frozen_gate_pipeline.py`. The test covers label-free two-pass
histograms and six per-channel threshold shapes on tiny synthetic graphs,
plus complete validation metric aggregation for a small CPU teacher. It does
not run official data or establish benchmark results.
