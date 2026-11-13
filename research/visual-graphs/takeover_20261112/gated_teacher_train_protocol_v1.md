# CIFAR10 PyG GatedGCN teacher training protocol v1

## Scope and claim

Train a stronger CIFAR10 graph teacher using all 45,000 official PyG GNNBenchmarkDataset training graphs and all 5,000 official validation graphs. The test split is never opened by this script. This is a PyG operator/data adaptation of the public ResGatedGCN configuration, not an exact reproduction of the original DGL experiment.

The source architecture and settings are `public_baselines/gated_net.py`, `gated_layer.py`, `benchmark_mlp_readout.py`, and `gated_cifar.json`. The adapted model uses four gated layers, hidden and output widths 70, residual connections, batch normalization, zero dropout, mean graph pooling, and the public two-hidden-layer MLP readout. The public CIFAR configuration specifies seed 41, batch size 128, Adam at 0.001, zero weight decay, plateau factor 0.5 and patience 10, minimum learning rate 1e-5, and a 12 hour budget. This run fixes the training cap at 200 epochs and 12 hours.

## Dataset semantics

The original public DGL pipeline constructs CIFAR superpixel nodes from mean pixel channels plus two-dimensional coordinates. It constructs directed top-k neighborhoods from a symmetric similarity matrix and attaches scalar affinities derived from color and coordinate distances. This script consumes the existing PyG CIFAR10 train/validation split and its stored `x`, optional `pos`, `edge_index`, and `edge_attr`; it does not recreate the public DGL preprocessing or recompute those affinities. The two implementations have documented small-sample feature/edge differences. Therefore numerical results are evidence about this PyG adaptation only.

Before training, the script requires 45,000 train and 5,000 validation examples and a nonempty `edge_attr`; dimensions are inferred from the first training graph and the model checks feature widths while forwarding. It records processed data hashes, source hashes, effective dimensions, and framework versions. No fallback replaces absent edge features with ones.

## Training and selection

- Seed Python, NumPy, CPU Torch, and CUDA Torch generators with 41.
- Shuffle all official training indices each epoch with batch size 128; require exactly 45,000 labels/graphs in every completed epoch.
- Evaluate the full validation split in deterministic order after each epoch; require exactly 5,000 examples.
- Step `ReduceLROnPlateau` on validation loss with factor 0.5 and patience 10.
- Save `last.pt` after each epoch. Save `best.pt` whenever validation top-1 strictly improves. Ties retain the earlier checkpoint.
- Stop after the learning rate drops below 1e-5, 200 epochs, or 12 elapsed hours. Write `done.json` only after normal termination.
- Record per-epoch train/validation counts, loss, accuracy, learning rate, elapsed time, and epoch time in `epochs.jsonl`.

The output directory is the short fixed name `full_teacher_runs/gated_cifar10_seed41_full_v1`; the script refuses to reuse an existing directory. It writes metadata and logs with exclusive creation. An interrupted run has no terminal `done.json` and is not eligible as a completed teacher.

## Operational constraints

Default full training requires CUDA and opens only CIFAR10 train and validation splits. `--smoke` instead runs one synthetic CPU optimizer/evaluation pass; it does not load official data or create a run directory. The smoke mode is an engineering check, not a result. The script stores training checkpoints and may be resumed only after a separately reviewed resume implementation; automatic resume is not part of v1.

## Conversion limitation

This teacher has dynamic edge states: sigmoid gates depend on current source/target node states and edge representations, followed by target-wise normalization. Edge BatchNorm and edge residuals also change those states layer by layer. The existing fixed-threshold staged conversion for residual GCN branches and MLP activations cannot directly express these operators. A separate conversion design and validation are required; this training protocol does not imply SNN compatibility.
