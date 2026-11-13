# CIFAR10 gated teacher seed protocol v2

## Purpose and scope

This protocol adds the predeclared CIFAR10 teacher-seed replications 42 and 43 using the unchanged `gated_teacher_train.py` design and `gated_teacher_model.py`. It creates a PyG adaptation of the public ResGatedGCN configuration; it is not an exact reproduction of the original DGL preprocessing or training pipeline. The seed-41 run remains the original reference run. This trainer does not alter conversion or student-training protocols.

## Frozen data, architecture, and optimization

- Dataset: PyG `GNNBenchmarkDataset` CIFAR10, all 45,000 official training graphs and all 5,000 official validation graphs. The test split is never opened. Each completed epoch must consume exactly 45,000 train graphs; each validation must consume exactly 5,000 graphs in deterministic order.
- Architecture: `GatedTeacher`, node and edge feature dimensions inferred from the official training sample; hidden width 70, output width 70, four gated layers, 10 classes, mean graph readout, residual connections, batch normalization, and dropout 0. This matches the v1 adapted model and `public_baselines/gated_cifar.json` settings.
- Optimization: Adam, learning rate 0.001, weight decay 0; `ReduceLROnPlateau` on validation loss with factor 0.5, patience 10, minimum learning rate 1e-5; batch size 128.
- Budget/stopping: actual cap is 200 epochs and 12 elapsed hours. Stop on learning rate `<=1e-5`, the epoch cap, or time cap, whichever occurs first. Seed 41 stopped at epoch 100 when the scheduler reached the minimum learning rate; 100 is not the configured epoch cap.
- Checkpoint selection is strict best validation accuracy; tied scores retain the earlier checkpoint. `last.pt` is saved each epoch. A terminal `done.json` is written only after normal termination.

## Seed semantics and provenance

`--seed` accepts only 42 or 43. It seeds Python, NumPy, CPU Torch, and all CUDA Torch generators before dataset/model initialization. The DataLoader uses the seeded Torch RNG for shuffle with `num_workers=0`, so the same seed controls initialization and train order. Metadata names this explicitly as a **teacher initialization and training-order seed**. It is not an SNN student-seed variance study. For context, existing student experiments use student seed 41; this trainer records that reference but does not apply or vary a student seed.

The unique output directories are `full_teacher_runs/gated_cifar10_seed42_full_v2` and `full_teacher_runs/gated_cifar10_seed43_full_v2`. Any existing target directory causes refusal before dataset access. Each run metadata contains a source manifest and digest covering this v2 trainer, this v2 protocol, unchanged `gated_teacher_train.py`, `gated_teacher_model.py`, `gated_teacher_train_protocol_v1.md`, the public `gated_net.py`, `gated_layer.py`, `benchmark_mlp_readout.py`, and `gated_cifar.json`, plus processed data hashes, split counts, seed roles, model/optimizer settings, framework versions, and device. `provenance.source_sha256` mirrors the same relative-path/hash mapping for compatibility with the existing calibration source validator; no calibration/evaluator code is changed.

## Checks and execution

`--smoke` builds only four tiny synthetic CPU graphs, performs one optimizer/evaluation pass, reports the source manifest, and does not load an official dataset or create a run directory. `check_gated_teacher_seed_v2.py` checks CLI smoke provenance, no official split load, deterministic same-seed/different-seed model initialization, source manifest coverage, directory refusal semantics, and the legacy source validator against a temporary synthetic completed-run fixture. These are engineering checks; no full training is part of validation.

Full runs require CUDA and are launched only after separate review. Do not modify v1 sources or calibration/evaluation files as part of this protocol.
