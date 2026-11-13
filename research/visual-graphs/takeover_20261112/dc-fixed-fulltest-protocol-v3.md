# Fixed-GCN source-unit adaptation: frozen full-test endpoints

Purpose: complete the official-test fidelity endpoints for the already validated graph adaptation. This is a baseline/reference exercise, not a new proposed-method trial, parameter search, final review or scientific gate pass.

## Frozen models and data

Use seed 41 full-data, four-layer, width-146 residual GCN teachers in full_teacher_runs/MNIST_seed41_full and CIFAR10_seed41_full. Preserve their existing best checkpoints and calibration_hist_v1.json. Calibration used the complete official training split (MNIST 55,000 / CIFAR10 45,000), two passes, 512 bins, q=0.999 per-channel thresholds; do not recalibrate. Do not train or select a new model. The official test split contains all 10,000 ordered graphs per dataset, including the final partial batch, with batch size 128; no masking, subsampling or new split.

Reference each teacher's existing teacher_full_test_predictions.npz plus teacher_full_test.json. Require exact ordered index/label correspondence and same-checkpoint metadata/hash. Recomputed ANN logits must agree within rtol1e-5/atol1e-4 before accepting an output. These test examples have prior project exposure; all results are descriptive exposed-test endpoints, not blind confirmation.

## Source and mechanism boundary

Use dcgs_fixed_seed_eval_v3.py, which imports the accepted v2 primitives and fixed forward. Author MTH_diff is unchanged from commit5e0e4c73f1d0e8e17d81befa24ff947deec1f4be. Fixed T grid is1/2/4/8; six zero-latch warmups per batch per T run are excluded from active ticks, with6T active unit calls. Deliver affine offsets once, and decode output by sum(event_t/t). Strip labels before the converted forward; teacher logits are used only for external fidelity comparison.

Call the result a PROJECT SOURCE-UNIT GRAPH ADAPTATION. Common full-training histogram calibration differs from the author's threshold iteration. The pinned CNN/ReLU converter source uses ReLU replacements, while its generic non-ReLU strategy also inserts pre-linear identity units; the current fixed-graph placement is a manually validated ReLU-branch strategy. This is not the author's original full-model experiment, and these results cannot stand in for a faithful generic dynamic-gate DCGS comparison. Signed eight-magnitude events are graded, not binary IF. Keep all dense graph/readout cost caveats visible.

## Execution and output

After root source/CPU acceptance and after GPU queue14938 is terminal, run the two endpoints sequentially on the local GPU. Reserved outputs: dc_mn_test1.json/.npz and dc_cf_test1.json/.npz. Refuse existing or partial outputs; no automatic overwrite/restart after a failed reference or input guard. Save every T result, predictions, teacher outputs, event counters, source-unit call accounting, input/source/checkpoint/calibration/reference/data hashes, runtime device/versions and elapsed diagnostic time. Rehash bound inputs at the end. The elapsed times are not a controlled latency benchmark.

No accuracy-dependent selection or follow-up tuning is authorized by this endpoint protocol. Interpret fidelity only within the matched checkpoint and scope; source adaptation fidelity does not establish proposed-method novelty, a baseline win, efficiency, full Algorithm2 reproduction or final CCFA>=7 acceptance. Existing full-validation seed42/43 records may be evaluated separately with the accepted same forward and their own frozen full-train calibration; do not confuse these full-data models with the historical narrow/two-layer secondary track.
