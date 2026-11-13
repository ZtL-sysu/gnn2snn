# Gated result-to-claim review v1

Date: 2026-11-13  
Status: **provisional — no paper claim audit available** in the requested artifact set. Independent result-to-claim review; no manuscript score or pass assigned.

## Verdicts

### Narrow feasibility claim

- **Claim:** A full-data visual GatedGCN can preserve its ANN accuracy on CIFAR10 at causal T<=8, within 1.5 percentage points, using oracle-free graph-adapted differential coding.
- **claim_supported:** **yes** (for this frozen teacher, architecture, dataset, and reported protocol).
- **confidence:** **medium**. The full official validation and test logits support the numerical claim, but this is one teacher seed and the test split was previously exposed.
- **what_results_support:** Teacher accuracy is 70.66% on 5,000 validation graphs and 69.20% on 10,000 test graphs. Differential T=4 gives 70.72% validation (+0.06 pp) and 69.10% test (-0.10 pp); T=8 gives 70.66% validation (0.00 pp) and 69.16% test (-0.04 pp). Both satisfy the declared 1.5 pp threshold. Teacher agreement is 99.62%/99.84% on validation and 99.70%/99.90% on test for (T=4/8), respectively. The evaluator code removes labels before conversion; the differential forward rejects labeled input and does not call the teacher forward for conversion. The result is therefore evidence of an oracle-free forward path by code inspection, although the teacher is separately evaluated for comparison.
- **what_results_dont_support:** This is not evidence of a general result across GNN operators, graph vision datasets, or independently trained teachers. It does not establish hardware latency, energy, sparse execution, or “efficient” execution. The test comparison is not an independent blind confirmation: both JSON reports explicitly mark prior validation exposure, and the test report marks prior test exposure.
- **suggested_claim_revision:** “On one CIFAR10 GatedGCN teacher (seed 41), an oracle-free differential hybrid matched teacher accuracy within 0.10 pp on the full validation and historically exposed test splits at T=4 and T=8, while recomputing dense analog gates each tick.”

### Broader novelty and efficiency claim

- **Claim:** A new efficient, operator-agnostic GNN-to-SNN method suitable for a CVPR contribution.
- **claim_supported:** **no**.
- **confidence:** **high** that these results do not establish this claim.
- **what_results_support:** They establish a strong, narrow fidelity result for the present GatedGCN diagnostic.
- **what_results_dont_support:** The design and source audit state that differential-coding algebra is established, and the released DCGS source covers fixed operators rather than this dynamic normalized-gate conversion. A single adapted model does not establish operator-agnostic scope or novelty. The current code retains dense analog edge controllers, sigmoid, target normalization, message reduction, and dense affine work every tick, and uses signed graded eight-level source events rather than binary spikes. The dense-cost audit counts executed linear MACs as embedding + T*body, versus embedding + body for the ANN (approximately 4x / 8x dense linear MACs at (T=4/8) for the stated cohort); additional nonlinear, scatter, state, and threshold work is excluded. Event counts are not a hardware-cost proxy.
- **suggested_claim_revision:** Restrict this experiment’s conclusion to low-step fidelity for one dynamic-gate visual GNN. Keep novelty and efficiency as unproven research hypotheses; do not call the present dense hybrid efficient or operator-agnostic.

## Evidence checked

- Recomputed accuracies, teacher agreement, and logit MSE from both NPZ archives with the local Conda environment `dyg-cold` (CPU-only analysis). For the principal rows, recomputed values agree with the JSON: validation DC T=4 70.72%, T=8 70.66%; test DC T=4 69.10%, T=8 69.16%. Half-init terminal IF at T=8 is 25.72% validation and 25.91% test; zero-init terminal IF is 10.38% and 10.00%, respectively. These controls show the failure of these binary causal IF variants, not superiority over all conversion baselines.
- NPZ shapes are 5,000 × 10 for validation and 10,000 × 10 for test, with labels and teacher logits present. JSON records complete splits, checkpoint SHA256 `d26df76d…5598b40`, calibration SHA256 `b28485dd…2cf8a9e`, teacher epoch 34, and author-source commit `5e0e4c73f1d0e8e17d81befa24ff947deec1f4be`. The differential source SHA256 in the report matches the local file: `66a8a306…14a7a1e`.
- Test archive alignment was checked against `full_teacher_runs/gated_cifar10_seed41_full_v1/gtest_v1_pred.npz`: ordered indices, labels, and teacher argmax predictions all match. Its sidecar reports teacher test accuracy 69.20%, checkpoint epoch 34, and the same test prediction SHA256 recorded in `gd_test1.json`. The report also records that author-source worktree content matches the pinned source blob.
- The NPZ embedded report predates the sidecar’s archive SHA field by construction; use `gd_val1.json`/`gd_test1.json` for archive verification metadata.

## Missing evidence and next smallest experiment

1. Re-run the predeclared validation protocol with an independently trained GatedGCN seed, train-derived thresholds, preserved full-validation predictions, and no test-driven selection. This is the smallest check of seed robustness; report paired accuracy, agreement, logit error, and signed graded event accounting.
2. Before using “operator-agnostic,” apply a frozen protocol to at least one distinct GNN operator/family and graph dataset, with explicit operator-specific mapping and accuracy criterion.
3. Before using “efficient,” compare end-to-end wall latency and an explicit operation/state/memory account against the ANN and relevant conversion baselines. A genuinely efficient-method claim also needs an implementation that reduces executed work; the current dense formula already shows more dense linear MACs at (T>1).

**Routing:** retain the narrow feasibility result with the scope and exposure caveats above; supplement for reproducibility and broader scope. Do not advance a new efficient/operator-agnostic-method claim from these results.

