# Claim-to-evidence map

All paths below are repository-relative; exact numerical data and hashes are preserved.

| Manuscript claim | Primary evidence | Qualification |
|---|---|---|
| Table 1 high-prefix fidelity | `research/visual-graphs/takeover_20261112/{fixed-gcn-test-three-seed-summary-v1.json,gated-three-teacher-seed-summary-v1.md}` and audit `affine-cal-and-fidelity-artifact-audit-v1.md` | Three teacher seeds, exposed full test, separate adapters |
| First-site disagreement gap closure | `research/20261114_story/analysis/ideal/aggregate.csv` | full_validation, T1, causal continuous oracle, nonadditive groups |
| 332/811 and 296/928 adverse margin changes | `research/20261115_decision/runs/intervention_v2/{MNIST41,CIFAR1041}/graphs.csv` | first-tick MSE, repeated within exposed validation graph, one teacher |
| Bootstrap intervals, T2 reuse | `research/20261115_decision/analysis/phase3/intervention_cluster_ci.csv` | graph-cluster resampling; not seed CI |
| Example val437 and allocation metric inversions | `research/20261115_decision/RQ1_diagnostic.md`, `analysis/cached/per_graph_diagnostics.csv` | all examples retained; not selected aggregate estimate |
| Three-teacher calibration | `research/20261114_story/analysis/frozen_test/aggregate.csv`, `analysis/confirmation/aggregate.csv` | horizon-specific fits; historical test |
| Ordinary shared-prefix controls | `research/20261115_decision/analysis/phase3/main_table.csv`, `paired.csv` | shared vector; teacher41 val512 |
| Ranking third-competitor effect | `research/20261115_decision/analysis/optimization_audit/results.json` | training-only optimization check |
| Source TI controls | `research/20261115_decision/analysis/source_ti/summary.md`, `source_ti.py` | pinned source algorithm; graph adaptation, not ImageNet reproduction |
| Local temporal equality/global changes | `research/20261115_decision/analysis/prefix_diagnostic/endpoints.csv`, `prefix_codec.py`, `run_prefix_diagnostic.py` | 128 training graphs, local T2 bitwise equality only |
| Scalar counterexample | `research/20261115_decision/test_prefix_network.py` | executable source recurrence, not population result |
| Centered shared theta | `research/visual-graphs/takeover_20261112/affine_{centered,first_analog}_shared_s{1,8}_tr1.json` | same threshold vectors, 45k train, no accuracy labels |
| Timing | `research/20261114_story/costs/cost_*.json`, `COST_DEFINITIONS.md`; critical `analysis/allocation_mechanism_cost.csv` | distinct experiments, cached batches, no energy |
| Node context | `paper/cvpr2027/early/raw/`, `six_dataset_summary.csv`, `sha256_manifest.json` | copied unchanged audited archive; aggregate-only SNN |

Main mathematical relations are explicitly standard identities/sufficient conditions. Graph degree descriptors and incremental in-sample R² come from `research/20261115_decision/analysis/cached/summary.json`. No new graph-specific performance or novelty is inferred from their values.

Separate centered full-validation campaign: `data/centered_validation/` preserves summaries, full metric CSV, latency and cost ledgers from branch agent/visual-conversion/20261113-wb-resume at 58a2578, with original byte hashes. Three-teacher fidelity is supported; joint complete-cost/latency advantage is not. Shared-theta training evidence and this validation matrix are separate.
