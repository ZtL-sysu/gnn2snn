# Targeted prior-art screen — 2026-11-14

This is a positioning screen, not an exhaustive novelty clearance. Primary publication pages and author code were consulted.

| Source | Verified scope | Consequence for this campaign |
|---|---|---|
| [Li et al., ICML 2021, A Free Lunch From ANN](https://proceedings.mlr.press/v139/li21d.html) | Post-conversion layerwise calibration of parameters using ANN knowledge and conversion-error propagation | Local activation reconstruction and potential calibration are established; this control must be meaningful. |
| [Bojkovic et al., AISTATS 2024](https://proceedings.mlr.press/v238/bojkovic24a.html) | Activation-distribution-driven threshold and initial-potential initialization, theoretical and practical optimization | Initial membrane calibration itself is not a new contribution. |
| [Chen et al., FAS, 2025](https://arxiv.org/abs/2502.04405v2) | Coarse-to-fine ANN-to-SNN calibration for language models, following full-parameter fine-tuning | Combining local and final-output information is not intrinsically new; architecture/domain differences require specific evidence. |
| [Spiking Graph Convolutional Networks, IJCAI 2022](https://www.ijcai.org/proceedings/2022/0338.pdf) | Directly trained graph SNN for node classification, explicitly without conversion | Distinct task/training route; do not present graph spiking computation as new. |
| [One-Spike SNN, 2024](https://arxiv.org/abs/2403.08786) | Phase coding/base manipulation, including GCN conversion | Graph conversion has prior art; topology-specific novelty cannot be inferred from replacing CNN with GCN. |
| [Lan et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lan_Efficient_Converted_Spiking_Neural_Network_for_3D_and_2D_Classification_ICCV_2023_paper.html) | Conversion for 3D and 2D classification, threshold initialization and adaptive firing | Broader graph-like/point-cloud classification conversion must be considered in a full review. |
| [Bojkovic et al., ICML 2025](https://proceedings.mlr.press/v267/bojkovic25a.html) | Temporal misalignment and probabilistic spiking mitigation | Low-T error attribution requires evidence separating transient effects from graph-specific amplification. |
| [Pinned DCGS author implementation](https://github.com/h-z-h-cell/ANN-to-SNN-DCGS/tree/5e0e4c73f1d0e8e17d81befa24ff947deec1f4be) | MTH_diff eight-level signed differential events | Source-unit adaptation is the baseline, not a full original-converter reproduction. |

Current defensible distinction to test: the effect of objective choice under identical, small conversion-parameter and optimization budgets on full-graph classification with strong frozen teachers. A positive screen supports pursuing this experiment; it does not establish a novel graph-aware algorithm. Structure-sensitive counterfactuals and faithful stronger calibration comparisons would still be required.
