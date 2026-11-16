# Comparator Literature Table (S4)

## Adaptation of the selection rubric

The submitted venue is a conference rather than an SCI journal. The f-submit-v1 rubric was therefore adapted by replacing SCI/JCR and impact-factor points with verifiable peer-reviewed top-venue status. No journal impact factor, quartile, or SCI status is claimed. Scores are:

| Criterion | Points |
|---|---:|
| Topic/method similarity | 0–5 |
| Same or adjacent application domain | 0–3 |
| Publication recency | 0–3 |
| Top-venue strength relative to field | 0–4 |
| Bibliographic-source confidence | 0–3 |
| Abstract/full-text or section-access usefulness | 0–2 |
| **Total** | **0–20** |

Recency is scored against a 2026 CVPR-style submission: 3 for 2024–2025, 2 for 2023, 1 for 2022, 0 for older work with historical justification. The final set is frozen before manuscript edits.

## Ranking of the candidate pool

| Rank | Candidate short name | Similarity | Domain | Recency | Venue | Source | Access | Total | Selected |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | Temporal Misalignment | 5 | 3 | 3 | 4 | 3 | 2 | 20 | Yes |
| 2 | Inference-Scale Complexity | 5 | 3 | 3 | 4 | 3 | 2 | 20 | Yes |
| 3 | Differential Coding | 5 | 3 | 3 | 4 | 3 | 2 | 20 | Yes |
| 4 | A Graph is Worth 1-bit Spikes | 5 | 3 | 3 | 4 | 3 | 2 | 20 | Yes |
| 5 | Are Conventional SNNs Really Efficient? | 4 | 3 | 3 | 4 | 3 | 2 | 19 | Yes |
| 6 | PD-Quant | 4 | 3 | 2 | 4 | 3 | 2 | 18 | Yes |
| 7 | Comparing Decision-Making Mechanisms | 4 | 3 | 3 | 4 | 3 | 2 | 19 | Yes |
| 8 | VISTREAM | 4 | 3 | 3 | 4 | 3 | 2 | 19 | Yes |
| 9 | Efficient Event-Based Object Detection | 3 | 3 | 3 | 4 | 3 | 2 | 18 | Yes |
| 10 | FTBC | 4 | 3 | 3 | 4 | 3 | 1 | 18 | Yes |
| 11 | Faster and Stronger | 4 | 3 | 3 | 4 | 3 | 2 | 19 | No |
| 12 | Efficient ANN-Guided Distillation | 4 | 3 | 3 | 4 | 3 | 2 | 19 | No |
| 13 | Spike-driven Transformer V2 | 3 | 3 | 3 | 4 | 3 | 2 | 18 | No |
| 14 | Threaten SNNs via Rate and Temporal Information | 4 | 2 | 3 | 4 | 2 | 1 | 16 | No |
| 15 | A2Q | 4 | 3 | 2 | 4 | 3 | 2 | 18 | No |
| 16 | Reducing ANN-SNN Conversion Error | 4 | 2 | 2 | 3 | 3 | 1 | 15 | No |
| 17 | Data Driven Threshold and Potential Initialization | 3 | 2 | 3 | 3 | 3 | 2 | 16 | No |
| 18 | Brain-Inspired SNNs for Energy-Efficient Object Detection | 2 | 3 | 3 | 4 | 3 | 2 | 17 | No |
| 19 | Towards Effective and Sparse Adversarial Attack on SNNs | 3 | 3 | 3 | 4 | 3 | 2 | 18 | No |
| 20 | Spiking Transformer | 2 | 3 | 3 | 4 | 3 | 2 | 17 | No |
| 21 | A Free Lunch From ANN | 3 | 2 | 0 | 4 | 3 | 2 | 14 | No |
| 22 | Degree-Quant | 3 | 3 | 0 | 4 | 3 | 2 | 15 | No |
| 23 | Error-Aware Conversion | 4 | 2 | 3 | 3 | 3 | 1 | 16 | No |
| 24 | Hybrid ANN-SNN Architecture for Low-Power Visual Perception | 3 | 3 | 3 | 2 | 3 | 2 | 16 | No |
| 25 | Neuromorphic Lip-Reading with Signed Spiking GRUs | 2 | 2 | 3 | 2 | 3 | 2 | 14 | No |
| 26 | High-Fidelity Conversion via Closed-Loop CKA Distillation | 4 | 3 | 3 | 4 | 1 | 1 | 16 | No |

Selection balance: the ten selected records cover conversion calibration and temporal dynamics (1, 3, 10), prediction-aware quantization (6), critical efficiency auditing (2, 5, 8), graph/spike representation (4), visual decision-mechanism diagnosis (7), and decision-plus-cost evaluation in vision (9). Ranking alone did not determine the final set: items 11, 12, 13, and 19 have strong scores but duplicate patterns already represented by selected papers, while the final set provides broader narrative coverage.

## Frozen set of exactly 10 comparators

### 1. Temporal Misalignment in ANN-SNN Conversion and its Mitigation via Probabilistic Spiking Neurons

- **Authors/year:** Velibor Bojkovic, Xiaofeng Wu, Bin Gu, 2025.
- **Venue:** 42nd International Conference on Machine Learning (ICML 2025), PMLR 267:4791–4822.
- **URL/DOI:** https://proceedings.mlr.press/v267/bojkovic25a.html
- **Venue evidence:** PMLR proceedings page with editors, volume, pages, and OpenReview link.
- **Relevance rationale:** Identifies event rearrangement and temporal misalignment as a conversion phenomenon, directly adjacent to our matched-endpoint codeword test. It supplies a precedent for naming a mechanism without reducing it to a new architecture.
- **Recency rationale:** Published in 2025, one year before the intended 2026 submission.
- **Access notes:** Metadata and full abstract verified on PMLR; full PDF linked from the proceedings page. Used for narrative structure only.
- **Writing/narrative pattern to study:** Observation-first opening (`identify a phenomenon`) → biologically plausible mitigation → theoretical and empirical validation. Our paper borrows the observation-first ordering, while retaining its audit framing and avoiding a mitigation claim.

### 2. Inference-Scale Complexity in ANN-SNN Conversion for High-Performance and Low-Power Applications

- **Authors/year:** Tong Bu, Maohua Li, Zhaofei Yu, 2025.
- **Venue:** CVPR 2025, pp. 24387–24397.
- **URL/DOI:** https://openaccess.thecvf.com/content/CVPR2025/html/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.html ; doi:10.1109/CVPR52734.2025.02271
- **Venue evidence:** CVF Open Access page; DOI cross-checked against DBLP.
- **Relevance rationale:** Explicitly separates conversion cost from training cost and examines energy and deployment implications, complementing our warning that local sparsity and event counts do not determine whole-network cost.
- **Recency rationale:** 2025 main-conference paper.
- **Access notes:** Complete abstract and bibliographic record verified; full paper available from CVF.
- **Writing/narrative pattern to study:** Name a practical assumption, show why common accounting is incomplete, then present a framework and broad task validation. Our revision uses the sharper distinction between reported proxy and complete cost in the activity section.

### 3. Differential Coding for Training-Free ANN-to-SNN Conversion

- **Authors/year:** Zihan Huang, Wei Fang, Tong Bu, Peng Xue, Zecheng Hao, Wenxuan Liu, Yuanhong Tang, Zhaofei Yu, Tiejun Huang, 2025.
- **Venue:** ICML 2025, PMLR 267:25235–25264.
- **URL/DOI:** https://proceedings.mlr.press/v267/huang25i.html
- **Venue evidence:** PMLR proceedings page.
- **Relevance rationale:** The DCGS event unit is the source family for the frozen graph adapters audited in the manuscript; its differential representation is essential to interpreting signed events and incomplete prefixes.
- **Recency rationale:** 2025 main-conference paper.
- **Access notes:** PMLR metadata and abstract verified; full PDF linked. No new technical claim is imported.
- **Writing/narrative pattern to study:** Concise motivation from repeated transmission redundancy, then representation change, then conversion fidelity. Our method description already follows this lineage; revision tightens the separation between source method and diagnostic adaptation.

### 4. A Graph is Worth 1-bit Spikes: When Graph Contrastive Learning Meets Spiking Neural Networks

- **Authors/year:** Jintang Li, Huizhe Zhang, Ruofan Wu, Zulun Zhu, Baokun Wang, Changhua Meng, Zibin Zheng, Liang Chen, 2024.
- **Venue:** ICLR 2024 (poster), pp. 39351–39374.
- **URL/DOI:** https://openreview.net/forum?id=LnLySuf1vp
- **Venue evidence:** OpenReview record marked ICLR 2024 poster with publication date; page range cross-checked in indexing results.
- **Relevance rationale:** Closest published intersection of graph representation learning and sparse spiking codes. It helps position our graph-to-spike audit relative to learned binary graph representations.
- **Recency rationale:** 2024 top-conference paper.
- **Access notes:** Full abstract, authors, TL;DR, and code link verified on OpenReview; PDF accessible.
- **Writing/narrative pattern to study:** Start from an overlooked resource burden, introduce a representation, pair a compact expressiveness argument with efficiency evidence. Our related-work grouping follows the same capability-first organization.

### 5. Are Conventional SNNs Really Efficient? A Perspective from Network Quantization

- **Authors/year:** Guobin Shen, Dongcheng Zhao, Tenglong Li, Jindong Li, Yi Zeng, 2024.
- **Venue:** CVPR 2024, pp. 27538–27547.
- **URL/DOI:** https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.html
- **Venue evidence:** CVF Open Access page and CVPR virtual poster record.
- **Relevance rationale:** A critical audit of a widely accepted efficiency claim. It is the closest narrative analogue to our question of whether a conventional local fidelity proxy really establishes decision fidelity.
- **Recency rationale:** 2024 main-conference paper.
- **Access notes:** Full abstract and metadata verified; PDF accessible from CVF.
- **Writing/narrative pattern to study:** Interrogative title, fair-comparison framing, unified perspective, revised resource metric. We borrow only the direct, question-led framing; our title keeps the scope explicit and avoids an rhetorical question.

### 6. PD-Quant: Post-Training Quantization Based on Prediction Difference Metric

- **Authors/year:** Jiawei Liu, Lin Niu, Zhihang Yuan, Dawei Yang, Xinggang Wang, Wenyu Liu, 2023.
- **Venue:** CVPR 2023, pp. 24427–24437.
- **URL/DOI:** https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.html
- **Venue evidence:** CVF Open Access page; exact title, authors, pages verified.
- **Relevance rationale:** Explicitly argues that local reconstruction can produce suboptimal quantization parameters and uses prediction differences. It is the most important conceptual precedent for our local-versus-decision distinction, while our paper changes no quantization parameter.
- **Recency rationale:** 2023 is three years before the intended 2026 submission; retained because of exceptional topical fit.
- **Access notes:** Abstract and full PDF verified on CVF.
- **Writing/narrative pattern to study:** State the local metric's limitation in one sentence, then introduce a decision-aware alternative. Our related work must state clearly that PD-Quant optimizes parameters whereas this paper audits invariants of fixed conversions.

### 7. Comparing the Decision-Making Mechanisms by Transformers and CNNs via Explanation Methods

- **Authors/year:** Mingqi Jiang, Saeed Khorram, Li Fuxin, 2024.
- **Venue:** CVPR 2024, pp. 9546–9555 (oral session record).
- **URL/DOI:** https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Comparing_the_Decision-Making_Mechanisms_by_Transformers_and_CNNs_via_Explanation_CVPR_2024_paper.html
- **Venue evidence:** CVF Open Access page and CVPR 2024 oral session page.
- **Relevance rationale:** A mechanism-discovery paper organized around two diagnostic methods and dataset-wide statistics rather than a new backbone. This is the strongest structural analogue for our audit paper.
- **Recency rationale:** 2024 main-conference paper.
- **Access notes:** Abstract, PDF, supplement, and oral-session record verified.
- **Writing/narrative pattern to study:** Define two measurable properties, introduce dataset-wide procedures, report consequences of architectural choices, and close by connecting the properties back to model families. Our paper parallelizes this with two information losses and two controlled interventions.

### 8. VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network

- **Authors/year:** Kang You, Ziling Wei, Jing Yan, Boning Zhang, Qinghai Guo, Yaoyu Zhang, Zhezhi He, 2025.
- **Venue:** CVPR 2025, pp. 8796–8805.
- **URL/DOI:** https://openaccess.thecvf.com/content/CVPR2025/html/You_VISTREAM_Improving_Computation_Efficiency_of_Visual_Streaming_Perception_via_Law-of-Charge-Conservation_CVPR_2025_paper.html
- **Venue evidence:** CVF Open Access page with complete author list and abstract.
- **Relevance rationale:** Uses differential encoding and neuron-state structure to make efficiency claims in a visual task, providing a current example of how temporal coding and resource accounting are connected in CVPR.
- **Recency rationale:** 2025 main-conference paper.
- **Access notes:** Abstract and metadata verified; full paper linked.
- **Writing/narrative pattern to study:** Practical bottleneck → unproven assumption → named principle → encoding mechanism → efficiency evidence. Our cost section borrows the explicit linkage between coding choice and measurable resource proxy.

### 9. Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention

- **Authors/year:** Soikat Hasan Ahmed, Jan Finkbeiner, Emre Neftci, 2025.
- **Venue:** CVPR 2025, pp. 13970–13979.
- **URL/DOI:** https://openaccess.thecvf.com/content/CVPR2025/html/Ahmed_Efficient_Event-Based_Object_Detection_A_Hybrid_Neural_Network_with_Spatial_CVPR_2025_paper.html ; doi:10.1109/CVPR52734.2025.01304
- **Venue evidence:** CVF Open Access page; DOI cross-checked against DBLP.
- **Relevance rationale:** Shows the current CVPR convention for pairing accuracy, ablations, and actual neuromorphic hardware measurements when claiming efficiency. Our paper lacks chip measurements and can learn from this paper's precise separation of proxy operations, on-chip realization, and deployment claims.
- **Recency rationale:** 2025 main-conference paper.
- **Access notes:** Full abstract, PDF, and metadata verified.
- **Writing/narrative pattern to study:** Present the hybrid design, then let ablations and hardware measurements support separate parts of the claim. Our revision uses a similar division between decision evidence and cost evidence without overstating deployment.

### 10. FTBC: Forward Temporal Bias Correction for Optimizing ANN-SNN Conversion

- **Authors/year:** Xiaofeng Wu, Velibor Bojkovic, Bin Gu, Kun Suo, Kai Zou, 2024.
- **Venue:** ECCV 2024, Proceedings Part LXIX, pp. 155–173.
- **URL/DOI:** https://doi.org/10.1007/978-3-031-72890-7_10
- **Venue evidence:** Springer/ACM proceedings record with exact authors, pages, and DOI.
- **Relevance rationale:** Directly relevant to temporal error compensation in ANN-to-SNN conversion and already conceptually connected to our FTBC-related historical analysis and calibration controls.
- **Recency rationale:** 2024 main-conference paper.
- **Access notes:** Springer/ACM metadata and abstract verified; preprint PDF accessible. Section-level full text was not separately extracted in this run, hence access score 1.
- **Writing/narrative pattern to study:** Theoretical expectation statement → lightweight forward correction → broad empirical validation. We borrow the compact linkage between theory and intervention, while preserving our narrower scope.

## Scope statement

These papers are comparators for writing and narrative structure. Their results, claims, and experimental conditions are not used as evidence for any scientific statement in the submitted manuscript, and no bibliography entry is added solely because a paper appears in this table.
