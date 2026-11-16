# Candidate Search Log (S3)

Date: 2026-11-16.

## Scope and queries

The submitted manuscript is a mechanism-audit paper at the intersection of ANN-to-SNN conversion, graph neural networks, finite-event coding, temporal state, decision fidelity, and inference cost. The manuscript-facing search inputs were:

- field: computer vision / machine learning, efficient and trustworthy neural inference;
- subfield: ANN-to-SNN conversion and spiking neural networks;
- core problem: whether local reconstruction metrics determine downstream graph decisions and activity under finite-event approximations;
- task/domain: visual superpixel graph classification, with classical node-classification context;
- method family: frozen graph conversion, differential coding, calibration, residual coding, temporal/event analysis;
- target venue style: CVPR/ICCV/ECCV/NeurIPS/ICML/ICLR main-conference papers;
- keywords: `ANN-SNN conversion`, `low latency SNN`, `temporal misalignment`, `spiking graph neural network`, `graph-to-spike`, `event timing`, `prediction difference quantization`, `decision mechanism analysis`, `bit budget`, `inference complexity`, `superpixel graph`, `hybrid SNN-ANN`, `differential encoding`;
- exclusions: event-camera application papers without architectural or evaluation relevance; neuromorphic hardware releases without a scientific evaluation question; medical/neuroscience spike analysis; unrelated graph-generation or quantum-GNN work; workshop reports when a main-conference record on the same topic was available.

Searches were run through the available `web_search` interface. Source preference was: CVF Open Access, PMLR, OpenReview, ACM/Springer proceedings, then author repositories or DBLP only to resolve bibliographic metadata. The protocol's SCI/JCR fields are journal-specific and inapplicable to a conference comparator set; they were replaced with explicit peer-reviewed top-venue evidence, and no impact factor or quartile was invented.

## Candidate pool (26)

| # | Candidate | Year / venue | Source URL | Screen result |
|---:|---|---|---|---|
| 1 | Temporal Misalignment in ANN-SNN Conversion and its Mitigation via Probabilistic Spiking Neurons | 2025 / ICML | https://proceedings.mlr.press/v267/bojkovic25a.html | Final 10. Direct temporal-state precedent. |
| 2 | Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation | 2025 / ICML | https://proceedings.mlr.press/v267/hao25c.html | Candidate. Strong conversion method; narrower fit than final set. |
| 3 | Differential Coding for Training-Free ANN-to-SNN Conversion | 2025 / ICML | https://proceedings.mlr.press/v267/huang25i.html | Final 10. Source method family of the graph adapter. |
| 4 | Inference-Scale Complexity in ANN-SNN Conversion for High-Performance and Low-Power Applications | 2025 / CVPR | https://openaccess.thecvf.com/content/CVPR2025/html/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.html | Final 10. Cost-accounting precedent. |
| 5 | Efficient ANN-Guided Distillation: Aligning Rate-based Features of Spiking Neural Networks through Hybrid Block-wise Replacement | 2025 / CVPR | https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Efficient_ANN-Guided_Distillation_Aligning_Rate-based_Features_of_Spiking_Neural_Networks_CVPR_2025_paper.html | Candidate. Feature-alignment method; less diagnostic. |
| 6 | PD-Quant: Post-Training Quantization Based on Prediction Difference Metric | 2023 / CVPR | https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.html | Final 10. Key local-versus-prediction proxy precedent. |
| 7 | Are Conventional SNNs Really Efficient? A Perspective from Network Quantization | 2024 / CVPR | https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.html | Final 10. Critical cost-audit narrative. |
| 8 | A Hybrid ANN-SNN Architecture for Low-Power and Low-Latency Visual Perception | 2024 / CVPR Workshop | https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Aydin_A_Hybrid_ANN-SNN_Architecture_for_Low-Power_and_Low-Latency_Visual_Perception_CVPRW_2024_paper.html | Excluded: workshop venue and task-specific hybrid model. [Added during verification: the CVF record identifies it as a CVPR 2024 workshop paper.] |
| 9 | Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention | 2025 / CVPR | https://openaccess.thecvf.com/content/CVPR2025/html/Ahmed_Efficient_Event-Based_Object_Detection_A_Hybrid_Neural_Network_with_Spatial_CVPR_2025_paper.html | Final 10. Strong decision-plus-hardware evaluation arc. |
| 10 | VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network | 2025 / CVPR | https://openaccess.thecvf.com/content/CVPR2025/html/You_VISTREAM_Improving_Computation_Efficiency_of_Visual_Streaming_Perception_via_Law-of-Charge-Conservation_CVPR_2025_paper.html | Final 10. Differential encoding and efficiency story. |
| 11 | Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer | 2025 / CVPR | https://www.openaccess.thecvf.com/content/CVPR2025/html/Guo_Spiking_Transformer_Introducing_Accurate_Addition-Only_Spiking_Self-Attention_for_Transformer_CVPR_2025_paper.html | Candidate. Spiking operator design, low mechanistic overlap. |
| 12 | Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients | 2025 / CVPR | https://openaccess.thecvf.com//content/CVPR2025/html/Lun_Towards_Effective_and_Sparse_Adversarial_Attack_on_Spiking_Neural_Networks_CVPR_2025_paper.html | Candidate. Temporal-state intervention, different question. |
| 13 | Brain-Inspired Spiking Neural Networks for Energy-Efficient Object Detection | 2025 / CVPR | https://openaccess.thecvf.com/content/CVPR2025/html/Li_Brain-Inspired_Spiking_Neural_Networks_for_Energy-Efficient_Object_Detection_CVPR_2025_paper.html | Candidate. SNN detector; weak fit to conversion evaluation. |
| 14 | Comparing the Decision-Making Mechanisms by Transformers and CNNs via Explanation Methods | 2024 / CVPR | https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Comparing_the_Decision-Making_Mechanisms_by_Transformers_and_CNNs_via_Explanation_CVPR_2024_paper.html | Final 10. Dataset-wide diagnostic design and mechanism-first narrative. |
| 15 | A Graph is Worth 1-bit Spikes: When Graph Contrastive Learning Meets Spiking Neural Networks | 2024 / ICLR | https://openreview.net/forum?id=LnLySuf1vp | Final 10. Graph-plus-spike representation and theory/efficiency balance. |
| 16 | Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips | 2024 / ICLR | https://openreview.net/forum?id=1SIBN5Xyw7 | Candidate. Influential architecture paper; method focus differs. |
| 17 | Neuromorphic Lip-Reading with Signed Spiking Gated Recurrent Units | 2024 / CVPR Workshop | https://openaccess.thecvf.com/content/CVPR2024W/EVW/html/Dampfhoffer_Neuromorphic_Lip-Reading_with_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.html | Excluded: workshop and application-specific signed spiking model. |
| 18 | FTBC: Forward Temporal Bias Correction for Optimizing ANN-SNN Conversion | 2024 / ECCV | https://doi.org/10.1007/978-3-031-72890-7_10 | Final 10. Direct temporal-bias correction precedent. |
| 19 | Threaten Spiking Neural Networks through Combining Rate and Temporal Information | 2024 / ICLR | https://openreview.net/forum?id=xv8iGxENyI | Candidate. Rate-versus-temporal analysis, but adversarial setting. Metadata verified only through secondary indexing. |
| 20 | High-Fidelity ANN-to-SNN Conversion via Closed-Loop CKA Distillation | 2026 / ICML | Local bibliography entry; no verified public source URL resolved in this session | Excluded from final set: title/year appear in the local bibliography, but a unique public source URL could not be verified. It is retained only as a screened candidate, never as a comparator. |
| 21 | A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration | 2021 / ICML | https://proceedings.mlr.press/v139/li21d.html | Candidate. Foundational calibration; outside recency window. |
| 22 | Data Driven Threshold and Potential Initialization for Spiking Neural Networks | 2024 / AISTATS | https://proceedings.mlr.press/v238/bojkovic24a.html | Candidate. Threshold initialization; less close to decision audit. |
| 23 | Reducing ANN-SNN Conversion Error through Residual Membrane Potential | 2023 / AAAI | https://doi.org/10.1609/aaai.v37i1.25071 | Candidate. Residual readout precursor; local bibliography already cited. |
| 24 | Error-Aware Conversion from ANN to SNN via Post-training Parameter Calibration | 2024 / IJCV | https://doi.org/10.1007/s11263-024-02046-2 | Candidate. Journal paper; excluded from conference-style comparator set. |
| 25 | Degree-Quant: Quantization-Aware Training for Graph Neural Networks | 2021 / ICLR | https://openreview.net/forum?id=NSBrFgJAHg | Candidate. Graph quantization precursor; outside recency window. |
| 26 | A2Q: Aggregation-Aware Quantization for Graph Neural Networks | 2023 / ICLR | https://openreview.net/forum?id=7L2mgi0TNEP | Candidate. Graph aggregation-aware quantization, already cited. |

## Notes on verification

- CVF Open Access pages supplied exact titles, author lists, years, venues, abstracts, and often page ranges or DOI records for items 4, 6, 7, 9, 10, 11, 12, 13, and 14.
- PMLR supplied exact proceedings metadata and abstracts for items 1, 2, and 3.
- OpenReview supplied exact titles, author lists, dates, venue status, abstracts, and TL;DRs for items 15 and 16.
- Springer/ACM supplied exact bibliographic metadata and the DOI for item 18.
- The local manuscript bibliography was used only to cross-check items already present in the paper; it did not introduce new scientific claims.
- All scientific claims in the submitted manuscript remain tied to its existing recorded experiments and citations. The comparator set is used solely to study organization, evidence presentation, and prose.

## Targeted verification URLs

During the S4 metadata check, the following authoritative records were queried and matched the frozen table:

- Temporal Misalignment: https://proceedings.mlr.press/v267/bojkovic25a.html
- Differential Coding: https://proceedings.mlr.press/v267/huang25i.html
- Inference-Scale Complexity: https://openaccess.thecvf.com/content/CVPR2025/html/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.html
- PD-Quant: https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.html
- A Graph is Worth 1-bit Spikes: https://openreview.net/forum?id=LnLySuf1vp
- Are Conventional SNNs Really Efficient: https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.html
- Comparing the Decision-Making Mechanisms: https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Comparing_the_Decision-Making_Mechanisms_by_Transformers_and_CNNs_via_Explanation_CVPR_2024_paper.html
- FTBC: https://doi.org/10.1007/978-3-031-72890-7_10
- VISTREAM: https://openaccess.thecvf.com/content/CVPR2025/html/You_VISTREAM_Improving_Computation_Efficiency_of_Visual_Streaming_Perception_via_Law-of-Charge-Conservation_CVPR_2025_paper.html
- Efficient Event-Based Object Detection: https://openaccess.thecvf.com/content/CVPR2025/html/Ahmed_Efficient_Event-Based_Object_Detection_A_Hybrid_Neural_Network_with_Spatial_CVPR_2025_paper.html
