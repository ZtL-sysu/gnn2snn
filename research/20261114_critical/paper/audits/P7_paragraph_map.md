# P7 paragraph map

Paragraph roles and evidence boundaries for the assembled manuscript. Table and figure markers remain in the manuscript for rendering.

## Critical-Site Sensitivity and Realizable Calibration in GCN-to-SNN Conversion

## Abstract

- P1: Low-timestep conversion of graph convolutional networks (GCNs) to spiking neural networks (SNNs) is constrained by temporal coding capacity and graph propagation: a local activation error can be amplified, attenuated, or redirected by the next graph operator
## 1 Introduction

- P1: Graph convolution couples representation error to topology
- P2: The question matters especially under low temporal budgets
- P3: We study this gap in GCN-to-SNN conversion
- P4: The empirical answer is bounded
## 2 Related Work

- P1: ANN-to-SNN conversion uses calibration data and pretrained ANN activations to reduce mismatches introduced by spiking dynamics
- P2: Another line of work adapts quantization to graph structure
## 3 Method

## 3.1 Teachers and graph conversion

- P1: We studied frozen four-layer residual graph convolutional networks (GCNs) for MNIST and CIFAR10 superpixel graph classification
- P2: H_{\ell+1}=H_\ell+\operatorname{ReLU}(A H_\ell W_\ell^{\mathsf T}+b_\ell),
- P3: \]
- P4: where evaluation batch normalization is folded into the weights and biases
- P5: The baseline replaces each activation site with a differential event unit
- P6: m=v+c_t+a-b,\qquad \widetilde C=C+b.
- P7: \]
- P8: A positive event at level $k$ is eligible when $m\geq 0.75\theta 2^{-k}$
- P9: v\leftarrow m-s,\quad C\leftarrow\widetilde C+s,\quad a\leftarrow a+c_t/t,\quad b\leftarrow b+s/t.
- P10: \]
- P11: The decoded activation is $\sum_t s_t/t$
## 3.2 Calibration objectives and controls

- P1: Calibration changes initial membrane offsets while preserving the teacher weights, channel thresholds, and hard event decisions in the forward pass
- P2: The all-site local objective averages six sitewise mean-squared errors, each divided by the batchwise teacher activation mean square with denominator floor $10^{-6}$
- P3: We also compared fixed source-percentile settings
- P4: A separate T1 hard-offset diagnostic uses one sweep over 146 channel offsets and 33 fixed candidates per coordinate on $[-0.5,0.5]$, retaining the incumbent when no hard-loss decrease is found
## 3.3 Continuous causal diagnostic

- P1: To measure conditional sensitivity separately from finite-event realizability, we define an ideal recovery intervention on the current converted signal, not on teacher hidden states
- P2: d_t=t\bigl(\operatorname{ReLU}(u_t)-\operatorname{ReLU}(u_{t-1})\bigr).
- P3: \]
- P4: The weighted prefix $\sum_{t\leq T}d_t/t$ equals $\operatorname{ReLU}(u_T)$
## 3.4 First-site alphabet and residual allocation

- P1: The alphabet-capacity probe changes only the first site
- P2: The residual-allocation candidate likewise changes only the first site, at its first active tick
- P3: -2\left\langle A E W_{\mathrm{next}}^{\mathsf T},\; A_{:,i}(c_iW_{\mathrm{next}}^{\mathsf T})\right\rangle
- P4: -\left\|A_{:,i}(c_iW_{\mathrm{next}}^{\mathsf T})\right\|^2.
- P5: \]
- P6: Local and message gains are normalized by graphwise teacher energy, averaged with equal weight across the graphwise gain terms, and ranked once for top-$k$ selection
## 3.5 Evaluation scope

- P1: Historical full experiments use teachers with seeds 41, 42, and 43
## 4 Experiments and Results

## 4.1 Baseline fidelity and conditional sensitivity

- P1: Across the historical three-teacher test evaluation, the ANN teachers reached 91.0233% mean accuracy on MNIST (sample SD 0.2650%) and 54.9400% on CIFAR10 (0.0700%)
- P2: The ideal causal diagnostic indicates substantial but site-dependent conditional recoverability at $T=1$
## 4.2 Calibration objectives and realizability

- P1: The teacher-41 T1 screen separates first-site targeting from the fitting objective (Table \ref{tab:realizability})
- P2: Hard training loss sometimes increased under the surrogate despite finite gradients and actual parameter changes
- P3: Greater first-site representational flexibility improved MNIST but did not rescue the structural objective
## 4.3 Fixed-opportunity residual allocation

- P1: Residual allocation was evaluated against baseline, random selection, local-gain selection, identity-operator selection, and shuffled-operator selection, all using the same per-graph opportunity rule
- P2: At $T=1$, actual-message allocation reached 79.49% on MNIST and 49.22% on CIFAR10
- P3: At $T=2$, message allocation achieved 88.67% on MNIST and 53.32% on CIFAR10, versus baseline values of 89.26% and 53.91%
- P4: Actual-message residual allocation reduced mean normalized immediate next-message MSE to 0.001280 on MNIST and 0.001717 on CIFAR10, compared with 0.001378 and 0.002018 for local allocation
- P5: The practical expansion rule required at least 0.5 percentage points over local selection and the stronger local accuracy reference on both tasks, superiority to feature/permuted controls, and at most 0.5 points of T2 regression relative to the identical frozen T1 base
## 4.4 Source adaptation and diagnostic cost

- P1: The source-percentile transfer exposed a configuration-sensitive adverse case: the four-level, threshold-scale-4 image preset produced very low $T=1$ accuracy on both tasks and remained weak at $T=2$
- P2: The allocation path also added diagnostic runtime in the measured implementation
## 5 Discussion

- P1: The results separate sensitivity to graph propagation from the ability to realize a useful correction with a finite event code
- P2: The membrane-calibration screen did not establish a consistent advantage for the actual-message objective over local fitting
- P3: The fixed-opportunity allocation experiment provides a direct test of whether using the actual graph mapping adds value beyond selecting residual corrections
- P4: The opportunity budget also limits what can be inferred about resource efficiency
- P5: Capacity and configuration further constrain interpretation
## 6 Conclusion

- P1: In the tested low-timestep GCN-to-SNN setting, critical-site sensitivity did not translate into an established, realizable graph-specific classification advantage

## Structural checks

- Main arc retained: Abstract plus numbered sections 1–6.
- Four table markers retained: historical, realizability, allocation, mechanism_cost.
- Four figure callouts: framework, ideal_sensitivity, allocation_contrasts, mechanism_runtime.
- Actual-message residual allocation is credited with lower immediate message MSE; membrane calibration is not credited with that result.
- Detailed controls are retained in tables/figure; prose centers on baseline and local comparisons.
- Single-teacher exploratory uncertainty, validation/test exposure, source mismatch, nonbinary events, and diagnostic runtime disclosures are retained.
