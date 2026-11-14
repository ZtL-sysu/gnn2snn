## 3 Method

### Teachers, graph propagation, and evaluation scopes

We studied frozen four-layer residual GCNs for MNIST and CIFAR10 superpixel graph classification. Node features and positions are concatenated and embedded to 146 channels. For each original directed edge (u	o v), the propagation matrix is
[
A_{vu}=rac{1}{sqrt{max(d_{mathrm{out}}(u),1)max(d_{mathrm{in}}(v),1)}} ,
]
with no added self-loops. Each residual branch computes
[
H_{ell+1}=H_ell+operatorname{ReLU}(A H_ell W_ell^	op+b_ell),
]
where evaluation batch normalization is folded into (W_ell,b_ell). Mean pooling feeds linear layers (146	o73	o36	o10), with ReLU after the two hidden readout layers. Teacher weights remain fixed in every conversion and calibration probe.

The historical operating-point comparison used three independently trained teachers (seeds 41, 42, and 43) on the official train/validation/test partitions (MNIST: 55,000/5,000/10,000; CIFAR10: 45,000/5,000/10,000). The subsequent method screens use teacher 41, 512 fixed training graphs for calibration and 512 fixed validation graphs for comparison. Both official validation and test sets had been exposed in historical work; the new method selection did not load test labels or outputs. Thus the 512-graph comparisons are exploratory screens on one teacher, not independent confirmation. The previously exposed 4,488 validation graphs outside the initial screen are not an external holdout.

### Differential graded-event conversion

At each of the four node branches and two readout activations, the baseline code represents a scalar activation with zero or one of eight signed amplitudes (q_k=	heta 2^{-k}), (k=0,ldots,7), where (	heta) is the channel threshold. At active tick (t), let (v) be the membrane, (C) the cumulative emitted amplitude, (a) the accumulated input, and (b) the accumulated decoded output. With new current (c),
[
m=v+c+a-b,qquad widetilde C=C+b.
]
The selected signed event (s) uses the available quantized levels: positive emission requires (mge 0.75q_k), while negative emission additionally requires (widetilde Cge q_k). Following hard selection,
[
vleftarrow m-s,quad Cleftarrowwidetilde C+s,quad
aleftarrow a+c/t,quad bleftarrow b+s/t.
]
The decoded activation after (T) active ticks is (sum_{t=1}^{T}s_t/t). Raw input and affine bias are injected once, at the first active tick. These are signed graded scalar events, not binary spikes. (T) denotes active ticks.

### Finite calibration and alphabet probes

Calibration freezes teacher weights and channel thresholds, and fits initial membrane offsets (	heta
ho), initialized at zero and clipped to ([-0.5,0.5]). The six site widths (four node sites of 146 channels and two hidden readout sites of 73 and 36 channels) yield 693 offset parameters; a first-site-only fit has 146. The standard screen uses the same 512 training graphs, batch size 32, and 64 Adam updates at learning rates 0.001 and 0.01. The forward pass retains hard event decisions; a ReLU membrane surrogate supplies gradients. These finite updates are a calibration probe, not a guarantee of minimizing the discontinuous deployed objective.

For a site with teacher activation (h) and decoded activation (hat h), the local term is the batch mean of squared error divided by the teacher batch mean-square, with denominator floor (10^{-6}). All-site local fitting averages this normalized term over six sites. First-site message fitting averages the normalized first-site local error and the next-message error
[
mathcal L_{mathrm{msg}}=
rac{operatorname{MSE}(A E W_{mathrm{next}}^	op)}
{operatorname{mean_square}(A H_{mathrm{teacher}}W_{mathrm{next}}^	op)},
qquad E=hat H-H_{mathrm{teacher}},
]
using the same (10^{-6}) denominator floor. Feature-only and structural controls replace (A) by the identity and by a fixed within-graph conjugation (P^{-1}AP), respectively. The exact fixed-operator identity is (Delta Z=A E W_{mathrm{next}}^	op); it isolates the next linear preactivation perturbation, while the residual identity and later nonlinearities remain separate. It is a proxy for downstream effects, not an exact final-logit or classification model.

We also tested a larger first-site alphabet: eight nonnegative, channel-specific levels plus zero, with nearest-neighbor midpoint decisions; other sites retained the baseline quantizer. Sixteen weighted Lloyd iterations fit 1,168 scalar levels. A separate hard-objective probe started from a common local-Lloyd initialization and made one 1,168-coordinate sweep, testing step fractions 1, 0.5, and 0.25 and retaining an incumbent only when the actual hard loss decreased. These alphabet and hard-loss probes use more fitted storage and optimization work than the 146-scalar first-site fit and 64-update Adam screen; they are distinct capacity/optimization diagnostics, not budget-matched comparisons.

### Ideal recovery and fixed-opportunity residual allocation

The ideal-recovery diagnostic continuously adjusts selected site activations to the teacher values and measures how much of the converted model's disagreement with its teacher predictions closes. It is a causal sensitivity diagnostic with effectively unrestricted activation precision, not a realizable event policy.

The finite allocation policy instead considers the residual already available at the first site and first tick. For current preactivation (z), baseline event (s), and (E=s-operatorname{ReLU}(z)), the correction (c_i) at candidate node (i) is the nearest signed level to (-E_i) from zero and the eight geometric amplitudes (	heta/(4cdot2^k)), (k=0,ldots,7). For the fixed next operator, adding that single-node correction changes message-error energy by the exact single-candidate amount
[
g_i=-2leftlangle A E W_{mathrm{next}}^	op,,
A_{:,i}(c_i W_{mathrm{next}}^	op)
ight
angle
-left|A_{:,i}(c_i W_{mathrm{next}}^	op)
ight|_F^2.
]
The local analogue evaluates the same change before propagation. Gains are normalized by graphwise teacher energy, averaged with equal weight over local and message criteria, and ranked to select (lceil n/4
ceil) nodes in each graph. This gives an opportunity budget of at most 146 additional scalar events per graph, because only the first site is supplemented. Local, feature-only, shuffled, and actual-message policies share the candidate set and opportunity budget. Since selected corrections interact through the squared norm, the single-node gains are not additive; ranking is a one-pass heuristic. Corrections update membrane, cumulative emission, and decoded output consistently. T1-fitted base offsets are reused at T2.

An opportunity is an upper bound on eligible extra events, not an equal realized event count or an energy measurement: zero corrections and quantized amplitudes make actual nonzero counts policy-dependent. Accuracy, teacher agreement, and logit error therefore assess different consequences from the local or message losses. The message objective can improve its chosen representation metric without improving the hard argmax classification.

## 4 Experiments and Results

### Low-tick operating points in the historical three-teacher comparison

The historical full-set comparison establishes the operating regime for these teachers, rather than validating the later method screens. Table 1 reports mean test accuracy across seeds 41/42/43 with sample standard deviation across teachers. The official test set had prior exposure, so these results are descriptive context.

| Dataset and conversion | T1 accuracy (%) | T2 accuracy (%) | T4 accuracy (%) | T8 accuracy (%) |
|---|---:|---:|---:|---:|
| MNIST baseline | 71.44 ± 2.80 | 90.16 ± 0.49 | 91.04 ± 0.26 | 91.01 ± 0.26 |
| MNIST node calibration | 78.50 ± 2.10 | 90.08 ± 0.26 | 91.00 ± 0.28 | 91.02 ± 0.25 |
| MNIST graph calibration | 78.70 ± 2.19 | 90.26 ± 0.35 | 91.01 ± 0.28 | 91.03 ± 0.27 |
| CIFAR10 baseline | 46.80 ± 1.22 | 54.07 ± 0.24 | 54.91 ± 0.06 | 54.96 ± 0.04 |
| CIFAR10 node calibration | 49.92 ± 0.14 | 54.18 ± 0.14 | 54.96 ± 0.09 | 54.95 ± 0.08 |
| CIFAR10 graph calibration | 49.75 ± 0.68 | 54.37 ± 0.26 | 54.97 ± 0.05 | 54.97 ± 0.11 |

The corresponding ANN teacher accuracies were 91.02 ± 0.27% on MNIST and 54.94 ± 0.07% on CIFAR10 (three-teacher mean ± sample SD). At T1, calibration variants improved on the baseline in these descriptive averages, while at T2 and above all three variants were close to teacher accuracy. The small and mixed T2 differences do not establish that graph calibration is better than node calibration; nor do the historical full-set values provide an independent estimate of the new policy's performance.

### Ideal sensitivity does not establish finite-event realizability

At T1, continuous ideal recovery at the first site closed 54.08% of the teacher-prediction disagreement gap on MNIST and 23.54% on CIFAR10. Extending the intervention to all four node sites closed 80.42% and 45.18%, respectively, whereas adjusting only the two readout sites closed 8.82% and 14.91%. Jointly correcting all six sites closes the defined gap by construction. This site contrast shows that the diagnostic sensitivity is concentrated in the node pathway for these teachers. Because the intervention directly restores continuous teacher activations, it does not measure whether a finite signed-event correction can realize the same effect.

### Finite calibration and enlarged alphabets

The 512-graph teacher-41 screens yielded mixed outcomes across datasets and optimization configurations. With the standard 64-update first-site fits, MNIST T1 accuracy was 69.73% for baseline, 73.63% for local, 73.83% for message, 72.66% for shuffled-operator, and 73.63% for feature-only fitting. On CIFAR10 the corresponding values were 45.12%, 45.51%, 43.95%, 45.51%, and 45.31%. When fitting all six sites, local/message accuracy was 75.00%/74.41% on MNIST and 47.85%/48.83% on CIFAR10. These are 512-graph screen accuracies, not seed averages; the relative ordering changes with dataset and scope.

Higher learning-rate and optimizer checks did not produce a stable message advantage: for example, first-site local/message/shuffled T1 accuracies were 75.59%/75.39%/76.17% on MNIST and 46.68%/45.70%/46.88% on CIFAR10. In the separate hard-loss probe, first-site local/message/shuffled/identity results were 69.73%/70.70%/70.90%/73.24% on MNIST and 45.12%/44.92%/45.70%/47.46% on CIFAR10. Thus a surrogate-based message fit did not consistently outperform local or counterfactual fits, and lowering the actual hard objective was not equivalent to improving classification. These probes also differ in optimization work and should not be pooled as matched trials.

The enlarged first-site alphabet likewise did not yield a consistent structural ranking. Under the weighted-Lloyd screen, local/message/shuffled/identity accuracies were 82.23%/82.23%/82.03%/82.03% on MNIST and 48.44%/47.46%/47.85%/48.24% on CIFAR10; under the subsequent full-all-site screen they were 83.20%/82.03%/81.84%/81.64% and 47.46%/46.09%/47.27%/49.22%, respectively. The 1,168 learned levels and larger optimization/storage budget show that increasing first-site alphabet flexibility can improve some screen accuracies, but neither the message target nor the actual graph operator consistently led these comparisons. The author-configuration percentile diagnostic also performed poorly (22.46% CIFAR10 and 23.44% MNIST), while the matched eight-level version gave 46.29% and 71.88%; this adverse contrast underscores sensitivity to the coding configuration rather than evidence of a robust method effect.

### Fixed-budget allocation: representation gains and mixed classification effects

For the allocation comparison, all policies had identical candidate nodes and an opportunity budget of at most 146 extra scalar events per graph, but their actual nonzero-event counts could differ. At T1, message-based allocation reduced mean normalized actual next-message MSE on the 512 validation graphs relative to local allocation: from 0.001378 to 0.001280 on MNIST and from 0.002018 to 0.001717 on CIFAR10. These are one-teacher point estimates without seed-level intervals. This is evidence that the selected message proxy can be reduced in this screen; it does not establish a corresponding classification benefit.

| Dataset | T | Policy | Accuracy (%) | Paired difference vs message (pp; descriptive 95% graph interval) |
|---|---:|---|---:|---:|
| MNIST | 1 | baseline | 75.00 | −4.49 [−7.88, −1.11] |
| MNIST | 1 | local allocation | 77.15 | −2.34 [−4.93, 0.25] |
| MNIST | 1 | feature-only allocation | 76.76 | −2.73 [−5.69, 0.22] |
| MNIST | 1 | shuffled-operator allocation | 79.30 | −0.20 [−2.82, 2.43] |
| MNIST | 1 | actual-message allocation | 79.49 | reference |
| MNIST | 2 | baseline | 83.01 | +0.59 [−1.08, 2.26] |
| MNIST | 2 | local allocation | 83.40 | +0.98 [−0.51, 2.46] |
| MNIST | 2 | feature-only allocation | 83.20 | +0.78 [−0.75, 2.31] |
| MNIST | 2 | shuffled-operator allocation | 83.20 | +0.78 [−0.75, 2.31] |
| MNIST | 2 | actual-message allocation | 82.42 | reference |
| CIFAR10 | 1 | baseline | 47.85 | −1.37 [−4.73, 1.99] |
| CIFAR10 | 1 | local allocation | 48.63 | −0.59 [−3.53, 2.36] |
| CIFAR10 | 1 | feature-only allocation | 49.22 | 0.00 [−2.92, 2.92] |
| CIFAR10 | 1 | shuffled-operator allocation | 47.27 | −1.95 [−4.87, 0.96] |
| CIFAR10 | 1 | actual-message allocation | 49.22 | reference |
| CIFAR10 | 2 | baseline | 52.15 | +0.59 [−1.33, 2.50] |
| CIFAR10 | 2 | local allocation | 51.17 | −0.98 [−2.55, 0.60] |
| CIFAR10 | 2 | feature-only allocation | 51.95 | −0.20 [−1.87, 1.47] |
| CIFAR10 | 2 | shuffled-operator allocation | 51.56 | −0.59 [−2.07, 0.90] |
| CIFAR10 | 2 | actual-message allocation | 51.56 | reference |

Accuracy is on the same fixed 512 validation graphs for teacher 41. Paired intervals are descriptive normal intervals for graph sampling, not intervals over teacher seeds, multiple comparisons, or external generalization. At MNIST T1, message allocation exceeded baseline and local allocation by 4.49 and 2.34 percentage points, but shuffled-operator allocation was 0.20 points higher. On CIFAR10 T1, message and feature-only allocation tied at 49.22%, while local allocation reached 48.63%. The gains therefore did not consistently distinguish the actual graph operator from local or counterfactual controls.

At T2, the T1-fitted offsets were reused. Message allocation regressed below the baseline by 0.586 points on MNIST and CIFAR10. On MNIST it was also below local allocation by 0.977 points and below both feature-only and shuffled controls by 0.781 points. On CIFAR10, message allocation matched shuffled allocation and was 0.977 points above local allocation, leaving no consistent cross-dataset pattern. Except for the MNIST T1 message-versus-baseline comparison, the displayed paired intervals cross zero. These exploratory graph-sampling intervals do not establish equivalence, and the isolated interval does not account for teacher-seed variation or multiple comparisons. The T2 decline also shows that the T1-selected corrections did not yield a stable benefit as active ticks increased.

The fixed-operator proxy has a direct representation-level interpretation, but classification passes through residual addition, later nonlinearities, readout, and an argmax. Consequently, lower next-message MSE need not change a decision boundary, while an event correction can alter a logit in ways not summarized by message energy. The hard-loss, accuracy, and teacher-agreement measures are not interchangeable outcomes.

### Diagnostic computational scope

On the RTX 3060 diagnostic implementation with cached batch size 32, median synchronized runtime over 20 randomized-interleaved repeats after five warmups was 0.668 ms/graph for MNIST local allocation and 0.834 ms/graph for message allocation; CIFAR10 medians were 0.806 and 0.991 ms/graph. Policy instrumentation was included, while teacher forward passes and data transfer were excluded. These timings are implementation-specific diagnostics, not optimized deployment bounds or energy measurements; the extra opportunity budget likewise does not establish equal physical cost.