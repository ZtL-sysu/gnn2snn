## 3 Method

### 3.1 Teachers and graph conversion

We studied frozen four-layer residual graph convolutional networks (GCNs) for MNIST and CIFAR10 superpixel graph classification. Node features and positions are concatenated and embedded into 146 channels. Each of four node branches computes
\[
H_{\ell+1}=H_\ell+\operatorname{ReLU}(A H_\ell W_\ell^{\mathsf T}+b_\ell),
\]
where evaluation batch normalization is folded into the weights and biases. For every original directed edge $u\to v$, the propagation matrix has $A_{vu}=1/\sqrt{\max(d_{\mathrm{out}}(u),1)\max(d_{\mathrm{in}}(v),1)}$; no additional self-loops are inserted. Mean pooling is followed by linear layers of widths 146, 73, 36, and 10, with ReLU after the two hidden readout layers. The teacher weights remain fixed throughout all conversion and calibration comparisons.

The baseline replaces each activation site with a differential event unit. There are four node sites and two readout sites. At a site with channel threshold $\theta$, each active tick may emit zero or a signed graded amplitude $\theta 2^{-k}$, for $k=0,\ldots,7$. These are scalar amplitudes rather than binary spikes. At active tick $t$, given incoming drive $c_t$, membrane $v$, cumulative event sum $C$, and auxiliary states $a,b$, the pre-event states are
\[
m=v+c_t+a-b,\qquad \widetilde C=C+b.
\]
A positive event at level $k$ is eligible when $m\geq 0.75\theta 2^{-k}$. Negative emission requires both $m\leq-0.75\theta 2^{-k}$ and $\widetilde C\geq\theta 2^{-k}$. The largest eligible magnitude is selected. After hard event selection $s$, the states update as
\[
v\leftarrow m-s,\quad C\leftarrow\widetilde C+s,\quad a\leftarrow a+c_t/t,\quad b\leftarrow b+s/t.
\]
The decoded activation is $\sum_t s_t/t$. Raw inputs and affine biases are introduced once, at the first active tick. We use $T$ for the number of active ticks.

### 3.2 Calibration objectives and controls

Calibration changes initial membrane offsets while preserving the teacher weights, channel thresholds, and hard event decisions in the forward pass. The common histogram thresholds were estimated on the full training split before these screens. An offset is initialized to zero and parameterized as $\theta\rho$, with $\rho$ clipped to $[-0.5,0.5]$. Fitting all six sites uses 693 offsets; first-site-only fitting uses 146. We used the same 512 fixed-index training calibration graphs, batch size 32, and 64 Adam updates for the screening comparisons, with learning rates 0.001 and 0.01 treated as separate conditions. The backward pass uses a ReLU membrane surrogate; it does not replace hard events in the forward computation.

The all-site local objective averages six sitewise mean-squared errors, each divided by the batchwise teacher activation mean square with denominator floor $10^{-6}$. For first-site message calibration, we average the normalized next-message error and the corresponding local loss. If $E$ is the converted-minus-teacher error at that site, the immediate preactivation perturbation under the fixed folded graph convolution is $A E W_{\mathrm{next}}^{\mathsf T}$. Its squared norm is normalized by the mean square of $A H_{\mathrm{teacher}}W_{\mathrm{next}}^{\mathsf T}$. This identity isolates the linear propagation contribution: the residual identity path and subsequent nonlinearities remain separate, so the objective is a local proxy rather than an exact final-classification loss. The feature control substitutes the identity for $A$; the structural control uses a fixed within-graph conjugation $P^{-1}AP$. Local, feature-only, permuted, and actual-operator message policies therefore distinguish local reconstruction from dependence on the actual graph mapping.

We also compared fixed source-percentile settings. The GCN comparator estimates per-channel 0.999 quantiles, floored at $10^{-6}$, from frozen post-ReLU teacher activations on the 512 training graphs. It is an adaptation of the source prescription, not a complete reproduction of its original converter. The original image example uses four levels, threshold scale 4 and 32 ticks. We transfer its level/scale constants to $T=1$ and $T=2$ only as a configuration diagnostic. Its 512-graph threshold estimate also differs from the inherited full-training histogram, so this is not a fully matched reproduction of the source pipeline. A second preset uses eight levels and threshold scale 1 to match the local event alphabet; no validation-based scale search is used.

A separate T1 hard-offset diagnostic uses one sweep over 146 channel offsets and 33 fixed candidates per coordinate on $[-0.5,0.5]$, retaining the incumbent when no hard-loss decrease is found. Its 4,818 candidate evaluations are not budget-matched to 64 Adam updates. This coarse grid can miss small-offset minima and does not establish a global capacity bound.

### 3.3 Continuous causal diagnostic

To measure conditional sensitivity separately from finite-event realizability, we define an ideal recovery intervention on the current converted signal, not on teacher hidden states. For incoming converted increments $c_t$, let $u_t=u_{t-1}+c_t/t$ and emit
\[
d_t=t\bigl(\operatorname{ReLU}(u_t)-\operatorname{ReLU}(u_{t-1})\bigr).
\]
The weighted prefix $\sum_{t\leq T}d_t/t$ equals $\operatorname{ReLU}(u_T)$. Thus, recovery is causal with respect to the current incoming signal and requires no teacher activation overwrite; at other idealized sites, incoming signals may still differ from the teacher. The empty intervention is the uncalibrated conversion, and replacing all six sites was empirically verified to reproduce the folded ANN. Single-site and grouped effects are conditional and nonadditive. We use this continuous construction only as a diagnostic, not as a realizable SNN coding policy.

### 3.4 First-site alphabet and residual allocation

The alphabet-capacity probe changes only the first site. It replaces the positive amplitude ladder with eight nonnegative channel-specific values plus zero, using nearest-neighbor midpoint decisions while retaining signed differential emissions; all later sites retain their original quantizer. Sixteen weighted Lloyd iterations fit 1,168 scalar values. A separate hard-objective refinement starts from the same local-Lloyd solution and makes one 1,168-coordinate sweep, considering step fractions 1, 0.5, and 0.25; an incumbent is retained unless the actual hard loss decreases. This diagnostic has a larger fitting and storage budget than the 64-step Adam calibration and the 146 first-site offsets. “Full-message” refinement minimizes the complete squared message objective, including cross terms, but still modifies only the first site. Separate six-site local and message membrane probes each use 693 offsets.

The residual-allocation candidate likewise changes only the first site, at its first active tick. At the first site, preactivation $z$ and its current hard event $s$ are already available. We compute $E=s-\operatorname{ReLU}(z)$ from this same incoming converted signal; no teacher hidden activation is queried. Computing that ReLU is an additional analog operation. A candidate correction $c$ is the nearest signed quantization of $-E$ from the eight geometric amplitudes $\theta/(4\cdot 2^k)$, $k=0,\ldots,7$, together with zero. For a candidate at node $i$, the exact single-node reduction in squared next-message error is
\[
-2\left\langle A E W_{\mathrm{next}}^{\mathsf T},\; A_{:,i}(c_iW_{\mathrm{next}}^{\mathsf T})\right\rangle
-\left\|A_{:,i}(c_iW_{\mathrm{next}}^{\mathsf T})\right\|^2.
\]
Local and message gains are normalized by graphwise teacher energy, averaged with equal weight across the graphwise gain terms, and ranked once for top-$k$ selection. Candidate interactions make joint gains nonadditive. The selected nodes are the top $\lceil n/4\rceil$ nodes in each graph. Each selected node has up to 146 extra scalar event slots, giving a per-graph opportunity cap of $146\lceil n/4\rceil$ slots. This is an upper bound on extra per-channel events, not an equal-energy constraint: policies can use different numbers of nonzero extras. The extra events update membrane, cumulative output, and decoded output consistently. The same offsets fitted at $T=1$ are used at both $T=1$ and $T=2$.

### 3.5 Evaluation scope

Historical full experiments use teachers with seeds 41, 42, and 43. Official train/validation/test counts are 55,000/5,000/10,000 for MNIST and 45,000/5,000/10,000 for CIFAR10. Their test-set results describe the established baseline and teacher-seed variation; both official validation and test sets had historical exposure. New method screens use teacher 41, 512 training calibration graphs, and a fixed-index validation subset of 512 graphs. Test data were not loaded for selecting the new methods. The resulting comparisons are exploratory, not independent external confirmation; the 4,488 other previously exposed validation graphs do not constitute an independent external set. We report accuracy, agreement with each method's own teacher, and logit mean-squared error where measured. Historical uncertainty is the sample standard deviation across three teacher seeds; new paired intervals describe graph sampling for one teacher only, not seed variation, multiplicity, or external generalization.

## 4 Experiments and Results

### 4.1 Baseline fidelity and conditional sensitivity

Across the historical three-teacher test evaluation, the ANN teachers reached 91.0233% mean accuracy on MNIST (sample SD 0.2650%) and 54.9400% on CIFAR10 (0.0700%). Baseline conversion was strongly timestep-dependent at the shortest prefix: MNIST accuracy was 71.4367% (2.8010%) at $T=1$ and 90.1567% (0.4850%) at $T=2$; CIFAR10 was 46.7967% (1.2240%) and 54.0700% (0.2427%), respectively. At $T=4$ and $T=8$, baseline accuracy approached the corresponding teacher performance on both datasets. All-site activation and graph-logit calibration share the same event code and 693 initial-state offsets; they change the fitting objective rather than the coding scheme. These historical figures are descriptive context from the 10,000 official test graphs per dataset and were not used to select the later methods.

[TABLE:historical]

The ideal causal diagnostic indicates substantial but site-dependent conditional recoverability at $T=1$. Recovering the first site closed 54.08% of the teacher-prediction disagreement gap on MNIST and 23.54% on CIFAR10; recovery at node sites 0–3 closed 80.42% and 45.18%, while head sites 4–5 closed 8.82% and 14.91%, respectively. Replacing all six sites closed 100% of the gap on both datasets, consistent with the folded-ANN verification. These are conditional continuous interventions, averaged over three teachers on historically exposed validation data; unequal intervention group sizes and nonadditivity preclude interpreting them as additive site contributions or finite-event gains.

[TABLE:realizability]

### 4.2 Calibration objectives and realizability

The teacher-41 T1 screen separates first-site targeting from the fitting objective (Table \ref{tab:realizability}). With 146 offsets and learning rate 0.001, first-local/message accuracy was 73.63/73.83\% on MNIST and 45.51/43.95\% on CIFAR10. The corresponding 693-offset all-local/message values were 75.00/74.41\% and 47.85/48.83\%. The higher-rate all-local control reached 78.13\% and 49.41\%. Thus first-site sensitivity did not translate into a consistent advantage from first-site-only fitting or the message term. These are single-teacher validation accuracies, not seed averages.

Hard training loss sometimes increased under the surrogate despite finite gradients and actual parameter changes. A bounded hard-offset search was therefore used as an optimization diagnostic; it did not establish a consistent actual-graph advantage over the feature and permuted controls. Its coarse grid and different update budget limit the interpretation: failure of this search is not a proof that the parameter family has no useful solution.

Greater first-site representational flexibility improved MNIST but did not rescue the structural objective. Local/diagonal-message Lloyd alphabets achieved 82.23/82.23\% on MNIST and 48.44/47.46\% on CIFAR10. After one matched hard-coordinate refinement, local/full-message accuracies were 83.20/82.03\% and 47.46/46.09\%. The full-message objective includes cross terms, but all these alphabet interventions remain restricted to the first site. Their 1,168 fitted values and additional optimization work differ from the offset baselines; they are capacity and objective diagnostics, not evidence of superiority under the same fitting budget.

### 4.3 Fixed-opportunity residual allocation

Residual allocation was evaluated against baseline, random selection, local-gain selection, identity-operator selection, and shuffled-operator selection, all using the same per-graph opportunity rule. The budget provides 146 extra scalar slots per selected node and $146\lceil n/4\rceil$ per graph. Across the 512-graph validation subset, this totals 1,344,222 slots over 9,207 selected nodes for MNIST and 2,225,186 slots over 15,241 selected nodes for CIFAR10. Actual message-policy nonzero extras were 612,232 and 926,039, respectively; thus equal opportunity did not imply equal realized event counts or energy.

At $T=1$, actual-message allocation reached 79.49% on MNIST and 49.22% on CIFAR10. The paired accuracy difference versus baseline was +4.4922 percentage points on MNIST (descriptive graph-sampling interval 1.1087 to 7.8757) and +1.3672 points on CIFAR10 (−1.9932 to 4.7275). Against local allocation, the corresponding differences were +2.3438 points (−0.2472 to 4.9347) and +0.5859 points (−2.3569 to 3.5288). Against identity controls, they were +2.7344 (−0.2243 to 5.6930) and 0.0000 (−2.9183 to 2.9183); against shuffled controls, +0.1953 (−2.4316 to 2.8223) and +1.9531 (−0.9602 to 4.8665). The MNIST comparison with a stronger local-learning-rate condition was +1.3672 points (−1.9046 to 4.6390), while the CIFAR10 comparison was −0.1953 (−3.6440 to 3.2533). All intervals are paired, descriptive intervals over graphs for one teacher and cross zero except the MNIST allocation-versus-baseline comparison; they do not establish seed-level reliability or a general structural benefit.

At $T=2$, the absolute accuracies were 88.67% for message allocation, 89.26% for baseline, 89.65% for local allocation, 89.45% for identity, and 89.45% for shuffled selection on MNIST. On CIFAR10, the corresponding values were 53.32%, 53.91%, 52.34%, 53.13%, and 52.73%. The paired message-minus-baseline difference was −0.5859 points on MNIST (−2.2554 to 1.0836) and −0.5859 on CIFAR10 (−2.5012 to 1.3293); message-minus-local was −0.9766 (−2.4582 to 0.5051) on MNIST and +0.9766 (−0.6011 to 2.5542) on CIFAR10. The corresponding differences versus identity and shuffled selection were −0.7812 and −0.7812 points on MNIST, and +0.1953 and +0.5859 on CIFAR10; each interval crossed zero. The direction changes across datasets and controls, and these single-teacher comparisons do not show robust structural superiority.

[TABLE:allocation]

The actual-graph policy reduced mean normalized next-message MSE to 0.001280 on MNIST and 0.001717 on CIFAR10, compared with 0.001378 and 0.002018 for the local selector. Every policy is evaluated under the true operator for this diagnostic. Improved immediate message reconstruction therefore coexists with a failed joint classification screen; it is not a result of the earlier membrane-fitting comparison.

The practical expansion rule required at least 0.5 percentage points over local selection and the stronger local accuracy reference on both tasks, superiority to feature/permuted controls, and at most 0.5 points of T2 regression relative to the identical frozen T1 base. Neither task passed the joint rule. This is an exploratory selection criterion, not a significance test; the failed screen did not trigger additional teacher or backbone expansion.


### 4.4 Source adaptation and diagnostic cost

The source-percentile transfer exposed a configuration-sensitive adverse case: the four-level, threshold-scale-4 image preset produced very low $T=1$ accuracy on both tasks and remained weak at $T=2$. With the eight-level, scale-1 event-matched preset, CIFAR10 reached 46.289062% at $T=1$ and 55.078125% at $T=2$; MNIST reached 71.875000% and 88.085938%. This supports treating the transferred image configuration as a diagnostic rather than attributing its performance to graph structure. Alphabet probes likewise have a different resource scope from the low-dimensional offset fits: first-site Lloyd fitting stores 1,168 scalars, and the optional hard-objective sweep adds a full coordinate pass; neither is an all-site alphabet intervention. The measured evidence therefore does not isolate a structural gain at matched capacity across these separate probes.

The allocation path also added diagnostic runtime in the measured implementation. On an RTX 3060 with cached batches of 32, median synchronized host time per graph was 0.6678 ms for MNIST local allocation and 0.8343 ms for message allocation; for CIFAR10 it was 0.8063 and 0.9914 ms. These medians used 20 randomized-interleaved repeats after five warmups; repeat variation is retained in raw records, with no population confidence interval. Timing includes policy instrumentation and excludes teacher forward and data transfer. It is neither an energy measurement nor an optimized deployment bound.

[TABLE:mechanism_cost]
