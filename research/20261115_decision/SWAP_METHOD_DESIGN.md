# Evidence-dependent next prototype: learned event reassignment

Status: **implemented and tested; method gate FAILED; family stopped after one alignment audit**. Results: `analysis/learned_swap/summary.md`; source/target audit: `analysis/learned_swap/alignment_audit.json`. The original pre-run design and thresholds below are retained without retrospective relaxation. This is not a reopened loss-coefficient search. The actual coding change is a same-channel swap of two existing nonzero node events; no new event magnitude or support is introduced at the first site. Oracle feasibility passed the pre-run capacity gate, but CIFAR training accuracy did not improve under the full nonincrease constraints, and strict equal-event feasible swaps were absent. This admitted one bounded falsifiable prototype, not a promising-method claim or confirmation.

## Why this is a different mechanism

The quantized first-site graph mean and channelwise amplitude histogram stay fixed. Different assignment to nodes changes propagation through graph columns and later quantization boundaries. We now ask whether a cheap predictor can exploit *available SNN state plus cached graph structure* to select a useful assignment without online teacher/full-network lookahead. This differs from changing MSE weights, first membrane offsets, adding residual events, or changing the amplitude codebook.

## Fixed experiment before any deployment evaluation

- Use the existing first128 training graphs and8-swap candidate banks, no additional calibration labels or fitting data. Source data are `runs/swap_oracle/*/candidates.npz`.
- Baseline is unchanged strong local T1 lr.01 offsets, same vector at all prefixes. Candidate proposals use only existing event values, graph membership/outdegree and an arbitrary per-graph RNG nonce. The nonce is not a predictor input, target, label or teacher signal; random candidate selection is conditioned on the graph/state, not on an official split index's semantics. All controls share identical proposals/nonces.
- A predictor executes once at the first active site using pre-swap emitted values and residual membrane state. It must not evaluate ReLU teacher compensation or any downstream continuous teacher/SNN lookahead. Online inference is one real converted-network pass; all selection overhead is included in timing.
- Cache actual normalized graph column sums `c=1ᵀA` and two-hop readout mass `c2=1ᵀA²` when a graph arrives. Record cold graph-cache cost and warm inference separately; no hidden assumption that arbitrary new graphs are cached free.
- Fit a single fixed ridge regressor (λ=1e-3, input standardization from training only, intercept unpenalized). No restarts/λ/feature subset/threshold tuning. Do not select best epochs or alter the failed old Gate2.

## Same-capacity predictors and available features

Base13 pair features (all from first-site SNN state/current graph): normalized event difference; normalized membrane difference; squared event difference; their product; two individual normalized amplitudes; normalized membrane sum; common outgoing degree; per-graph channel mean event; channel nonzero fraction; channel mean membrane; log node count; normalized event sum. Add channel one-hot146.

Three conditioning slots:
- Non-graph control: event-difference×membrane-square-difference, event-difference×membrane-sum, event-difference×absolute-membrane-difference.
- Real graph: event-difference×column-mass-difference, event-difference×mass-weighted-membrane-difference, event-difference×two-hop-mass-difference.
- Shuffled graph: same three formulas using graph-within-node-permuted masses; inference task graph/labels never change.

All have16+146=162 features, plus intercept, and8 output heads (1304 fitted coefficients). Normalization/storage and added graph-cache bytes are counted separately. The extra capacity is controlled by comparing predictors, not by pretending unchanged693-offset calibration has the same parameter count.

Pre-evaluation literature amendment: TopGQ/DAC2026 §4.2 already proposes TopPIN degree/mean-reciprocal-neighbor-degree grouping. Add a **TopPIN-feature/decision** sixth matched arm: conditioning slots are event-difference×in-degree-difference, event-difference×mean-reciprocal-in-neighbor-degree-difference, and event-difference×degree-weighted-membrane-difference. This is an explicit directed-GCN descriptor adaptation, not reproduction of TopGQ's complete scale-absorption quantizer. It uses the same162features/8outputs/ridge budget. Descriptor novelty alone is not a contribution. See RELATED_WORK.md for primary links and temporal-permutation/AIF overlap limits.

Targets: decision variants predict actual hard-forward T1/T2 all-competitor teacher-margin changes; local variants predict negative first-site local-MSE change for the two score heads. All variants also predict the same six actual cost changes (events/amplitude/branch-delivery proxy at T1/T2). No true training label enters fitting. Six arms: non-graph/local, real-graph/local, non-graph/decision, real-graph/decision, shuffled-graph/decision, TopPIN-feature/decision; unchanged strong local is a reference. Exactly the same sample count, ridge solver and output dimension for every arm.

At deployment choose the maximum positive predicted T1 score among8 pairs, requiring predicted T2 score and all six predicted cost constraints nonnegative/nonpositive as appropriate; otherwise no swap. Prediction is not a proof of true cost feasibility: measure real downstream counts and report violations. The offline oracle feasibility guard is **not** available online. Never replay per-graph oracle choices or precomputed teacher logits as the deployed selector.

## Evaluation, cost and stopping gate

Evaluate fixed rules once on the already-exposed validation512 for each task, same T1/T2 parameters/choice. T4/T8 functional checks only if lowT gate passes. Initial additional wall cap10 minutes, serial RTX3060, no teacher training or test inference. Retain every arm/endpoint and source/config/feature hash.

To admit expansion, full real-graph/decision must improve actual T1 accuracy >=0.5pp on both tasks over strong local and same-capacity non-graph/decision, have nonworse teacher agreement, T2 accuracy loss<=0.5pp, and outperform shuffled-graph/decision and the TopPIN-feature/decision descriptor control. Actual nonzero counts and amplitude/delivery costs must not increase at T1/T2 under the claimed fixed-event budget; otherwise only a transparent Pareto comparison is possible, not the promised budget-matched method. Measure added predictor arithmetic/parameters/cache/cold+warm latency and prohibit hardware-energy claims. Graph-only/local arm isolates structure from task objective.

If the learned selector fails, audit feature/target alignment and single-pass hard-state behavior once, then stop this2-event/8-candidate family. No larger bank, more swaps, new labels, more graph samples, weaker budget guard or loss retuning. A new distinct mechanism requires a separate scientific decision. Passing this prototype still does not establish novelty, multi-teacher stability or external confirmation.
