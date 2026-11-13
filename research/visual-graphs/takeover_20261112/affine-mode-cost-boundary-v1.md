# Affine-mode cost boundary review (v1)

## Scope

This is a prospective operation-count boundary for the three affine-base reference modes, not a measured implementation result. It uses the frozen CIFAR10 seed-41 GatedGCN shape: input dimensions 5 node/1 edge, hidden width (d=70), four equal-width residual graph layers, (E=8N), and a 70→35→17→10 analog graph head. It assumes a future shared input-projection accumulator for coded streams. The current Python prototype still evaluates dense operators and does not demonstrate this cache or energy savings.

The formulas below use the same rough 45 nm / 0.9 V arithmetic convention as the existing screen: ADD 0.9 pJ and MUL 3.7 pJ. Keep the predeclared ≥5× arithmetic-energy target unchanged. These figures are terms in that model, not hardware energy, runtime, or a universal lower bound.

## Per-mode cached-projection accounting

Let (K^h_l(T),K^e_l(T)) be cumulative nonzero scalar events through prefix (T) in layer (l)'s coded node and edge input streams. For centered coding use (K^q_l,K^r_l) for the raw cumulative residual streams. Each node stream is consumed once by each of A/B/D/E in that layer; each edge stream is consumed once by C. Under the optimistic pre-scaled event-bank assumption, one event updates one output accumulator per output channel:

\[
P_{\rm gen}(T)=d\sum_{l=0}^{3}(4K^h_l(T)+K^e_l(T)),\\
P_{\rm first}(T)=d\sum_{l=1}^{3}(4K^h_l(T)+K^e_l(T)),\\
P_{\rm center}(T)=d\sum_{l=1}^{3}(4K^q_l(T)+K^r_l(T)).
\]

These are event projection ADDs. There is no prior-layer 3/2/1 fanout factor: each code is consumed only by its own graph layer. The formulas presume that a stream is shared by all four node maps in that layer, that event levels/signs select the right pre-scaled column, and that initialization bias/base values have already been installed. They do not count source encoding, event decode/time scaling, accumulator memory traffic, or bank construction.

| Mode | Coded input sites | One-time affine/base terms before gate/readout | What can be cached exactly | Gate-controller layer evaluations in the declared schedule* |
|---|---:|---|---|---:|
| `generic` | 8 (node+edge at layers 0–3) | (910N) MACs for the node/edge embeddings | Embeddings can be computed once; layer 0's decoded input can still change across ticks, so do not presume its projections, gates, or controller outputs are static | (4T) |
| `first_analog` | 6 (layers 1–3) | (910N+1{,}960N=2{,}870N) MACs | First-layer raw affine projections and the complete first-layer controller/output, since its input is fixed | (1+3T) |
| `centered` | 6 (layers 1–3) | (7{,}840N+350B) MACs, plus 5-channel raw-node pooling; no node/edge base embedding tensors need materialization | First-layer controller/output; all four layers' raw-feature-composed affine maps; pooled node base from raw features | (1+3T) |

The (910N) embedding term for the explicit generic/first-analog paths is (350N+70E): 5→70 node embeddings plus 1→70 edge embeddings. For a fused first layer, A/B/D/E can each compose with the 5→70 node embedding and cost (4·5·70N=1{,}400N); C composes with the 1→70 edge embedding and costs (70E=560N). Their sum is (1{,}960N). The separate (910N) embedding is needed by `first_analog`, which stores full h/e raw prefixes for its later coders and residual skips, so this explicit path is (2{,}870N), before biases and gate arithmetic.

For the optimized `centered` realization, retain only raw residuals q/r: q0=r0=0; q1=ReLU(J0), r1=ReLU(K0). Compose each of the four layers' A/B/D/E maps with the original 5-wide node features and C with the 1-wide edge feature. This costs 1,960N per layer, or 7,840N total, and supplies J0/K0 without materializing h0/e0. At later layers the consumer term W h0+b is already part of the raw-feature-composed affine map; add W decoded(q/r) from the coded event bank. For the final mean head, pool h0+q4 as pool(h0)+pool(q4): pool the five raw node features then apply the shared 5→70 embedding, costing 350B MACs plus the five-channel raw-feature pool. The edge base is not needed by the final readout. This is the concrete low-dimensional fusion path behind the protocol's (7,840N+350B) expression; it must preserve the raw-q/r skip updates and include the raw pools.

An explicit-materialization alternative computes h0/e0 per node/edge (910N MACs) and still performs the four layers' fused raw affine maps (7,840N), giving 8,750N before the readout-base pool. It is a valid higher-cost reference path, not a necessary centered lower term. Count one declared implementation; do not charge both the 350B pooled base and per-node base embedding if reusing a materialized h0 path.

For the optimized centered route above, (350B) is the pooled-base MAC term because h0 is not materialized. If h0 is instead materialized, pool its 70 channels (addition/division work, no 350B embedding MAC) and include the 910N embedding cost in the affine path. Do not charge both routes. The final analog head is (3{,}215B) MACs once per endpoint: (70·35+35·17+17·10). Prefix diagnostic logits at T1/T2/T4 incur three head calls during profiling; separate those costs from an endpoint, which runs the head once.

## Shared arithmetic terms and a comparable boundary

The prior arithmetic screen's known controller term for one GatedGCN layer is

\[
C(N,E,d)=(4Ed+Nd)\;\text{ADD} + Ed\;\text{MUL}.
\]

The ADDs cover the two edge-logit sums, edge numerator/denominator reductions and (A(h)) plus normalized message; epsilon is fused into the initialized denominator accumulator. The MUL covers gate×message. At (E=8N,d=70), this is (4{,}151) pJ/N per layer under the cited arithmetic convention, excluding sigmoid, division, BatchNorm, ReLU and memory. The normalization denominator is node-target/channel shaped, so the required division count is Nd, not Ed. Under the declared conservative cache schedule, generic evaluates four controllers per tick; first-analog and centered reuse layer 0 and evaluate layers 1–3 each tick. These are schedule counts, not universal unavoidable costs: other layers may be memoized if their full input/controller state is identical, with any equality checks, cache storage, reads, invalidations and maintenance charged. Reuse is a measured implementation property, not assumed from the representation name.

The last layer's edge output is unused by the graph head: layer 3 uses its incoming edge state to form gates, but no layer consumes the edge state it produces. An output-only implementation can omit the final edge branch BN/ReLU/residual result. Let S_T count only needed raw skip ADDs:

\[
S_{gen}=dT(4N+3E),\\
S_{first}=d((1+3T)N+(1+2T)E),\\
S_{center}=d(3TN+2TE+B).
\]

Generic keeps node skips at all four layers and edge skips for layers 0–2. First-analog caches layer 0 once, then updates node skips through layer 3 and edge skips through layer 2. Centered stores raw q/r skips: q starts at zero so layer 0 needs no q addition; q node updates remain at layers 1–3, q edge at 1–2, and the final node base/residual join can be done once at graph level (the dB term). Pool reductions/division are separate. The earlier rough d(N+E)-per-layer schedule overcharges a centered layer-0 q skip and a final unused edge skip; this refined schedule assumes the output-only cache realization. If the implementation materializes full hidden states, charge those additions instead.

For reference, the existing optimized ANN linear comparator is (179{,}270N+3{,}215B) MACs: embeddings (910N), four fused first-layer maps (1{,}960N), and three remaining dense graph layers (176{,}400N), plus the head. The arithmetic screen lists 18,872 pJ/N of known non-MAC work: four gate-controller layers (4·4{,}151=16{,}604), plus four node/edge skip sums (4·0.9·70(N+E)/N=2{,}268). The fourth edge skip is output-unused; pruning it gives 18,368 pJ/N (four gate layers plus four node and three edge skip sums). Thus 18,872 is a reproducible screen convention, while 18,368 is the more favorable output-only skip count under the same gate model. The respective 20% comparators differ by 100.8 pJ/N before adding the identical head term. The ANN MAC count already includes the head once.

Under that same optimistic convention, a candidate known-term expression is:

\[
E_{\rm known}/N=4.6B_{\rm MAC}/N+0.9P(T)/N+g_T·4{,}151+0.9S_T/N+14{,}789(B/N),\\
E_{\rm ANN,screen}/N=4.6·179{,}270+18{,}872+14{,}789(B/N),
\]

where B_MAC is the row's one-time base term, P is its projected event ADD count, and g_T=4T for generic or 1+3T for first-analog/centered. S_T is the mode-specific skip count above. This carries the single endpoint head once. It is intentionally incomplete: unpriced terms below can be substantial. Use measured per-layer K(T) from the frozen profile; no event counts or advantage are inferred in this review. A logits-only ANN with its final edge skip pruned replaces 18,872 by 18,368.

## Costs this lower-bound screen leaves open

- **Graded source work.** Each coded site has one zero-latch per batch and (T) active calls. Per scalar, the pinned MTH units maintain membrane, input/output accumulators and perform eight-level threshold decisions; signed line coding can produce both signs. At the full-train protocol's 352 batches, the site call counts are 2,816 warmups + 11,264 active calls for generic, and 2,112 warmups + 8,448 active calls for each six-site mode at T4. These are call counts, not scalar work totals; scalar costs scale with N/E and d. Threshold comparisons, event selection, Σevent/t scaling and state reads/writes must be priced.
- **The 1/t decoder.** The event projection formula assumes each graded update contributes (W s_t/t). Pre-scaling W by theta/2^k alone does not include 1/t. Implementations must either price the per-update scale/shift or count and store time-specific banks (for T4 prefixes, t=1,2,3,4); a power-of-two shift is not silently free under the current energy model.
- **Banks and memory.** At width 70, four node maps plus one edge map make five 70×70 matrices per coded layer. Eight source levels imply up to 196,000 FP32 bank values per layer, before sign duplication or time-specific scaling: 784,000 values (3.14 MB) for generic, 588,000 (2.35 MB) for either six-site mode. This is a storage illustration, not a prescribed hardware layout. It excludes original weights, centered static-base accumulators, source state, bank construction, address/routing logic, cache writes and reads.
- **Gates and reductions.** Sigmoid (Ed), denominator division (Ed), scatter/index reductions, gate denominator updates, edge-message multiplication, and graph pooling remain. The controller formula includes its declared ADD/MUL subset only. Each raw skip and branch ReLU must be charged at its actual reuse rate.
- **BatchNorm.** Node eval BN is affine on A(h)+normalized-message and its per-channel gamma can be folded into A/B outputs (with beta in the affine offset) without changing the gate. Edge BN is applied to z after gates use sigmoid(z); folding its gamma into C/D/E alone would change those gates. Keep pre-BN z for sigmoid and separately apply gamma*z+beta to the edge-branch path, or count the extra edge projections/banks needed for two paths. The final layer edge BN/ReLU/skip is output-unused and can be pruned. If node BN or nonfinal edge BN is executed explicitly after folding its mean into the offset, charge one MUL and one ADD per activation: (N+E)d pairs for both node+edge BN, and Ed pairs for edge BN alone, per layer evaluation. A literal subtract-mean, multiply, add-beta sequence uses one additional ADD per element. Apply the same fold-or-execute choice to ANN and all candidates; do not count edge BN as free by folding it through the gate.
- **Bias/setup and precision.** The MAC table excludes affine biases, BN folding/setup, quantized bank generation, weight storage/precision, accumulator widths, overflow handling, graph-index routing and potentially unfavorable numerical effects. Static first-layer cacheability is a property of the representation, not evidence that the Python profile actually reuses it.

## Decision boundary

The old projected-branch fanout rejection does not decide these three input-coded paths. Conversely, the smaller site count or lower formula alone does not establish a ≥5× result. The fair decision requires the six protocol-fixed mode/scale profiles, their measured per-layer event and row counts, a same-boundary accounting of event projection ADDs, source/(1/t)/bank/memory work, raw skips, controller/gate/sigmoid/division/reductions, BN policy, pooling and one endpoint head call. Keep the profile's extra prefix-head calls separate from endpoint cost. No present result establishes an efficient backend, measured energy or a method advantage.

Evidence inspected: `affine-base-fulltrain-protocol-v1.md`; `../DERIVATION_PACKAGE_v2.md`; `gated_affine_base_v1.py`; `gated_teacher_model.py`; `gated-dual-arithmetic-screen-v1.md`; and `gated-differential-dense-cost-v1.md`. The last two files provide the prior event accounting and the unfused dense prototype formula respectively; their numbers use different execution paths and must not be interchanged.
