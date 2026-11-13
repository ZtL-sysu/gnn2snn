# Centered residual coding: derivation package v1

Status: **COHERENT AS STATED for the real-arithmetic representation identities only.** Coding accuracy, novelty, efficient implementation, and the fivefold target are unproven. This is an internal candidate derivation, not an accepted manuscript theorem or an experiment result.

## Target and invariant object

Explain whether the existing equal-width identity-residual GCN/GatedGCN can represent each layer's accumulated positive residual as one coded state, avoiding separate downstream projection of every earlier branch. The organizing object is the actual decoded residual state and its complete projected update cost; a low spike count is not substituted for end-to-end energy.

Current reference: `takeover_20261112/gated_dual_code_v1.py` separately codes node and edge ReLU branches. A branch then contributes to every later projection, giving downstream fanout3/2/1/0 for four layers. The following changes coding placement, not the author differential neuron. No implementation or official-data result for this centered placement exists yet.

## Assumptions and notation

For layers l=1,...,L, the teacher has identity residuals and nonnegative updates:

\[
h_l=h_{l-1}+\phi(J_l(h_{l-1},e_{l-1})),\quad
e_l=e_{l-1}+\phi(K_l(h_{l-1},e_{l-1})),\quad\phi=\mathrm{ReLU}.
\]

All states have constant width. The embeddings h0/e0 may be signed. For fixed GCN, omit e and let J contain the fixed normalized graph map, affine map and eval BN. For GatedGCN, J/K contain its own current C/D/E logits, sigmoid, normalized message, A/B and eval BN. No teacher hidden states or labels are supplied to a converted forward. Dropout is disabled in eval. These assumptions match the current public-architecture adaptations; arbitrary signed updates, nonidentity skips, changing widths and nonlinear readout need separate treatment.

Let q_l=h_l-h0, r_l=e_l-e0 and q0=r0=0. Let hats denote converted states, and s_l(t) a signed source event at active tick t, after a zero-input source latch. A time step retains the existing sequential layer order; this is not a hardware-latency guarantee.

## Derivation map

Positive additive residuals -> centered nonnegative states -> identity ReLU at cumulative state -> changed coding placement -> source decoded event stream -> exact linear projection-cache identity -> conditional operation accounting. The first three arrows are algebra; inserting an approximate source unit changes the model and requires experiments.

## 1. Exact centered teacher representation

Induction yields

\[
q_l=q_{l-1}+\phi(J_l(h0+q_{l-1},e0+r_{l-1}))\ge0,
\]

and the corresponding expression for r_l. Thus wrapping the right-hand side in an additional ReLU is an identity in real arithmetic. With exact ReLU at every cumulative state, h0+q_l and e0+r_l equal the teacher layer by layer, for any graph and any fixed inputs satisfying the assumptions. This statement does not require the graph-dependent gates to be linear.

The centered coded candidate instead forms, in its own trajectory,

\[
a_l(t)=\hat q_{l-1}(t)+\phi(J_l(h0+\hat q_{l-1}(t),e0+\hat r_{l-1}(t))),
\]

delivers t[a_l(t)-a_l(t-1)] to that layer's source unit, and decodes
\(\hat q_l(t)=\sum_{s\le t}s_l(s)/s\). Edge residual r is analogous. This codes a cumulative residual, whereas the existing dual pilot codes only phi(J_l). It is a different approximation, not an algebraic guarantee of the same SNN outputs.

## 2. Source-state identity and positivity

For the pinned MTH_diff with zero latch/bias, let its current target be a(t), decoded output u(t)=sum(s(t)/t), membrane m(t), and cumulative_out c(t), all after active tick t. In real arithmetic:

\[
m(t)=t[a(t)-u(t)],\qquad c(t)=t u(t),\qquad exp\_in(t)=a(t).
\]

Proof: before emitting at t, the membrane update is
\((t-1)[a(t-1)-u(t-1)]+t[a(t)-a(t-1)]+a(t-1)-u(t-1)=t[a(t)-u(t-1)]\).
Subtracting the event gives the first identity. The cumulative update adds u(t-1) and s(t) to (t-1)u(t-1), giving t u(t). The exp_in update adds a(t)-a(t-1).

The source's negative event of magnitude v is allowed only when pre-event cumulative_out=t u(t-1)>=v. Hence u(t)>=0 by induction. This justifies a_l(t)>=0 for the proposed own-trajectory centered inputs. Floating-point identities need numerical checks and may differ at threshold ties; no bitwise kernel equivalence is inferred. These identities are derived from an established source neuron and are not claimed as a new coding invention.

## 3. Exact projection-cache identity

For a later affine projection with matrix W and bias b,

\[
W(h0+\hat q_{l-1}(t))+b
=Wh0+b+\sum_{s\le t}W s_{l-1}(s)/s.
\]

Thus the cache can start at the projected fixed base and update from ONE preceding-layer event stream. No separate decomposition into every earlier branch is required. For the existing separate-branch code, these would be L(L-1)/2 branch-to-layer streams; for centered code they are L-1, with the crucial caveat that their event counts and quantization errors differ.

The small raw embedding can be composed with each projection offline; applying that composition to each graph remains a real cost. An event with magnitude theta_c/(t*2^k) still requires a multiply-add unless a corresponding prescaled weight column or supported scale/shift mechanism actually exists. Count bank construction, bank storage/reads, signed routing, accumulations and decoder operations. Grant the ANN the same legal folding/fusions. Algebra does not establish a runtime or energy gain.

For the four-layer width-d Gated model, an event-projection estimate for centered streams is

\[
d\sum_{l=2}^4[4K^h_{l-1}(T)+K^e_{l-1}(T)],
\]

where K is the ACTUAL cumulative scalar event count in the new trajectory. The separate-branch reference weights each branch by its number of downstream layers. Do not substitute reference counts for centered counts. Dynamic sigmoid, normalized weighted products, scatter, division, source units and mean-pool/MLP costs remain. Full h/e materialization can sometimes be avoided by a cache; it must not be counted as universally mandatory in a lower bound.

## 4. Error statement and falsification

Define local coding error eta_l(t)=hat q_l(t)-a_l(t). The actual teacher-state error includes eta_l plus the incoming residual error and the change in the nonlinear current J_l. No uniform smaller-error claim follows from moving the quantizer: cumulative residual ranges can be larger, and newly quantized incoming states change future gates. Any global bound must explicitly control these terms and gate normalization, rather than assuming fixed graph propagation.

Falsifiers: signed residual updates invalidate positivity; nonidentity skips invalidate the shared fixed base; source-negative guards or decoder changes can invalidate positivity; cumulative coding may degrade low-T accuracy; denser centered events, banks or gate work may erase the fanout saving. A strong baseline may already use an equivalent representation. All are live possibilities.

## Next evidence gate

First finish the already-running separate-branch full-training profile14938. Independently screen prior art for residual quantizer placement and centered shared bases. A synthetic real-arithmetic/float64 check can test the identities and explicit counterexamples without opening official data. Only if these survive and the measured cost screen leaves a credible route should a new versioned centered converter/calibration experiment be declared. No validation/test selection or new result is authorized by this note alone; experiments still follow the current scope and protocol process.
