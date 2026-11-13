# Centered residual derivation: bounded proof/source review

**Scope:** Read-only review of `../DERIVATION_PACKAGE.md`, `check_centered_derivation_v1.py`, `centered_math_v1.json`, and the pinned `MTH_diff` source. This is a real-arithmetic/source-algebra audit only. It does not review a converter implementation, open official data, or infer accuracy, novelty, efficiency, or energy savings.

## Verdict

**Limited mathematical acceptance:** the centered residual identity, the pinned source-state identities, and the affine projection update identity are correct in real arithmetic under the package’s stated assumptions. The source-code evidence is consistent with the pinned commit. The claims must stay scoped to identity residuals, equal widths, nonnegative ReLU branch updates, fixed eval-time affine projections, a zero latch/bias and the unchanged MTH differential source. No fatal algebra error was found in these statements.

The one-line comparison “L−1 streams rather than L(L−1)/2 streams” is an incidence-count statement for a proposed cache representation, not an end-to-end operation or energy result. It only counts event-stream-to-later-layer affine-projection dependencies. Nonlinear dynamic gates, normalization, gathers/scatters, routing, source operations, cache/state management, decoder work, event density, and bank costs remain. Nothing here establishes that the new representation is accurate or computationally beneficial.

## Proof checks

### 1. Centering the residual states

For an identity residual node update

`h_l = h_(l−1) + ReLU(J_l(h_(l−1), e_(l−1)))`,

set `q_l = h_l − h_0`, `q_0=0`. Then

`q_l = q_(l−1) + ReLU(J_l(h_0+q_(l−1), e_0+r_(l−1)))`.

Since `q_0≥0` and each ReLU output is nonnegative, induction gives `q_l≥0`; hence `ReLU(q_(l−1)+ReLU(J_l))` equals the unwrapped sum. The edge proof for `r_l` is identical. This establishes exact real-arithmetic teacher reconstruction layer by layer, even when `J_l/K_l` includes a nonlinear, state-dependent GatedGCN controller: the argument uses only the outer ReLU branch and identity residual, not linearity of the controller.

Scope conditions are material: every layer must have the compatible identity skip and constant state width; the update placed inside the residual must be nonnegative after its final ReLU; BN must use fixed evaluation statistics; dropout must be disabled. A signed update, projection/learned skip, changing width, or other skip form breaks this shared-base argument unless separately rederived. The package correctly excludes these cases.

### 2. Pinned `MTH_diff` identities and positivity

Let the zero latch set `bias=0`, `exp_in=0`, `exp_out=0`; for active tick `t≥1`, feed `x_t=t(a_t−a_(t−1))` with `a_0=0`. Let `u_t=Σ_(s≤t) s_s/s` and let `c_t` be source `cumulative_out`. The source updates `exp_in` by `x_t/t`, `exp_out` by `s_t/t`, adds prior `exp_out` to `cumulative_out` before its negative-event test, then adds `s_t` after the test. Thus, assuming the identities at `t−1`,

- `exp_in_(t−1)=a_(t−1)` and `exp_out_(t−1)=u_(t−1)`;
- pre-event membrane is `(t−1)(a_(t−1)−u_(t−1)) + t(a_t−a_(t−1)) + a_(t−1)−u_(t−1) = t(a_t−u_(t−1))`;
- pre-event `cumulative_out` is `(t−1)u_(t−1)+u_(t−1)=t u_(t−1)`;
- after event `s_t`, `m_t=t(a_t−u_t)` and `c_t=t u_t`, because `u_t=u_(t−1)+s_t/t`;
- `exp_in_t=a_t` and `exp_out_t=u_t` follow by their source accumulators.

The positivity step also checks out for the pinned eight-level code. Its negative threshold mask at level `k` is the product of `1{m≤−0.75λ_k}` and `1{c_pre≥λ_k}`, where `λ_k=θ/2^k`, `θ>0`. As k increases, λ decreases, so both conditions are nested from false to true. The source differences adjacent nested masks (`neg_true`) can select at most one level λ_j; if selected, the eligibility mask ensures `c_pre≥λ_j`, so the negative event magnitude cannot exceed available cumulative output. Positive events only increase it. Therefore `c_t≥0` inductively and `u_t=c_t/t≥0` for `t>0`. This justifies the proposed centered input `qhat_(l−1)(t)+ReLU(J_l(...))≥0` in exact arithmetic, provided its preceding decoded residual is the source’s correctly reset, correctly decoded nonnegative state.

This is a real-arithmetic result, not a floating-point invariant. In finite precision, separately accumulated `cumulative_out` and `Σs_t/t` can differ by rounding; a small negative decoded residual or threshold-tie discrepancy is not ruled out by the derivation. Keep any implementation’s numerical guard explicit and test ties, extreme θ scales, and float32/device behavior. The report’s float64 maximum membrane discrepancy (8.41e−12) is numerical evidence for its synthetic cases, not a formal runtime bound.

### 3. Affine projection update and stream-count comparison

For a fixed affine map `W(·)+b`, linearity gives exactly

`W(h_0+qhat_l(t))+b = Wh_0+b + Σ_(s≤t) W s_l(s)/s`.

So one cached projected base plus the decoded event increments from the immediately preceding layer’s *aggregate centered residual stream* reproduces that projection in real arithmetic. This is per affine map; bias belongs in the base once. Evaluation BN can be folded into a fixed affine map or applied consistently after it. If projection weights, BN statistics, or the source trajectory change between updates, this identity does not certify cache validity.

For a four-layer stack, each earlier branch in the separate-branch representation can feed multiple later-layer projections (per node/edge modality, six earlier-branch-to-later-layer incidences in total: 3+2+1). In the centered representation, each layer exports one aggregate residual stream to the next layer, yielding three such inter-layer incidences per modality. This is a valid representation-level dependency reduction. It is not automatically a reduction in total multiply-adds: the cumulative streams may be denser or have larger ranges; each event can require a weighted multiply and accumulation; and D/E projections may require edge gathers/routing. The formula `dΣ_(l=2..4)[4K^h_(l−1)(T)+K^e_(l−1)(T)]` is valid as an event-to-affine-projection MAC estimate only when `K` counts actual scalar events in the **new centered trajectory**, every listed projection updates the corresponding d outputs, and the accounting consistently treats shared node projections/gathers. It excludes dynamic sigmoid, normalized weighted messages, reductions, divisions, pooling, classifier/readout, MTH, routing, memory, initialization and cache management. The package flags these omitted costs and correctly does not call the formula a full-model lower bound or energy result.

## Code and saved-check coverage

- The JSON’s script and derivation SHA-256 values match the present files. The pinned external source `C:\Users\xhf\AppData\Local\Temp\dcgs-5e0e4c73\neurons\relu_neuron.py` hashes to `ae9018fb…dcc89851`, matching the JSON; the saved commit and git blob IDs also agree.
- The check covers a zero latch and eight active differential inputs across seven positive θ values from `1e−3` to `1e3`; it asserts membrane/cumulative/input/projection identity residuals below `1e−10` and decoded rate minimum ≥`−1e−10`. Saved maxima are 8.41e−12 (membrane), 9.09e−13 (cumulative), 4.55e−13 (input), and 2.27e−13 (projection).
- A synthetic four-layer, width-7, directed GatedGCN-like recurrence with random affine weights, fixed affine BN, sigmoid gates, normalized messages and residual ReLU gives centered-vs-teacher max error `1.78e−15`, with no negative centered states. This exercises the induction in a nonlinear graph controller setting, but not the real teacher implementation, isolated/empty-edge cases, nonlinear pooling/readout, or any centered MTH-coded graph trajectory.
- The projection test is a generic matrix identity with signed source events; it does not implement the proposed graph projection cache or measure its event counts. The two scalar counterexample assertions (signed update and doubled skip) illustrate the boundary but are not structural model-level counterexample tests.
- The check constructs an `MTH_diff` whose threshold is a trainable `Parameter` and does not wrap the synthetic forward in `torch.no_grad()`. This does not alter the tested forward identities, but builds autograd state and makes the check’s memory/runtime irrelevant to inference. No production converter is present here.

## Remaining limits / recommended wording

Use the derivation only as: **“Under equal-width identity residuals with nonnegative ReLU branch updates and an unchanged zero-latch source unit, the cumulative residual has a nonnegative centered representation; a fixed affine projection of that representation can be updated from one aggregate preceding-layer event stream in real arithmetic.”**

Do not infer accuracy preservation, bounded whole-network error, operator-agnostic novelty, fewer total operations, energy savings, or fivefold performance. The check is mathematical/source evidence only. The centered placement has not been implemented or evaluated on any official data; follow the derivation package’s sequencing constraint and wait for the current full-training T4-prefix profile and its cost review before deciding whether to implement it.

**Verdict:** Clear for the stated real-arithmetic identities and their assumptions; limited for source-forward numerical evidence to the recorded synthetic float64 cases; no acceptance of an implementation, benchmark, energy, novelty, or manuscript claim.
