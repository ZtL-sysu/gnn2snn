# Affine-base hoisting with shared consumer coding — candidate v2

Status: planned representation and synthetic prototype design. No official-data result, novelty verdict, fidelity or energy acceptance. Preserve DERIVATION_PACKAGE.md and its independently reviewed float64 evidence unchanged. This version changes an important placement choice: residual SKIPS retain their own unquantized graded prefix; only matrix consumers see coded inputs. It is not the v1 proposal that quantized the state forwarded through the skip.

## Motivation from actual evidence

The completed branch-coded profile has high repeated projection work. At T1, optimistic prescaled projection ADDs plus the specified fixed-base work already exceed the fivefold allowance; T2/T4 projection ADDs alone do. This rejects that projected-branch representation under the documented model, not every other encoding. See takeover_20261112/dual-train-profile-findings-v1.md and gated-dual-arithmetic-screen-v1.md. A candidate must remove measured branch-to-many-layer fanout, rather than tune a stale-cache tolerance.

DCGS's paper Algorithm2 and generic non-ReLU source already establish input coding before linear consumers. Its CNN source uses ReLU-unit replacement. Neither generic pre-linear coding, common-subexpression sharing nor signed/dyadic coding is a new claim here. The candidate hypothesis is whether hoisting the graph's known low-dimensional static embedding base out of every coded matrix input is useful and distinguishable after fair controls. Primary-source screen: takeover_20261112/centered-residual-prior-art-v1.md.

## Three matched synthetic/reference modes

Use the same frozen equal-width identity-residual GatedGCN weights, eval BN, full graph, sequential layer order and analog graph-level MLP head in ALL three modes. No labels or teacher hidden states enter converted forward. The analog head is explicit: this is a graded/event-coded graph backbone with analog controller/readout, not a binary fully spiking network. Its complete arithmetic must be counted; no method advantage can be attributed to a different head policy.

1. **Generic shared consumer code:** each layer's node/edge raw prefix is coded by the author's signed identity MTH_diff_line; one decoded node stream is shared by A/B/D/E, one edge stream by C. Eight coded sites for four layers. Native sigmoid, normalized messages, BN and ReLU implement graded nonlinear prefixes. This is a source-inspired graph reference with the declared common calibration/head, not an original full-model DCGS reproduction.
2. **First layer analog:** same reference, but leave the first graph layer's inputs uncoded. Code raw node/edge prefixes only before layers1–3 (six sites). This isolates the effect of exact first-layer computation and a reduced encoder count.
3. **Affine-base hoisted:** write h_l=h0+q_l and e_l=e0+r_l. Code the nonnegative cumulative raw residual q_l/r_l only before layers1–3 using the unchanged author ReLU MTH_diff. Feed h0+decoded(q_l), e0+decoded(r_l) to matrix consumers. Six sites. The raw residual skip accumulates its own current ReLU updates; it is not replaced by the decoded consumer stream. The first graph layer is analog as in mode2.

Raw inputs, embeddings and the first graph layer are computed by this model's own operators; no full teacher forward is called to initialize gates, branches or intermediate states. Coding thresholds will require a new declared calibration experiment if this candidate advances; synthetic checks can use fixed positive vectors without official data.

## Real-arithmetic recurrences

Let U_l(t),V_l(t) be the RAW graded node/edge prefixes before layer l in generic modes. Let C_l be that layer's signed identity source decoder. When coded, matrix inputs are bar U_l=C_l[U_l] and bar V_l=C_l[V_l]; otherwise they are U_l,V_l. The raw skip update is

\[
U_{l+1}=U_l+\mathrm{ReLU}(J_l(\bar U_l,\bar V_l)),\qquad
V_{l+1}=V_l+\mathrm{ReLU}(K_l(\bar U_l,\bar V_l)).
\]

It is NOT bar U_l plus the branch. Preserving the raw skip is necessary for the stated reference dataflow.

For the proposed base-hoisted mode, q0=r0=0 and

\[
\bar U_l=h0+C_l[q_l],\quad\bar V_l=e0+C_l[r_l],
\]
\[
q_{l+1}=q_l+\mathrm{ReLU}(J_l(\bar U_l,\bar V_l)),\quad
r_{l+1}=r_l+\mathrm{ReLU}(K_l(\bar U_l,\bar V_l)).
\]

The q_l/r_l on the right are RAW current prefixes, not decoded C_l[q_l]. Therefore they remain nonnegative because all added branches are nonnegative, even if a floating-point decoder has a tiny signed roundoff error. Each source sees t[a_l(t)-a_l(t-1)] after a zero latch and its output is decoded as sum(event/t), exactly as the existing source-unit integration.

With identity replacement C_l[a]=a, all three modes recover the teacher layer by layer. No quantized-mode equivalence follows. Moving the origin, range, negative-event guard and quantizer placement can improve or worsen errors. Nonidentity skips, changing widths, signed residual updates or training BN need separate derivations.

## Conditional projection representation

For base-hoisted matrix consumers,

\[
W(h0+C_l[q_l](t))+b=Wh0+b+\sum_{s\le t}W s_l(s)/s.
\]

There is one input event stream per layer and modality, shared by the node projections. No separate routing of every previous branch is required. Bases can be composed through the 5/1-wide raw embeddings, with real per-graph application, storage and initialization costs. First-layer controllers can be cached only in modes2/3; mode1's encoded embedding inputs can change with time. The reference prototype still executes dense linears and cannot demonstrate efficiency by itself. A sparse/cache implementation, event-bank format, actual events and full arithmetic ledger are later gates.

## Synthetic acceptance before official data

Use the pinned source MTH_diff and MTH_diff_line without editing author files. Verify zero latch, reset, prefix T1/T2/T4/T8, source call counts, site/level partitions, input label and invalid-threshold rejection. With identity replacement, all three modes must match the actual teacher on every tick after disabling its forward method as an oracle. A zero-output fake encoder must show that residual skips retain raw h0 and accumulated raw branches; it must fail if decoded inputs are incorrectly substituted into the skip. Compare shared consumers against identical duplicated source coders at fixed thresholds if needed to check sharing semantics. No official graph inference/calibration is declared by these synthetic tests.

## Decision boundary

This candidate is justified for a bounded synthetic implementation because it targets the measured repeated projection structure. A successful synthetic check is not a novelty, cost or accuracy result. Before any new training/calibration/validation campaign, review the exact baseline placement, proposed mechanism and actual prospective cost boundary. Keep both original datasets, competitive full-data teachers and the separately corrected historical track in the final task scope. Final independent CCFA>=7 remains unachieved.
