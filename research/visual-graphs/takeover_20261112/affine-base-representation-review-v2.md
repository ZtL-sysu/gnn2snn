# Affine-base representation v2: bounded algebra review

## Verdict

**Limited mathematical acceptance in exact real arithmetic under the package assumptions.** The v2 recurrences correctly retain raw residual skips while using decoded residuals only as matrix-consumer inputs. This is a representation identity for the exact-decoder control; it does not establish approximate-code fidelity, implementation correctness, efficiency, hardware latency, or novelty. This review covers the two derivation packages only; it does not inspect the in-flight helper implementation.

## Recurrence and identity checks

For generic/first-analog modes, the stated update

\[
U_{l+1}=U_l+\operatorname{ReLU}(J_l(\bar U_l,\bar V_l))
\]
(and its edge analogue) preserves the teacher skip path. Replacing it by `bar U_l + branch` would change the approximate network and is correctly prohibited. With identity replacement \(C_l[x]=x\), the coded consumer equals the teacher consumer and induction across layers gives exact teacher states. Leaving the first layer analog also preserves exactness there; coding later inputs with the identity replacement preserves it layer by layer.

For affine-base hoisting, \(q_0=r_0=0\), and each update adds a ReLU branch to the **raw** current prefix. Thus \(q_l,r_l\ge0\) by induction in real arithmetic, regardless of signed values in \(h_0,e_0\). A small negative floating-point decoder output does not invalidate that raw-prefix positivity: it can perturb a consumer input \(h_0+C[q]\), but it is not added to the raw skip. The exact-decoder case \(C[q]=q\) recovers the teacher recurrence. No such equality follows for nonidentity coding.

The projection identity is valid for an affine consumer with fixed \(W,b\), when the decoder equals the sum of the source's scaled events through the current tick:

\[
W(h_0+C[q](t))+b=Wh_0+b+\sum_{s\le t}Ws(s)/s.
\]

It justifies one shared node input stream feeding A/B/D/E and one edge stream feeding C, provided each consumer reads the same state and event history. Bias is part of the fixed-base term and must be included once. The source-event statement \(s(t)\) is the current input \(t[a(t)-a(t-1)]\); equality to the decoded target is conditional on the pinned coder's reset/zero-latch, exact arithmetic, decoder, and threshold behavior. Threshold ties and finite-precision event decisions are outside this algebraic identity and require source-level synthetic checks.

## Comparison and scope

V1 centered the residual state that is forwarded through the skip. V2 deliberately changes this: it keeps the raw accumulated residual as the skip and codes only the matrix-consumer input. That distinction is real and avoids feeding a slightly negative decoded residual into the skip. V1's positivity/source argument cannot simply be cited as an implementation proof for v2; v2's raw recurrence supplies the needed positivity argument directly.

The three modes share weights, graph order, eval BN, and analog graph-level MLP head, which is a useful controlled comparison. The first-layer-analog mode is the direct control for affine-base hoisting: both have six coded sites and the same first analog layer. Generic consumer coding has eight sites, so its comparison also changes first-layer treatment and site count; it is a broader reference, not a matched-cost isolation. Threshold calibration must be declared and applied comparably across modes. Keeping the same analog head prevents head-policy differences among these modes, but supports no claim of a fully spiking model or cost advantage over an ANN unless the head and all other arithmetic are included under a fair common boundary.

Base hoisting removes repeated appearance of the fixed embedding from later coded matrix inputs only algebraically. It does not remove per-graph base composition/application, event projection accumulations, source-unit work, dynamic sigmoid/gate normalization, scatter, nonlinearities, dense A/B/D/E/C work, or analog readout. The stated prototype executes dense operations; the identity alone is not a cost saving. Event counts and approximate accuracy must be measured on each mode's own trajectory.

## Claim boundary / remaining checks

The package appropriately frames affine-base hoisting as a hypothesis. Generic pre-linear coding, differential events, signed/dyadic coding, and common-subexpression sharing are not claimed as novel. No novelty conclusion follows here; any distinct contribution would need a prior-art screen and demonstrated benefit of this specific graph embedding-base treatment. V2 also remains specific to equal-width identity-residual, nonnegative ReLU residual updates and eval-mode operators; signed updates, nonidentity skips, changing widths, training BN, or different readout paths need separate derivations.

Before relying on the algebra in an implementation, check per-forward/per-layer state reset and zero latch, exact event-to-decoder behavior including signed events and threshold ties, that the skip uses the raw current \(q,r\) rather than decoded values, and that shared consumers receive exactly the same coded state. Synthetic exact-identity and deliberately corrupted-skip cases are appropriate. These checks would validate implementation behavior, not establish coded fidelity or efficiency.
