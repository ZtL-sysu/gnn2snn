# Stable normalized-gate perturbation bound

## Result

A real-arithmetic bound holds for the GatedGCN target-wise normalized sigmoid aggregate, including its positive denominator epsilon. It avoids a lower bound on individual sigmoid gates and therefore does not blow up when gates are tiny. It is a local perturbation inequality, not a floating-point certificate, accuracy guarantee, efficiency result, or novelty claim.

For an incoming edge set \(I(v)\), channelwise logits \(z_i\), messages \(b_i\), gates \(g_i=\sigma(z_i)>0\), and \(\epsilon>0\), write
\[
m(z,b)=\frac{\sum_{i\in I(v)}g_i b_i}{\sum_{i\in I(v)}g_i+\epsilon}.
\]
This equals an ordinary normalized weighted average after adding one dummy item with fixed weight \(\epsilon\) and message zero. If \(\max_i|z_i'-z_i|\le\eta\), and \(b_i'\) are perturbed messages on the same incoming edges, then
\[
\boxed{|m(z',b')-m(z,b)|\le \max_i|b_i'-b_i|+R(b')\tanh(\eta/2)},
\]
where \(R(b')=\max(\{0\}\cup\{b_i'\})-\min(\{0\}\cup\{b_i'\})\). The maximum message difference is zero for an empty set. This uses the **converted** message range, so the bound does not require teacher hidden states or teacher messages.

## Proof

For finite real \(z\),
\[
\frac{d}{dz}\log\sigma(z)=1-\sigma(z)\in(0,1).
\]
The mean-value theorem gives \(|\log(g_i'/g_i)|\le\eta\), so each real edge weight ratio \(u_i=g_i'/g_i\) lies in \([e^{-\eta},e^{\eta}]\). The dummy weight ratio is exactly one and lies in the same interval.

Let \(p,p'\) be the normalized distributions over real edges plus the dummy item. Then \(p_i'=p_i u_i/\bar u\), where \(\bar u=\sum_i p_i u_i\), so the likelihood ratios \(r_i=p_i'/p_i\) have max/min ratio at most \(e^{2\eta}\). For distributions with likelihood ratio \(r\) of mean one and range ratio at most \(\kappa\), the standard bounded-likelihood-ratio inequality is \(\mathrm{TV}(p,p')\le(\sqrt\kappa-1)/(\sqrt\kappa+1)\). It follows by maximizing the two-endpoint bound \((1-a)(b-1)/(b-a)\) over \(b/a\le\kappa\). Taking \(\kappa=e^{2\eta}\) yields \(\mathrm{TV}(p,p')\le\tanh(\eta/2)\).

Now decompose using the converted messages in the weight-change term:
\[
\sum_i p_i'b_i'-\sum_i p_ib_i
=\sum_i p_i(b_i'-b_i)+\sum_i(p_i'-p_i)b_i'.
\]
The first term is bounded by \(\max_i|b_i'-b_i|\). For the second, the oscillation of \(b'\) over the support, including dummy value zero, times total variation bounds the expectation difference. This gives the boxed inequality. Signed messages are allowed: including zero in the range handles either sign.

For \(\epsilon\downarrow0\), the result remains valid for a nonempty finite incoming set because all real sigmoid gates are strictly positive and the dummy item can be removed. For \(\epsilon=0\) and no incoming edges, the original expression is \(0/0\); it is undefined unless the implementation separately defines the aggregate as zero. With \(\epsilon>0\), an empty incoming set has aggregate zero in both paths and error zero. Tiny positive gates do not weaken the real-arithmetic bound. Floating-point underflow, zeroed gates, denominator rounding, overflow, and accumulation error are outside this proof.

A targeted NumPy float64 check in Conda `dyg-cold` tested 20,000 synthetic cases with signed messages, logits spanning approximately \([-650,650]\), perturbations bounded by \(\eta\), epsilon from zero (nonempty cases) to \(10^{-300}\), and empty sets with positive epsilon. The computed TV and message differences never exceeded their bounds. This is a sanity check, not proof or a floating-point guarantee.

## Applying the bound to raw-skip v2

Use the eval-mode layer from [gated_teacher_model.py](</C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision/takeover_20261112/gated_teacher_model.py>):
\[
z_\ell=C_\ell(e_{\ell-1})+D_\ell(h_{\ell-1})[src]+E_\ell(h_{\ell-1})[dst],\quad
b_\ell=B_\ell(h_{\ell-1}),
\]
\[
J_\ell=A_\ell(h_{\ell-1})+m(z_\ell,b_\ell),
\quad
h_\ell=h_{\ell-1}+\mathrm{ReLU}(BN^h_\ell(J_\ell)),\quad
 e_\ell=e_{\ell-1}+\mathrm{ReLU}(BN^e_\ell(z_\ell)).
\]
The recurrence below is for identity residual skips and fixed eval-mode parameters. It is source-free: it propagates bounds from known weights, calibration/decoder residuals, and converted trajectory values. It never substitutes teacher hidden states into the converted forward.

Let \(\delta^h_\ell,\delta^e_\ell\) be componentwise bounds on the difference between the converted **raw-skip** states and teacher states. Let \(\kappa^h_\ell,\kappa^e_\ell\) bound the local decoder error, decoder minus that converted state’s own raw target, at the consumer input. By triangle inequality, the consumer-input error envelopes are
\[
q^h_\ell=\delta^h_\ell+\kappa^h_\ell,\qquad q^e_\ell=\delta^e_\ell+\kappa^e_\ell.
\]
These are bounds on magnitudes (nonnegative vectors); the decoder residual is not assumed to have a favorable sign. For each edge \(u\to v\) and channel \(k\), weighted absolute projections give the logit bound
\[
\eta_{uv,k}^{\ell}=\sum_j|C_{\ell,kj}|q^e_{\ell-1,uv,j}
+\sum_j|D_{\ell,kj}|q^h_{\ell-1,u,j}
+\sum_j|E_{\ell,kj}|q^h_{\ell-1,v,j}.
\]
Likewise \(d^B_{u,k}=\sum_j|B_{\ell,kj}|q^h_{\ell-1,u,j}\) bounds the message difference. For target \(v\), set \(\eta^\ell_{v,k}=\max_{u\in N^-(v)}\eta^\ell_{uv,k}\), and let \(R^\ell_{v,k}\) be the range of the **converted** \(B_\ell(\bar h_{\ell-1})\) messages on incoming edges plus zero. Then
\[
d^m_{v,k}=\max_{u\in N^-(v)}d^B_{u,k}+R^\ell_{v,k}\tanh(\eta^\ell_{v,k}/2)
\]
bounds the normalized aggregate difference. For an empty incoming set take \(d^m_{v,k}=0\).

The node and edge pre-BN branch-error envelopes are
\[
d^J_\ell=|A_\ell|q^h_{\ell-1}+d^m_\ell,
\qquad d^z_\ell=\eta^\ell.
\]
In eval mode, batch norm is a fixed affine map with coordinate scale \(s=\gamma/\sqrt{var+\epsilon_{BN}}\); the offsets cancel in differences. ReLU is 1-Lipschitz. Thus raw residual branch errors satisfy
\[
\beta^h_\ell=|s^h_\ell|\,d^J_\ell,\qquad
\beta^e_\ell=|s^e_\ell|\,d^z_\ell,
\]
with multiplication elementwise. The raw-skip recurrence is
\[
\delta^h_\ell\le\delta^h_{\ell-1}+\beta^h_\ell,\qquad
\delta^e_\ell\le\delta^e_{\ell-1}+\beta^e_\ell.
\]
For an unencoded exact input (and exact first analog layer, if used), initialize its raw and decoder errors to zero. Otherwise initialize with explicit input/decoder bounds. For layers without batch norm use scale one. Non-residual or shape-changing skips require a different recurrence.

## Limits of a certificate

The recurrence is valid only if its decoder residual bounds \(\kappa\), converted message ranges \(R\), fixed weights, and eval-BN scales are valid for the evaluated forward. It may be loose: absolute-weight propagation ignores cancellation, and the TV term uses a worst-case logit perturbation per incoming edge. Computing and storing per-edge bounds/ranges has runtime and memory cost that must be counted if deployed; an offline certificate also needs stated coverage and arithmetic precision. The proof is in real arithmetic and does not certify source MTH decisions or floating-point evaluation.

A final logit/accuracy certificate additionally needs bounds through mean pooling and the analog MLP head (e.g. weight-norm propagation), then a teacher/candidate class-margin condition. The present local layer bound alone does not prove class agreement. No inference, energy, efficiency, novelty, or accuracy claim is made here.



## Reproducible check artifact

The synthetic check is now reproducible with [check_stable_gate_bound_v1.py](</C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision/takeover_20261112/check_stable_gate_bound_v1.py>) in Conda dyg-cold; its JSON report is [stable_gate_bound_check_v1.json](</C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision/takeover_20261112/stable_gate_bound_check_v1.json>). Fixed seed 20261113, 20,000 randomized cases plus 24 boundary/empty/extremal checks passed with zero failures. Script SHA-256: 6a9ddd3ef94bd8d4bc705f3dcde367ef599b8343ff5d216c15b2548456b80d4d.

