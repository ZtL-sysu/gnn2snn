# D1 — Theory of GNN→SNN rate conversion (derivation + numerical verification)

This note derives the conversion error of the D1 spiking-GNN conversion and, crucially,
records every place where the first draft of the theory was **wrong and caught by the
numerical check**. All symbols match `protocols/snn_gnn_experiment.py` exactly.

## 0. Setup

Two-layer GCN. `A_hat` is the (fixed) message-passing operator — symmetric normalised
adjacency with self-loops for GCN, but the derivation only needs `A_hat` to be fixed and
non-negative. Write `a_n := A_hat[n, :]`, `rho_n := ||a_n||_1`, `||a_n||_2` its Euclidean norm.

```
ANN:    B1 = X W1 + b1 ;  H1 = relu(B1) ;  B2 = A_hat H1 W2 + b2 ;  pred = argmax_c B2[n,c]
SNN:    layer-1 IF units receive the constant per-step current B1[n,h] + beta, threshold
        theta = 1, reset-by-subtraction. Layer 2 accumulates the instantaneous messages:

            inst_t = A_hat ( s1_t W2 ) + b2,      s1_t in {0,1}^{N x H}
            mem    = sum_t inst_t = A_hat ( count1 W2 ) + T b2 ,   count1 = sum_t s1_t
        =>  B2_hat := mem / T = A_hat ( count1/T ) W2 + b2 .
```

**Exact error identity** (no approximation):

```
        B2_hat - B2 = A_hat ( E1 W2 ),        E1 := count1/T - relu(B1).            (1)
```

## 1. Lemma 1 — the IF unit is an exact floor quantiser on `[0, theta]`

For a constant per-step input `c` with `0 <= c <= theta`, the membrane after `t` steps obeys
`u_t = t c - theta * count_t` with `u_t in [0, theta)`, hence

```
        count_T = floor( T c / theta ),   U := T c - theta*count_T in [0, theta)  ("residual")
```

so for `theta = 1`:  `count/T = c - U/T`, i.e. `E1 = -U/T in (-1/T, 0]`.

Three regimes matter and all three are *exactly* what the code does:

| regime | rate | `E1` |
|---|---|---|
| `c < 0` | never fires, `count = 0` | `0 - relu(c) = 0` — **ReLU is implemented for free** |
| `0 <= c <= theta` | `floor(Tc)/T` | `-U/T`, magnitude `< 1/T` |
| `c > theta` | saturates at 1 spike/step | `1 - c`: **unbounded**, the only escape from the `1/T` bound |

*Verified:* `count == floor(T*c)` and `u in [0,theta)` for 20 000 random `c in [0,1]` at every
`T in {1,2,4,8,16}`. Measured fraction of `relu(B1)` exceeding `theta`: 0.00–0.01 % on the
Planetoid graphs with GCN/mean aggregation, 0.66–1.01 % with sum aggregation.

> **Correction 1 (caught numerically).** The draft theory claimed that symmetric
> normalisation makes rows sum to 1. It does not: `rho` measured 5.75 (Cora) / 4.16 (CiteSeer).
> Row sum 1 is a property of *random-walk* normalisation `D^-1 A`, not of `D^-1/2 A D^-1/2`.

## 2. Lemma 2 — what FTBC actually removes, and `beta* = 1/(2T)`

IF is deterministic, so `E1` is a deterministic function of `B1`; the probabilistic content
enters only through the *population* distribution of the residuals `U`. Under

> **Assumption 1** (phase randomisation, standard in conversion analyses): the residuals
> `U[m,h]` are approximately uniform on `[0,1)` and independent across `(m,h)`.

we get `E[E1] = -1/(2T)`. Adding a per-step bias shifts the floor argument to `T(c+beta)`:

```
        E[E1] = beta - 1/(2T)          =>   beta* = 1/(2T)  is the unique bias zeroing it
        |E1 - E[E1]| <= 1/(2T)         (half-width, same with or without the bias)
```

So **FTBC does not shrink the fluctuation; it removes a systematic, degree-weighted
shrinkage of the decision margin**, and `beta* = 1/(2T)` is the closed-form optimum.

*Verified (Cora, T = 8, 7 seeds):* population mean error `mu_E` moves from `-0.0493` (no bias)
to `-0.0011` (FTBC), a **45x reduction**, and the measured mean margin error moves from
`-0.0413` to `+0.0034`, i.e. the systematic term is not merely reduced but cancelled to within
the sign of the sampling noise. Residual statistics support Assumption 1 in the unbiased arm:
`var(U) = 0.0823` vs the uniform value `1/12 = 0.0833`, and `mean(U) = 0.481` vs `0.5`.

> **Correction 2 (caught numerically).** The first version bounded `|E1| <= 1/T` globally.
> That fails on the few saturated units (`|E1|` up to ~0.83). Those units are deterministic,
> so they are now peeled out and added back as an explicit, exactly computable term
> (`sat_bound` in `theory_verify.py`) instead of inflating the fluctuation half-width.

## 3. Lemma 3 — neighbourhood averaging: worst case vs concentration

For node `n`, winner `c*`, rival `c`, put `DW := W2[:,c*] - W2[:,c]`. From (1),

```
 margin error := (B2_hat - B2)[n,c*] - (B2_hat - B2)[n,c]
               = sum_{m,h} a_n[m] * DW[h] * E1[m,h]
               = rho_n * sum_h mu_h DW[h]                 [SYSTEMATIC, removed by FTBC]
                 + sum_{m,h} a_n[m] * DW[h] * xi[m,h]     [FLUCTUATION, xi zero-mean]
```

* **Worst case:** `|margin error| <= rho_n * ||E1||_max * ||DW||_1`.
* **Concentration (Hoeffding, independent bounded `xi` of half-width `w`):**
  with probability `>= 1 - delta`,
  `|fluctuation| <= w * ||a_n||_2 * ||DW||_2 * sqrt(2 ln(2/delta))`.

The factor `||a_n||_2` (0.47 on Cora) rather than `rho_n` (up to 5.75) is the quantitative
form of "temporal averaging over the neighbourhood": the per-unit quantisation errors are
averaged, not accumulated.

> **Correction 3 (caught numerically).** Using a single scalar `mu` for the systematic term
> reproduced its magnitude but flipped its sign (predicted `+0.044` vs measured `-0.048`).
> The error `E1` is *spatially correlated* through graph homophily, so the per-hidden-unit
> means `mu_h` must be used, and even then the node-level systematic is only a proxy.
> The margin guarantee below is unaffected, because the fluctuation half-width `w` covers it —
> the theorem is a *sufficient* condition and was never violated in any run.

## 4. Theorem — margin recovery guarantee

If, for every rival class `c`,

```
   Delta_n := B2[n,c*] - max_{c != c*} B2[n,c]
            >  |systematic_n| + sat_bound_n + w * ||a_n||_2 * ||DW_c||_2 * sqrt(2 ln(2/delta))
```

then `argmax(B2_hat[n,:]) = argmax(B2[n,:])` with probability `>= 1 - delta`. Equivalently the
required number of timesteps is `T = O( ||a_n||_2 ||DW||_2 / Delta_n )`, and FTBC removes the
additional `rho_n |sum_h DW_h| / (2T)` systematic term (i.e. it halves the T needed when the
systematic dominates).

*Verified (7 seeds):* the predicted guaranteed fraction is below the measured SNN/ANN agreement
in every (dataset, T, variant) cell — e.g. Cora T=8: predicted `>= 0.680`, measured `0.962`
(FTBC); predicted `>= 0.548`, measured `0.886` (no bias). The bound is conservative, as
expected; §7.3 (in `results/theory_tightness.json`) quantifies by how much.

> **Correction 4 (from the architecture study).** The draft predicted sum aggregation
> (`rho` up to 169) would fail. It does not (Cora: 0.767 vs ANN 0.7694). Reason: the margin
> `Delta_n` scales with the same aggregation operator, so the *relative* error is
> `rho`-invariant. What matters is `||a_n||_2 ||DW||_2 / (T Delta_n)`, not `rho` alone.

## 5. Scope conditions (stated as positive boundaries)

* The guarantees hold for message passing with a **fixed** non-negative linear operator
  (GCN, mean/GraphSAGE-mean, sum/GIN verified); learned per-edge attention would have to be
  frozen into a fixed operator first.
* Lemma 1 needs activations in `[0, theta]`; with `theta = 1` and the tested graphs the
  violating mass is <= 1 %. Larger activation scales call for the standard per-layer
  rescaling (the theory then carries an extra factor `1/lambda`).
* Assumption 1 is an approximation; it is *checked* per run (`var(U)` vs `1/12`), not assumed
  silently.
* The reported energy proxies are spike counts and timesteps, not measured joules on
  neuromorphic hardware.

## 6. Mouthpiece check (theory analyses exactly the run experiment)

`theory_verify.py` recomputes `B2_hat` itself and compares its accuracy with the accuracy
`convert()` reports: the gap is `0.00e+00` in every cell. The two Landau-style notebooks can
therefore not drift apart: if a future change makes the theory and the experiment disagree,
this check fails loudly.


<!-- BEGIN GENERATED 7-SEED TABLES -->

## 7. Numerical verification (7 seeds)

`protocols/theory_verify.py`. Every row is a *sufficient* condition, so the requirement is
`measured agreement >= predicted`. The `mouthpiece gap` (this file's own `B2_hat` vs the
accuracy `convert()` reports) is `0.00e+00` in **every** cell: the theory analyses exactly
the shipped computation, and the check fails loudly if they ever drift apart.

| dataset | T | variant | beta | beta*=1/2T | mu_E | w | predicted>= | measured agreement | verdict |
|---|---|---|---|---|---|---|---|---|---|
| cora | 1 | ftbc | 0.500 | 0.5000 | -0.1496 | 0.3754 | 0.0010 | 0.2229 | OK |
| cora | 1 | nobias | 0.000 | 0.5000 | -0.1600 | 0.7781 | 0.0000 | 0.1314 | OK |
| cora | 2 | ftbc | 0.250 | 0.2500 | -0.0287 | 0.3115 | 0.0457 | 0.8319 | OK |
| cora | 2 | nobias | 0.000 | 0.2500 | -0.1548 | 0.3722 | 0.0069 | 0.1730 | OK |
| cora | 4 | ftbc | 0.125 | 0.1250 | -0.0069 | 0.1420 | 0.4111 | 0.9321 | OK |
| cora | 4 | nobias | 0.000 | 0.1250 | -0.0918 | 0.1643 | 0.2143 | 0.6880 | OK |
| cora | 8 | ftbc | 0.062 | 0.0625 | -0.0011 | 0.0655 | 0.6804 | 0.9624 | OK |
| cora | 8 | nobias | 0.000 | 0.0625 | -0.0493 | 0.0744 | 0.5484 | 0.8861 | OK |
| cora | 16 | ftbc | 0.031 | 0.0312 | -0.0003 | 0.0322 | 0.8286 | 0.9851 | OK |
| cora | 16 | nobias | 0.000 | 0.0312 | -0.0252 | 0.0365 | 0.7447 | 0.9457 | OK |
| citeseer | 1 | ftbc | 0.500 | 0.5000 | -0.0988 | 0.3604 | 0.0000 | 0.1724 | OK |
| citeseer | 1 | nobias | 0.000 | 0.5000 | -0.0992 | 0.7064 | 0.0000 | 0.1544 | OK |
| citeseer | 2 | ftbc | 0.250 | 0.2500 | -0.0493 | 0.3546 | 0.0000 | 0.6087 | OK |
| citeseer | 2 | nobias | 0.000 | 0.2500 | -0.0990 | 0.3602 | 0.0000 | 0.1646 | OK |
| citeseer | 4 | ftbc | 0.125 | 0.1250 | -0.0041 | 0.1506 | 0.1051 | 0.9017 | OK |
| citeseer | 4 | nobias | 0.000 | 0.1250 | -0.0742 | 0.1513 | 0.0359 | 0.4663 | OK |
| citeseer | 8 | ftbc | 0.062 | 0.0625 | +0.0002 | 0.0696 | 0.4233 | 0.9151 | OK |
| citeseer | 8 | nobias | 0.000 | 0.0625 | -0.0392 | 0.0716 | 0.2546 | 0.7910 | OK |
| citeseer | 16 | ftbc | 0.031 | 0.0312 | -0.0003 | 0.0333 | 0.6810 | 0.9594 | OK |
| citeseer | 16 | nobias | 0.000 | 0.0312 | -0.0195 | 0.0350 | 0.5580 | 0.8950 | OK |
| pubmed | 1 | ftbc | 0.500 | 0.5000 | -0.1065 | 0.4178 | 0.0056 | 0.6484 | OK |
| pubmed | 1 | nobias | 0.000 | 0.5000 | -0.1189 | 0.8713 | 0.0004 | 0.4294 | OK |
| pubmed | 2 | ftbc | 0.250 | 0.2500 | -0.0319 | 0.3292 | 0.0419 | 0.8569 | OK |
| pubmed | 2 | nobias | 0.000 | 0.2500 | -0.1127 | 0.4136 | 0.0221 | 0.5494 | OK |
| pubmed | 4 | ftbc | 0.125 | 0.1250 | -0.0051 | 0.1472 | 0.3194 | 0.9344 | OK |
| pubmed | 4 | nobias | 0.000 | 0.1250 | -0.0723 | 0.1820 | 0.1503 | 0.7956 | OK |
| pubmed | 8 | ftbc | 0.062 | 0.0625 | -0.0011 | 0.0685 | 0.6680 | 0.9757 | OK |
| pubmed | 8 | nobias | 0.000 | 0.0625 | -0.0387 | 0.0839 | 0.4739 | 0.9210 | OK |
| pubmed | 16 | ftbc | 0.031 | 0.0312 | -0.0002 | 0.0324 | 0.8466 | 0.9881 | OK |
| pubmed | 16 | nobias | 0.000 | 0.0312 | -0.0199 | 0.0393 | 0.7433 | 0.9569 | OK |

**Theorem respected in 30/30 cells.**

Systematic error `mu_E` against the predicted `-1/(2T)` (scaled by the active-unit fraction):

```
cora      T4: nobias -0.0918 -> ftbc -0.0069 | T8: nobias -0.0493 -> ftbc -0.0011 | T16: nobias -0.0252 -> ftbc -0.0003
citeseer  T4: nobias -0.0742 -> ftbc -0.0041 | T8: nobias -0.0392 -> ftbc +0.0002 | T16: nobias -0.0195 -> ftbc -0.0003
pubmed    T4: nobias -0.0723 -> ftbc -0.0051 | T8: nobias -0.0387 -> ftbc -0.0011 | T16: nobias -0.0199 -> ftbc -0.0002
```

### 7.1 Architecture / aggregation generality (7 seeds)

| dataset | aggregation | rho mean/max | `||a_n||_2` | frac(relu(B1)>theta) | ANN | T4 | T8 | T16 |
|---|---|---|---|---|---|---|---|---|
| cora | gcn | 0.93/5.7 | 0.469 | 0.01% | 0.8060 | 0.7911 | 0.8021 | 0.8023 |
| cora | mean | 1.00/1.0 | 0.510 | 0.00% | 0.8093 | 0.8024 | 0.8080 | 0.8079 |
| cora | sum | 4.90/169.0 | 2.105 | 0.67% | 0.7701 | 0.7646 | 0.7667 | 0.7733 |
| citeseer | gcn | 0.96/4.2 | 0.557 | 0.00% | 0.7031 | 0.6839 | 0.6880 | 0.6960 |
| citeseer | mean | 1.00/1.0 | 0.584 | 0.00% | 0.7026 | 0.6857 | 0.6843 | 0.6999 |
| citeseer | sum | 3.77/100.0 | 1.871 | 0.61% | 0.6529 | 0.6313 | 0.6470 | 0.6513 |

### 7.2 The theory yields a better hyper-parameter: beta* = 1/(2T) (7 seeds)

| dataset | ANN | grid T4/T8/T16 | beta* T4/T8/T16 | delta T4/T8/T16 |
|---|---|---|---|---|
| cora | 0.8060 | 0.7911/0.8021/0.8023 | 0.7943/0.8026/0.8046 | +0.0031/+0.0004/+0.0023 |
| citeseer | 0.7031 | 0.6839/0.6880/0.6960 | 0.6929/0.6907/0.7021 | +0.0090/+0.0027/+0.0061 |
| pubmed | 0.7757 | 0.7681/0.7684/0.7729 | 0.7739/0.7740/0.7771 | +0.0057/+0.0056/+0.0043 |

Better in 9/9 cells and it removes the calibration search, so the method becomes parameter-free. Adopted as the default (`--bias analytic`).

<!-- END GENERATED 7-SEED TABLES -->

### 7.3 How loose is the bound, and where (7 seeds, delta = 0.05)

`protocols/theory_tightness.py` decomposes the bound into its three additive terms and compares
each against the realised margin error. `viol` = fraction of test nodes with
`actual > bound` (must stay inside the `delta` budget); `b/a` = median `bound / actual`;
`H/p95` = Hoeffding half-width divided by the realised 95th percentile of the fluctuation.

| cell | certificate | measured | viol | b/a | syst share | fluct share | sat share | H/p95 |
|---|---|---|---|---|---|---|---|---|
| cora ftbc T=4 | 0.411 | 0.854 | 0.001 | 5.73x | 2.9 % | 96.9 % | 0.0 % | 1.78 |
| cora ftbc T=8 | 0.680 | 0.933 | 0.000 | 6.68x | 1.4 % | 98.5 % | 0.0 % | 2.04 |
| cora nobias T=8 | 0.548 | 0.831 | 0.000 | 3.48x | 26.1 % | 73.1 % | 0.0 % | 1.54 |
| citeseer ftbc T=8 | 0.423 | 0.859 | 0.000 | 6.31x | 1.0 % | 99.0 % | 0.0 % | 1.99 |
| pubmed ftbc T=1 | 0.006 | 0.562 | 0.003 | 5.84x | 11.0 % | 84.0 % | 2.3 % | 1.64 |
| pubmed ftbc T=8 | 0.668 | 0.947 | 0.000 | 6.69x | 0.7 % | 99.2 % | 0.0 % | 2.33 |

Three conclusions, all used in Discussion §5.2 of the manuscript:

1. **Sound.** The node-level violation rate is 0.0-0.6 %, uniformly inside the `delta = 0.05`
   budget; the sufficiency claim is never contradicted, only conservative.
2. **The slack is located.** 73-99 % of every bound is the Hoeffding fluctuation term; the
   systematic term is dominant only in the *unbiased* arm (26 % vs 1.4 % on Cora at T=8), which
   is an independent confirmation of Lemma 2 from a different measurement direction.
3. **Structural, not a bug.** Hoeffding applies the worst-case half-width `w = 1/(2T)` to every
   `(m,h)` term as if the sum were adversarial, while the residuals are near-uniform so the
   realised sum is a variance sum: the Hoeffding half-width is 1.5-2.3x the realised 95th
   percentile. A Bernstein/variance bound would tighten the constant by roughly `sqrt(12) ~ 3.5`.
   This is reported as a known limitation rather than tuned away.
