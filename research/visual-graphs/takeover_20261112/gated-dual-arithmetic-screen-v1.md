# Dual-code arithmetic screen: measured prefix counts

## Scope and operation model

This is a lower-bound screen for the **projected-branch cache** proposal, not the literal dense pilot and not a global hardware lower bound. It uses the completed label-free T4 profile's T1/T2/T4 prefix counters on all 45,000 official CIFAR10 training graphs. No fidelity/accuracy result follows from event counts.

For 32-bit arithmetic, use Horowitz's rough 45 nm / 0.9 V values from [“Computing’s Energy Problem”](https://doi.org/10.1109/ISSCC.2014.6757323): 0.9 pJ/add, 3.7 pJ/multiply; 4.6 pJ/MAC is their sum. The event projection estimate is deliberately optimistic: each event is assumed to update a pre-scaled signed-dyadic weight-column bank with one accumulator ADD (0.9 pJ). This bank is not implemented or measured. It excludes source MTH state/comparisons, event decoding and time scaling, bank construction/storage/reads, memory, scatter routing, pooling additions, sigmoid/division, and other special functions. They are unpriced, not zero.

## Counts and formulas

For layer \(i\in\{0,1,2,3\}\), let \(K^h_i(T)\) and \(K^e_i(T)\) be the profile's cumulative `nonzero_scalar_events` at the four node and four edge ReLU sites through prefix T. The proposed projected representation gives each node stream four projection consumers and each edge stream one; remaining-layer fanout is 3/2/1/0. Therefore the optimistic output-accumulator updates are

\[
D_T=70\sum_{i=0}^{3}(3-i)\bigl(4K^h_i(T)+K^e_i(T)\bigr),\qquad E_T=0.9D_T\text{ pJ}.
\]

This uses exact per-layer counts, not a common event-density approximation. No update is charged for layer 3's projections because there is no later gated layer to consume them; readout/event decoding remains outside this term.

| Prefix T | N | directed E | B | node events by layer 0–3 | edge events by layer 0–3 | optimistic projection ADDs | projection pJ/N |
|---:|---:|---:|---:|---|---|---:|---:|
| 1 | 5,294,119 | 42,352,952 | 45,000 | 166,781,446; 179,158,760; 179,747,061; 186,003,104 | 1,615,084,080; 1,525,277,138; 1,490,576,242; 1,631,146,591 | 947,801,290,380 | 161,126.18 |
| 2 | 5,294,119 | 42,352,952 | 45,000 | 322,211,904; 363,423,436; 375,879,816; 397,815,493 | 3,081,331,586; 3,035,964,866; 3,017,525,031; 3,330,925,473 | 1,862,762,938,470 | 316,669.62 |
| 4 | 5,294,119 | 42,352,952 | 45,000 | 503,417,840; 642,326,482; 696,535,283; 742,829,729 | 4,622,226,709; 5,120,942,390; 5,414,201,304; 6,130,860,829 | 3,044,197,329,530 | 517,513.41 |

Every prefix has 45,000 graphs, 352 batches, final batch 72 graphs, all ten sites, finite event counts, and exact positive+negative=nonzero=level-event partitions. Prefix counters are monotone. The T4 run's actual execution is T4; T1/T2 are recorded cumulative prefixes, not three independent runs.

## Cache work and fair ANN boundary

The cached baseline terms are deliberately favorable. Set \(E=8N\), hidden width d=70. The raw-input base projection is \(7,840N\) MACs: four 5→70 node maps PER layer across four layers (5,600N), plus one 1→70 edge map per layer across four layers (280E=2,240N). Pooling the five raw node channels and applying the shared node embedding adds \(350B\) MACs, or \(36,064+1,610(B/N)\) pJ/N. Offline weight composition/storage and raw-feature pooling ADDs are not included.

The known dense gate arithmetic per layer is \((4Ed+Nd)\) ADD plus \(Ed\) MUL after exact fusion of denominator epsilon into its initial accumulator: 4,151 pJ/N/layer. The projected cache can memoize **all first-layer controller work** because its input \((h_0,e_0)\) is static. The optimistic T-tick controller charge is therefore \(4,151(1+3T)\) pJ/N, not \(16,604T\). This assumes first-layer projections, sigmoid, gate products, normalization and edge current can all be reused exactly; it still charges layers 1–3 each tick. MLP linears repeat every tick: add \(14,789T(B/N)\) pJ/N to the candidate. The readout pool and unlisted special-function costs remain unknown.

For a matched optimized ANN lower comparator, fold the first layer through raw inputs but retain hidden residual states in dense form. Its linear work is \(179,270N+3,215B\) MACs, plus the same known gate pattern at one pass (18,872 pJ/N); hence
\[
E_{ANN,known}/N=843,514+14,789(B/N),\quad 0.2E_{ANN,known}/N=168,727.94\text{ pJ/N}
\]
for this profile's \(B/N=0.008499998\). This uses the existing fair-operation boundary; sigmoid/division and memory remain unpriced on both paths.

The candidate lower bound is
\[
E_{cand,known}/N=36,064+1,610(B/N)+4,151(1+3T)+0.9D_T/N+14,789T(B/N).
\]

| Prefix T | base pJ/N | cached gate pJ/N | projection pJ/N | repeated MLP pJ/N | candidate known pJ/N | 20% ANN pJ/N | excess over target | candidate / ANN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36,077.68 | 16,604 | 161,126.18 | 125.71 | 213,933.57 | 168,727.94 | 45,205.63 | 0.2536 |
| 2 | 36,077.68 | 29,057 | 316,669.62 | 251.41 | 382,055.72 | 168,727.94 | 213,327.78 | 0.4529 |
| 4 | 36,077.68 | 53,963 | 517,513.41 | 502.83 | 608,056.92 | 168,727.94 | 439,328.98 | 0.7208 |

The profile therefore rejects **this projected-branch cache arithmetic path** against the 5× target even at T1, under favorable 0.9-pJ ADD projection updates and the static-first-layer cache. T1 alone is a lower bound for longer prefixes: measured event projections are monotone, repeated MLP work grows, and the three dynamic controller layers do not shrink. T2/T4 provide increasing measured lower bounds. Added state, event, bank, pooling, or special-function cost only worsens this comparison. No T8 run or validation run is justified for this same cost path; pursue another representation only if it removes the measured high-fanout edge/node projection updates or changes the matched hardware boundary with evidence. This is not a no-go for other encodings, hardware, or energy models.

For precision, **projection updates alone do not exceed the fivefold allowance at T1**: 161,126.18 < 168,727.94 pJ/N. The fixed-base plus projection subtotal is already 197,203.86 pJ/N, above that allowance before gate/MLP/encoder work. At T2 and T4 the projection-update subtotal alone exceeds the allowance. These are statements about the specified projected-branch representation and matched arithmetic model; they are not lower bounds for every possible event representation or hardware.

## Integrity and limits

`dual_tr1.json` records `complete`, `all_45000_official_train_graphs_label_free_fixed_order`, CPU/GPU runtime metadata (`device=cuda` for the completed profile), `labels_passed=false`, and `validation_or_test_opened=false`. All 10 start/end project source hashes, train processed hashes, teacher checkpoint/metadata/done hashes, node-calibration hash and edge-calibration hash were rechecked against the current files; start/end bounds match. This audit read only profile JSON and hashed files; it did not open graph data or execute inference. The pinned-source identity is recorded by the artifact. Event counts are not an accuracy test, sparse-kernel benchmark, or measured chip energy.
