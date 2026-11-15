# Fixed-event reassignment feasibility — pre-run protocol

This is a new **training-only capacity/mechanism diagnostic**, not admission of a deployable Graph-aware method after failed Gate2. No loss/temperature/learning-rate tuning is reopened. Previous goal turn made concrete progress: source TI component implemented,16 endpoints verified and pushed at08642ea. Long-term algorithm discovery remains unachieved.

## Distinct mechanism and graph hypothesis

Swap two **existing nonzero** first-tick events at different nodes of the same graph/channel and equal outgoing degree. This preserves the original amplitude alphabet, every nonzero coordinate, channelwise amplitude multiset, first-site number of events and edge-delivered amplitude multiset. No added residual event or Lloyd amplitude is allowed. Swap state updates must propagate through membrane/cumulative/exp_out, just as real events do.

Graph-specific motivation: a same-channel swap preserves first-site graph mean. Its next normalized message need not have zero graph mean because the two columns of the actual propagation matrix can have different sums. Subsequent quantization gates also differ by neighborhood. Thus node assignment, unlike amplitude histogram alone, could alter graph-level decisions. This is an elementary linear observation, not a new theorem or proof of useful algorithmic benefit.

## Fixed budget and controls

Two tasks, existing teacher41, strong all-position local T1 lr.01 parameter vector from the previous objective run. **First128 training indices** of the already fixed training512, batch32; no validation or test split loading. Eight deterministic graph-ID-seeded candidate pairs per graph, independent of labels/logits/teacher. Identical candidate bank and forward budget for all controls. No weight fitting or new teacher. Source thresholds/offsets remain fixed at all prefixes. Additional wall-time cap5 minutes, serialGPU; preserve failures/no feasible candidates.

Evaluate baseline and all8 swaps at T1/T2, actual hard full-network forward. For each candidate record all-site nonzero events, amplitude sum and residual-branch graph-delivery proxy (each GCN branch's events can enter remaining graph layers), separately from first-site exact conservation. The delivery proxy excludes continuous embedding, normalization/readout/control arithmetic and is not chip energy. No online cost advantage is inferred from the oracle.

Three explicitly separate budget sets:
1. **exact_events**: total nonzero event counts at T1 AND T2 equal baseline per graph, amplitude sum and branch-delivery proxy no larger at each prefix;
2. **nonincrease**: those counts, amplitude sums and branch-delivery proxy all no larger at T1/T2;
3. **first_site_only**: only exact first-site conservation, deliberately NOT a fixed-total-budget result.

All policies share a T2 teacher-class/all-competitor margin nondegradation feasibility guard. Include unchanged baseline in every candidate set. Select among feasible identical candidates using four arms: local reconstruction minimum, true next-message MSE minimum, graph-conjugate-permuted message MSE minimum (task graph unchanged), and teacher full-competitor T1-margin maximum. Teacher/logits/continuous target are offline diagnostics for **every** arm, not a permitted deployment selector. True labels are stored only for post-selection description and never enter candidate generation, costs, guard or scores.

Numerical tolerances fixed before analysis: integer event/delivery counts exact; amplitude sums measured in float64 allow only1e-9 relative roundoff (`1e-9*max(1,baseline)`), and the T2 margin guard allows1e-7 logit arithmetic roundoff. No tolerance is tuned using results.

## Readout and admission rules

Report every candidate/policy at each budget, graph IDs/coordinates, full logits and counts, feasible nontrivial candidates and graph-cluster uncertainty. An oracle gain is an optimistic **existence upper bound** by construction, not generalization, significance, a deployable algorithm, or independent graph contribution. Graph-message vs permuted control is a falsifier, not a theorem. No exact-cost claim from first-site support alone.

Predeclared diagnostic continuation threshold: under **nonincrease** costs and shared T2 guard, >10% of graphs on each task have some T1 teacher-margin improvement >=0.1 logit, and oracle T1 teacher-agreement gain >=0.5pp with no T2 agreement loss. Only then consider a deployment predictor/encoding design with a new written gate and matched non-graph controls. Exact-events feasible rates are reported even if too low. Failure stops this two-event reassignment family; no increase in candidate count, graphs, swap count or relaxed cost tolerance. Passing alone does not override Gate2 or establish a new method.
