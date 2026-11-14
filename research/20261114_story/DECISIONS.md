# Research decisions

## 2026-11-14: independent strong-teacher screen

Fetched origin/research/2026-11-14-graph-aware-conversion-plan and created independent branch research/20261114-graph-calibration. The supplied working directory is a historical material directory, not a Git checkout. Existing main has unrelated untracked work, left intact.

The previous graph-logit calibration on weaker teachers did not beat activation reconstruction (56.70 versus 56.78%, descriptive overlapping validation). Therefore no assumed method success, and no repeated broad tuning. Current test summaries suggest a substantial strong-teacher T1 deficit and a small T2 deficit. Audit raw archives independently before use.

Selected the smallest shared-parameter initial-membrane experiment. Exact hard-forward equivalence is required before optimizing. Direct graph-logit matching is a task-aware distillation objective; it alone does not establish graph-structure-specific novelty. Core causal and novelty claims require additional evidence.

Found another task's scheduled queue process on the local GPU host. It currently has no Python compute child. Do not stop or edit it; recheck pending queue and compute processes before launching. New campaign must avoid resource contention.
