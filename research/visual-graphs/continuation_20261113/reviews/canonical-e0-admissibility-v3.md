# Canonical E0 admissibility review — proposed v3 rule

**Date:** 2026-11-13
**Scope:** Read-only review of `logs/canonical-fullrepeat-v1/report.json`, `logs/fullval-v3-difference.json`, retained E0 failure logs, and the user’s immutable E0 guidance. This is a single-reviewer assessment, not a multi-reviewer gate and not a paper review.
**Decision:** The proposed v3 rule is scientifically justified for **aggregate baseline-metric reproduction**, if formally versioned at P0 before E1 validation. It must preserve and disclose the historical DC output drift. **E0 v1 and v2 remain failed; the current evidence is not an E0 pass under either version.**

## Numeric evidence

| Evidence artifact | Observed result | Meaning |
|---|---|---|
| `logs/e0-fullval-v1-failure.md` / `logs/e0_fullval_v1.log` | On all 5,000 validation graphs, teacher comparison passed; DC T4 had exact reference classes but failed tight logits on 2,836/50,000 values (5.67%), max absolute difference 0.15234947. V1 stopped before checking T8. | V1 fails its added historical tight-logit gate; it does not establish a T8 result. |
| `logs/fullval-v3-difference.json` | Full 5,000: T4 accuracy 70.72% equals reference, zero changed classes, max logit delta 0.15234947. T8 accuracy 70.66% equals reference, **one changed class at index 1850** (1/5,000 = 0.02%), max logit delta 0.02459240. Status is `v2_failed`. | User-guide aggregate endpoints match exactly, but V2’s exact historical-class gate fails at T8. Same aggregate accuracy does not imply exact samplewise predictions. |
| `logs/canonical-fullrepeat-v1/report.json` | Status `canonical_repeat_pass`; all seven arrays (indices, labels, teacher logits, ANN T4/T8, DC T4/T8) are bitwise equal to the saved canonical raw arrays; each max absolute difference is zero. Canonical and actual archive SHA-256 both equal `7c0727a2902bb2c05a0ec4930734b0bfa05e0905c4b86a6c976155a0c702fcc3`. All declared dependency/data hashes match before and after. | The current deterministic implementation reproduced its own full-split output exactly. `evaluation_exit_code: 1` is the expected V2 historical-class assertion failure; `canonical_repeat_pass` is not an E0 pass and does not erase the T8 historical mismatch. |

The user’s guidance defines E0 as reproducing the known seed-41 full-validation **numbers**, with floating tolerance written before the run. It does not require identical historical DC logits or exact per-example DC classes. Contract v1/v2 added those stronger DC invariants. The teacher path remains subject to the frozen tight teacher-logit check; the current comparisons report that teacher check passed.

## Is v3 an acceptable correction?

**Yes, for this narrow E0 purpose, with explicit limits.** A v3 rule requiring (1) the exact full-validation aggregate accuracies 70.66/70.72/70.66%, (2) exact canonical indices/labels, (3) tight teacher-logit comparison, (4) bitwise repeatability of the canonical full-run output under the same deterministic setup, and (5) unchanged source/data/teacher/calibration/evaluator hashes is a defensible operationalization of the user’s metric-reproduction requirement. It sets no new, enlarged DC-logit tolerance and does not alter the method’s frozen `0.3 pp AND complete-cost/latency` criterion, scale, seeds, or scope.

This is a versioned correction after seeing failures, so its provenance must say so. Its scientific justification is that the exact-old-logit/per-example-DC invariant was locally added, while the authoritative E0 question is whether the canonical evaluation reproduces the documented aggregate metrics reliably. The newly measured canonical repeat provides direct evidence for repeatability; the known index-1850 class drift remains disclosed. Do not say “exact reproduction,” “same historical predictions,” or “tight DC-logit match.” Describe the narrower result as **reproduction of the historical aggregate baseline metrics with canonical repeatability, alongside one changed historical T8 class and quantified DC-logit differences**.

The current full run and repeat can be adjudicated under v3 only after the root formally reopens and accepts a v3 P0 gate. Keep the run’s original `v2_failed` / `unvalidated` status intact. Do not rewrite it as a v1/v2 pass or repeat it until the old class happens to match. If v3 is not formally accepted, E0 remains failed/pending under the active protocol.

## Required v3 acceptance record

1. Freeze v3 before E1 validation and name it metric-level historical reproduction, not trajectory reproduction. Retain v1 and v2 unchanged as superseded criteria and preserve their failure logs.
2. Bind the existing canonical raw archive, its repeat report, `fullval-v3-difference.json`, reference NPZ, evaluator, checkpoint, calibration, author source and processed data by actual hashes. Confirm the current report’s before/after dependency equality rather than relying on filenames.
3. Require exact full-split indices/labels and exact reported teacher/DC4/DC8 aggregate accuracies; keep teacher-logit tight comparison. Report DC maximum/mean/RMS logit differences, counts outside v1’s old tight criterion, and the changed class index against the historical reference. Do not substitute a looser DC-logit threshold.
4. Treat bitwise equality as a same-environment repeatability claim only. It does not imply cross-device reproducibility, historical per-example equivalence, or method correctness.
5. Keep E1 and all later claims under their original frozen gates. E0 admissibility supplies no evidence for residual-prefix accuracy, efficiency, novelty, or cost advantage.

**Verdict:** Recommend accepting the *v3 rule* for future/current E0 adjudication after formal P0 versioning and input-hash audit. The evidence supports canonical full-run repeatability and exact guide-level aggregate metrics; it also documents an actual historical T8 class mismatch. Therefore no pass is asserted here. No new method test was run.
