# Overview
Goal: a credible GNN→SNN method paper, conditional on genuine algorithmic evidence. Current phase: critical-site/message calibration screening complete; broader research and paper integration remain active. Branch `research/20261114-critical-site`. Updated 2026-11-14.

## Verified
- Prior campaign is preserved at `research/20261114_story/`; 142 frozen input hashes rechecked unchanged.
- New 110 screening endpoints completed on train/validation only. See [decision record](../research/20261114_critical/DECISIONS.md) and [raw-derived summaries](../research/20261114_critical/analysis/SUMMARY.md).
- Message membrane calibration, diagonal weighted alphabets, and full hard-message alphabet refinement did not demonstrate independent structure benefit. Full-message T1 versus equally refined local: MNIST −1.171875pp, CIFAR10 −1.3671875pp. Stop extending these losses.
- Alphabet capacity improves MNIST T1 substantially, but CIFAR10 and novelty requirements remain unresolved. No successful method-paper claim.
- User-requested f-skills source c0637ddc installed locally: 9 skills, 286 source files verified, original directories backed up. Latest tests: 55/57 pass with UTF8 and child-Git main initialization; two Windows path-separator expectations fail. No global Git configuration changed.

## Hypothesis / Pending
Position-sensitive precision allocation might offer a better budget tradeoff, but known graph quantization and coding precedents must be confronted. Do not simply rename Lloyd quantization or mixed timesteps. Current extended method exploration and full paper quality audit are incomplete.

## Key Decisions
User asks for stage-wise GitHub delivery; do not wait for the final paper to push valid work. Newly updated research workflow is being migrated from real evidence, with pending stages left pending. [Handoff](agents/codex-critical-20261114.md) contains exact next actions; [paper evidence roles](../research/20261114_critical/EVIDENCE_MAP.md) separates reusable motivation from new claims.
