# Current ordered research state

Updated 2026-11-13T14:00:52.234160+00:00

Goal active. Experiments local RTX3060 only; Conda. research-github-sync checkpoints on agent branch, no main merge.

|Stage|State|
|---|---|
|P0|passed|
|P1|passed|
|P1A|passed|
|P1B|passed|
|P2|passed|
|P3|passed|
|P3A|passed|
|P4|running|
|P5|pending|
|P5A|pending|
|P6|pending|
|P7|pending|
|P8|pending|
|P9|pending|
|P10|pending|
|P11|pending|
|P12|pending|
|P13|pending|
|P14|pending|
|P15|pending|
|P16|pending|
|P17|pending|
|P18|pending|
|P19|pending|
|P20|pending|
|P21|pending|

P0–P3A passes admit bounded frozen evidence collection only; P1B/P3A current decision is same-agent-role-pass. The fa1403b independent admission REVIEW was one byte and is NOT credited. Final substantive independent CCFA manuscript review remains required.

Completed E1 first group: seed41 official5000, centered/first_analog/generic × T1/2/4/8, 12 NPZs. Source/model/cal/data before-after checks passed; primary numeric recomputation passed. Centered T4/T8 accuracy70.66/70.70%. Other controls/seeds/sharedtheta/completecost and E2–E6 remain pending; no StoryA or paper acceptance.

Execution26002/PID23080 completed exit0, no remaining GPU experiment process. First resource probe128graphs9.285sec/peak allocated2545697280bytes is not a latency benchmark. No newtest inference.

GitHub measured checkpoint e9541860648110004115d7c86ceaaba11b84b70b verified on origin/agent/visual-conversion/20261113-codex-vision. Includes complete sample predictions and truthful limitations. Historical E0 v1/v2 failures remain failures; v3 reproduces aggregate metrics and canonical bitwise output with one historical T8class drift.

Next: implementation delegate in isolated agent/visual-conversion/20261113-luna-controls writes remaining-control evaluator only, no official jobs. Primary reviews code and executes fixed fullvalidation; then fulltrain label-free calibrations for teachers42/43. Original artifacts protected, failed routes logged.

## Resume note 2026-11-13T15:14:02Z (readable view only; JSON unchanged)

User resumed the frozen goal on local RTX 3060 with research-github-sync checkpoints. Contract v4 remains active; `--before P4` passes; P4 stays `running` with no pass flag added. Agent `20261113-wb-resume` works in an isolated worktree/branch. Implemented `code/eval_controls_validation_v2.py` (the interrupted luna-controls task) plus 4 passing synthetic tests, bound to both frozen calibrations, and completed one official full-5000 seed41 control grid (3 rows x 4 timesteps, exit 0, 346 s). Evidence: `results/e1-seed41-controls-v1/` and collaboration checkpoint `20261113T151402Z-controls-result`. Cross-run bitwise agreement with the earlier independent generic run confirmed for all four T. E1 acceptance is still NOT reached: seeds 42/43, shared-theta controls, historical six-site DC-adapt aggregation, complete known-operation cost and matched wall-clock remain. No Story decision, no test inference, no paper stage entered.
