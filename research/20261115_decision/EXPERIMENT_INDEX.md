# Experiment and reproduction index

Base cb742f2, branch research/20261115-decision-aware. All scripts run using local conda `dyg-cold`. From this worktree:

```powershell
$py = 'C:/Users/xhf/miniconda3/envs/dyg-cold/python.exe'
& $py -X utf8 research/20261115_decision/analyze_cached.py
& $py -X utf8 research/20261115_decision/audit_history.py
& $py -X utf8 research/20261115_decision/run_interventions.py --dataset MNIST --output research/20261115_decision/runs/reproduction/intervention/MNIST41
& $py -X utf8 research/20261115_decision/run_interventions.py --dataset CIFAR10 --output research/20261115_decision/runs/reproduction/intervention/CIFAR1041
& $py -X utf8 research/20261115_decision/run_objectives.py --dataset MNIST --output research/20261115_decision/runs/reproduction/objectives/MNIST41
& $py -X utf8 research/20261115_decision/run_objectives.py --dataset CIFAR10 --output research/20261115_decision/runs/reproduction/objectives/CIFAR1041
& $py -X utf8 research/20261115_decision/audit_optimization.py --output research/20261115_decision/analysis/optimization_reproduction
& $py -X utf8 research/20261115_decision/summarize_phase3.py
& $py -X utf8 research/20261115_decision/run_source_ti.py --dataset MNIST --output research/20261115_decision/runs/reproduction/source_ti/MNIST41
& $py -X utf8 research/20261115_decision/run_source_ti.py --dataset CIFAR10 --output research/20261115_decision/runs/reproduction/source_ti/CIFAR1041
& $py -X utf8 research/20261115_decision/summarize_source_ti.py
& $py -X utf8 research/20261115_decision/run_swap_oracle.py --dataset MNIST --output research/20261115_decision/runs/reproduction/swap_oracle/MNIST41
& $py -X utf8 research/20261115_decision/run_swap_oracle.py --dataset CIFAR10 --output research/20261115_decision/runs/reproduction/swap_oracle/CIFAR1041
& $py -X utf8 research/20261115_decision/summarize_swap_oracle.py
& $py -X utf8 research/20261115_decision/run_learned_swap.py --dataset MNIST --output research/20261115_decision/runs/reproduction/learned_swap/MNIST41
& $py -X utf8 research/20261115_decision/run_learned_swap.py --dataset CIFAR10 --output research/20261115_decision/runs/reproduction/learned_swap/CIFAR1041
& $py -X utf8 research/20261115_decision/summarize_learned_swap.py
& $py -m pytest research/20261115_decision/test_intervention.py research/20261115_decision/test_objectives.py -q
```

Model runners refuse to overwrite output directories. `audit_optimization.py` likewise refuses an existing output; choose a fresh checkout/output before reproduction. CPU summaries deliberately rebuild derived files only in this new research directory. Historical directories are never written. Raw datasets/weights remain external, original hashed assets at `C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision`.

|Experiment|Raw output|Data / fit / cost|Validation exposure|
|---|---|---|---|
|Archived CPU endpoint reconstruction|`analysis/cached/*`, `analysis/historical/audit.json`|No model fitting/inference; saved logits and source hashes, plus CPU val graph statistics|Old test exposure explicitly retained, no new test dataset load|
|First hard-event intervention|`runs/intervention/{MNIST41,CIFAR1041}`|512 val graphs each; 4 interventions/graph; T1/T2; no fitting|Same exposed indices as historical critical|
|Trace correction reproduction|`runs/intervention_v2/*`|Same interventions, now coordinate/actual-change fields, deterministic-control name corrected|Reproduction, not additional distinct scientific trials|
|Five matched objectives|`runs/objectives/{MNIST41,CIFAR1041}`|512 train, batch32,64 Adam updates lr.01;693 offsets; no labels.40 total accuracy endpoints (2 tasks×5 arms×4 T), including20 T1/T2 selection endpoints and20 high-T regressions|All 40 reported, no best-epoch/temperature/restart selection|
|Single optimization/capacity audit|`analysis/optimization_audit/results.json`|Training only;4 interpolations between zero and fitted offsets for each arm;no new fit or validation selection|Zero new validation endpoints|
|Automatic tables/paired CI/cost scatter|`analysis/phase3/*`|Raw NPZ recomputation,2000 graph-bootstrap samples; actual events reduced in int64|Not teacher-seed CI or independent confirmation|
|Source threshold iteration follow-up|`runs/source_ti/{MNIST41,CIFAR1041}`, `analysis/source_ti/*`|Two predeclared source calibration horizons H1/H2, n256/512; same training512; zero optimizer updates; sixteen H×T endpoints, all same-vector prefixes|Eight additional T1/T2 selection endpoints and eight high-T functional checks, all exposed val512; no test inference|
|Finite-event reassignment capacity oracle|`runs/swap_oracle/*/candidates.npz`, `analysis/swap_oracle/*`|First128 training graphs/task,8 swaps+baseline each,sharedT1/T2,zero fitting.48 descriptive budget/policy/prefix rows from two banks. Wall10.6/15.3seconds|No val/test load; selected and evaluated same training graphs; optimistic teacher/future-cost oracle, not deployable performance|
|Single-pass learned reassignment|`runs/learned_swap/*`, `analysis/learned_swap/*`|6 predictors+unchanged reference,T1/T2,28 val endpoints;ridgeλ.001 once;128×8 incremental calibration candidates,prior512 baseline calibration shared;source/feature/targets/models saved|Same exposed val512,no labels in fit,no new test/teacher training;one alignment audit,no post-failure tuning;run wall53.5/75.7seconds|

Teacher and calibration/source/data SHA checks are inherited from frozen loaders; each new objective run records their identity, fixed train/val indices, parameters, loss trace, gradient norms, full offset NPZ, T1/2/4/8 outputs and wall times. Original implementation snapshots are recoverable at commit `7cb5021`; modified intervention tracing is separately hashed in v2 metadata. `verify_phase3.py` verifies all recorded script identities against current code or immutable Git objects and builds a final artifact manifest.

Important: original `results.json` event totals used float32 reduction. Raw **per-graph** event counts are exactly representable; the canonical main table sums them in int64, recording the small correction per endpoint. This is a summation audit, not a changed event definition. Full GPU evaluation timing includes the offline teacher and instrumentation; it is not pure SNN latency. No total-energy claim. New training/inference is serial and uses existing teachers, well below the initial 30-minute admission cap; no teacher retraining, paid API job or external submission.
