# Primary full-manuscript checks

No numerical contradiction was found in 32 reloaded rescue endpoints or five table-source hashes. Eleven citation keys resolve to eleven bibliography records; six main sections and all eight insertion anchors are present. Independent full review is recorded separately.

Concrete pre-revision actions from implementation inspection:

1. REWRITE Method3.2: define all-message as one half of the six-site normalized local mean plus one half of the first-site message objective. It fits all693 offsets but the message term targets only the first site's immediate next message. Current text defines the first-site objective but leaves this all-site variant implicit. Source: objective.py loss_terms.
2. REWRITE Method3.3: state initial u0=0 and disagreement-gap closure 1-d_intervention/d_base, with teacher predictions as the reference. This avoids confusing disagreement closure with accuracy-gap recovery.
3. REWRITE Method3.5: explicitly name descriptive 95 percent paired-normal intervals and their paired correctness variable. Current figures state95 percent but prose only says intervals.
4. REWRITE Results4.4: include recorded offline fitting cost with its timing boundary, rather than reporting online allocation time alone. T1 screen fit-loop times on RTX3060 range1.1214–1.7101s for the listed local/message first/all-site fits, one measurement each,64updates. Teachers/threshold estimation/data setup are outside these cached fitting loops; these values are not repeated benchmarks or total calibration-pipeline costs. Sources: runs/screen/{MNIST,CIFAR10}41/*_T1_training.json and run_critical.py lines248–266.
5. REWRITE visual references: introduce each numbered visual explicitly in nearby prose; figure/table anchors alone do not supply interpretation.

These are bounded definition/presentation repairs with existing evidence. They do not reopen experiments or repair the lack of demonstrated algorithmic advantage. Publication novelty, teacher breadth and independent holdout remain unresolved research limitations.
