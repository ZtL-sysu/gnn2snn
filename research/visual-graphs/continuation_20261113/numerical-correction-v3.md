# Canonical baseline reproduction protocol v3

Purpose: satisfy the user guide's E0 reported-number reproduction using one tested, stable numerical path; do not claim exact reproduction of a historical nondeterministic trajectory.

Preserved failures: v1 DC4 tight-logit comparison failed; v2 DC8 historical-class equality failed on graph1850. Neither is a pass. Graph1850 was misclassified in both versions (label2, historicalprediction4,new3). All fullvalidation aggregate targets exactly match: teacher70.66%,DC470.72%,DC870.66%.

New diagnostic evidence, before candidate validation: two complete5000 deterministic replays are bitwise identical on indices, labels, teacher_logits, ann_T4/8 and dc_adapt_T4/8. Archive SHA7c0727a2902bb2c05a0ec4930734b0bfa05e0905c4b86a6c976155a0c702fcc3 is identical. All actual bound sources/data/checkpoint/calibration hashes match before/after. Fullreport: logs/canonical-fullrepeat-v1/report.json. Batch14 repeated independently also matched this canonical output exactly. These checks support current-platform deterministic repeatability; no causal attribution to a specific floating-point kernel is established.

Replacement E0 acceptance, fixed before the next verification run:
1. Exactly5000 canonical indices/labels; teacher70.66%,DC470.72%,DC870.66% exactly, no enlarged accuracy tolerance.
2. Teacher logits agree with historical reference at original rtol1e-5,atol1e-4.
3. All arrays must match the frozen deterministic canonical reference BITWISE, including DC4/DC8 logits, not merely aggregate accuracy. Reference imported as inputs/canonical-baseline-val.npz with its bound hash. This reference is baseline-only, generated using unchanged model/calibration/source; no candidate hyperparameter was selected.
4. Historical DC class mismatch counts, every changed index, max/mean/RMS logit difference and number outside original tight tolerance are mandatory report fields. No exact-historical-class, tight-historical-logit or historical-event-count reproduction claim. A historicalclass difference is reported, not silently erased.
5. Complete before/after source/model/calibration/data/reference checks, label-free inputs, sequential128batches and pinned author source remain required. These criteria do not imply full cost, efficiency or candidate advantage.

This replaces two internally introduced historical numerical invariants absent from the user's aggregate E0 criterion. It is based on measured repeatability and transparent provenance rather than picking a favorable nondeterministic realization. Main method precision/cost target, scale1, three seeds, full datasets, required controls, independent final paper review≥7 and all E1–E6 scope stay unchanged. Do not transfer v1/v2 pass flags. Reopen P0, reaccept unchanged specification, then accept only actual v3 evidence. Method experiments wait for baseline/feasibility/review acceptance.
