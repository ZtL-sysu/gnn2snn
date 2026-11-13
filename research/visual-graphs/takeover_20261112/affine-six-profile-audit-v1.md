# Affine-base six-profile audit v1

Status: independent audit of all six saved training-only profiles. No labels or ground-truth accuracy were used. Reported values are teacher-class agreement, logit MSE, teacher-to-candidate KL and prospective single-stream projection ADDs.

## Prefix results

| Mode | Scale | T | Teacher-class agreement | Logit MSE | Teacher→candidate KL | Prospective projection ADDs |
|---|---:|---:|---:|---:|---:|---:|
| generic | 1 | 1 | 0.504333333 | 9.52466240885 | 1.35907060634 | 1221830889930 |
| generic | 1 | 2 | 0.896822222 | 0.627219934276 | 0.078371995702 | 2382262847370 |
| generic | 1 | 4 | 0.993155556 | 0.00278574884142 | 0.000334634198804 | 4144486996790 |
| generic | 8 | 1 | 0.501022222 | 9.65883665918 | 1.39192356639 | 1072157740080 |
| generic | 8 | 2 | 0.903311111 | 0.567961173009 | 0.0712755196892 | 1967197191470 |
| generic | 8 | 4 | 0.978777778 | 0.03159696111 | 0.00382554566858 | 2823253578740 |
| first_analog | 1 | 1 | 0.828088889 | 1.42733627831 | 0.193488336022 | 910964813850 |
| first_analog | 1 | 2 | 0.968044444 | 0.071385779594 | 0.00872635737301 | 1749326014100 |
| first_analog | 1 | 4 | 0.997488889 | 0.000577553377547 | 6.21227368016e-05 | 2750670384490 |
| first_analog | 8 | 1 | 0.818577778 | 1.55190995169 | 0.21371104933 | 763425954690 |
| first_analog | 8 | 2 | 0.955688889 | 0.117903541119 | 0.0151816977824 | 1318300803610 |
| first_analog | 8 | 4 | 0.984822222 | 0.0157967339257 | 0.00194042224182 | 1764528777060 |
| centered | 1 | 1 | 0.838400000 | 1.33854894832 | 0.178934550775 | 665113351420 |
| centered | 1 | 2 | 0.968200000 | 0.0673267897413 | 0.00812781691469 | 1294074700240 |
| centered | 1 | 4 | 0.997511111 | 0.000451377096459 | 4.82345197481e-05 | 2087346292710 |
| centered | 8 | 1 | 0.832511111 | 1.45713174636 | 0.191906153784 | 594269624830 |
| centered | 8 | 2 | 0.961911111 | 0.0971115578684 | 0.011772570941 | 1024486000230 |
| centered | 8 | 4 | 0.987844444 | 0.0107361596483 | 0.00134538627248 | 1355761013810 |

Agreement is a fraction, not top-1 accuracy. ADDs are `70 × Σ(4 × node events + edge events)` across coded input sites; this is one prospective arithmetic component, not total compute, latency or energy.

## Frozen conditional-control decision

| Scale | Agreement ≥ and MSE ≤ at all prefixes | At least one strict fidelity improvement | Projection ADDs ≤ at all prefixes | Shared-θ controls |
|---:|---|---|---|---|
| 1 | True | True | True | TRIGGER |
| 8 | True | True | True | TRIGGER |

Overall: **TRIGGER**. Decisions use unrounded JSON values and the conservative all-prefix rule. A trigger only authorizes the four shared-threshold train-only control trajectories in the frozen protocol.

## Integrity and scope

All six profiles passed status, coverage (45,000 graphs / 352 batches / final batch 72), node/edge rows (5,294,119 / 42,352,952), prefix, runtime, input-hash, teacher/checkpoint, calibration, author-source, threshold, event-partition, call-count, metric-sum and projection-ADD checks. Calibration SHA-256: `1618e3b46f75646ab051229b0d954815b5e28bf1729cbbf2042e4d596766a511`.

The profiles are fixed-order, label-free training-split diagnostics for one frozen seed-41 CIFAR10 GatedGCN teacher. No ground-truth accuracy, validation or test is used. This dense reference does not establish efficient execution, latency, energy or novelty.

## Profile JSON hashes

- `generic` scale 1: `c868dafd87087a2c8d9ccdc0e7598738d83edf09b96937c7aae4c8e35c44c3d6`
- `generic` scale 8: `333a9c67d9c7cbafba2459f4cf5869c4e77b03fe52a665028355b5d5f9d7cef3`
- `first_analog` scale 1: `eeac398bada5745a12d224abb931cced2f5a2717021579dad4226296785bd0e1`
- `first_analog` scale 8: `20d818f320b66d57d544279ac284bdb050521554eeee51e2db34a6b2d109841d`
- `centered` scale 1: `7b5ae70b31a17c9f0308a682c84f05471d34ac94a325cf3ed9b156c6bc1e59af`
- `centered` scale 8: `4ee03b1be19af59c10f684f8a8cd36ec61d523f5865aa711720cabfbb0eac368`
