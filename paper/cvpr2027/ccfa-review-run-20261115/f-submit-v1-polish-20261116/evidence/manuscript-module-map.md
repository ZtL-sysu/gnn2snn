# Manuscript Module Map (S1)

## Scope and frozen baseline

This map describes the immutable baseline under `sources/`. It is used only to guide structural, prose, layout, figure-placement, and metadata work in a later working copy. No new experiment is proposed and no scientific claim, numerical result, equation, citation, or disclosure is added here.

Manuscript: `sources/main.tex`  
Supplement: `sources/supplement.tex`  
Main bibliographic database: `sources/references.bib`  
Venue style: `sources/cvpr.sty` (`review` option in both documents)

## Main-document module map

| Module | Source location | Function in the argument | Items that must remain stable |
|---|---|---|---|
| Title | `main.tex:11` | Names the object of study and the audit scope: finite events, decision fidelity, temporal fidelity, graph-to-spike conversion. | Exact title wording unless a purely typographic line-break fix is required; no method claim may be added. |
| Author / anonymity block | `main.tex:7-14` | Preserves double-blind presentation. | `Anonymous authors`, `\paperID{-----}`, `\confName{CVPR}`, `\confYear{2027}`, and line-number switching stay anonymous and metadata-consistent. |
| Abstract | `main.tex:16-18` | States the proxy problem, two audited invariants, the directional and source-recurrence explanations, the calibration scope, and the reporting implication. | All conditional rates and scopes: 40.9% / 31.9% among message-improving interventions; dependence on intervention distributions; exact local decoded tensors and equal local event counts; no universal frequency or new-converter claim. |
| Introduction | `main.tex:20-38` | Moves from graph mixing and finite events to two distinct information losses, then motivates controlled hard-event interventions and introduces the central finding. | 332/811, 296/928, 54.1%/23.5%, 8.8%/14.9%; teacher agreement versus ground-truth accuracy; local events versus whole-network activity; the checklist framing and its evidence scope. |
| Concept figure | `main.tex:32-38` (`figures/concept.pdf`) | Makes the spatial decision-boundary loss and temporal state-history loss visible before technical definitions. | Schematic status of the left panel and executable-recurrence status of the right panel; source file and caption semantics. |
| Related Work | `main.tex:40-47` | Separates conversion/calibration/readout, differential and phase coding, temporal misalignment and prediction-driven quantization, and graph-dependent approximation. | PD-Quant is an optimization method; One-Spike is a single-spike phase converter and is neither implemented nor matched here; static propagation, feature-dependent gating, and classical node classification are separate settings. Citation keys remain unchanged. |
| Analysis design / event semantics | `main.tex:49-70` | Defines the frozen residual GCN, graph normalization, signed scalar events, whole-network activity, DCGS-derived recurrence, decoder, and meaning of `T`. | Eqs. (1)-(2); event alphabet; no self-loops; `T` means active ticks rather than measured latency; whole-network activity is the total number of nonzero scalar events and is not energy, memory traffic, or effective spike count. |
| Graph message / decision analysis | `main.tex:71-97` | Shows that an exact message identity becomes a scalar norm, while decisions depend on signed projections and margins. | Eqs. (3)-(6); $\Delta M=AEW^\top$ as elementary linear algebra; Eq. (5) is the exact decision condition; Eq. (6) is only sufficient; the norm does not order two actual errors; linearization may fail in the full converter. |
| Terminal equivalence analysis | `main.tex:98-124` | Demonstrates that matched integrated input, alphabet, horizon, and local event count still permit a different decoded result; transfers the test to legal two-tick codewords in the graph adapter. | Scalar counterexample values and threshold; codeword pair $(q,-q/2)\leftrightarrow(q/2,q/2)$, endpoint $3q/4$, unweighted amplitudes $3q/2$ and $q$; eligibility under actual source replay; explicit statement that first-prefix profile and absolute amplitude are uncontrolled. |
| Interventions and evidence scope | `main.tex:116-124` | Defines the three evidence levels, split roles, continuous position diagnostic, hard-event rules, confidence intervals, and calibration controls. | Exposed validation/test data are descriptive rather than new blind evidence; 512 graphs / 2,048 first-tick attempts; 128 training graphs; centered profile uses 45,000 training graphs; teacher is consulted only after replay; graph-cluster resampling retains all four interventions from a graph. |
| Results: operating regime | `main.tex:126-152`, Table 1, Figs. 3 | Establishes high four/eight-tick fidelity and localizes a short-prefix bottleneck. | Table 1 mean/SD values; three-teacher historical exposed tests; GatedGCN is a separate hybrid adapter; $T=1$ values 71.44%/46.80%; 54.08%/23.54%, 80.4%/45.2%, 8.8%/14.9% recoveries; continuous diagnostic status. |
| Results: message--decision mismatch | `main.tex:153-179`, Table 2, Fig. 4 | Primary mechanism result: reducing the actual first-tick message MSE can worsen the teacher-class margin. | Table 2 counts/rates/intervals; conditions on message-improving interventions; $T=2$ reuses the same MSE condition; conditional-frequency scope; MNIST graph 437 example values; stratum ranges and graph-level counts. |
| Results: calibration controls | `main.tex:180-202`, Table 3 | Shows that sensitivity diagnostics and matched objectives do not identify a consistently superior correction target. | Table 3 values and matched resources; historical per-horizon fitting versus later shared-prefix fitting; $+$0.20/$+$0.18 and $-$0.18/$+$0.19 differences; paired intervals include zero; top-two outside rates 4.69%→22.95% and 9.47%→27.34%; source TI numbers and adaptation caveats. |
| Results: temporal equivalence | `main.tex:203-218`, Fig. 5 | Shows that exact local two-tick equality and local event count do not determine downstream decisions or whole-network activity. | 128 training graphs; eligible coordinate counts; 4,351,056→4,499,463 and 6,876,233→7,219,258 total events; 6/128, 16/128, 2/128, 10/128 prediction changes; +3.41%, +4.99%, CIFAR10 1.33% decrease (6,876,233→6,785,108); no timing/amplitude isolation. |
| Results: residual centering | `main.tex:219-236`, Fig. 6 | Presents centering as a useful representation change with partial and bounded evidence. | Eq. (7) is an affine identity; caching/shared streams are implementation opportunities only; shared thresholds; 45,000 training graphs; +2.04 points and $9.099\times10^{11}$→$6.650\times10^{11}$; 5/6 comparisons and the $-0.0044$ exception; separate validation campaign values and nine unresolved cost items. |
| Results: cost / applicability | `main.tex:237-241` | Separates low `T`, local sparsity, partial arithmetic proxies, and measured dense execution. | RTX 3060 timings and protocol; allocation timings from a distinct experiment; no chip energy; dense kernels do not skip zero events. |
| Discussion and Conclusion | `main.tex:242-248` | Recovers the two scoped information losses, proposes an evaluation checklist, and states boundaries without claiming a new converter. | Fixed residual GCNs, one teacher for core hard tests, previously used validation/training graphs; open effects on other teachers/tasks/converters/topologies; no matched One-Spike comparison; no conversion-wide frequency. |
| LLM Usage Statement | `main.tex:250-252` | Gives an explicit tool-use and responsibility disclosure under anonymity. | Preserve the full disclosure verbatim in declaration mode `preserve`; no model names, dates, or responsibility statements may be altered without author instruction. |
| Bibliography | `main.tex:254-255`, `references.bib` | Supports the related-work distinctions and cost-context statements. | All citation keys and bibliographic facts; no new citations may be introduced for stylistic reasons. |

## Supplement module map

| Module | Source location | Function | Items that must remain stable |
|---|---|---|---|
| Evidence and reproducibility | `supplement.tex:10-32`, Table 1 | Distinguishes evidence families, splits, quantities, derived files, intervention rules, and statistics. | All split sizes, teacher identifiers, data-provenance caveats, selection rules, bootstrap protocol, and anonymized package descriptions. |
| Graph models and calibration | `supplement.tex:33-69`, Tables 2-3 | Documents datasets, model variants, fitting conventions, source-linked threshold iteration, and controls. | 55,000/5,000 and 45,000/5,000 partitions; architecture widths; 693 offsets / 512 graphs / 64 updates; DCGS commit; 100 updates and fallback details; table values. |
| Directional and temporal derivations | `supplement.tex:70-91` | Provides assumptions and algebra behind the margin bound, executable counterexample, and a fixed-operator gate caveat. | Linear-suffix assumptions; Cauchy--Schwarz condition labelled as illustrative rather than new theory; exact emitted-event values; gate inequality assumptions and exclusions. |
| Classical node context | `supplement.tex:92-111`, Table 4 | Places FTBC-style bias compensation and membrane readout in earlier node-classification experiments. | $\theta N_T+u_T=Tc$; $0\leq u_T<\theta$; corrected analytic offset under stated assumptions; all six dataset rows; audit provenance and separation from visual adapters. |
| Residual centering and complete costs | `supplement.tex:112-123` | Defines the partial projection proxy and separates implementation/cost regimes. | $70\sum(4N_{\rm node}+N_{\rm edge})$ proxy exclusions; scale/timing protocols; centered/raw MSE and addition ratios; validation campaign values; nine unresolved accounting items. |
| Failure boundaries and research scope | `supplement.tex:124-130` | Documents two implemented selectors and the narrow empirical conclusion. | They are counterexamples, not impossibility results; no new algorithm or deployment guarantee is claimed. |

## Figure and table inventory

### Main text

- Fig. 1 `figures/concept.pdf`: conceptual two-panel schematic; left is schematic, right is an executable recurrence example.
- Fig. 2 `figures/temporal_mechanism.pdf`: legal two-tick codeword equivalence and prefix difference.
- Fig. 3 `figures/position_recovery.pdf`: full-validation, three-teacher conditional recovery at `T=1`; nonadditive site groups.
- Fig. 4 `figures/hard_intervention_stratified.pdf`: adverse margin rates by coordinate rule and amplitude direction; one teacher, repeated interventions are not independent.
- Fig. 5 `figures/temporal_decision_effects.pdf`: paired downstream decisions under exact local `T=2` equality; training-only diagnostic.
- Fig. 6 `figures/centered_tradeoff.pdf`: centered versus raw fidelity and prospective coded-projection additions on the full CIFAR10 training split.
- Table 1: high-prefix historical full-test accuracy.
- Table 2: core hard-intervention result.
- Table 3: matched historical full-test calibration comparison.

### Supplement

- Table 1: evidence-family scope.
- Table 2: shared-prefix ordinary controls.
- Table 3: source-threshold iteration graph controls.
- Table 4: audited historical node-classification context.

## Citation inventory and role

All cited work must remain attributed to its stated role:

- Conversion / calibration / bias and residual readout: `li2021`, `bojkovic2024`, `wu2024ftbc`, `li2024erroraware`, `hao2023residual`.
- Differential coding and single-spike phase conversion: `huang2025`, `hwang2025onespike`.
- Temporal order and prediction-driven parameter choice: `bojkovic2025`, `liu2023pdquant`.
- Short-horizon closed-loop alignment: `li2026cka`.
- Graph quantization / operator-aware approximation: `degreequant2021`, `a2q2023`.
- Visual graph benchmark and cost context: `dwivedi2023`, `bu2025complexity`.

## Scientific boundary checklist for all later edits

1. Keep the paper a controlled mechanism audit, not a new converter or evaluation of the original DCGS/One-Spike systems.
2. Preserve distinctions among teacher agreement, ground-truth accuracy, local event count, whole-network activity, active ticks, measured latency, and energy.
3. Preserve separation among historical exposed tests, exposed validation diagnostics, training-only diagnostics, and the full-validation centered campaign.
4. Preserve the exact numerics in the abstract, introduction, results, captions, tables, and supplement; edits must not round selectively or mix experiment families.
5. Preserve all explicit caveats about conditional rates, intervention distributions, uncontrolled first-prefix/amplitude in temporal exchanges, partial cost proxies, and exposed data.
6. Preserve equations, citation keys, reference facts, figure data sources, the LLM Usage Statement, and anonymity metadata.
