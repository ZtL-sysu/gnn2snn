# Comparator Section and Narrative Patterns (S5)

## Method

The frozen ten-paper comparator set was read at the level of title, abstract, introduction openings, section sequencing, result framing, and conclusion callbacks available from CVF, PMLR, OpenReview, and publisher records. The goal is to learn how recent top-conference papers compress a contribution and distribute evidence across modules. No result from these papers is transferred into the scientific argument of the submitted manuscript.

## Cross-comparator patterns

| Pattern | Representative comparators | Observable convention | Application to `A Finite-Event Audit...` |
|---|---|---|---|
| Opening hook | 5, 7, 1 | Open with a precise tension or phenomenon, not a long field history. The first paragraph gives readers the object of study. | Replace broad background with a concrete finite-event scenario: a smaller message error or matched local code can still change the graph decision. |
| Problem/gap compression | 6, 5, 2 | One or two sentences identify what a common metric assumes and what it omits. | Keep the two information losses as one paired gap: local reconstruction ignores signed decision direction and consumed temporal history. |
| Contribution grammar | 1, 7, 3 | `We identify/observe/audit` is used for discoveries; `we propose` is reserved for methods actually introduced. | Preserve `audit`, `test`, and `show`; avoid implying a new converter or a universal frequency. |
| Result-first versus method-first | 7, 5 vs. 3, 8 | Mechanism papers lead with diagnostic definitions and findings; method papers lead with representation or algorithm. | Use mechanism-first throughout. Centered residual coding remains a complementary diagnostic, not a competing thesis. |
| Figure/table storyboarding | 7, 9, 2 | Concept figures define the question; main tables carry the decisive comparison; analysis figures decompose heterogeneous effects. | Keep the concept figure for two information losses, Table 2 for the decisive adverse-rate result, and stratify rather than narrate every cell. |
| Paired evidence | 7, 1 | Two methods or observations are defined symmetrically, then each answers one half of the central question. | Pair hard message interventions with exact codeword exchanges and state explicitly what each invariant controls. |
| Uncertainty and scope | 9, 2 | Numerical uncertainty and resource scope sit next to the affected claim rather than in a generic disclaimer. | Retain cluster-bootstrap intervals with rates, keep training-only temporal evidence labeled, and define whole-network activity precisely. |
| Limitation placement | 1, 7 | Boundaries appear where the interpretation is introduced and are condensed again near the end. | Keep first-prefix/amplitude confounding beside the temporal result; keep converter/task scope in Discussion. |
| Conclusion callback | 7, 5 | Return to the named quantities or properties from the opening without adding a new result. | Close by calling back decision, prefix, and activity as three separately reportable quantities. |

## Module-specific lessons

### Title and abstract

Recent diagnostic papers favor titles that name both the object and the audited property. Our title already does this. The abstract should follow a four-move arc: (i) state the proxy and the missing downstream quantity; (ii) give the first intervention and its conditional rate; (iii) give the second exact-invariant counterexample and its consequence; (iv) state the evaluation implication. The current abstract has all four moves but places several qualifications inside the first result sentence; the revision can make the arc easier to parse without changing scope.

### Introduction

Comparator 7 shows that a mechanism paper can spend its opening on the scientific distinction rather than on application background. Comparator 5 shows the value of an explicit fairness question. Our introduction should therefore move faster from graph propagation to the two information losses, present the central claim once, and avoid repeating the contribution list at the end of the section.

### Related work

Comparators 1, 3, 6, and 10 are positioned by capability: temporal mismatch, differential representation, prediction-aware quantization, and bias correction. Our related work should be grouped the same way and should state precise non-identity: PD-Quant optimizes quantization parameters; One-Spike is a single-spike phase-coding converter; our interventions change existing events in a frozen DCGS-derived graph adapter.

### Analysis/method

Comparator 4 pairs a representation argument with efficiency evidence and labels the argument's scope. Our equations should retain the same discipline: `Delta M = A E W^T` and the margin condition are decision bookkeeping, while the codeword equivalence is exact only at the local terminal tensor and event count. The text must explicitly stop short of claiming equality at every consumed tick.

### Results

Comparator 7 organizes results around the diagnostic properties rather than around experiments chronologically. Our Results section should foreground the question in each subsection opening, put full rates in tables, and use prose for direction and interpretation. Historic test results establish an operating regime; they should not be presented as a new method comparison.

### Discussion and conclusion

Comparator 9 separates model performance, ablations, and hardware evidence. Comparator 2 separates algorithmic cost from training cost. Our discussion should similarly separate decision fidelity, whole-network activity, and measured simulation time. The conclusion should not introduce a new efficiency or generality claim.

## Guardrails for the S7 edit

1. Preserve all numbers, confidence intervals, split descriptions, equations, citation keys, and the LLM Usage Statement.
2. Do not turn the audit into a method paper by renaming interventions or controls.
3. Do not remove immediately necessary qualifications: pooled rates are conditional on the intervention distribution; temporal exchanges leave first-prefix profile and absolute amplitude uncontrolled; historical tests were previously exposed.
4. Do not add references from the comparator set unless the manuscript already cites them.
5. Prefer sentence-level clarity, stronger topic sentences, and tighter section transitions over added claims.
