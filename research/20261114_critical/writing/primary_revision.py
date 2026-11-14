"""Apply primary-authored, request-bound scientific edits for the live ARC19 call."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCHANGE=ROOT/'writing/arc_run_v7/native-exchange-19'
request=EXCHANGE/'call_01_request.json'
task=json.loads(json.loads(request.read_text(encoding='utf-8'))['messages'][-1]['content'])
s=task['current_manuscript'].replace('\r','')
s=re.sub(r'\n{3,}','\n\n',s)
def replace(old,new):
    global s
    assert s.count(old)==1,(old[:70],s.count(old))
    s=s.replace(old,new)
replace('including approaches for spiking language models and single-spike conversion [fas2025, onespike2024].',
        'including spiking language models [fas2025] and single-spike phase coding, which has also been applied to GCN conversion [onespike2024].')
replace('They do not resolve whether targeting the actual next graph message improves on matched local and operator controls in the setting studied here.',
        'These coding and calibration precedents motivate the present conditional comparison of local reconstruction and actual-operator message targets.')
replace('## 3 Method\n\n[FIGURE:framework]',
        r'## 3 Method'+'\n\n'+r'Figure \ref{fig:framework} summarizes the frozen computation and the alternative diagnostic and finite-event tests.'+'\n\n[FIGURE:framework]')
replace('For first-site message calibration, we average the normalized next-message error and the corresponding local loss.',
        r'Let $L_0$ be the first-site normalized local error, $\overline L$ the six-site local mean, and $L_m$ the normalized first-site next-message error. First-message fitting minimizes $(L_0+L_m)/2$ with 146 offsets; all-message fitting minimizes $(\overline L+L_m)/2$ with 693 offsets. Both message variants target the first site\'s immediate next message; the latter adds no later-site message terms.')
replace('The feature control substitutes the identity for $A$; the structural control uses a fixed within-graph conjugation $P^{-1}AP$.',
        r'The feature control substitutes the identity for $A$; the structural control uses a fixed within-graph conjugation $P^{-1}AP$. A CPU generator initialized with seed 20261116 draws permutations successively within each fixed batch; an identity draw for a graph with more than one node is rolled. The same graph and batch order shares $P$ across arms. The task graph and labels remain fixed while the operator is relabeled relative to the features.')
replace(r'let $u_t=u_{t-1}+c_t/t$ and emit',r'set $u_0=0$, let $u_t=u_{t-1}+c_t/t$, and emit')
replace('Single-site and grouped effects are conditional and nonadditive.',
        r'For each teacher, let $d_B$ and $d_I$ be the fractions of predictions disagreeing with that teacher under baseline and intervention on the same graphs. Disagreement-gap closure is $1-d_I/d_B$, averaged across teachers before reporting its sample SD. Single-site and grouped effects are conditional and nonadditive.')
replace('The alphabet-capacity probe changes only the first site.',
        'The diagnostic families change different resources: offset fits change 146 or 693 initial-state scalars, alphabet fits change 1,168 first-site amplitudes, and allocation changes which nodes receive extra first-tick events while preserving the fitted offsets. Comparisons within each family isolate its stated objective or control; comparisons across families assess different capacities and fitting budgets. The alphabet-capacity probe changes only the first site.')
replace('These comparisons are exploratory; the other 4,488 previously exposed validation graphs do not constitute an independent external set.',
        'Many candidate families and settings reuse this fixed subset for comparisons and the expansion decision, so the endpoints are selection-exposed and descriptive. The other 4,488 previously exposed validation graphs do not constitute an independent external set.')
replace('Historical uncertainty is the sample standard deviation across three teacher seeds; new paired intervals describe graph sampling for one teacher only, not seed variation, multiplicity, or external generalization.',
        r'Historical uncertainty is the sample standard deviation across three teacher seeds. For a paired accuracy contrast on $n=512$ graphs, define $q_i=\mathbf{1}(\hat y_i^{\mathrm{actual}}=y_i)-\mathbf{1}(\hat y_i^{\mathrm{control}}=y_i)$. The estimate is $100\overline q$ percentage points, with descriptive 95\% normal interval $100(\overline q\pm1.96s_q/\sqrt n)$, where $s_q$ is the sample SD. These intervals describe graph sampling for one teacher; they do not account for selection, multiplicity, teacher variation or external generalization.')
replace('Across the historical three-teacher test evaluation,',r'Table \ref{tab:historical} summarizes the historical three-teacher test evaluation. Across these teachers,')
replace('The ideal causal diagnostic indicates substantial but site-dependent conditional recoverability at $T=1$.',
        r'The ideal causal diagnostic in Figure \ref{fig:ideal_sensitivity} indicates substantial but site-dependent conditional recoverability at $T=1$.')
replace('\n\n[TABLE:realizability]\n\n### 4.2 Calibration objectives and realizability', '\n\n### 4.2 Calibration objectives and realizability')
replace('These are single-teacher validation accuracies, not seed averages.',
        'These are single-teacher validation accuracies, not seed averages.\n\n[TABLE:realizability]')
replace('The allocation path also added diagnostic runtime in the measured implementation.',
        r'Figure \ref{fig:mechanism_runtime} and Table \ref{tab:mechanism_cost} separate immediate message reconstruction from measured implementation cost. The allocation path added diagnostic runtime.')
replace('[TABLE:mechanism_cost]\n\n## 5 Discussion',
        '[TABLE:mechanism_cost]\n\nThe recorded offline T1 fitting loops for the first-site and all-site local/message screens took 1.1214–1.7101 s for 64 Adam updates on cached batches of 32 on the RTX 3060. Each value is a single observed fit-loop duration, including gradient checks and trace collection, rather than a repeated speed benchmark. Teacher-target caching, threshold estimation and data setup fall outside these loops. These one-time fitting observations are reported separately from per-graph allocation timing and do not estimate total pipeline cost.\n\n## 5 Discussion')
replace('The evidence supports a bounded conclusion about these tested methods and conditions, while leaving their reliability across teachers and independently sampled data unresolved.',
        'The evidence supports a bounded empirical conclusion about these tested methods and conditions. It leaves reliability across teachers and independent tasks, complete source-converter comparisons, and the novelty and breadth required for a method contribution unresolved.')
assert len(re.findall(r'^## [1-6] ',s,re.M))==6
assert s.count('[TABLE:')==s.count('[FIGURE:')==4
assert not any(ord(c)<32 and c not in '\n\t' for c in s)
answer={'manuscript':s,'notes':'Primary native assistant took over after Luna usage-limit failure. Targeted scientific revisions only: exact objectives, permutation, initialized recovery and closure, paired95% intervals and selection exposure, offline fitting boundary, comparator roles, primary-source positioning, visual narration. Original numbers/equations/citations preserved except added definitions and verified fit-loop observations. No new experiment or independent review is claimed.'}
with (EXCHANGE/'call_01_response.json').open('x',encoding='utf-8') as f:json.dump(answer,f,indent=2,ensure_ascii=False)
with (EXCHANGE/'call_01_binding.json').open('x',encoding='utf-8') as f:json.dump(dict(request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),execution_identity={'role':'primary native assistant','model':'not exposed','reasoning_effort':'not exposed','reason':'Luna usage limit; primary authoring of targeted scientific revision'}),f,indent=2)
print('Bound primary revision:',len(s.split()),'words')
