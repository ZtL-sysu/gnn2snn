from pathlib import Path
import json,shutil,hashlib
H=Path(__file__).resolve().parent
S=Path('C:/code/gnn2snn-wb-resume/review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix/results')
D=H/'data/centered_validation';D.mkdir(exist_ok=True)
files=['e1-fullrow-summary-v1/e1_fullrow_summary_v1.json','e1-fullrow-summary-v1/METRICS.csv','e3-e5-v1/e3_e5_criterion_v1.json','e3-e5-v1/e5_latency_v1.json','e3-e5-v1/e3_cost_ledger_v1.json']
manifest=[]
for x in files:
 dst=D/x.replace('/','__');shutil.copyfile(S/x,dst);manifest.append({'branch':'agent/visual-conversion/20261113-wb-resume','commit':'58a2578','source':str(S/x),'copy':str(dst.relative_to(H)),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
(D/'manifest.json').write_text(json.dumps(manifest,indent=2))
p=H/'main.tex';s=p.read_text(encoding='utf-8');anchor='\\subsection{Activity, simulation time, and applicability}'
para=r'''A separate three-teacher campaign evaluates centered conversion on all 5,000 exposed CIFAR10 validation graphs. At $T=4/8$, centered mean accuracy is 70.53\%/70.56\%, within 0.07/0.05 points of the strongest recorded controls. However, its matched dense implementation takes 6.245/12.049 ms per graph against 6.213/3.132 for the strongest controls at those horizons. That campaign also retains nine unresolved cost-accounting items. It therefore extends the evidence for centered fidelity beyond training, while leaving a complete efficiency advantage unestablished. These controls use their own calibration and head conventions; they are kept separate from the shared-threshold training comparison in Fig.~\ref{fig:centered} and from the historical test adapters in Table~\ref{tab:main}.

'''
if 'A separate three-teacher campaign' not in s:s=s.replace(anchor,para+anchor)
p.write_text(s,encoding='utf-8')
p=H/'supplement.tex';s=p.read_text(encoding='utf-8');anchor='\\section{Failure Boundaries and Research Scope}'
para=r'''\paragraph{Separate full-validation centered campaign.}
An earlier independent branch retains a three-teacher, 5,000-graph CIFAR10 validation matrix with raw/shared-threshold coding, exact-first-layer and source-adapter controls. Its centered $T=4/8$ mean accuracies are 70.5267\%/70.5600\%; strongest controls reach 70.5933\%/70.6067\%. Several raw trajectories are numerical aliases (per-consumer, generic and common-subexpression-only) and are not independent method wins. At $T=4$ the strongest accuracy control is centered shared-threshold; at $T=8$ it is the six-site source adapter. Their matched wall times are 6.2130 and 3.1315 ms per graph versus centered's 6.2449 and 12.0486. Timing uses batch 128, the full 5,000 graphs, one warmup pass and three timed repeats. These times belong to a distinct implementation and are not comparable with the cached-batch fixed-GCN timings.

The corresponding declared arithmetic proxy is also not strictly lower at both horizons under either its addition-based or conservative MAC convention. Nine unresolved items include event banks, cache maintenance and actual memory traffic. This campaign supplies validation fidelity, while the shared-threshold training experiment isolates the coordinate change. Neither establishes a complete runtime or hardware-energy advantage. We use this evidence to retain centering as a meaningful representation result without promoting the partial projection proxy to overall efficiency.

'''
if 'Separate full-validation centered campaign' not in s:s=s.replace(anchor,para+anchor)
p.write_text(s,encoding='utf-8')
p=H/'中文说明.md';s=p.read_text(encoding='utf-8').replace('但只有单教师、完整训练集诊断，缺少验证精度、完整成本与可辨识新颖性。','共享阈值的机制比较为单教师完整训练集诊断；另一个独立分支已有三教师全量验证保真，但完整成本和同批计时未建立优势，可辨识新颖性仍不足。').replace('centered 仅支持训练保真与部分投影代理；','centered 支持共享阈值训练诊断及另行三教师验证保真，但未建立完整效率优势；')
p.write_text(s,encoding='utf-8')
p=H/'CLAIMS.md';s=p.read_text(encoding='utf-8');s+='\nSeparate centered full-validation campaign: `data/centered_validation/` preserves summaries, full metric CSV, latency and cost ledgers from branch agent/visual-conversion/20261113-wb-resume at 58a2578, with original byte hashes. Three-teacher fidelity is supported; joint complete-cost/latency advantage is not. Shared-theta training evidence and this validation matrix are separate.\n';p.write_text(s,encoding='utf-8')
print('Integrated three-teacher validation and matched cost evidence without changing historical sources')

