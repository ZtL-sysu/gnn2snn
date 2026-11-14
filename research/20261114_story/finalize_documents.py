"""Populate the research report and manuscript from validated, frozen evidence."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def select(rows,**keys):
    found=[r for r in rows if all(str(r[k])==str(v) for k,v in keys.items())]
    if len(found)!=1:raise ValueError((keys,len(found)))
    return found[0]


def pm(row,metric='accuracy_percent',latex=False):
    symbol=r'$\pm$' if latex else '±'
    return f"{float(row[metric+'_mean']):.2f} {symbol} {float(row[metric+'_sample_sd']):.2f}"


def main():
    val=read(ROOT/'analysis/confirmation/aggregate.csv')
    test=read(ROOT/'analysis/frozen_test/aggregate.csv')
    test_seed=read(ROOT/'analysis/frozen_test/per_seed.csv')
    ideal=read(ROOT/'analysis/ideal/aggregate.csv')
    costs=[json.loads(p.read_text()) for p in sorted((ROOT/'costs').glob('cost_*.json'))]
    def v(ds,t,m,scope='full_validation'):return select(val,dataset=ds,T=t,method=m,scope=scope)
    def q(ds,t,m):return select(test,dataset=ds,T=t,method=m,scope='historically_exposed_test')
    def o(ds,t,m):return select(ideal,dataset=ds,T=t,method=m,scope='full_validation')
    def ms(ds,condition):
        return statistics.mean(next(x['per_graph_ms_median'] for x in c['latency'] if x['condition']==condition) for c in costs if c['dataset']==ds)
    head=ROOT/'paper'
    abstract=r'''Frozen graph classifiers expose a practical question for low-timestep conversion: does matching final graph logits improve calibration beyond reconstructing local activations? We compare six conditions using the same pretrained residual GCNs, training examples and update budget on MNIST and CIFAR10 superpixel classification. Learned conditions calibrate 693 initial-membrane values while retaining hard signed eight-level events. Across three teacher seeds per task, frozen test differences between graph-logit and local reconstruction objectives range from $-0.18$ to $+0.20$ percentage points at one or two active ticks; validation and test results differ in direction. These official splits have historical project exposure, so the endpoints are descriptive. Output error correlates with prediction flips more closely than node error, while ideal-site interventions locate a substantial short-prefix deficit upstream: restoring the first node nonlinearity alone closes about 54\% and 24\% of the one-tick teacher-disagreement gaps. Graph-level objective choice consequently requires evidence beyond error correlation. Synchronized RTX 3060 measurements show that the current diagnostic conversion core takes about 2.4--3.2 times the teacher's time at one tick. The study separates fidelity, calibration effectiveness and software execution cost, with small objective-dependent gains and no general method advantage established in the tested regime.'''
    abstract=abstract.replace('Learned conditions calibrate 693','Matched membrane conditions calibrate 693').replace('2.4--3.2','2.4--3.1')
    (head/'abstract_results.tex').write_text(abstract+'\n',encoding='utf-8')

    lines=[r'\input{result_tables.tex}',r'\subsection{Fidelity and objective choice}',
           r'Tables~\ref{tab:validation}--\ref{tab:test} retain all six conditions. At $T=1$, local reconstruction improves mean test accuracy from '+f"{float(q('MNIST',1,'baseline')['accuracy_percent_mean']):.2f}\\% to {float(q('MNIST',1,'node')['accuracy_percent_mean']):.2f}\\% on MNIST and from {float(q('CIFAR10',1,'baseline')['accuracy_percent_mean']):.2f}\\% to {float(q('CIFAR10',1,'node')['accuracy_percent_mean']):.2f}\\% on CIFAR10.",
           r'The additional effect of the graph objective is much smaller. Graph-minus-local test differences at $T=1/2$ are '+f"{float(q('MNIST',1,'paired_graph-minus-node')['accuracy_percent_mean']):+.2f}/{float(q('MNIST',2,'paired_graph-minus-node')['accuracy_percent_mean']):+.2f} percentage points on MNIST and {float(q('CIFAR10',1,'paired_graph-minus-node')['accuracy_percent_mean']):+.2f}/{float(q('CIFAR10',2,'paired_graph-minus-node')['accuracy_percent_mean']):+.2f} on CIFAR10.",
           r'On full validation, the corresponding means are $-0.10/-0.13$ and $-0.27/+0.43$ points. Excluding the 512 current-screen graphs reduces the CIFAR10 $T=2$ mean to $+0.33$ points; its three-seed interval spans zero. The fixed mixture also lacks a consistent advantage across tasks and splits. Its strongest short-prefix test gain over local calibration is $+0.45$ points on MNIST $T=1$, compared with $-0.10$ on full validation. These outcomes fall short of the screening criterion of a stable, practically meaningful gain in multiple settings.',
           r'By $T=4/8$, uncalibrated conversion is already close to teacher accuracy. The small remaining changes should be read alongside teacher agreement and reconstruction error rather than ranked as reliable improvements.',
           r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{../figures/accuracy_curves.pdf}\caption{Full validation fidelity across three teacher seeds. Error bars are sample SD. Teacher reference bands use the corresponding validation mean and SD. All optimized conditions use the frozen smaller-step configuration.}\label{fig:accuracy}\end{figure*}',
           r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{../figures/paired_objective.pdf}\caption{Graph-minus-local accuracy differences on full validation and the 4,488 examples outside the current screen. Small markers show each teacher seed; intervals use the paired three-seed $t$ distribution with two degrees of freedom. Both subsets have historical exposure.}\label{fig:paired}\end{figure*}',
           r'\subsection{Error association and conditional location}',
           r'On the screen-excluded examples, graph-logit MSE is more correlated with flips than the four-node-site error aggregate in all twelve task/seed/low-prefix cells. Including the two MLP sites raises the local proxy correlation, while the output proxy remains stronger on average. Teacher margin is more predictive still: mean flip-ranking AUC for negative margin is 0.77/0.96 on MNIST $T=1/2$ and 0.73/0.90 on CIFAR10, compared with 0.67/0.61 and 0.59/0.59 for graph-logit MSE. These are marginal associations on the same graphs.',
           r'The ideal interventions identify a different question: which quantizers conditionally cause the observed discrepancy? At $T=1$, restoring all four node sites closes '+f"{100*float(o('MNIST',1,'nodes0_3')['agreement_gap_closed_mean']):.1f}\\%/{100*float(o('CIFAR10',1,'nodes0_3')['agreement_gap_closed_mean']):.1f}\\% of the MNIST/CIFAR10 disagreement gaps, versus {100*float(o('MNIST',1,'head4_5')['agreement_gap_closed_mean']):.1f}\\%/{100*float(o('CIFAR10',1,'head4_5')['agreement_gap_closed_mean']):.1f}\\% when restoring only the two readout sites.",
           r'The first node site alone closes about 54.1\%/23.5\%, and gives the largest individual-site accuracy recovery in both tasks. This supports an important upstream quantization effect in this architecture rather than a dominant readout-only explanation. At $T=2$, node and readout group effects are closer on CIFAR10. Restoring all sites reproduces the folded teacher exactly in these runs. These continuous-valued interventions restore different numbers of sites and can interact; their effects are conditional rather than additive.',
           r'\input{ideal_table.tex}',r'\subsection{Calibration and software cost}',
           r'Local and graph conditions share their inference computation and differ only in fitted initial states. Their measured times are nearly coincident (Figure~\ref{fig:latency}). The one-tick baseline core averages '+f"{ms('MNIST','baseline_T1'):.3f}/{ms('CIFAR10','baseline_T1'):.3f} ms per graph on MNIST/CIFAR10, versus {ms('MNIST','teacher'):.3f}/{ms('CIFAR10','teacher'):.3f} for the teachers.",
           r'The conversion core at $T=8$ averages '+f"{ms('MNIST','baseline_T8'):.3f}/{ms('CIFAR10','baseline_T8'):.3f} ms per graph. These are means of three per-seed batch medians divided by 32, with raw repeated timings retained. The diagnostic core returns site activations as well as logits, so these ratios describe its actual implementation rather than a best-achievable backend bound.",
           r'The 64-update local/graph calibration loops take about 1.1--2.6 seconds per low-prefix condition, excluding shared data loading, checks and target preparation. Dense graph projections and edge reductions run at every tick regardless of zero-event counts. The arithmetic ledger records those projections and reductions as a partial lower bound; it excludes neuron-state arithmetic and memory traffic and therefore cannot serve as a complete energy estimate. Physical energy remains unmeasured.',
           r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{../figures/latency.pdf}\caption{Synchronized diagnostic-core runtime on a fixed cached batch of 32 graphs. Points summarize three teacher-seed medians; error bars show their sample SD. Each condition has ten warmups and thirty randomized interleaved repeats. Data transfer is excluded.}\label{fig:latency}\end{figure*}',
           r'\subsection{Failure conditions}',
           r'The initial $10^{-2}$ step size increased MNIST $T=2$ hard training logit MSE from 0.685 to 0.966; the single smaller-step correction improved optimization but did not produce a stable graph-objective advantage. Shuffling targets substantially degrades fidelity at every tested prefix, establishing target-correspondence sensitivity. That result is compatible with ordinary distillation and does not identify a topology-specific mechanism. The unchanged original screen, all per-seed endpoints and individual-site interventions accompany the summaries.']
    lines=[line.replace('The 64-update local/graph calibration loops take about','The three-seed mean times of the 64-update local/graph calibration loops are about') for line in lines]
    lines=[line.replace(r'\begin{figure*}[t]',r'\begin{figure*}[p]') if '../figures/latency.pdf' in line else line for line in lines]
    lines.append(r'\input{per_seed_table.tex}')
    (head/'generated_results.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    table=[r'\begin{table*}[t]\centering\footnotesize',r'\begin{tabular}{llrrrrr}\toprule',r'Task & $T$ & Baseline & First node (1) & All node (4) & Readout (2) & All sites (6) \\ \midrule']
    for ds in ['MNIST','CIFAR10']:
        for t in [1,2]:table.append(f'{ds} & {t} & '+' & '.join(pm(o(ds,t,m),latex=True) for m in ['baseline','site0','nodes0_3','head4_5','all6'])+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Ideal-site validation accuracy (percent), mean $\pm$ sample SD over three teachers. Parentheses give numbers of restored nonlinearities. These are continuous-valued diagnostic interventions with zero fitted offsets, not deployable spiking competitors or budget-matched algorithms. All six ideal sites recover the teacher.}\label{tab:ideal}\end{table*}']
    (head/'ideal_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    individual=[r'\begin{table*}[p]\centering\footnotesize',r'\begin{tabular}{llrrrrrrrr}\toprule',
                r'Task & Seed & $T$ & ANN & Uncalibrated & Local & Graph & Mixed & Output & Shuffled \\ \midrule']
    for ds in ['MNIST','CIFAR10']:
        for seed in [41,42,43]:
            for t in [1,2,4,8]:
                base=select(test_seed,dataset=ds,seed=seed,T=t,method='baseline',scope='historically_exposed_test')
                values=[float(base['teacher_accuracy_percent'])]+[float(select(test_seed,dataset=ds,seed=seed,T=t,method=m,scope='historically_exposed_test')['accuracy_percent']) for m in ['baseline','node','graph','mix','output','shuffled']]
                individual.append(f'{ds} & {seed} & {t} & '+' & '.join(f'{x:.2f}' for x in values)+r' \\')
            individual.append(r'\addlinespace[2pt]')
    individual += [r'\bottomrule\end{tabular}',r'\caption{Individual frozen test accuracies (percent) underlying Table~\ref{tab:test}. Each row uses the same teacher checkpoint and all 10,000 official test graphs, with no fitting at test time. The official test split has historical exposure.}\label{tab:individual}\end{table*}']
    (head/'per_seed_table.tex').write_text('\n'.join(individual)+'\n',encoding='utf-8')
    (head/'conclusion_results.tex').write_text(r'''Local activation calibration captures substantial one-tick fidelity gains in the tested frozen GCN family. Direct graph-logit matching yields smaller changes that depend on task, seed and split. Better output-error correlation consequently does not suffice to establish a better calibration objective. Ideal interventions identify an important early node-quantization effect, while dense software execution remains costly even at short prefixes. The evidence supports a bounded empirical study and motivates stronger source-faithful controls, structure-preserving mechanism tests and broader confirmation.'''+ '\n',encoding='utf-8')

    report=['# GNN2SNN 本轮研究报告','', '**结论：选择路线 C，交付实证技术报告和论文草稿；当前证据不足以支撑新方法论文投稿。**','',
            '日期：2026-11-14。已完成两数据集 × 三教师种子 × T1/T2/T4/T8 的六条件公平校准、完整验证与冻结测试，以及六组成本测量和十种理想化位置干预。未重新训练教师。所有优化在训练集完成，验证集仅进行一次步长修正；测试阶段没有更新或选择参数。官方验证/测试集有历史暴露，最终测试应视作冻结后的描述性端点。','',
            '主要材料：[论文 PDF](paper/main.pdf)、[论文源码](paper/main.tex)、[完整实验矩阵](experiment_matrix.csv)、[结论证据索引](claims_to_evidence.csv)、[全部文件哈希](ARTIFACTS_SHA256.csv)、[复现说明](README.md)。','',
            '## 1. 核心假设是否成立？','',
            '部分相关性动机成立，方法优越性未成立。输出 logits MSE 与预测翻转的关联高于四个节点层的误差汇总，补入两个 MLP 层后差距缩小；分类 margin 的预测力更强。这些关联没有证明读出层是主要误差源。理想化干预显示，T1 的首个节点量化位置尤其重要，读出侧单独恢复不能解释大部分损失。','',
            '图级目标没有在两任务、多个教师和不同划分上稳定超过局部目标。混合目标同样不稳定。不能把“误差指标更贴近输出”直接转成“优化该指标得到更好的转换”。','',
            '## 2. 相对最强公平基线有何优势？','',
            '完整冻结测试准确率如下，单位 %，均值 ± 三教师种子样本标准差。原始转换、局部、图级与混合条件共享阈值、教师和图；三个学习目标共享 693 参数、512 训练图和 64 步。输出偏置对照只有 10 参数。','',
            '|任务|T|原始转换|局部目标|图级目标|混合目标|输出偏置|','|---|---:|---:|---:|---:|---:|---:|']
    for ds in ['MNIST','CIFAR10']:
        for t in [1,2]:report.append(f'|{ds}|{t}|'+ '|'.join(pm(q(ds,t,m)) for m in ['baseline','node','graph','mix','output'])+'|')
    report += ['', '图级减局部的冻结测试差值为 MNIST T1/T2 +0.20/+0.18 pp，CIFAR10 T1/T2 −0.18/+0.19 pp。相应完整验证差值为 −0.10/−0.13 和 −0.27/+0.43 pp。部分小收益是真实记录，但没有达到预设的跨设置稳定、具有实际意义的 ≥0.5 pp 筛选标准。局部校准本身在 T1 的较大收益，不能算成图级校准的独特贡献。', '',
               '完整每种子结果、配对差和区间：[验证](analysis/confirmation/per_seed.csv)、[冻结测试](analysis/frozen_test/per_seed.csv)。所有六条件，包括失败的打乱目标控制，都保留在完整表中。', '',
               '## 3. 在什么条件下有收益？','',
               '图级目标的局部小收益依赖任务、时间步与划分。排除本轮筛选 512 图后，CIFAR10 T2 的均值仅 +0.33 pp，三种子 t95 区间 [−0.28,+0.93] pp；其他三个低 T 单元均值为负。完整测试上的少量正差不能改变验证方向不稳和效应偏小的事实。T4/T8 原始转换已饱和，微小差异不宜叙述成可靠方法提升。', '',
               '## 4. 是否存在真实计算效率收益？','',
               '没有观察到当前实现相对 ANN 的 GPU 加速。以下是每种子 30 次同步测量的中位数，再对三个种子取均值，单位 ms/graph（32 图缓存批次的批时间除以 32）：','',
               '|任务|ANN|转换 T1|转换 T2|转换 T4|转换 T8|','|---|---:|---:|---:|---:|---:|']
    for ds in ['MNIST','CIFAR10']:report.append('|'+ds+'|'+'|'.join(f'{ms(ds,k):.3f}' for k in ['teacher','baseline_T1','baseline_T2','baseline_T4','baseline_T8'])+'|')
    report += ['', '局部和图级条件的推理代码相同、延迟近乎重合。低 T 的 64 步优化循环约 1.1–2.6 秒，另计共享加载、校验和教师目标准备；不是零成本。事件是带符号、八档幅值的标量事件，不是二值脉冲。零事件不会跳过本实现的密集投影。运算表只是明确标注的部分下界，不能代替完整能耗分母。**硬件能耗：未测量。**', '',
               '[成本口径](COST_DEFINITIONS.md)、[原始重复计时](costs/)、[逐条件校准成本](analysis/confirmation/cost_calibration.csv)。诊断实现额外返回层激活，ANN/转换时间比不能外推为最优部署实现的界。', '',
               '## 5. 哪些实验失败，为什么？','',
               '- 初始 lr=0.01 在 MNIST T2 使硬训练 logits MSE 由 0.685 升至 0.966，存在优化步长问题。只进行了一次十倍降步长修正，全部对照同步改变，之后冻结；没有继续搜索。',
               '- 图级与混合目标相对局部目标没有稳定优势。当前只能评价此参数化和 surrogate 优化器，不能推导所有图级目标都无效。',
               '- 非确定性 CUDA 在 T4 的源码对齐检查出现随运行改变的偏差。确定性设置后四个 T 均精确对齐，失败运行没有进入科学汇总。',
               '- seed42 首次启动因历史校准文件存于 rep/M42/cal.json 而失败；仅修复路径解析，权重、阈值和预算均未改变。',
               '- 打乱目标明显破坏保真度，这是对应关系控制的预期结果；它证明目标配对有用，不能证明图结构特异性。', '',
               '[研究决策与失败记录](DECISIONS.md)、[初始筛选](analysis/screen/)、[一次修正筛选](analysis/low_lr_screen/)、[失败原始记录](runs/failures/)。', '',
               '## 6. 论文应选择哪条路线？','',
               '**路线 C。** 以“强教师下低时间步保真度、校准目标、量化位置与软件成本”为范围明确的实证研究。A 所需的稳定跨设置优势与图结构特异性未建立；现有局部小收益不足以将 B 包装为新方法贡献。线性 RA(EW) 恒等式是动机，不是新理论；没有向动态门控骨干扩张其适用范围。', '',
               '候选标题：① Fidelity, Calibration, and Cost in Low-Timestep GCN-to-SNN Conversion；② When Output Error Does Not Identify a Better Calibration Objective；③ Locating Quantization Error in Converted Graph Classifiers。', '',
               '## 7. 当前是否足以投稿？','',
               '**暂不足。** 已形成可复查的实验闭环与完整草稿，但新颖性、源方法的完整强对照、独立未暴露确认和骨干覆盖仍不足。负结果本身不构成创新。草稿应作为技术报告和后续研究基础，不能按新 SOTA、硬件加速或节能论文投稿。', '',
               '## 8. 最值得追加的三个实验','',
               '1. **源方法忠实强对照。** 在现有六教师上，核验并适配公开校准/阈值迭代方法，保留相同训练样本和评价流程；同时报告参数量、更新数与实际校准时间。区分“目标无效”和“当前 surrogate 优化有限”。',
               '2. **图结构关联的保任务反证。** 在固定图、标签、教师和参数预算下，比较传播/读出敏感度加权与图内打乱相同权重值的控制；结合首层量化参数干预。目标是识别结构关联，而不是通过破坏图本身制造差异。',
               '3. **外部与骨干确认。** 预先固定方案，在更强 GatedGCN（已有三检查点）及一个独立任务/未暴露确认集检验 T1/T2；动态门控只使用实际非线性目标与对应编码审计。若改走效率主线，应另做优化后端或实际硬件测量。', '',
               '## 9. 下一步如何提高论文质量？','',
               '先补上述决定新颖性与一般性的证据，再决定投稿定位。保留首层误差定位这一可检验发现；不要把图级相关性、局部精度小差和事件稀疏性捆成一个未证明的“高效图感知转换”结论。按最终新证据同步改摘要、贡献、实验和局限。', '',
               '## 机制结果与可信起点','',
               '下表给出 T1 理想干预关闭教师不一致缺口的平均比例。它使用连续值的理想 ReLU，保持图、权重和输入不变，无拟合；是定位诊断，不是可部署 SNN。分组恢复的非线性数不同，效应不能相加。','',
               '|任务|首个节点位置（1）|全部节点位置（4）|读出位置（2）|全部位置（6）|','|---|---:|---:|---:|---:|']
    for ds in ['MNIST','CIFAR10']:report.append('|'+ds+'|'+'|'.join(f"{100*float(o(ds,1,m)['agreement_gap_closed_mean']):.1f}%" for m in ['site0','nodes0_3','head4_5','all6'])+'|')
    report += ['', '全部六位置恢复在所有六教师上重现 ANN；空干预逐位复现原转换。首次节点位置是 T1 最大的单位置准确率恢复项。该结论定位当前固定 GCN 的条件量化效应，不能直接宣称传播放大规律或跨骨干定理。', '',
               '[干预完整统计](analysis/ideal/aggregate.csv)、[六位置误差与 margin 分组](analysis/mechanism_v2/)、[初始四节点汇总保留版本](analysis/mechanism/)。', '',
               '九个现有教师均有可复用检查点。GCN 历史测试均值为 MNIST 91.02%、CIFAR10 54.94%；GatedGCN 为 68.77%。Gated 现有转换包含模拟门控，仅有 T4/T8 记录，未将其混入本轮固定 GCN 的新方法汇总。公开 evidence ZIP 已包含九个教师权重与六套 GCN 阈值，可以按 README 恢复，不必重训。', '',
               '[GCN 原始资产审计](audit_baselines.md)、[Gated 资产审计及缺失项](gated_baseline_audit.md)、[相关工作定位](RELATED_WORK.md)。本轮不覆盖历史稿件和其他任务工作区。']
    val_seconds=sum(json.loads(p.read_text())['total_elapsed_seconds'] for p in (ROOT/'runs/confirmation').glob('*/DONE.json'))
    test_seconds=sum(json.loads(p.read_text())['total_elapsed_seconds'] for p in (ROOT/'runs/frozen_test').glob('*/DONE.json'))
    ideal_seconds=sum(json.loads(p.read_text())['elapsed_sec'] for p in (ROOT/'runs/ideal').glob('*/metadata.json'))
    report += ['', '## 预算与停止条件', '',
               f'确认验证运行记录合计 {val_seconds/60:.1f} 分钟，冻结测试 {test_seconds/60:.1f} 分钟，位置干预 {ideal_seconds/60:.1f} 分钟（均含各运行的读取、校验与评估，不是纯 CUDA 核时间）。两轮小筛选、准备和成本计时另有记录。总量远低于 48 小时上限。按无稳定方法优势的止损规则结束本轮方法扩张，完成证据闭环，未为了用满预算重训教师或扩大调参。']
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Populated report and complete manuscript sections from validated evidence')


if __name__=='__main__':main()
