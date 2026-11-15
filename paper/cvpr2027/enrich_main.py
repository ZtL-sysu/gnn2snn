from pathlib import Path
p=Path(__file__).with_name('main.tex');s=p.read_text()
anchor='This perspective connects to early node-conversion analysis.'
s=s.replace(anchor,'For reported interventions, the scalar teacher-class margin is the converted logit of $y$ minus the largest converted competing logit. The competing class can change after intervention. Fixing the teacher\'s original runner-up can miss a newly dominant class. Ground-truth accuracy answers a different question: a converted prediction can disagree with an incorrect teacher and become correct. We retain both metrics instead of treating improved accuracy as evidence of improved conversion fidelity.\n\n'+anchor)
anchor='This is not confined to teachers with a small top-two margin.'
table=r'''\begin{table}[t]
\caption{Core hard-intervention result. Among changes lowering first-tick message MSE, how often does the final teacher-class margin worsen? Each task has 512 graphs and 2,048 attempts. $T=2$ reuses the same interventions and MSE condition. Intervals are 95\% graph-cluster bootstrap.}
\label{tab:hard}\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrl}\toprule
Task & Readout & Adverse / improved & Rate [interval] (\%)\\\midrule
MNIST & $T=1$ & 332 / 811 & 40.94 [37.28,44.57]\\
 & $T=2$ & 395 / 811 & 48.71 [45.02,52.42]\\
CIFAR10 & $T=1$ & 296 / 928 & 31.90 [28.66,35.31]\\
 & $T=2$ & 409 / 928 & 44.07 [40.54,47.60]\\\bottomrule
\end{tabular}\end{table}

'''
s=s.replace(anchor,table+anchor)
anchor='\\subsection{Identical local decoding can change the downstream network}'
extra=r'''The source threshold-iteration control provides a different correction mechanism. Calling its original moment estimates, 100 update steps and fallback rules yields one-tick accuracies of 71.48\%/45.31\% on MNIST/CIFAR10 with calibration horizon one. At four ticks, the same fitted thresholds yield 91.21\%/54.88\%. This comparison establishes that high-prefix fidelity and a severe first-prefix gap coexist under an alternative source-linked calibration, beyond the histogram preset. Its graph placement and eight-level configuration are adaptations; the results do not evaluate the original complete source converter on its published benchmarks.

\paragraph{Does graph structure add a decision signal?}
The actual-message objective uses the task operator, whereas the feature-only control omits propagation and the conjugated control relabels it relative to features. Actual-message allocation is close to conjugated allocation on MNIST ($79.49\%$ versus $79.30\%$), and equal to feature-only allocation on CIFAR10 ($49.22\%$). These controls weaken an attribution of aggregate gains specifically to using the correct adjacency. They are especially necessary on superpixel graphs: outdegree is fixed by construction, so outdegree variance cannot explain graph-to-graph changes. Reconstructed indegree and neighborhood variables do vary, but adding them after teacher margin gives small in-sample incremental $R^2$ values (0.004--0.035 across tasks and prefixes). This is a descriptive association, rather than independent prediction evidence.

'''
s=s.replace(anchor,extra+anchor)
s=s.replace('The upper-left/lower-left split','The upper-left/lower-left split')
p.write_text(s)
p=Path(__file__).with_name('supplement.tex');s=p.read_text();start=s.index('For the main text recurrence with zero state');end=s.index('For currents $(0.01,1.58)$',start);s=s[:start]+s[end:];p.write_text(s)
print('Added core result table, directional definition and source/operator controls')
