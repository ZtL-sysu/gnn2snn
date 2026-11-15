from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(3.25,1.7))
for vals,col,mark,name in [([1,-.5],'#32688e','o','word A'),([.5,.5],'#d9822b','s','word B')]:
    axs[0].plot([1,2],vals,color=col,marker=mark,lw=1,label=name)
    axs[1].plot([1,2],[vals[0],vals[0]+vals[1]/2],color=col,marker=mark,lw=1)
for ax in axs:ax.set_xticks([1,2]);ax.set_xlabel('Tick');ax.axhline(0,color='gray',lw=.5)
axs[0].set_ylabel('Event / q');axs[1].set_ylabel('Decoded prefix / q');axs[0].legend(frameon=False,fontsize=6)
axs[1].annotate('same endpoint',xy=(2,.75),xytext=(1.1,.45),fontsize=6,arrowprops={'arrowstyle':'->','lw':.5})
fig.tight_layout(pad=.4);fig.savefig(HERE/'figures/temporal_mechanism.pdf',bbox_inches='tight');fig.savefig(HERE/'figures/temporal_mechanism.png',dpi=300,bbox_inches='tight');plt.close(fig)
p=HERE/'main.tex';s=p.read_text()
anchor='\\subsection{Interventions, controls, and evidence scope}'
new=r'''\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{figures/temporal_mechanism.pdf}
\caption{The local equivalence test. Two legal codewords have two nonzero events and the same decoded endpoint $3q/4$, but present different first-prefix values to downstream consumers. Source replay verifies eligibility before replacement; the rest of the network is then executed without resetting its states.}
\label{fig:temporal-mechanism}
\end{figure}

'''
if 'fig:temporal-mechanism' not in s:s=s.replace(anchor,new+anchor)
# Tie figure to substantive analysis and clarify established prior work.
s=s.replace('Both decode to $3q/4$ and contain two events.','Both decode to $3q/4$ and contain two events (Fig.~\\ref{fig:temporal-mechanism}).')
s=s.replace('Equality only at its terminal tick is weaker.','Equality only at its terminal tick is weaker. This is consistent with prior temporal-misalignment results~\\cite{bojkovic2025}; our diagnostic isolates the ambiguity while enforcing exact local decoding and local event-count equality inside the graph adapter.')
p.write_text(s)
p=HERE/'supplement.tex';s=p.read_text().replace('[breaklinks,colorlinks','[pagebackref,breaklinks,colorlinks')
s=s.replace('\\begin{tabular}{llrrr}',r'\setlength{\tabcolsep}{3pt}\begin{tabular}{llrrr}')
s=s.replace('\\begin{tabular}{llll}',r'\setlength{\tabcolsep}{5pt}\begin{tabular}{llll}')
p.write_text(s)
print('Finalized mechanism figure and local formatting')
