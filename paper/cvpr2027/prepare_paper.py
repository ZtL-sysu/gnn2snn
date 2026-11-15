"""Deterministic manuscript-only assets; run in the local dyg-cold Conda env."""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def entries(path):
    text=path.read_text(encoding='utf-8')
    starts=list(re.finditer(r'(?m)^@\w+\s*\{\s*([^,]+),',text))
    return {m.group(1):text[m.start():starts[i+1].start() if i+1<len(starts) else len(text)].strip() for i,m in enumerate(starts)}
refs=entries(ROOT/'research/20261114_critical/paper/references.bib')
refs.update(entries(ROOT/'paper/reference-audit/references-corrections-v6.bib'))
keys=['li2021','bojkovic2024','wu2024ftbc','huang2025dcgs','hwang2025onespike','bojkovic2025','degreequant2021','a2q2023','dwivedi2023']
# Preserve one DCGS citation key throughout the new manuscript.
refs['huang2025']=refs['huang2025dcgs'].replace('{huang2025dcgs,','{huang2025,',1)
keys[keys.index('huang2025dcgs')]='huang2025'
(HERE/'references.bib').write_text('\n\n'.join(refs[k] for k in keys)+'\n',encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,
    'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(3.25,1.65),gridspec_kw={'width_ratios':[1,1.05]})
ax=axs[0]
ax.axvspan(.6,1.5,color='#faece5',zorder=0)
ax.axvline(.6,color='#bd4b4b',lw=1,ls='--')
ax.add_patch(Circle((0,0),1.1,fill=False,lw=.6,color='#9ca3af',ls=':'))
for e,col,label in [((.15,.95),'#32688e','larger'),((.7,.15),'#bd4b4b','smaller')]:
    ax.annotate('',xy=e,xytext=(0,0),arrowprops={'arrowstyle':'->','lw':1.6,'color':col})
    ax.text(e[0]-.15,e[1]+.12,label,color=col,fontsize=7)
ax.scatter([0],[0],s=12,c='black',zorder=5)
ax.set(xlim=(-.35,1.2),ylim=(-.25,1.35),xlabel='decision direction',ylabel='orthogonal direction')
ax.set_title('(a) Norm omits direction')
ax.set_xticks([]);ax.set_yticks([])
ax=axs[1]
vals=[.5078125,.75]
ax.bar([0,1],vals,width=.55,color=['#32688e','#d9822b'])
ax.axhline(.6,color='#bd4b4b',ls='--',lw=1)
ax.text(.5,.615,'boundary',ha='center',fontsize=7,color='#bd4b4b')
ax.set_xticks([0,1],['history A','history B'])
ax.set(ylim=(0,.9),ylabel='decoded output')
ax.set_title('(b) Same integral = 0.8')
ax.text(.5,.84,'two events each',ha='center',fontsize=7)
fig.subplots_adjust(left=.06,right=.99,bottom=.26,top=.83,wspace=.58)
fig.savefig(HERE/'figures/concept.pdf',bbox_inches='tight',pad_inches=.02)
fig.savefig(HERE/'figures/concept.png',dpi=300,bbox_inches='tight',pad_inches=.02)
plt.close(fig)
print('Prepared verified bibliography and numeric concept figure')
