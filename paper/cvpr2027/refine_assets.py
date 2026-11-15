from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import build_assets as b
b.main()
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7})
def finish(fig,name):
    fig.tight_layout(pad=.4)
    fig.savefig(b.FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.02)
    fig.savefig(b.FIG/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)
d=b.read_csv('ideal');d=d[(d.scope=='full_validation')&(d['T']==1)]
methods=['site0','site1','site2','site3','site4','site5','nodes0_3','head4_5']
fig,ax=plt.subplots(figsize=(3.25,2.1))
for ds,offset,col,mark in [('MNIST',-.08,'#32688e','o'),('CIFAR10',.08,'#d9822b','s')]:
    q=d[d.dataset==ds].set_index('method').loc[methods]
    ax.errorbar(np.arange(8)+offset,100*q.agreement_gap_closed_mean,yerr=100*q.agreement_gap_closed_sample_sd,fmt=mark,ms=3,capsize=2,lw=.7,color=col,label=ds)
ax.set_xticks(range(8),['N1','N2','N3','N4','R1','R2','Nodes','Head'],rotation=35)
ax.set_ylabel('Disagreement gap closed (%)');ax.legend(frameon=False,fontsize=7);ax.grid(axis='y',alpha=.2)
finish(fig,'position_recovery')
fig,axs=plt.subplots(1,2,figsize=(3.25,2.1),sharey=True)
for ax,ds,key in zip(axs,['MNIST','CIFAR10'],['intervention_mnist','intervention_cifar']):
    q=b.read_csv(key);q=q[q['T']==1]
    x=q.changed_message_mse-q.baseline_message_mse;y=q.changed_margin-q.baseline_margin
    ax.scatter(x*1000,y,s=2,alpha=.3,c='#32688e',linewidths=0,rasterized=True)
    ax.axhline(0,c='gray',lw=.5);ax.axvline(0,c='gray',lw=.5)
    ax.set_title(ds);ax.xaxis.set_major_locator(MaxNLocator(3));ax.set_xlabel(r'$\Delta$MSE ($10^{-3}$)')
    ax.text(.03,.98,'40.94% adverse' if ds=='MNIST' else '31.90% adverse',transform=ax.transAxes,va='top',fontsize=6.5)
axs[0].set_ylabel(r'$\Delta$ teacher-class margin')
finish(fig,'message_decision')
d=pd.read_csv(b.DATA/'centered_shared_theta.csv')
fig,axs=plt.subplots(1,2,figsize=(3.25,1.9))
for ax,col,title in zip(axs,['centered_over_first_analog_mse','centered_over_first_analog_adds'],['Logit MSE','Projection ADDs']):
    for scale,c,m in [(1,'#32688e','o'),(8,'#d9822b','s')]:
        q=d[d.scale==scale];ax.plot(q['T'],q[col],marker=m,color=c,ms=3,lw=1,label=f'scale {scale}')
    ax.axhline(1,ls='--',c='gray',lw=.6);ax.set_title(title);ax.set_xticks([1,2,4]);ax.set_xlabel('Prefix T')
axs[0].set_ylabel('Centered / raw-input');axs[1].legend(frameon=False,fontsize=6)
finish(fig,'centered_tradeoff')
d=pd.read_csv(b.DATA/'temporal_prefix_events.csv')
fig,ax=plt.subplots(figsize=(3.25,2.0))
for ds,off,c in [('MNIST',-.18,'#32688e'),('CIFAR10',.18,'#d9822b')]:
    vals=d[d.dataset==ds].delta_percent.to_numpy();ax.bar(np.arange(4)+off,vals,width=.35,color=c,label=ds)
ax.axhline(0,c='gray',lw=.6);ax.set_xticks(range(4),['Halve\nall','Halve\nhalf','Double\nall','Double\nhalf'])
ax.set_ylabel('Total event change (%)');ax.legend(frameon=False,fontsize=7)
finish(fig,'temporal_equivalence')
manifest=json.loads((b.HERE/'source_sha_manifest.json').read_text())
manifest['refinement_generator']='paper/cvpr2027/refine_assets.py'
manifest['refinement_sha256']=b.sha(Path(__file__))
manifest['outputs']=[{'path':str(p.relative_to(b.ROOT)).replace('\\','/'),'sha256':b.sha(p),'bytes':p.stat().st_size} for p in sorted([*b.DATA.glob('*.csv'),*b.FIG.glob('*.pdf'),*b.FIG.glob('*.png')])]
(b.HERE/'source_sha_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
