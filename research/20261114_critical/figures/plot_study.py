"""Source-bound publication plots for sensitivity, controls and implementation cost."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
COLORS=dict(baseline='#4D4D4D',random='#A4A9AC',local='#C6834D',identity='#8674A7',shuffled='#55968C',message='#356F9E')
LABELS=dict(baseline='Base',random='Random',local='Local',identity='Feature',shuffled='Permuted',message='Actual graph')
def read(path):
    with (ROOT/path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=300)
    plt.close(fig)
def main():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    paths=['inputs/historical/ideal.csv','analysis/rescue_paired.csv','analysis/rescue_endpoints.csv','analysis/allocation_mechanism_cost.csv']
    ideal=read(paths[0]);paired=read(paths[1]);endpoint=read(paths[2]);cost=read(paths[3])
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.6),sharey=True,constrained_layout=True)
    groups=['site0','nodes0_3','head4_5','all6'];labels=['First node\n(1 site)','All nodes\n(4 sites)','Readout\n(2 sites)','All\n(6 sites)']
    for ax,d,panel in zip(axes,['MNIST','CIFAR10'],['a','b']):
        r=[next(v for v in ideal if v['scope']=='full_validation' and v['dataset']==d and v['T']=='1' and v['method']==m) for m in groups]
        y=[100*float(v['agreement_gap_closed_mean']) for v in r];err=[100*float(v['agreement_gap_closed_sample_sd']) for v in r]
        ax.bar(np.arange(4),y,yerr=err,capsize=3,color=[COLORS['message'],COLORS['shuffled'],COLORS['identity'],'#B3BBC2'],width=.65,edgecolor='white',linewidth=.5)
        ax.set_xticks(np.arange(4),labels);ax.set_ylim(0,112);ax.set_title(f'{panel}  {d}',loc='left',fontweight='bold');ax.set_yticks([0,25,50,75,100]);ax.tick_params(axis='x',length=0)
        for i,value in enumerate(y):ax.text(i,value+err[i]+2,f'{value:.1f}',ha='center',fontsize=7.5)
    axes[0].set_ylabel('Teacher-disagreement gap closed (%)')
    save(fig,'ideal_sensitivity')
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.6),constrained_layout=True)
    controls1=['baseline','local','identity','shuffled','strong_local_lr01'];controls2=['baseline','local','identity','shuffled']
    names={'baseline':'Same T1 base','local':'Local selector','identity':'Feature selector','shuffled':'Permuted selector','strong_local_lr01':'Stronger local fit'}
    for col,d in enumerate(['MNIST','CIFAR10']):
        for row,t in enumerate([1,2]):
            ax=axes[row,col];controls=controls1 if t==1 else controls2
            values=[next(r for r in paired if r['dataset']==d and r['T']==str(t) and r['control']==m) for m in controls]
            mean=np.array([float(v['delta_pp']) for v in values]);lo=np.array([float(v['paired_normal_low']) for v in values]);hi=np.array([float(v['paired_normal_high']) for v in values])
            ax.errorbar(mean,np.arange(len(values)),xerr=[mean-lo,hi-mean],fmt='o',ms=4,color=COLORS['message'],elinewidth=1,capsize=2)
            ax.axvline(0,color='#999999',ls='--',lw=.7);ax.set_yticks(np.arange(len(values)),[names[x] for x in controls]);ax.invert_yaxis();ax.set_xlim(-6,9 if t==1 else 6)
            ax.set_title(f"{'abcd'[row*2+col]}  {d}, T{t}",loc='left',fontweight='bold');ax.set_xlabel('Actual-graph accuracy difference (pp)')
    save(fig,'allocation_contrasts')
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.6),constrained_layout=True)
    sources=[]
    offsets=dict(random=(4,-11),local=(4,-12),identity=(4,-11),shuffled=(4,5),message=(4,8))
    for row,d in enumerate(['MNIST','CIFAR10']):
        ax=axes[row,0]
        for m in ['random','local','identity','shuffled','message']:
            r=next(v for v in cost if v['dataset']==d and v['policy']==m)
            acc=next(v for v in endpoint if v['dataset']==d and v['phase']=='allocation' and v['method']==m and v['T']=='1')
            x=1000*float(r['actual_normalized_message_mse_after']);y=float(acc['accuracy_percent'])
            ax.scatter(x,y,s=26,color=COLORS[m],zorder=3);ax.annotate(LABELS[m],(x,y),xytext=offsets[m],textcoords='offset points',fontsize=7,color=COLORS[m])
        ax.margins(x=.28,y=.25);ax.set_xlabel(r'Actual normalized message MSE ($\times10^{-3}$)');ax.set_ylabel('T1 accuracy (%)');ax.set_title(f"{'ac'[row]}  {d}",loc='left',fontweight='bold')
        path=ROOT/'analysis/allocation_diagnostics'/f'{d}_summary.json';sources.append(path)
        timing=json.loads(path.read_text(encoding='utf-8'))['timing'];policies=list(COLORS)
        times=[np.array(timing['results'][m]['raw_wall_ms'])/timing['batch_graphs'] for m in policies]
        ax=axes[row,1];boxes=ax.boxplot(times,patch_artist=True,widths=.55,showfliers=False,medianprops={'color':'#333333','linewidth':.9})
        for patch_,m in zip(boxes['boxes'],policies):patch_.set_facecolor(COLORS[m]);patch_.set_alpha(.5)
        rng=np.random.default_rng(20261119)
        for i,(m,values) in enumerate(zip(policies,times),1):
            assert abs(np.median(values)-timing['results'][m]['median_ms_per_graph'])<1e-10
            ax.scatter(i+rng.uniform(-.10,.10,len(values)),values,s=6,alpha=.55,color=COLORS[m],linewidths=0)
            ax.annotate(f'{np.median(values):.3f}',(i,np.median(values)),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6)
        ax.set_xticks(np.arange(1,7),[LABELS[m] for m in policies],rotation=25,ha='right');ax.set_ylabel('Synchronized time (ms/graph)');ax.set_title(f"{'bd'[row]}  {d}, cached batch32",loc='left',fontweight='bold')
    save(fig,'mechanism_runtime')
    sources+=[ROOT/x for x in paths]
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n',encoding='utf-8')
    print('Generated three scientific plots (PDF/SVG/PNG)')
if __name__=='__main__':main()
