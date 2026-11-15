"""Rebuild tables, gate decision and cost plot from raw arrays; no model fitting."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_cached import align_indices, accuracy_bootstrap, sha256

HERE=Path(__file__).resolve().parent
OUT=HERE/'analysis'/'phase3'
METHODS=['local_t1','local_joint','logits','kl','ranking']


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261115)
    rows=[]; paired=[];sources={};gate={};clusters=[];examples=[]
    for ds in ('MNIST','CIFAR10'):
        root=HERE/'runs'/'objectives'/f'{ds}41'
        meta=json.loads((root/'metadata.json').read_text())
        assert meta['test_loaded'] is False and all(meta['source_exact'].values())
        arrays={}
        raw_results=json.loads((root/'results.json').read_text())
        for method in METHODS:
            for T in (1,2,4,8):
                path=root/f'{method}_T{T}.npz'
                sources[str(path.relative_to(HERE))]=sha256(path)
                a=dict(np.load(path));arrays[method,T]=a
                assert np.array_equal(a['indices'],meta['val_indices'])
                base=arrays['local_t1',1]
                assert np.array_equal(a['labels'],base['labels']) and np.array_equal(a['teacher'],base['teacher'])
                p=a['logits'].argmax(1); ap=a['teacher'].argmax(1); y=a['labels']
                raw=next(r for r in raw_results if r['method']==method and r['T']==T)
                acc=float(100*np.mean(p==y));agree=float(100*np.mean(p==ap))
                assert acc==raw['accuracy_percent'] and agree==raw['teacher_agreement_percent']
                # Per-graph counts are exact float32 integers (<2**24); reduce in int64.
                assert np.all(a['events']==np.round(a['events'])) and a['events'].max()<2**24
                events=int(a['events'].astype(np.int64).sum())
                row=dict(dataset=ds,seed=41,method=method,T=T,accuracy_percent=acc,
                    teacher_agreement_percent=agree,teacher_accuracy_percent=float(100*np.mean(ap==y)),
                    events=events,event_reduction_delta_from_printed_result=events-raw['events'],
                    amplitude_sum=float(a['amplitude'].astype(np.float64).sum()),parameters=693,
                    parameter_bytes=2772,calibration_seconds=raw['calibration_seconds'],
                    teacher_correct_conversion_wrong=int(((ap==y)&(p!=y)).sum()),
                    teacher_wrong_conversion_correct=int(((ap!=y)&(p==y)).sum()))
                rows.append(row)
        for method in METHODS[1:]:
            for control in ('local_t1','local_joint'):
                for T in (1,2,4,8):
                    a,b=arrays[method,T],arrays[control,T]
                    perm=align_indices(b['indices'],a['indices'])
                    for metric,target in [('accuracy',b['labels']),('agreement',b['teacher'].argmax(1))]:
                        delta=(a['logits'][perm].argmax(1)==target).astype(float)-(b['logits'].argmax(1)==target).astype(float)
                        mean,lo,hi=accuracy_bootstrap(delta,rng)
                        paired.append(dict(dataset=ds,method=method,control=control,T=T,metric=metric,
                            difference_pp=mean,bootstrap95_low_pp=lo,bootstrap95_high_pp=hi))
        def metric(method,T,key):return next(r[key] for r in rows if r['dataset']==ds and r['method']==method and r['T']==T)
        gate[ds]={}
        for method in ('logits','kl','ranking'):
            gain=min(metric(method,1,'accuracy_percent')-metric(control,1,'accuracy_percent') for control in ('local_t1','local_joint'))
            t2=metric(method,2,'accuracy_percent')-metric('local_t1',2,'accuracy_percent')
            agreement=metric(method,1,'teacher_agreement_percent')-metric('local_t1',1,'teacher_agreement_percent')
            gate[ds][method]=dict(T1_min_accuracy_gain_pp=gain,T2_delta_pp=t2,T1_agreement_delta_pp=agreement,
                admission_pass=bool(gain>=.5 and t2>=-.5 and agreement>=0))
        ip=HERE/'runs'/'intervention_v2'/f'{ds}41'/'graphs.csv'
        with ip.open() as f: ir=list(csv.DictReader(f))
        assert all(x['actual_first_event_changed']=='True' for x in ir)
        sources[str(ip.relative_to(HERE))]=sha256(ip)
        for T in (1,2):
            r=[x for x in ir if int(x['T'])==T]
            ids=sorted(set(int(x['index']) for x in r));counts=[]
            for idx in ids:
                g=[x for x in r if int(x['index'])==idx]
                good=[x for x in g if float(x['changed_message_mse'])<float(x['baseline_message_mse'])-1e-12]
                worse=[x for x in good if float(x['changed_margin'])<float(x['baseline_margin'])-1e-7]
                counts.append([len(good),len(worse)])
                if T==1:
                    for x in good:
                        if int(x['baseline_pred'])==int(x['teacher_class']) and int(x['changed_pred'])!=int(x['teacher_class']):
                            examples.append(x)
            c=np.array(counts);draws=rng.integers(0,len(c),size=(2000,len(c)));tot=c[draws].sum(1)
            ratio=tot[:,1]/tot[:,0]
            clusters.append(dict(dataset=ds,T=T,graphs=len(c),message_improving_interventions=int(c[:,0].sum()),
                margin_worse=int(c[:,1].sum()),margin_worse_fraction=float(c[:,1].sum()/c[:,0].sum()),
                graph_cluster_bootstrap95_low=float(np.quantile(ratio,.025)),graph_cluster_bootstrap95_high=float(np.quantile(ratio,.975))))
    write_csv(OUT/'main_table.csv',rows);write_csv(OUT/'paired.csv',paired)
    write_csv(OUT/'intervention_cluster_ci.csv',clusters);write_csv(OUT/'all_lost_agreement_examples.csv',examples)
    status=dict(gate2_pass=any(all(gate[ds][m]['admission_pass'] for ds in gate) for m in ('logits','kl','ranking')),
        criteria='pre-run DECISIONS.md: both tasks gain>=0.5pp against both local controls, T2 loss<=0.5pp, T1 agreement nonworse',
        per_dataset=gate,conditional_gate3_admitted=False,
        limits='single teacher, selection-exposed validation512; graph-bootstrap CIs are descriptive, not teacher-seed/generalization uncertainty')
    (OUT/'gate.json').write_text(json.dumps(status,indent=2)+'\n')
    (OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
    lines=['# Raw-array summary','', 'Teacher41, exposed validation512; identical parameter vector at T1/2/4/8. All objectives are existing controls.','',
        '|Dataset|Objective|T1 acc/agreement %|T2 acc/agreement %|T4 acc %|T8 acc %|T1 events|',
        '|---|---|---:|---:|---:|---:|---:|']
    for ds in ('MNIST','CIFAR10'):
        for m in METHODS:
            rr={r['T']:r for r in rows if r['dataset']==ds and r['method']==m}
            lines.append(f"|{ds}|{m}|{rr[1]['accuracy_percent']:.4f}/{rr[1]['teacher_agreement_percent']:.4f}|{rr[2]['accuracy_percent']:.4f}/{rr[2]['teacher_agreement_percent']:.4f}|{rr[4]['accuracy_percent']:.4f}|{rr[8]['accuracy_percent']:.4f}|{rr[1]['events']}|")
    lines+=['','Gate2 admission: **'+('PASS' if status['gate2_pass'] else 'FAIL')+'**. See paired.csv for graph bootstrap intervals. No graph-specific new method established.']
    (OUT/'main_table.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'phase3-20261115'})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    colors=['#0072B2','#009E73','#D55E00','#CC79A7','#777777']
    for ax,ds in zip(axes,('MNIST','CIFAR10')):
        for m,c in zip(METHODS,colors):
            r=next(r for r in rows if r['dataset']==ds and r['method']==m and r['T']==1)
            ax.scatter(r['events']/1e6,r['accuracy_percent'],label=m,color=c,s=45)
            ax.annotate(m,(r['events']/1e6,r['accuracy_percent']),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set(title=ds+' T1 (teacher41; exposed val512)',xlabel='Actual nonzero events (millions)',ylabel='Accuracy (%)')
        ax.margins(x=.20,y=.12)
        ax.grid(alpha=.2)
    fig.savefig(OUT/'event_accuracy.png',dpi=180)
    svg=OUT/'event_accuracy.svg'
    fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
