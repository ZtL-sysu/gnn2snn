from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

SITES = [f'gcn_{i}' for i in range(4)]

def spearman(x, y):
    x=np.asarray(x,dtype=float).reshape(-1); y=np.asarray(y,dtype=float).reshape(-1)
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0: return float('nan')
    def ranks(a):
        order=np.argsort(a,kind='mergesort'); sorted_a=a[order]; r=np.empty(len(a),float)
        i=0
        while i<len(a):
            j=i+1
            while j<len(a) and sorted_a[j]==sorted_a[i]: j+=1
            r[order[i:j]]=(i+j-1)/2+1; i=j
        return r
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])

def read_run(run):
    meta_path=run/'metadata.json'; results_path=run/'results.json'; done_path=run/'DONE.json'
    if not meta_path.is_file() or not results_path.is_file() or not done_path.is_file():
        raise ValueError(f'{run}: missing metadata.json, results.json, or DONE.json')
    done=json.loads(done_path.read_text(encoding='utf-8'))
    if done.get('status')!='complete': raise ValueError(f'{done_path}: status is not complete')
    meta=json.loads(meta_path.read_text(encoding='utf-8')); results=json.loads(results_path.read_text(encoding='utf-8'))
    if not isinstance(results,list): raise ValueError(f'{results_path}: expected list')
    if done.get('rows')!=len(results): raise ValueError(f'{run}: DONE row count does not match results.json')
    found={}; expected=set(tuple(int(t) if i==1 else t for i,t in enumerate((row['method'],row['T']))) for row in results)
    if len(expected)!=len(results): raise ValueError(f'{run}: duplicate method/T rows in results.json')
    if any(int(t) not in meta.get('T',[]) for _,t in expected): raise ValueError(f'{run}: result T missing from metadata')
    for row in results:
        method=row['method']; t=int(row['T']); path=run/f'{method}_T{t}.npz'
        if not path.is_file(): raise FileNotFoundError(f'missing predictions {path}')
        found[(method,t)]=np.load(path,allow_pickle=False)
    return meta,found

def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--runs',type=Path,required=True,help='directory containing per-run output subdirectories'); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run_dirs=sorted(p for p in args.runs.iterdir() if p.is_dir()); loaded=[]
    for d in run_dirs:
        try:
            item=read_run(d)
            loaded.append((d,*item))
        except Exception as e: raise RuntimeError(f'Invalid/partial run cannot be included: {d}: {type(e).__name__}: {e}') from e
    if not loaded: raise RuntimeError('No valid run directories with metadata.json and results.json')
    identities=[(meta['dataset'],meta['seed']) for _,meta,_ in loaded]
    if len(set(identities))!=len(identities): raise ValueError('Duplicate dataset/seed run directories are not allowed')
    by_dataset={}
    for run,meta,_ in loaded:
        ds=meta['dataset']
        signature=(meta.get('steps'),len(meta.get('train_indices',[])),len(meta.get('val_indices',[])),tuple(meta.get('T',[])),meta.get('optimizer'),meta.get('lr'))
        if ds in by_dataset and by_dataset[ds][0]!=signature:
            raise ValueError(f'{run}: incompatible run settings for {ds}; expected {by_dataset[ds][0]}, got {signature}')
        by_dataset.setdefault(ds,(signature,run))
    summary=[]; mechanism=[]; paired_by_key={}; paired_values={}
    for run,meta,arrays in loaded:
        ds=meta['dataset']; seed=meta['seed']; sample_arrays={}
        for (method,t),npz in arrays.items():
            a={k:npz[k] for k in npz.files}; npz.close(); required={'logits','teacher','labels','indices','node_mse','graph_mse','margin','margin_shift','flip'}
            missing=required-set(a)
            if missing: raise ValueError(f'{run}: {method} T{t} missing {sorted(missing)}')
            n=len(a['labels']); finite=np.isfinite(a['logits']).all(1)&np.isfinite(a['teacher']).all(1)&np.isfinite(a['margin'])&np.isfinite(a['margin_shift'])&np.isfinite(a['graph_mse'])&np.isfinite(a['node_mse']).all(1)
            if a['node_mse'].ndim!=2 or a['node_mse'].shape[1]<4: raise ValueError(f'{run}: node_mse must be [graphs, >=4 sites]')
            if not finite.all(): raise ValueError(f'{run}: {method} T{t} contains {int((~finite).sum())} non-finite graph rows')
            if not np.isfinite(a['graph_mse']).all() or not np.isfinite(a['flip']).all(): raise ValueError(f'{run}: {method} T{t} has non-finite diagnostics')
            if len(a['indices'])!=n or len(a['logits'])!=n or len(a['teacher'])!=n: raise ValueError(f'{run}: inconsistent sample counts in {method} T{t}')
            idx=np.asarray(a['indices']); order=np.argsort(idx); idx=idx[order]; y=np.asarray(a['labels'])[order]; z=np.asarray(a['logits'])[order]; teacher=np.asarray(a['teacher'])[order]
            if len(np.unique(idx))!=n: raise ValueError(f'{run}: duplicate validation indices in {method} T{t}')
            sample_arrays[(method,t)]={'a':a,'idx':idx,'y':y,'z':z,'teacher':teacher,'correct':(z.argmax(1)==y),'agree':(z.argmax(1)==teacher.argmax(1))}
        # Within-run pairings use identical indices, labels and teacher predictions.
        anchors=list(sample_arrays.values())
        if anchors:
            first=anchors[0]
            for other in anchors[1:]:
                if not (np.array_equal(first['idx'],other['idx']) and np.array_equal(first['y'],other['y']) and np.array_equal(first['teacher'],other['teacher'])):
                    raise ValueError(f'{run}: method/T bundles disagree on indices, labels, or teacher logits')
        for (method,t),d in sample_arrays.items():
            base=sample_arrays.get(('baseline',t))
            if base is not None and len(base['idx'])==len(d['idx']) and np.array_equal(base['idx'],d['idx']) and np.array_equal(base['y'],d['y']):
                delta=d['correct'].astype(float)-base['correct'].astype(float); bdelta=d['agree'].astype(float)-base['agree'].astype(float)
                d['paired_baseline_mean_pp']=float(delta.mean()*100); d['paired_baseline_sd_pp']=float(delta.std(ddof=1)*100) if len(delta)>1 else 0.; d['paired_baseline_agreement_delta_pp']=float(bdelta.mean()*100)
            else: d['paired_baseline_mean_pp']=float('nan'); d['paired_baseline_sd_pp']=float('nan'); d['paired_baseline_agreement_delta_pp']=float('nan')
            node=sample_arrays.get(('node',t))
            if node is not None and len(node['idx'])==len(d['idx']) and np.array_equal(node['idx'],d['idx']) and np.array_equal(node['y'],d['y']):
                delta=d['correct'].astype(float)-node['correct'].astype(float); d['paired_node_mean_pp']=float(delta.mean()*100); d['paired_node_sd_pp']=float(delta.std(ddof=1)*100) if len(delta)>1 else 0.
            else: d['paired_node_mean_pp']=float('nan'); d['paired_node_sd_pp']=float('nan')
            z=d['z']; teacher=d['teacher']; y=d['y']; a=d['a']; order=np.argsort(a['indices']);
            r={'dataset':ds,'seed':seed,'method':method,'T':t,'n_graphs':len(y),'failure_sample_count':0,'accuracy_percent':float((z.argmax(1)==y).mean()*100),'teacher_agreement_percent':float((z.argmax(1)==teacher.argmax(1)).mean()*100),'logit_mse':float(np.mean((z.astype(float)-teacher.astype(float))**2)),'paired_vs_node_accuracy_delta_pp':d['paired_node_mean_pp'],'paired_vs_node_graph_delta_sd_pp':d['paired_node_sd_pp'],'paired_vs_baseline_accuracy_delta_pp':d['paired_baseline_mean_pp'],'paired_vs_baseline_graph_delta_sd_pp':d['paired_baseline_sd_pp'],'paired_vs_baseline_agreement_delta_pp':d['paired_baseline_agreement_delta_pp']}
            summary.append(r)
            if method in ('node','graph','mix','shuffled','output','baseline'):
                order=np.argsort(a['indices']); g=a['graph_mse'][order]; flip=a['flip'][order].astype(float); abs_shift=np.abs(a['margin_shift'][order]); margins=a['margin'][order]
                for i,site in enumerate(SITES):
                    x=a['node_mse'][order,i]; mechanism.append({'dataset':ds,'seed':seed,'method':method,'T':t,'site':site,'n_graphs':len(x),'spearman_node_mse_vs_flip':spearman(x,flip),'spearman_node_mse_vs_abs_margin_shift':spearman(x,abs_shift),'spearman_graph_mse_vs_flip':spearman(g,flip),'spearman_graph_mse_vs_abs_margin_shift':spearman(g,abs_shift),'spearman_teacher_margin_vs_flip':spearman(margins,flip),'spearman_teacher_margin_vs_abs_margin_shift':spearman(margins,abs_shift)})
    # Aggregate by dataset/method/T over seeds, retaining each seed row above.
    for ds in sorted({r['dataset'] for r in summary}):
        for method in sorted({r['method'] for r in summary if r['dataset']==ds}):
            for t in sorted({r['T'] for r in summary if r['dataset']==ds and r['method']==method}):
                group=[r for r in summary if r['dataset']==ds and r['method']==method and r['T']==t]
                if len(group)<2: continue
                agg={'dataset':ds,'seed':'MEAN±SD','method':method,'T':t,'n_graphs':int(sum(r['n_graphs'] for r in group)),'failure_sample_count':int(sum(r['failure_sample_count'] for r in group))}
                for metric in ['accuracy_percent','teacher_agreement_percent','logit_mse','paired_vs_node_accuracy_delta_pp','paired_vs_node_graph_delta_sd_pp','paired_vs_baseline_accuracy_delta_pp','paired_vs_baseline_graph_delta_sd_pp','paired_vs_baseline_agreement_delta_pp']:
                    v=np.array([r[metric] for r in group],float); v=v[np.isfinite(v)]; agg[metric+'_mean']=float(v.mean()) if len(v) else float('nan'); agg[metric+'_sample_sd']=float(v.std(ddof=1)) if len(v)>1 else float('nan')
                summary.append(agg)
    fields=['dataset','seed','method','T','n_graphs','failure_sample_count','accuracy_percent','teacher_agreement_percent','logit_mse','paired_vs_node_accuracy_delta_pp','paired_vs_node_graph_delta_sd_pp','paired_vs_baseline_accuracy_delta_pp','paired_vs_baseline_graph_delta_sd_pp','paired_vs_baseline_agreement_delta_pp','accuracy_percent_mean','accuracy_percent_sample_sd','teacher_agreement_percent_mean','teacher_agreement_percent_sample_sd','logit_mse_mean','logit_mse_sample_sd','paired_vs_node_accuracy_delta_pp_mean','paired_vs_node_accuracy_delta_pp_sample_sd','paired_vs_baseline_accuracy_delta_pp_mean','paired_vs_baseline_accuracy_delta_pp_sample_sd']
    mech_fields=list(mechanism[0]) if mechanism else ['dataset','seed','method','T','site','n_graphs','spearman_node_mse_vs_flip','spearman_node_mse_vs_abs_margin_shift','spearman_graph_mse_vs_flip','spearman_graph_mse_vs_abs_margin_shift','spearman_teacher_margin_vs_flip','spearman_teacher_margin_vs_abs_margin_shift']
    write_csv(args.output/'results_summary.csv',summary,fields); write_csv(args.output/'mechanism_summary.csv',mechanism,mech_fields)
    note=['# Exploratory screen analysis','',f'Completed run directories included: {len(loaded)}. Failed sample count: 0; runs with invalid or non-finite artifacts cause the analysis to stop.', '', 'These are exploratory results on historically exposed official validation data. Paired differences are within-run, graph-paired comparisons over the exact saved validation indices; mean/sample SD summaries across teacher seeds are descriptive. They do not establish an independent test result, confirmation, novelty, efficiency, or causality. Do not infer graph-structure specificity from shuffled-target results alone.','','## Interpretation note','','The mechanism table reports Spearman associations across validation graphs. Teacher margin is included as a simple confound/context diagnostic because low-margin graphs are more likely to flip and may also show larger reconstruction errors. These marginal correlations are descriptive; they are not adjusted causal effects and do not account for teacher-seed uncertainty.']
    (args.output/'analysis.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
    print(f'Wrote {len(summary)} result rows, {len(mechanism)} mechanism rows from {len(loaded)} complete runs.')
if __name__=='__main__': main()
