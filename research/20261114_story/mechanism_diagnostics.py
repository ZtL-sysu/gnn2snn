"""Descriptive error associations and graph-group diagnostics; no causal claims."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def association(x,y):
    if np.ptp(x)==0 or np.ptp(y)==0: return '', ''
    return float(spearmanr(x,y).statistic),float(roc_auc_score(y,x))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[];groups=[];paired=[]
    excluded=set(np.random.default_rng(20261115).permutation(5000)[:512].tolist())
    for run in sorted(args.runs.iterdir()):
        if not (run/'DONE.json').is_file(): raise RuntimeError('incomplete run '+str(run))
        meta=json.loads((run/'metadata.json').read_text())
        for t in [1,2]:
            with np.load(run/f'baseline_T{t}.npz',allow_pickle=False) as z:
                a={key:z[key] for key in z.files}
            if len(a['indices'])!=5000 or not np.array_equal(np.sort(a['indices']),np.arange(5000)):
                raise RuntimeError('incomplete validation coverage')
            for scope,mask in [('full_validation',np.ones(5000,dtype=bool)),
                               ('screen_excluded',np.array([int(i) not in excluded for i in a['indices']]))]:
                flip=a['flip'][mask].astype(int)
                local=a['node_mse'][mask,:4]
                signals={f'node_site_{i}_mse':local[:,i] for i in range(4)}
                signals['node_relative_mse']=np.mean(local/np.maximum(local.mean(0),1e-12),axis=1)
                signals['graph_logit_mse']=a['graph_mse'][mask]
                signals['negative_teacher_margin']=-a['margin'][mask]
                rho={}
                for name,x in signals.items():
                    corr,auc=association(x,flip);rho[name]=corr
                    rows.append(dict(dataset=meta['dataset'],seed=meta['seed'],T=t,scope=scope,
                                     signal=name,n_graphs=len(flip),flip_count=int(flip.sum()),
                                     spearman_with_flip=corr,flip_prediction_auc=auc))
                paired.append(dict(dataset=meta['dataset'],seed=meta['seed'],T=t,scope=scope,
                                   graph_minus_node_spearman=float(rho['graph_logit_mse']-rho['node_relative_mse'])))
                for name in ['nodes','mean_degree','degree_variance','margin']:
                    x=a[name][mask]
                    edges=np.unique(np.quantile(x,np.linspace(0,1,5)))
                    bins=np.searchsorted(edges[1:-1],x,side='right')
                    for group in np.unique(bins):
                        selected=bins==group
                        groups.append(dict(dataset=meta['dataset'],seed=meta['seed'],T=t,scope=scope,
                                           variable=name,quantile_bin=int(group),
                                           lower=float(x[selected].min()),upper=float(x[selected].max()),
                                           graphs=int(selected.sum()),flip_rate_percent=float(flip[selected].mean()*100),
                                           graph_logit_mse=float(a['graph_mse'][mask][selected].mean()),
                                           node_relative_mse=float(signals['node_relative_mse'][selected].mean())))
    write_csv(args.output/'error_associations.csv',rows)
    write_csv(args.output/'graph_groups.csv',groups)
    write_csv(args.output/'graph_vs_node_association.csv',paired)
    note='''# Mechanism diagnostics

Baseline T1/T2 only. Each row is a saved graph prediction, with the teacher's top-one class used to define margin and prediction flip. The final graph-logit MSE is an output-space discrepancy, not an isolated propagation/readout projection. The node aggregate averages four site MSE values after dividing each by its across-graph mean; raw site values are also reported. Correlations and AUC are marginal descriptive associations. Shared teacher margins, graph difficulty and class can confound them. This comparison cannot establish a graph-specific causal bottleneck.

Graph groups use data-dependent quartile boundaries of graph size, degree, degree variance and teacher margin. Duplicate boundaries are removed, so low-variation variables may produce fewer than four groups. Quantile boundaries and actual counts are saved. Full validation and the 4488 examples outside the current screen are kept separate; both have historical exposure. The small number of teacher seeds limits generalization.
'''
    (args.output/'README.md').write_text(note,encoding='utf-8')
    print('Wrote',len(rows),'associations and',len(groups),'group rows')


if __name__=='__main__':main()
