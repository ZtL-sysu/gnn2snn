"""Rebuild mechanism/cost table, checking per-graph counts and raw timings."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    out=ROOT/'analysis/allocation_diagnostics';rows=[]
    for dataset in ['MNIST','CIFAR10']:
        a=np.load(out/(dataset+'_per_graph.npz'));s=json.loads((out/(dataset+'_summary.json')).read_text())
        for i,method in enumerate(a['policy_names']):
            t=s['timing']['results'][str(method)];assert len(t['raw_wall_ms'])==20
            median=float(np.median(t['raw_wall_ms'])/s['timing']['batch_graphs'])
            assert abs(median-t['median_ms_per_graph'])<1e-12
            counts=s['allocation_count_checks'][str(method)]
            assert all(x['match'] and x['recomputed']==x['recorded'] for x in counts.values())
            assert int(a['nonzero_events'][:,i].sum())==counts['extra_nonzero_events']['recorded']
            rows.append(dict(dataset=dataset,policy=str(method),graphs=512,
                actual_normalized_message_mse_before=float(a['before_message_actual_normalized'][:,i].mean()),
                actual_normalized_message_mse_after=float(a['after_message_actual_normalized'][:,i].mean()),
                actual_local_mse_after=float(a['after_local'][:,i].mean()),median_ms_per_graph=median,
                baseline_median_ms_per_graph=s['timing']['results']['baseline']['median_ms_per_graph'],
                extra_nonzero_events=int(a['nonzero_events'][:,i].sum())))
    with (ROOT/'analysis/allocation_mechanism_cost.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Verified raw-derived mechanism/count/timing rows:',len(rows))
if __name__=='__main__':main()
