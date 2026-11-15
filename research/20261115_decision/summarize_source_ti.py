"""Recompute source-TI benchmark table from immutable per-graph outputs."""
import csv
import json
from pathlib import Path
import numpy as np
from analyze_cached import align_indices,accuracy_bootstrap,sha256

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'analysis/source_ti';out.mkdir(parents=True,exist_ok=True)
    rows=[];hashes={};rng=np.random.default_rng(20261116)
    for ds in ('MNIST','CIFAR10'):
        run=HERE/'runs/source_ti'/f'{ds}41'
        meta=json.loads((run/'metadata.json').read_text())
        assert meta['test_loaded'] is False and meta['labels_used'] is False
        assert json.loads((run/'DONE.json').read_text())['endpoints']==8
        for name,digest in meta['snapshots'].items():assert sha256(run/'source'/name)==digest
        results=json.loads((run/'results.json').read_text())
        for H in (1,2):
            fit=json.loads((run/f'H{H}_fit.json').read_text())
            assert fit['effective_steps_n']==H*256 and fit['scale_iterations']==100
            acceptance=json.loads((run/f'H{H}_source_acceptance.json').read_text())
            assert all(acceptance.values())
            for T in (1,2,4,8):
                path=run/f'H{H}_T{T}.npz';a=dict(np.load(path));hashes[str(path.relative_to(HERE))]=sha256(path)
                b=dict(np.load(HERE/'runs/objectives'/f'{ds}41'/f'local_t1_T{T}.npz'))
                perm=align_indices(b['indices'],a['indices'])
                a={k:v[perm] for k,v in a.items()}
                assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['teacher'],b['teacher'])
                acc=100*float(np.mean(a['logits'].argmax(1)==a['labels']))
                agree=100*float(np.mean(a['logits'].argmax(1)==a['teacher'].argmax(1)))
                raw=next(r for r in results if r['calibration_horizon']==H and r['T']==T)
                assert abs(acc-raw['accuracy_percent'])<1e-8 and abs(agree-raw['teacher_agreement_percent'])<1e-8
                events=int(a['events'].sum(dtype=np.int64));assert events==raw['events']
                delta=(a['logits'].argmax(1)==a['labels']).astype(float)-(b['logits'].argmax(1)==b['labels']).astype(float)
                mean,lo,hi=accuracy_bootstrap(delta,rng)
                row=dict(dataset=ds,seed=41,calibration_horizon=H,T=T,source_intended_operating_point=H==T,
                    accuracy_percent=acc,teacher_agreement_percent=agree,events=events,
                    source_minus_local_accuracy_pp=mean,paired_graph_bootstrap95_low_pp=lo,paired_graph_bootstrap95_high_pp=hi,
                    source_nan_fallback_channels=sum(s['source_nan_fallback_count'] for s in fit['sites']),
                    source_negative_to_zero_channels=sum(s['raw_negative_count'] for s in fit['sites']),
                    source_raw_zero_channels=sum(s['raw_zero_count'] for s in fit['sites']),
                    effective_quantization_count=H*256,optimizer_updates=0,
                    statistic_plus_TI_seconds=raw['statistic_plus_TI_seconds'])
                rows.append(row)
    with (out/'endpoints.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Source threshold iteration: raw-derived results','',
        'Teacher41, exposed val512 per dataset; source moments/TI/fallback run unchanged. Eight levels and GCN routing are explicit adaptations. H is calibration horizon; one fitted vector is shared at all evaluated T.','',
        '|Dataset|H|T|Accuracy %|Teacher agreement %|Actual events|Δ accuracy vs strong local pp [graph95%CI]|NaN fallback channels|',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in rows:lines.append(f"|{r['dataset']}|{r['calibration_horizon']}|{r['T']}|{r['accuracy_percent']:.4f}|{r['teacher_agreement_percent']:.4f}|{r['events']}|{r['source_minus_local_accuracy_pp']:+.4f} [{r['paired_graph_bootstrap95_low_pp']:.4f},{r['paired_graph_bootstrap95_high_pp']:.4f}]|{r['source_nan_fallback_channels']}|")
    lines+=['','These are source-algorithm controls, not a new graph encoding contribution, a literal ImageNet reproduction, or independent multi-teacher confirmation. No official test dataset or teacher retraining. No source-scale or level search.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'acceptance.json').write_text(json.dumps(dict(endpoints=len(rows),raw_array_metrics_recomputed=True,
        all_source_snapshot_hashes_match=True,all_source_forward_checks_pass=True,source_hashes=hashes,
        qualification='single teacher, selection-exposed validation; graph bootstrap descriptive only'),indent=2)+'\n')
    print(json.dumps(dict(endpoints=len(rows),intended_T1_T2=[r for r in rows if r['source_intended_operating_point']]),indent=2))


if __name__=='__main__':main()
