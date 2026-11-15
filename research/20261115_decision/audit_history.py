"""Recompute already-exposed archived baselines; no dataset test split or new inference."""
import csv
import json
from pathlib import Path
import numpy as np
from analyze_cached import sha256

HERE=Path(__file__).resolve().parent
STORY=HERE.parent/'20261114_story'


def main():
    out=HERE/'analysis'/'historical';out.mkdir(parents=True,exist_ok=True)
    rows=[];hashes={}
    with (STORY/'baseline_summary.csv').open() as f: gcn=list(csv.DictReader(f))
    for r in gcn:
        if not r['seed'].isdigit():continue
        path=Path(r['prediction_path']);side=Path(r['sidecar_path'])
        doc=json.loads(side.read_text());assert sha256(path)==doc['predictions_sha256']
        assert sha256(Path(r['checkpoint_path']))==r['checkpoint_sha256']==doc['checkpoint_sha256']
        hashes[str(path)]=sha256(path);hashes[str(side)]=sha256(side)
        a=dict(np.load(path));labels=a['labels'];teacher=a['teacher_logits']
        ta=100*float(np.mean(teacher.argmax(1)==labels));assert abs(ta-float(r['teacher_accuracy_percent']))<1e-7
        for T in (1,2,4,8):
            acc=100*float(np.mean(a[f'T{T}_logits'].argmax(1)==labels))
            assert abs(acc-float(r[f'T{T}_accuracy_percent']))<1e-7
            rows.append(dict(dataset=r['dataset'],seed=int(r['seed']),condition='original_conversion',T=T,
                teacher_accuracy_percent=ta,accuracy_percent=acc,n=len(labels),scope='historically exposed test logits only'))
        # The prior strong all-position control uses the historical frozen-test node arm.
        for T in (1,2,4,8):
            local=STORY/'runs'/'frozen_test'/f"{r['dataset']}{r['seed']}"/f'node_T{T}.npz'
            b=dict(np.load(local));hashes[str(local.relative_to(HERE.parent.parent))]=sha256(local)
            assert np.array_equal(labels,b['labels']) and np.array_equal(a['indices'],b['indices'])
            assert np.allclose(teacher,b['teacher_logits'],rtol=1e-5,atol=1e-4)
            rows.append(dict(dataset=r['dataset'],seed=int(r['seed']),condition='historical_all_position_local',T=T,
                teacher_accuracy_percent=ta,accuracy_percent=100*float(np.mean(b['converted_logits'].argmax(1)==labels)),n=len(labels),
                scope='historically exposed test logits; original local protocol, not new Phase3 calibration'))
    with (STORY/'gated_baseline_summary.csv').open() as f: gated=list(csv.DictReader(f))
    for r in gated:
        if not r['seed'].isdigit() or r['split']!='test' or r['method']!='analog_teacher':continue
        path=Path(r['predictions_path']);doc=json.loads(Path(r['sidecar_path']).read_text())
        assert sha256(path)==doc['predictions_sha256']
        assert sha256(Path(r['checkpoint_path']))==r['checkpoint_sha256']
        a=dict(np.load(path));ta=100*float(np.mean(a['teacher_logits'].argmax(1)==a['labels']))
        assert abs(ta-float(r['teacher_accuracy_percent']))<1e-7
        hashes[str(path)]=sha256(path)
        for T in (4,8):rows.append(dict(dataset='CIFAR10_GatedGCN',seed=int(r['seed']),condition='gated_hybrid',T=T,
            teacher_accuracy_percent=ta,accuracy_percent=100*float(np.mean(a[f'dc_T{T}'].argmax(1)==a['labels'])),n=len(a['labels']),
            scope='historically exposed test; analog gates retained; not pure SNN'))
    means=[]
    for ds,cond,T in sorted(set((r['dataset'],r['condition'],r['T']) for r in rows)):
        group=[r for r in rows if (r['dataset'],r['condition'],r['T'])==(ds,cond,T)]
        assert len(group)==3
        means.append(dict(dataset=ds,condition=cond,T=T,seeds=3,
            teacher_accuracy_mean=float(np.mean([r['teacher_accuracy_percent'] for r in group])),
            accuracy_mean=float(np.mean([r['accuracy_percent'] for r in group]))))
    (out/'audit.json').write_text(json.dumps(dict(archived_logits_only=True,new_test_inference=False,
        test_history_exposed=True,rows=rows,means=means,input_hashes=hashes),indent=2)+'\n')
    print(json.dumps(means,indent=2))


if __name__=='__main__':main()
