"""Recompute temporal diagnostics and fixed method gates from raw arrays."""
import csv
import json
from pathlib import Path
import numpy as np
from analyze_cached import accuracy_bootstrap, sha256

HERE=Path(__file__).resolve().parent
ARMS=['baseline','nongraph_local','graph_local','nongraph_decision','graph_decision','shuffled_decision','toppin_decision']


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def diagnostics():
    out=HERE/'analysis/prefix_diagnostic';out.mkdir(parents=True,exist_ok=True)
    rows=[];hashes={}
    for direction,folder in [('halve','prefix_diagnostic'),('double','prefix_double')]:
        for ds in ('MNIST','CIFAR10'):
            root=HERE/'runs'/folder/f'{ds}41'
            m=json.loads((root/'metadata.json').read_text())
            assert m['training_only'] and not m['validation_loaded'] and not m['test_loaded']
            assert m['optimizer_updates']==0 and m.get('direction','halve')==direction
            for name,digest in m['source_sha256'].items():assert sha256(root/'source'/name)==digest
            assert json.loads((root/'DONE.json').read_text())['endpoints']==6
            for policy in ('baseline','all_eligible','deterministic_half'):
                old=None
                for T in (1,2):
                    path=root/f'{policy}_T{T}.npz';a=dict(np.load(path));hashes[str(path.relative_to(HERE))]=sha256(path)
                    assert np.array_equal(a['indices'],m['training_indices'])
                    if T==1:old=a['selected_by_channel']
                    else:
                        assert np.array_equal(old,a['selected_by_channel'])
                        assert a['first_decode_error_max'].max()==0
                    rows.append(dict(dataset=ds,direction=direction,policy=policy,T=T,
                        accuracy_percent=float(100*np.mean(a['logits'].argmax(1)==a['labels'])),
                        teacher_agreement_percent=float(100*np.mean(a['logits'].argmax(1)==a['teacher'].argmax(1))),
                        events=int(a['events'].sum()),amplitude_sum=float(a['amplitude'].sum()),
                        deliveries=int(a['deliveries'].sum()),first_events=int(a['first_events'].sum()),
                        first_amplitude_sum=float(a['first_amplitude'].sum()),eligible=int(a['eligible_by_channel'].sum()),
                        occupied=int(a['occupied_by_channel'].sum()),selected=int(a['selected_by_channel'].sum()),
                        local_T2_decode_checked=T==2,local_decode_max_error=float(a['first_decode_error_max'].max())))
    write_csv(out/'endpoints.csv',rows)
    (out/'acceptance.json').write_text(json.dumps(dict(endpoints=24,training_only=True,
        T2_local_decode_exact=True,T1_decode_preserved=False,whole_network_cost_guaranteed=False,
        source_hashes=hashes),indent=2)+'\n')


def method():
    out=HERE/'analysis/temporal';out.mkdir(parents=True,exist_ok=True)
    rows=[];paired=[];audits=[];gates={};hashes={};rng=np.random.default_rng(20261119)
    for ds in ('MNIST','CIFAR10'):
        root=HERE/'runs/temporal'/f'{ds}41'
        m=json.loads((root/'metadata.json').read_text());timing=json.loads((root/'timing.json').read_text())
        assert not m['online_teacher'] and not m['test_loaded'] and not m['labels_used_for_training']
        assert m['controller_parameters']==584 and m['steps']==64 and m['lr']==.01
        assert json.loads((root/'DONE.json').read_text())['endpoints']==14
        for name,digest in m['source_sha256'].items():assert sha256(root/'source'/name)==digest
        arrays={}
        for arm in ARMS:
            for T in (1,2):
                path=root/f'{arm}_T{T}.npz';a=dict(np.load(path));arrays[arm,T]=a
                hashes[str(path.relative_to(HERE))]=sha256(path)
                assert np.array_equal(a['indices'],m['val_indices']) and len(a['labels'])==512
                assert np.isfinite(a['logits']).all()
                if T==2:
                    assert np.array_equal(a['choice_count'],arrays[arm,1]['choice_count'])
                    assert a['first_decode_error_max'].max()==0
                base=arrays['baseline',T]
                assert np.array_equal(a['teacher'],base['teacher']) and np.array_equal(a['labels'],base['labels'])
                rows.append(dict(dataset=ds,arm=arm,T=T,
                    accuracy_percent=float(100*np.mean(a['logits'].argmax(1)==a['labels'])),
                    teacher_agreement_percent=float(100*np.mean(a['logits'].argmax(1)==a['teacher'].argmax(1))),
                    events=int(a['events'].sum()),amplitude_sum=float(a['amplitude'].sum()),deliveries=int(a['deliveries'].sum()),
                    selected=int(a['choice_count'].sum()),new_parameters=0 if arm=='baseline' else 584,
                    warm_ms_per_graph=timing['results']['baseline/plain' if arm=='baseline' else arm+'/warm']['median_ms_per_graph'],
                    cold_ms_per_graph=timing['results']['baseline/plain' if arm=='baseline' else arm+'/cold']['median_ms_per_graph']))
        for arm in ARMS[1:]:
            train=json.loads((root/f'{arm}_training.json').read_text())
            model=np.load(root/f'{arm}_controller.npz')
            assert model.files==['weight'] and model['weight'].shape==(4,146)
            assert len(train['history'])==64 and all(np.isfinite(x['gradient_norm']) for x in train['history'])
            assert any(x['gradient_norm']>0 for x in train['history'])
            audits.append(dict(dataset=ds,arm=arm,initial_hard_loss=train['initial_full_train_hard_objective'],
                final_hard_loss=train['final_full_train_hard_objective'],gradients_finite_nonzero=True,parameters_finite=bool(np.isfinite(model['weight']).all())))
            for T in (1,2):
                a=arrays[arm,T]
                for control in ('baseline','nongraph_decision','shuffled_decision','toppin_decision','ordinary_joint_logits'):
                    c=dict(np.load(HERE/'runs/objectives'/f'{ds}41'/f'logits_T{T}.npz')) if control=='ordinary_joint_logits' else arrays[control,T]
                    assert np.array_equal(c['indices'],a['indices'])
                    for metric,cls in [('accuracy',a['labels']),('agreement',a['teacher'].argmax(1))]:
                        delta=(a['logits'].argmax(1)==cls).astype(float)-(c['logits'].argmax(1)==cls).astype(float)
                        mean,lo,hi=accuracy_bootstrap(delta,rng)
                        paired.append(dict(dataset=ds,arm=arm,control=control,T=T,metric=metric,difference_pp=mean,
                            graph_bootstrap95_low_pp=lo,graph_bootstrap95_high_pp=hi))
        def value(arm,T,key):return next(r[key] for r in rows if r['dataset']==ds and r['arm']==arm and r['T']==T)
        accgain=value('graph_decision',1,'accuracy_percent')-value('baseline',1,'accuracy_percent')
        ng=value('graph_decision',1,'accuracy_percent')-value('nongraph_decision',1,'accuracy_percent')
        prefix=value('graph_decision',2,'accuracy_percent')-value('baseline',2,'accuracy_percent')
        agreement=value('graph_decision',1,'teacher_agreement_percent')-value('baseline',1,'teacher_agreement_percent')
        structural=all(value('graph_decision',1,'accuracy_percent')>value(c,1,'accuracy_percent') for c in ('shuffled_decision','toppin_decision'))
        costs=all(value('graph_decision',T,k)<=value('baseline',T,k)+(1e-9*max(1,value('baseline',T,k)) if k=='amplitude_sum' else 0)
            for T in (1,2) for k in ('events','amplitude_sum','deliveries'))
        ordinary=float(100*np.mean(np.load(HERE/'runs/objectives'/f'{ds}41'/'logits_T1.npz')['logits'].argmax(1)==arrays['baseline',1]['labels']))
        ordinarygain=value('graph_decision',1,'accuracy_percent')-ordinary
        gates[ds]=dict(T1_gain_over_local_pp=accgain,T1_gain_over_nongraph_pp=ng,T1_gain_over_ordinary_logits_pp=ordinarygain,
            T2_delta_pp=prefix,T1_agreement_delta_pp=agreement,structure_control_pass=bool(structural),actual_cost_pass=bool(costs),
            admission_pass=bool(accgain>=.5 and ng>=.5 and ordinarygain>0 and prefix>=-.5 and agreement>=0 and structural and costs))
    write_csv(out/'endpoints.csv',rows);write_csv(out/'paired.csv',paired);write_csv(out/'fit_audit.csv',audits)
    gate=dict(method_gate_pass=all(g['admission_pass'] for g in gates.values()),per_dataset=gates,
        independent_confirmation=False,new_validation_endpoints=28,validation_endpoints_total=112,
        source_hashes=hashes,one_alignment_audit_completed=True,refitting_after_validation=False)
    (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    lines=['# Bounded temporal selector results','',
        'Teacher41, exposed validation512. Both prefixes share final 584 new coefficients. First-site T2 decode is exact; full-network output and costs need not match. T1 error array is a placeholder, not an equality assertion.','',
        '|Dataset|Arm|T1 accuracy/agreement %|T2 accuracy/agreement %|T1/T2 events|T1 warm/cold ms per graph|',
        '|---|---|---:|---:|---|---:|']
    for ds in ('MNIST','CIFAR10'):
        for arm in ARMS:
            r={x['T']:x for x in rows if x['dataset']==ds and x['arm']==arm}
            lines.append(f"|{ds}|{arm}|{r[1]['accuracy_percent']:.4f}/{r[1]['teacher_agreement_percent']:.4f}|{r[2]['accuracy_percent']:.4f}/{r[2]['teacher_agreement_percent']:.4f}|{r[1]['events']}/{r[2]['events']}|{r[1]['warm_ms_per_graph']:.3f}/{r[1]['cold_ms_per_graph']:.3f}|")
    lines+=['','Gate: '+('PASS' if gate['method_gate_pass'] else 'FAIL')+'. No post-validation fitting. Full costs, paired graph bootstrap uncertainty, and training alignment are in adjacent CSV/JSON. GPU wall time is not chip energy.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(gate,indent=2))


if __name__=='__main__':
    diagnostics()
    if (HERE/'runs/temporal/CIFAR1041/DONE.json').exists():method()
