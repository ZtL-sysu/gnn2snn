"""Raw-array effectiveness/cost gate and one fixed alignment audit, no refitting."""
import csv
import json
from pathlib import Path
import numpy as np
from analyze_cached import accuracy_bootstrap,sha256
from summarize_swap_oracle import class_margins
from swap_predictor import ridge_predict,choose_pairs

HERE=Path(__file__).resolve().parent
ARMS=['baseline','nongraph_local','graph_local','nongraph_decision','graph_decision','shuffled_decision','toppin_decision']
COSTS=[('events',1),('events',2),('amplitude',1),('amplitude',2),('deliveries',1),('deliveries',2)]


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    out=HERE/'analysis/learned_swap';out.mkdir(parents=True,exist_ok=True)
    rows=[];paired=[];audits=[];graphcost=[];gate={};hashes={};rng=np.random.default_rng(20261118)
    for ds in ('MNIST','CIFAR10'):
        root=HERE/'runs/learned_swap'/f'{ds}41'
        meta=json.loads((root/'metadata.json').read_text())
        assert not meta['online_teacher'] and not meta['online_downstream_lookahead'] and meta['online_full_passes']==1
        assert not meta['test_loaded'] and not meta['labels_used_for_fit']
        assert json.loads((root/'DONE.json').read_text())['endpoints']==14
        for name,digest in meta['source_sha256'].items():assert sha256(root/'source'/name)==digest
        bank=dict(np.load(HERE/'runs/swap_oracle'/f'{ds}41'/'candidates.npz'))
        timing=json.loads((root/'timing.json').read_text())
        assert timing['teacher_in_timed_block'] is False and timing['state_dependent_proposals_and_prediction_in_timed_block']
        fit={r['arm']:r for r in json.loads((root/'fit_audit.json').read_text())}
        arrays={}
        for arm in ARMS:
            for T in (1,2):
                path=root/f'{arm}_T{T}.npz';a=dict(np.load(path));arrays[arm,T]=a
                hashes[str(path.relative_to(HERE))]=sha256(path)
                assert np.array_equal(a['indices'],meta['val_indices'])
                if arm!='baseline':
                    base=arrays['baseline',T]
                    assert np.array_equal(a['labels'],base['labels']) and np.array_equal(a['teacher'],base['teacher'])
                else:
                    old=dict(np.load(HERE/'runs/objectives'/f'{ds}41'/f'local_t1_T{T}.npz'))
                    assert np.array_equal(a['logits'],old['logits'])
                    assert np.array_equal(a['events'],old['events'].astype(np.int64))
                if T==2:assert np.array_equal(a['choice'],arrays[arm,1]['choice'])
                pred=a['logits'].argmax(1);teacher=a['teacher'].argmax(1)
                rows.append(dict(dataset=ds,arm=arm,T=T,accuracy_percent=float(100*np.mean(pred==a['labels'])),
                    teacher_agreement_percent=float(100*np.mean(pred==teacher)),events=int(a['events'].sum(dtype=np.int64)),
                    amplitude_sum=float(a['amplitude'].sum()),branch_delivery_proxy=int(a['deliveries'].sum(dtype=np.int64)),
                    selected_swap_graphs=int((a['choice']>0).sum()),new_fitted_coefficients=0 if arm=='baseline' else fit[arm]['fitted_coefficients'],
                    model_numeric_storage_bytes=0 if arm=='baseline' else fit[arm]['stored_numeric_bytes'],
                    warm_ms_per_graph=timing['results']['baseline/plain' if arm=='baseline' else arm+'/warm']['median_ms_per_graph'],
                    cold_ms_per_graph=timing['results']['baseline/plain' if arm=='baseline' else arm+'/cold']['median_ms_per_graph']))
        for arm in ARMS[1:]:
            modelpath=root/f'{arm}_model.npz';calpath=root/f'{arm}_calibration.npz'
            model=dict(np.load(modelpath));cal=dict(np.load(calpath));X=cal['features'];valid=cal['valid'];target=cal['targets']
            assert X.shape==(8,128,162) and model['coef'].shape==(163,8)
            assert np.array_equal(cal['indices'],bank['indices'])
            expected_cost=np.stack([bank[f'{key}_T{T}'][1:]-bank[f'{key}_T{T}'][:1] for key,T in COSTS],axis=-1)
            assert np.array_equal(target[:,:,2:],expected_cost)
            if arm.endswith('decision'):
                cls=bank['teacher'].argmax(1)
                expected=np.stack([class_margins(bank[f'logits_T{T}'],cls)[1:]-class_margins(bank[f'logits_T{T}'],cls)[:1] for T in (1,2)],axis=-1)
            else:
                local=-(bank['local_mse'][1:]-bank['local_mse'][:1]);expected=np.stack([local,local],axis=-1)
            assert np.array_equal(target[:,:,:2],expected)
            pred=ridge_predict(model,X)
            assert np.allclose(pred,cal['predictions'],rtol=1e-12,atol=1e-12)
            design=np.column_stack(((X[valid]-model['mean'])/model['std'],np.ones(int(valid.sum()))))
            penalty=np.eye(163)*.001;penalty[-1,-1]=0
            rhs=design.T@target[valid]
            residual=np.linalg.norm((design.T@design+penalty)@model['coef']-rhs)/max(1.,np.linalg.norm(rhs))
            assert residual<1e-8
            chosen=choose_pairs(pred,valid);g=np.arange(128)
            train_joint=np.ones(128,dtype=bool)
            for key,T in COSTS:
                actual=bank[f'{key}_T{T}'][chosen,g]-bank[f'{key}_T{T}'][0]
                tolerance=1e-9*np.maximum(1,bank[f'{key}_T{T}'][0]) if key=='amplitude' else 0
                train_joint&=actual<=tolerance
            val_joint=np.ones(512,dtype=bool);prefix_margin_ok=np.ones(512,dtype=bool)
            for T in (1,2):
                a=arrays[arm,T];b=arrays['baseline',T]
                for key in ('events','amplitude','deliveries'):
                    tolerance=1e-9*np.maximum(1,b[key]) if key=='amplitude' else 0
                    val_joint&=a[key]<=b[key]+tolerance
                if T==2:
                    cls=b['teacher'].argmax(1)
                    prefix_margin_ok=class_margins(a['logits'][None],cls)[0]>=class_margins(b['logits'][None],cls)[0]-1e-7
                for control in ('baseline','nongraph_decision','shuffled_decision','toppin_decision'):
                    c=arrays[control,T]
                    for metric,targetclass in [('accuracy',a['labels']),('agreement',a['teacher'].argmax(1))]:
                        delta=(a['logits'].argmax(1)==targetclass).astype(float)-(c['logits'].argmax(1)==targetclass).astype(float)
                        mean,lo,hi=accuracy_bootstrap(delta,rng)
                        paired.append(dict(dataset=ds,arm=arm,control=control,T=T,metric=metric,difference_pp=mean,
                            graph_bootstrap95_low_pp=lo,graph_bootstrap95_high_pp=hi))
            selected=arrays[arm,1]['choice']>0
            audit=dict(dataset=ds,arm=arm,target_and_bank_alignment_exact=True,normal_equation_relative_residual=float(residual),
                training_mse_per_head=np.square(pred[valid]-cal['targets'][valid]).mean(0).tolist(),
                training_zero_change_mse_per_head=np.square(cal['targets'][valid]).mean(0).tolist(),
                training_selected=int((chosen>0).sum()),training_selected_true_joint_cost_feasible=int(((chosen>0)&train_joint).sum()),
                validation_selected=int(selected.sum()),validation_selected_true_joint_cost_feasible=int((selected&val_joint).sum()),
                validation_selected_T2_margin_nondegraded=int((selected&prefix_margin_ok).sum()))
            audits.append(audit)
            for g,idx in enumerate(arrays[arm,1]['indices']):
                graphcost.append(dict(dataset=ds,arm=arm,index=int(idx),selected=bool(selected[g]),
                    actual_both_prefix_costs_nonincreasing=bool(val_joint[g]),T2_teacher_margin_nondegraded=bool(prefix_margin_ok[g])))
            hashes[str(modelpath.relative_to(HERE))]=sha256(modelpath);hashes[str(calpath.relative_to(HERE))]=sha256(calpath)
        def metric(arm,T,key):return next(r[key] for r in rows if r['dataset']==ds and r['arm']==arm and r['T']==T)
        accgain=metric('graph_decision',1,'accuracy_percent')-metric('baseline',1,'accuracy_percent')
        nongraphgain=metric('graph_decision',1,'accuracy_percent')-metric('nongraph_decision',1,'accuracy_percent')
        t2=metric('graph_decision',2,'accuracy_percent')-metric('baseline',2,'accuracy_percent')
        agreement=metric('graph_decision',1,'teacher_agreement_percent')-metric('baseline',1,'teacher_agreement_percent')
        structural=all(metric('graph_decision',1,'accuracy_percent')>metric(control,1,'accuracy_percent') for control in ('shuffled_decision','toppin_decision'))
        costs=all(metric('graph_decision',T,key)<=metric('baseline',T,key)+(1e-9*max(1,metric('baseline',T,key)) if key=='amplitude_sum' else 0)
            for T in (1,2) for key in ('events','amplitude_sum','branch_delivery_proxy'))
        gate[ds]=dict(T1_gain_over_strong_local_pp=accgain,T1_gain_over_same_capacity_nongraph_pp=nongraphgain,
            T2_delta_pp=t2,T1_agreement_delta_pp=agreement,outperforms_shuffled_and_toppin=structural,
            aggregate_event_amplitude_delivery_budgets_pass=bool(costs),
            admission_pass=bool(accgain>=.5 and nongraphgain>=.5 and t2>=-.5 and agreement>=0 and structural and costs))
    csv_write(out/'endpoints.csv',rows);csv_write(out/'paired.csv',paired);csv_write(out/'per_graph_cost_audit.csv',graphcost)
    verdict=dict(method_gate_pass=all(v['admission_pass'] for v in gate.values()),per_dataset=gate,
        one_alignment_audit_completed=True,teacher_online=False,novel_algorithm_established=False,
        qualification='teacher41, exposed val512, one fixed ridge/pair family; no test/multi-teacher/external confirmation',source_hashes=hashes)
    (out/'gate.json').write_text(json.dumps(verdict,indent=2)+'\n')
    (out/'alignment_audit.json').write_text(json.dumps(dict(refitting_performed=False,hyperparameters_changed=False,
        source_bank_and_hard_baseline_replay='exact in runner for all128 training graphs',
        no_teacher_no_lookahead_unit_test='6*T calls with online teacher sentinel; same choices at prefixes',rows=audits),indent=2)+'\n')
    lines=['# Learned single-pass reassignment: bounded prototype results','',verdict['qualification'],'',
        '|Dataset|Arm|T1 accuracy/agreement %|T2 accuracy/agreement %|T1/T2 actual events|Warm/cold ms per graph (T1)|',
        '|---|---|---:|---:|---|---:|']
    for ds in ('MNIST','CIFAR10'):
        for arm in ARMS:
            rr={r['T']:r for r in rows if r['dataset']==ds and r['arm']==arm}
            lines.append(f"|{ds}|{arm}|{rr[1]['accuracy_percent']:.4f}/{rr[1]['teacher_agreement_percent']:.4f}|{rr[2]['accuracy_percent']:.4f}/{rr[2]['teacher_agreement_percent']:.4f}|{rr[1]['events']}/{rr[2]['events']}|{rr[1]['warm_ms_per_graph']:.3f}/{rr[1]['cold_ms_per_graph']:.3f}|")
    lines+=['','Predeclared method gate: **'+('PASS' if verdict['method_gate_pass'] else 'FAIL')+'**. One alignment audit completed without refitting/tuning. Full costs/paired intervals/constraint violations are in CSV and audit JSON. First-site conservation is not a guarantee of downstream costs.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(method_gate_pass=verdict['method_gate_pass'],per_dataset=gate,
        graph_decision_audits=[r for r in audits if r['arm']=='graph_decision']),indent=2))


if __name__=='__main__':main()
