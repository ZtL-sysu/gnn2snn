"""Final reproducibility acceptance; engineering success is not scientific success."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from analyze_cached import sha256

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def _load_npz(path):
    with np.load(path,allow_pickle=False) as f:
        return {key:f[key].copy() for key in f.files}


def _assert_metric_row(arrays,row,*,prefix=False):
    labels=arrays['labels']
    logits=arrays['logits']
    teacher=arrays['teacher']
    assert np.isfinite(logits).all() and np.isfinite(teacher).all()
    acc=100*np.mean(logits.argmax(1)==labels)
    agreement=100*np.mean(logits.argmax(1)==teacher.argmax(1))
    assert np.isclose(acc,float(row['accuracy_percent']),rtol=0,atol=1e-12)
    assert np.isclose(agreement,float(row['teacher_agreement_percent']),rtol=0,atol=1e-12)
    for key in ('events','deliveries'):
        assert arrays[key].dtype==np.int64
        if key in row:
            assert int(arrays[key].sum(dtype=np.int64))==int(row[key])
    assert arrays['amplitude'].dtype==np.float64 and np.isfinite(arrays['amplitude']).all()
    assert np.isclose(arrays['amplitude'].sum(dtype=np.float64),float(row['amplitude_sum']),rtol=1e-12,atol=1e-8)
    if prefix:
        for key in ('first_events','eligible_by_channel','occupied_by_channel','selected_by_channel'):
            assert arrays[key].dtype==np.int64
        for key,rowkey in (('first_events','first_events'),('first_amplitude','first_amplitude_sum'),
                           ('eligible_by_channel','eligible'),('occupied_by_channel','occupied_T1'),
                           ('selected_by_channel','selected')):
            if key=='first_amplitude':
                assert arrays[key].dtype==np.float64 and np.isfinite(arrays[key]).all()
            total=arrays[key].sum(dtype=np.float64 if 'amplitude' in key else np.int64)
            assert np.isclose(total,float(row[rowkey]),rtol=1e-12,atol=1e-8)


def verify_prefix_diagnostics(check):
    expected_policies=('baseline','all_eligible','deterministic_half')
    run_count=0
    for direction,folder in (('halve','prefix_diagnostic'),('double','prefix_double')):
        for ds in ('MNIST','CIFAR10'):
            run=HERE/'runs'/folder/f'{ds}41'
            assert run.is_dir(),f'missing required prefix run: {run}'
            done=json.loads((run/'DONE.json').read_text())
            assert done['endpoints']==6
            meta=json.loads((run/'metadata.json').read_text())
            assert meta.get('direction','halve')==direction and meta['training_only']
            assert meta['validation_loaded'] is False and meta['test_loaded'] is False
            assert meta['online_teacher'] is False and meta['optimizer_updates']==0
            ids=np.asarray(meta['training_indices'],dtype=np.int64)
            assert ids.shape==(128,) and len(np.unique(ids))==128
            objective=json.loads((HERE/'runs/objectives'/f'{ds}41'/'metadata.json').read_text())
            assert ids.tolist()==objective['train_indices'][:128]
            assert meta['baseline_parameters_sha256']==sha256(HERE/'runs/objectives'/f'{ds}41'/'local_t1_parameters.npz')
            assert meta['checkpoint_sha256']==objective['checkpoint_sha256']
            assert meta['calibration_sha256']==objective['calibration_sha256']
            # Train and val IDs are split-local indices and may numerically
            # overlap; provenance is established by the training-only run
            # metadata plus exact equality to the objective train prefix.
            assert len(objective['val_indices'])==512 and meta['validation_loaded'] is False
            for name,digest in meta['source_sha256'].items():
                assert sha256(run/'source'/name)==digest
            result_rows=json.loads((run/'results.json').read_text())
            assert len(result_rows)==6
            indexed={(r['policy'],int(r['T'])):r for r in result_rows}
            assert set(indexed)=={(p,T) for p in expected_policies for T in (1,2)}
            for policy in expected_policies:
                t1=_load_npz(run/f'{policy}_T1.npz');t2=_load_npz(run/f'{policy}_T2.npz')
                for T,a in ((1,t1),(2,t2)):
                    assert np.array_equal(a['indices'],ids)
                    assert a['logits'].shape==(128,10) and a['teacher'].shape==(128,10)
                    assert a['labels'].dtype==np.int64 and a['events'].dtype==np.int64
                    assert a['deliveries'].dtype==np.int64 and a['first_events'].dtype==np.int64
                    assert a['first_decode_error_max'].shape==(128,)
                    assert np.isfinite(a['first_decode_error_max']).all()
                    _assert_metric_row(a,indexed[(policy,T)],prefix=True)
                    assert np.isclose(float(a['first_decode_error_max'].max()),
                                      float(indexed[(policy,T)]['first_decode_max_abs']),rtol=0,atol=1e-12)
                    run_count+=1
                    if policy!='baseline':
                        assert np.array_equal(a['labels'],_load_npz(run/f'baseline_T{T}.npz')['labels'])
                        assert np.array_equal(a['teacher'],_load_npz(run/f'baseline_T{T}.npz')['teacher'])
                assert np.array_equal(t1['selected_by_channel'],t2['selected_by_channel'])
                assert np.all(t2['first_decode_error_max']==0)
                baseline1=_load_npz(run/'baseline_T1.npz')
                baseline2=_load_npz(run/'baseline_T2.npz')
                assert np.array_equal(t1['first_events'],baseline1['first_events'])
                assert np.array_equal(t2['first_events'],baseline2['first_events'])
            assert len({(r['policy'],r['T']) for r in result_rows})==6
    assert run_count==24
    out=HERE/'analysis/prefix_diagnostic'
    acceptance=json.loads((out/'acceptance.json').read_text())
    assert acceptance['endpoints']==24 and acceptance['training_only'] is True
    assert acceptance['T2_local_decode_exact'] is True
    assert len(acceptance['source_hashes'])==24
    with (out/'endpoints.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==24 and {r['direction'] for r in rows}=={'halve','double'}
    assert all(r['local_T2_decode_checked']=='True' and float(r['local_decode_max_error'])==0 for r in rows if r['T']=='2')
    for path,digest in acceptance['source_hashes'].items():
        assert sha256(HERE/path)==digest
    check['prefix_diagnostic_24_training_only_rows_raw_arrays_and_source_hashes']='pass'


def verify_temporal_runs(check):
    arms=('baseline','nongraph_local','graph_local','nongraph_decision','graph_decision','shuffled_decision','toppin_decision')
    new_arms=arms[1:]
    count=0
    for ds in ('MNIST','CIFAR10'):
        run=HERE/'runs/temporal'/f'{ds}41'
        assert run.is_dir(),f'missing required temporal run: {run}'
        meta=json.loads((run/'metadata.json').read_text())
        assert json.loads((run/'DONE.json').read_text())['endpoints']==14
        assert len(meta['train_indices'])==128 and len(meta['val_indices'])==512
        assert meta['test_loaded'] is False and meta['labels_used_for_training'] is False
        assert meta['online_teacher'] is False and meta['online_downstream_lookahead'] is False
        assert meta['controller_parameters']==584 and meta['trainable_parameters_per_arm']==584
        assert meta['steps']==64 and meta['lr']==0.01 and meta['clamping'] is False
        assert meta['arms']==list(new_arms)
        obj=json.loads((HERE/'runs/objectives'/f'{ds}41'/'metadata.json').read_text())
        assert meta['train_indices']==obj['train_indices'][:128]
        assert meta['val_indices']==obj['val_indices']
        assert meta['checkpoint_sha256']==obj['checkpoint_sha256']
        assert meta['calibration_sha256']==obj['calibration_sha256']
        assert meta['baseline_parameters_sha256']==sha256(HERE/'runs/objectives'/f'{ds}41'/'local_t1_parameters.npz')
        for name,digest in meta['source_sha256'].items():
            assert sha256(run/'source'/name)==digest
        all_arrays={}
        rows=json.loads((run/'results.json').read_text())
        row_map={(r['arm'],int(r['T'])):r for r in rows}
        assert len(rows)==14 and set(row_map)=={(arm,T) for arm in arms for T in (1,2)}
        for arm in arms:
            for T in (1,2):
                a=_load_npz(run/f'{arm}_T{T}.npz');all_arrays[(arm,T)]=a
                assert np.array_equal(a['indices'],np.asarray(meta['val_indices'],dtype=np.int64))
                assert a['indices'].dtype==np.int64 and a['labels'].dtype==np.int64
                assert a['logits'].shape==(512,10) and a['teacher'].shape==(512,10)
                assert a['events'].dtype==np.int64 and a['deliveries'].dtype==np.int64
                assert a['choice_count'].dtype==np.int64 and np.isfinite(a['amplitude']).all()
                _assert_metric_row(a,row_map[(arm,T)])
                assert int(a['choice_count'].sum(dtype=np.int64))==int(row_map[(arm,T)]['changed_coordinates'])
                assert row_map[(arm,T)]['controller_parameters']==(0 if arm=='baseline' else 584)
                if T==2:
                    assert np.all(a['first_decode_error_max']==0)
                    assert float(a['first_decode_error_max'].max())==float(row_map[(arm,T)]['first_decode_error_max'])
                assert np.array_equal(a['labels'],all_arrays.get(('baseline',T),a)['labels'])
                if ('baseline',T) in all_arrays:
                    assert np.array_equal(a['teacher'],all_arrays[('baseline',T)]['teacher'])
                count+=1
            assert np.array_equal(all_arrays[(arm,1)]['choice_count'],all_arrays[(arm,2)]['choice_count'])
        baseline_params=HERE/'runs/objectives'/f'{ds}41'
        for T in (1,2):
            current=all_arrays[('baseline',T)]
            old=_load_npz(baseline_params/f'local_t1_T{T}.npz')
            assert np.array_equal(current['logits'],old['logits'])
            assert np.array_equal(current['labels'],old['labels'])
            assert np.array_equal(current['teacher'],old['teacher'])
            assert np.array_equal(current['events'],old['events'].astype(np.int64))
            # Historical instrumentation accumulates in float32; temporal replay
            # sums float64 events. Bound this reduction-order difference tightly.
            assert np.allclose(current['amplitude'],old['amplitude'],rtol=5e-7,atol=1e-6)
        for arm in new_arms:
            model=_load_npz(run/f'{arm}_controller.npz')
            assert list(model)==['weight'] and model['weight'].shape==(4,146)
            assert np.isfinite(model['weight']).all()
            training=json.loads((run/f'{arm}_training.json').read_text())
            assert training['steps']==64 and training['parameters']==584 and training['clamping'] is False
            assert np.isfinite(training['initial_full_train_hard_objective'])
            assert np.isfinite(training['final_full_train_hard_objective'])
            assert len(training['history'])==64
            gradients=np.asarray([x['gradient_norm'] for x in training['history']],dtype=np.float64)
            assert np.isfinite(gradients).all() and np.any(gradients>0)
        timing=json.loads((run/'timing.json').read_text())
        assert timing['teacher_in_timed_block'] is False
        assert timing['cold_includes_temporal_cache_build'] is True
        assert timing['warm_cache_build_excluded'] is True
        expected_timing={'baseline/plain'}|{f'{a}/{t}' for a in new_arms for t in ('warm','cold')}
        assert set(timing['results'])==expected_timing
        for item in timing['results'].values():
            assert len(item['raw_ms'])==5 and np.isfinite(item['raw_ms']).all()
    assert count==28

    out=HERE/'analysis/temporal'
    assert out.is_dir(),f'missing summarize_temporal output: {out}'
    gate=json.loads((out/'gate.json').read_text())
    assert isinstance(gate['method_gate_pass'],bool)
    assert gate['new_validation_endpoints']==28 and gate['validation_endpoints_total']==112
    assert gate['independent_confirmation'] is False
    assert set(gate['per_dataset'])=={'MNIST','CIFAR10'}
    for result in gate['per_dataset'].values():
        for key in ('structure_control_pass','actual_cost_pass','admission_pass'):
            assert isinstance(result[key],bool)
    with (out/'endpoints.csv').open(newline='',encoding='utf-8') as f:
        summary_rows=list(csv.DictReader(f))
    assert len(summary_rows)==28
    assert {(r['dataset'],r['arm'],int(r['T'])) for r in summary_rows}=={
        (ds,arm,T) for ds in ('MNIST','CIFAR10') for arm in arms for T in (1,2)}
    for path,digest in gate['source_hashes'].items():
        assert sha256(HERE/path)==digest
    assert (out/'summary.md').is_file() and (out/'paired.csv').is_file() and (out/'fit_audit.csv').is_file()
    check['temporal_28_validation_endpoints_584_parameters_baseline_and_gate']='pass'


def script_identity(name,digest):
    current=HERE/name
    if sha256(current)==digest:return 'current'
    archived=subprocess.check_output(['git','show','7cb5021:research/20261115_decision/'+name],cwd=ROOT)
    assert hashlib.sha256(archived).hexdigest()==digest, 'unrecoverable executed source: '+name
    return 'git:7cb5021'


def main():
    assert Path(sys.executable).parent.name=='dyg-cold'
    check={}
    cached=json.loads((HERE/'analysis/cached/summary.json').read_text())
    assert sum(x['source_hash_and_metrics_verified'] for x in cached['raw_endpoint_audit'].values())==142
    assert cached['endpoint_audit']['mismatch_count']==0 and not cached['missing_inputs']
    with (HERE/'analysis/cached/input_hashes.csv').open() as f:
        for r in csv.DictReader(f):assert sha256(ROOT/r['path'])==r['sha256']
    check['historical_142_raw_endpoints']='pass'
    gate=json.loads((HERE/'analysis/phase3/gate.json').read_text())
    assert gate['gate2_pass'] is False and gate['conditional_gate3_admitted'] is False
    check['scientific_gate2']='FAIL, retained without changing thresholds'
    with (HERE/'analysis/phase3/main_table.csv').open() as f: table=list(csv.DictReader(f))
    assert len(table)==40
    for ds in ('MNIST','CIFAR10'):
        d=HERE/'runs/objectives'/f'{ds}41'
        meta=json.loads((d/'metadata.json').read_text())
        assert meta['test_loaded'] is False and all(meta['source_exact'].values())
        check[f'{ds}_objective_source']=script_identity('run_objectives.py',meta['script_sha256'])
        assert sha256(HERE/'decision_objectives.py')==meta['objectives_sha256']
        assert json.loads((d/'DONE.json').read_text())['endpoints']==20
        for row in [r for r in table if r['dataset']==ds]:
            a=dict(np.load(d/f"{row['method']}_T{row['T']}.npz"))
            assert np.isfinite(a['logits']).all()
            assert 100*np.mean(a['logits'].argmax(1)==a['labels'])==float(row['accuracy_percent'])
            assert int(a['events'].astype(np.int64).sum())==int(row['events'])
        # T1 strong local exactly reproduces old hard predictions (T2 old independently fitted).
        current=np.load(d/'local_t1_T1.npz')
        old=np.load(HERE.parent/'20261114_critical/runs/optimizer_check'/f'{ds}41'/'all_local_T1.npz')
        assert np.array_equal(current['indices'],old['indices'])
        assert np.array_equal(current['logits'],old['logits'])
        for ver in ('intervention','intervention_v2'):
            im=json.loads((HERE/'runs'/ver/f'{ds}41'/'summary.json').read_text())
            check[f'{ds}_{ver}_source']=script_identity('run_interventions.py',im['script_sha256'])
            assert im['test_loaded'] is False and im['first_site_nonzero_events_preserved']
        with (HERE/'runs/intervention'/f'{ds}41'/'graphs.csv').open() as f:v1=list(csv.DictReader(f))
        with (HERE/'runs/intervention_v2'/f'{ds}41'/'graphs.csv').open() as f:v2=list(csv.DictReader(f))
        assert len(v1)==len(v2)==4096
        for a,b in zip(v1,v2):
            assert b['actual_first_event_changed']=='True'
            for key in a:
                if key=='selector':assert (a[key]=='random' and b[key]=='deterministic_coordinate') or a[key]==b[key]
                else:assert a[key]==b[key],(ds,key,a[key],b[key])
    hashes=json.loads((HERE/'analysis/phase3/source_hashes.json').read_text())
    for path,digest in hashes.items():assert sha256(HERE/path)==digest
    changes=subprocess.check_output(['git','diff','--name-only','cb742f2'],cwd=ROOT,text=True).splitlines()
    collaboration_files={'collaboration/STATE.md','collaboration/TASKS.md','collaboration/agents/codex-decision-20261115.md'}
    assert all(p.startswith('research/20261115_decision/') or p in collaboration_files for p in changes),changes
    check['historical_directories_unchanged']=True
    check['all_40_objective_endpoints_and_v1_v2_trace_reproduction']='pass'
    history=json.loads((HERE/'analysis/historical/audit.json').read_text())
    assert history['new_test_inference'] is False and history['archived_logits_only']
    check['historical_three_teacher_archive_reconstruction']='pass'
    source_root=HERE/'runs/source_ti'
    if source_root.exists():
        for ds in ('MNIST','CIFAR10'):
            run=source_root/f'{ds}41'
            sm=json.loads((run/'metadata.json').read_text())
            assert sm['test_loaded'] is False and sm['labels_used'] is False
            assert json.loads((run/'DONE.json').read_text())['endpoints']==8
            for name,digest in sm['snapshots'].items():assert sha256(run/'source'/name)==digest
            for H in (1,2):
                fit=json.loads((run/f'H{H}_fit.json').read_text())
                assert fit['effective_steps_n']==H*256 and fit['scale_iterations']==100
                assert all(json.loads((run/f'H{H}_source_acceptance.json').read_text()).values())
                vectors=np.load(run/f'H{H}_thresholds.npz')
                for site in fit['sites']:
                    assert np.array_equal(vectors[f"threshold_{site['site']}"],site['converter_threshold_after_negative_to_zero_mapping'])
        sa=json.loads((HERE/'analysis/source_ti/acceptance.json').read_text())
        assert sa['endpoints']==16 and sa['raw_array_metrics_recomputed']
        for path,digest in sa['source_hashes'].items():assert sha256(HERE/path)==digest
        check['source_TI_16_endpoints_source_snapshots_moments_and_forward']='pass'
    swap_root=HERE/'runs/swap_oracle'
    if swap_root.exists():
        from summarize_swap_oracle import class_margins,budget_mask
        for ds in ('MNIST','CIFAR10'):
            run=swap_root/f'{ds}41'
            sm=json.loads((run/'metadata.json').read_text())
            assert sm['training_only'] and not sm['validation_loaded'] and not sm['test_loaded']
            assert sm['labels_used_for_selection'] is False and sm['candidate_swaps']==8
            paramfile=HERE/'runs/objectives'/f'{ds}41'/'local_t1_parameters.npz'
            assert sha256(paramfile)==sm['baseline_parameters_sha256']
            oldmeta=json.loads((paramfile.parent/'metadata.json').read_text())
            assert oldmeta['checkpoint_sha256']==sm['checkpoint_sha256']
            assert oldmeta['calibration_sha256']==sm['calibration_sha256']
            assert oldmeta['train_indices'][:128]==sm['training_indices']
            for name,digest in sm['source_sha256'].items():assert sha256(run/'source'/name)==digest
            a=dict(np.load(run/'candidates.npz'))
            assert sha256(run/'candidates.npz')==sm['candidate_npz_sha256']
            assert a['logits_T1'].shape==a['logits_T2'].shape==(9,128,10)
            assert np.array_equal(a['indices'],sm['training_indices'])
            assert a['events_T1'].dtype==a['events_T2'].dtype==np.int64
            margins={T:class_margins(a[f'logits_T{T}'],a['teacher'].argmax(1)) for T in (1,2)}
            for budget in ('exact_events','nonincrease','first_site_only'):
                assert budget_mask(a,margins,budget)[0].all()
        verdict=json.loads((HERE/'analysis/swap_oracle/gate.json').read_text())
        assert verdict['deployable_method_established'] is False
        assert verdict['independent_graph_contribution_established'] is False
        for path,digest in verdict['source_hashes'].items():assert sha256(HERE/path)==digest
        check['training_only_8_swap_banks_and_cost_oracle_provenance']='pass; not a deployment claim'
    learned_root=HERE/'runs/learned_swap'
    if learned_root.exists():
        arms=['baseline','nongraph_local','graph_local','nongraph_decision','graph_decision','shuffled_decision','toppin_decision']
        for ds in ('MNIST','CIFAR10'):
            run=learned_root/f'{ds}41'
            lm=json.loads((run/'metadata.json').read_text())
            assert not lm['online_teacher'] and not lm['online_downstream_lookahead'] and lm['online_full_passes']==1
            assert not lm['test_loaded'] and not lm['labels_used_for_fit']
            assert json.loads((run/'DONE.json').read_text())['endpoints']==14
            for name,digest in lm['source_sha256'].items():assert sha256(run/'source'/name)==digest
            fitted=json.loads((run/'fit_audit.json').read_text())
            assert len(fitted)==6 and all(f['fitted_coefficients']==1304 and f['examples']==1024 for f in fitted)
            for arm in arms:
                a=dict(np.load(run/f'{arm}_T1.npz'));b=dict(np.load(run/f'{arm}_T2.npz'))
                assert np.array_equal(a['indices'],lm['val_indices']) and np.array_equal(a['choice'],b['choice'])
                assert len(a['labels'])==512 and np.isfinite(a['logits']).all() and np.isfinite(b['logits']).all()
                if arm=='baseline':
                    for T,current in [(1,a),(2,b)]:
                        old=np.load(HERE/'runs/objectives'/f'{ds}41'/f'local_t1_T{T}.npz')
                        assert np.array_equal(current['logits'],old['logits'])
                        assert np.array_equal(current['events'],old['events'].astype(np.int64))
            timed=json.loads((run/'timing.json').read_text())
            assert not timed['teacher_in_timed_block'] and timed['state_dependent_proposals_and_prediction_in_timed_block']
        lg=json.loads((HERE/'analysis/learned_swap/gate.json').read_text())
        assert lg['method_gate_pass'] is False and lg['one_alignment_audit_completed']
        for path,digest in lg['source_hashes'].items():assert sha256(HERE/path)==digest
        check['single_pass_learned_28_endpoints_equal_capacity_and_source_alignment']='pass; scientific method gate FAIL'
    verify_prefix_diagnostics(check)
    verify_temporal_runs(check)
    result=dict(engineering_acceptance=True,bounded_research_delivery_complete=True,
        method_discovery_goal_achieved=False,checks=check,
        runtime=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),numpy=np.__version__),
        git_base='cb742f2bd30b2408cab5d09a7d9af8ff4d7c67c8',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip())
    (HERE/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    files={str(p.relative_to(HERE)):sha256(p) for p in sorted(HERE.rglob('*')) if p.is_file()
        and '__pycache__' not in p.parts and p.name!='artifact_manifest.json'}
    (HERE/'artifact_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(dict(artifacts=len(files),acceptance=result),indent=2))


if __name__=='__main__':main()
