"""Index configurations, measured outputs and their source hashes for review."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def main():
    current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    matrix=[];evidence=[]
    for phase in ['screen','low_lr_screen','confirmation','frozen_test','ideal']:
        folder=ROOT/'runs'/phase
        if not folder.exists():continue
        for run in sorted(folder.iterdir()):
            if not run.is_dir():continue
            if not (run/'DONE.json').is_file():raise RuntimeError('incomplete scientific run: '+str(run))
            m=json.loads((run/'metadata.json').read_text());done=json.loads((run/'DONE.json').read_text())
            results=json.loads((run/'results.json').read_text())
            if done['status']!='complete' or done['rows']!=len(results):raise RuntimeError('incomplete result grid')
            config=m.get('calibration_config',m)
            split='historically_exposed_test' if phase=='frozen_test' else 'historically_exposed_validation'
            for r in results:
                method=r.get('method',r.get('condition'));t=r['T']
                pred=run/f'{method}_T{t}.npz'
                if not pred.is_file():raise FileNotFoundError(pred)
                item=dict(run_id=run.name,phase=phase,dataset=m['dataset'],backbone='GCN',seed=m.get('seed',m.get('teacher_seed')),
                          T=t,method=method,split=split,graphs=r.get('graphs',r.get('val_graphs')),
                          calibration_samples=0 if phase=='ideal' or method=='baseline' else 512,
                          fitted_calibration_steps=0 if phase=='ideal' or method=='baseline' else config.get('steps'),
                          optimization_during_run=(phase not in ['frozen_test','ideal'] and method!='baseline'),
                          trainable_parameters=0 if phase=='ideal' else (10 if method=='output' else 0 if method=='baseline' else 693),
                          lr='' if phase=='ideal' or method=='baseline' else config.get('lr'),checkpoint_sha256=m.get('checkpoint_sha256'),
                          config=str((run/'metadata.json').relative_to(ROOT)),source_path=str(pred.relative_to(ROOT)),
                          source_sha256=sha(pred),commit=m.get('git_commit',current_commit),status='VERIFIED_OUTPUT_RECORD')
                matrix.append(item)
                if phase=='frozen_test' and method=='baseline' and t==1:
                    evidence.append(dict(claim='C0_teacher_quality',**item,metric='teacher_accuracy_percent',value=r['teacher_accuracy_percent']))
                for metric in ['accuracy_percent','teacher_agreement_percent','logit_mse','cross_entropy']:
                    if metric not in r:continue
                    evidence.append(dict(claim='C4_site_intervention' if phase=='ideal' else 'C2_fidelity_and_calibration',
                                         **item,metric=metric,value=r[metric]))
    for path in sorted((ROOT/'costs').glob('cost_*.json')) if (ROOT/'costs').exists() else []:
        m=json.loads(path.read_text())
        for row in m['latency']:
            method=row['condition'];t=int(method.split('_T')[1]) if '_T' in method else ''
            evidence.append(dict(claim='C5_software_latency',dataset=m['dataset'],backbone='GCN',seed=m['seed'],T=t,
                                 method=method,split='cached_validation_batch',graphs=m['graphs'],source_path=str(path.relative_to(ROOT)),
                                 source_sha256=sha(path),commit=current_commit,config=str(path.relative_to(ROOT)),
                                 metric='batch_median_ms_divided_by_graphs',value=row['per_graph_ms_median'],status='VERIFIED_MEASUREMENT_RECORD'))
    for source in ['analysis/confirmation/aggregate.csv','analysis/frozen_test/aggregate.csv','analysis/ideal/aggregate.csv']:
        path=ROOT/source
        with path.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['method'].startswith('paired_'):
                    metric='accuracy_percent_mean';claim='C2_paired_objective_difference'
                elif source.startswith('analysis/ideal/'):
                    metric='agreement_gap_closed_mean';claim='C4_site_intervention'
                else:continue
                if not row.get(metric):continue
                evidence.append(dict(claim=claim,dataset=row['dataset'],backbone='GCN',seed='three_teacher_mean',T=row['T'],
                                     method=row['method'],split=row['scope'],source_path=source,source_sha256=sha(path),
                                     config=source,commit=current_commit,metric=metric,value=row[metric],status='DESCRIPTIVE_DERIVED_RECORD'))
    path=ROOT/'gated_baseline_summary.csv'
    with path.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split']=='test' and row['T']=='4' and row['seed'] in ['41','42','43']:
                evidence.append(dict(claim='C0_teacher_quality',dataset='CIFAR10',backbone='GatedGCN',seed=row['seed'],T='',
                                     method='teacher',split='historically_exposed_test',source_path=path.name,source_sha256=sha(path),
                                     config='gated_baseline_audit.md',commit=current_commit,metric='teacher_accuracy_percent',
                                     value=row['teacher_accuracy_percent'],status='VERIFIED_ARCHIVED_RECORD'))
    for seed in [41,42,43]:
        for t in [1,2]:
            evidence.append(dict(claim='C0_gated_endpoint',dataset='CIFAR10',backbone='GatedGCN',seed=seed,T=t,
                                 source_path='gated_baseline_audit.md',commit=current_commit,metric='converted_accuracy_percent',
                                 value='',status='NEEDS_REPRO_NO_RAW_ENDPOINT'))
    evidence.append(dict(claim='C5_hardware_energy',source_path='COST_DEFINITIONS.md',commit=current_commit,
                         metric='hardware_joules',value='',status='NOT_MEASURED'))
    matrix_fields=['run_id','phase','dataset','backbone','seed','T','method','split','graphs','calibration_samples',
                   'fitted_calibration_steps','optimization_during_run','trainable_parameters','lr','checkpoint_sha256','config','source_path','source_sha256','commit','status']
    write(ROOT/'experiment_matrix.csv',matrix,matrix_fields)
    write(ROOT/'claims_to_evidence.csv',evidence,['claim',*matrix_fields,'metric','value'])
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--','research/20261114_story'],cwd=REPO,text=True).splitlines()
    files=[]
    for name in sorted(set(paths)):
        path=REPO/name
        if path.name=='ARTIFACTS_SHA256.csv' or not path.is_file():continue
        files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    write(ROOT/'ARTIFACTS_SHA256.csv',files,['path','bytes','sha256'])
    print('Indexed',len(matrix),'condition records,',len(evidence),'claim records and',len(files),'artifact hashes')


if __name__=='__main__':main()
