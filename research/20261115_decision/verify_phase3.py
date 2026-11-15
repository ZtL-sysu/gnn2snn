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
    assert all(p.startswith('research/20261115_decision/') for p in changes),changes
    check['historical_directories_unchanged']=True
    check['all_40_objective_endpoints_and_v1_v2_trace_reproduction']='pass'
    history=json.loads((HERE/'analysis/historical/audit.json').read_text())
    assert history['new_test_inference'] is False and history['archived_logits_only']
    check['historical_three_teacher_archive_reconstruction']='pass'
    result=dict(engineering_acceptance=True,method_discovery_goal_achieved=False,checks=check,
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
