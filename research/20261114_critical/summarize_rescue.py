"""Source-validated baseline/allocation endpoints and frozen stop/go decision."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    rows=[];contrasts=[];gates=[]
    for phase in ['percentile_v2','allocation']:
        for path in sorted((ROOT/'runs'/phase).iterdir()):
            meta=json.loads((path/'metadata.json').read_text());done=json.loads((path/'DONE.json').read_text())
            assert not meta['test_loaded'] and not done['test_loaded']
            for name,digest in meta['source_snapshots'].items():
                candidates=[f for f in (path/'source').iterdir() if f.name==name or f.name[3:]==Path(name).name]
                assert len(candidates)==1 and sha(candidates[0])==digest
            arrays={}
            if phase=='allocation':assert sha(path/'input_base_parameters.npz')==meta['base_parameters_sha256']
            for r in json.loads((path/'results.json').read_text()):
                method=r.get('method',r.get('condition'));tag=f"{method}_T{r['T']}";p=path/(tag+'.npz');a=np.load(p)
                acc=100*float((a['logits'].argmax(1)==a['labels']).mean());assert abs(acc-r['accuracy_percent'])<1e-5
                assert np.array_equal(a['indices'],meta['val_indices'])
                arrays[tag]=a
                counts=r.get('allocation',{})
                rows.append(dict(phase=phase,dataset=meta['dataset'],seed=meta['seed'],T=r['T'],method=method,accuracy_percent=acc,
                    teacher_agreement_percent=100*float((a['logits'].argmax(1)==a['teacher'].argmax(1)).mean()),
                    logit_mse=float(np.square(a['logits']-a['teacher']).mean()),selected_nodes=counts.get('selected_nodes',0),
                    opportunity_slots=counts.get('max_opportunity_slots',0),extra_nonzero_events=counts.get('extra_nonzero_events',0),
                    extra_edge_deliveries=counts.get('extra_edge_deliveries',0),diagnostic_eval_seconds=r['diagnostic_eval_seconds'],
                    source=str(p.relative_to(ROOT)),sha256=sha(p)))
            if phase!='allocation':continue
            for t in [1,2]:
                x=arrays[f'message_T{t}'];controls=['baseline','local','random','identity','shuffled']
                for control in controls+(['strong_local_lr01'] if t==1 else []):
                    y=np.load(ROOT/'runs/optimizer_check'/path.name/'all_local_T1.npz') if control=='strong_local_lr01' else arrays[f'{control}_T{t}']
                    assert np.array_equal(x['labels'],y['labels']) and np.array_equal(x['indices'],y['indices'])
                    d=(x['logits'].argmax(1)==x['labels']).astype(float)-(y['logits'].argmax(1)==y['labels']).astype(float)
                    mean=100*d.mean();se=100*d.std(ddof=1)/np.sqrt(len(d))
                    contrasts.append(dict(dataset=meta['dataset'],T=t,method='message',control=control,delta_pp=mean,
                        paired_normal_low=mean-1.96*se,paired_normal_high=mean+1.96*se,qualification='exploratory one teacher; no multiplicity correction or external confirmation'))
            lookup={(r['T'],r['control']):r['delta_pp'] for r in contrasts if r['dataset']==meta['dataset']}
            gate=dict(dataset=meta['dataset'],local_delta_T1=lookup[1,'local'],strong_local_delta_T1=lookup[1,'strong_local_lr01'],
                      identity_delta_T1=lookup[1,'identity'],shuffled_delta_T1=lookup[1,'shuffled'],same_base_delta_T2=lookup[2,'baseline'])
            gate['pass']=bool(gate['local_delta_T1']>=.5 and gate['strong_local_delta_T1']>=.5 and gate['identity_delta_T1']>0 and gate['shuffled_delta_T1']>0 and gate['same_base_delta_T2']>=-.5)
            gates.append(gate)
    out=ROOT/'analysis';save(out/'rescue_endpoints.csv',rows);save(out/'rescue_paired.csv',contrasts)
    decision=dict(gates=gates,continue_multi_teacher=all(g['pass'] for g in gates),
        scope='fixed quarter-node opportunity budget; current first-tick residual selector only',
        criterion='cross-task T1 local and strong baseline >=.5pp; true graph beats identity/shuffle; T2 regression <=.5pp',
        next='do not expand this selector if failed; complete mechanism/cost/novelty synthesis and conditional route decision')
    (out/'rescue_decision.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
    lines=['# Source baseline and allocation rescue','','All values: exploratory validation512, teacher41, no test. T1-fit base offsets are shared across T1/T2.','','|Dataset|T|Policy|Accuracy%|Extra nonzero events|Opportunity slots|','|---|---:|---|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['dataset']}|{r['T']}|{r['phase']}/{r['method']}|{r['accuracy_percent']:.4f}|{r['extra_nonzero_events']}|{r['opportunity_slots']}|")
    lines+=['','Predefined expansion gate: '+('PASS' if decision['continue_multi_teacher'] else 'FAIL')+'. No stable method claim.']
    (out/'RESCUE_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(endpoints=len(rows),contrasts=len(contrasts),decision=decision)))
if __name__=='__main__':main()
