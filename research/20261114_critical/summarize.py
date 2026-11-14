"""Recompute exploratory metrics and paired contrasts from saved predictions."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    rows=[];paired=[];runs=[]
    for phase in ['screen','optimizer_check','hard_probe','book_screen','full_book_screen']:
        for path in sorted((ROOT/'runs'/phase).glob('*')):
            if not (path/'DONE.json').exists():continue
            meta=json.loads((path/'metadata.json').read_text());done=json.loads((path/'DONE.json').read_text())
            assert not meta['test_loaded'] and not done['test_loaded']
            for name,digest in meta['source_snapshots'].items():assert sha(path/'source'/name)==digest
            arrays={}
            for result in json.loads((path/'results.json').read_text()):
                tag=f"{result['method']}_T{result['T']}";a=np.load(path/(tag+'.npz'))
                acc=100*float((a['logits'].argmax(1)==a['labels']).mean())
                assert abs(acc-result['accuracy_percent'])<1e-10
                arrays[tag]=a
                trainfile=path/(tag+'_training.json')
                if phase in ['book_screen','full_book_screen']:trainfile=path/(result['method']+'_training.json')
                tr=json.loads(trainfile.read_text()) if trainfile.exists() else {}
                if tr and phase in ['book_screen','full_book_screen']:
                    assert tr['teacher_frozen'] and not tr['test_loaded']
                    h=tr.get('objective_history',[])
                    assert all(y<=x+1e-5 for x,y in zip(h,h[1:]))
                    if phase=='full_book_screen':
                        r=tr['full_hard_refinement']
                        assert r['final_objective']<=r['initial_objective']+1e-5
                elif tr:
                    assert tr['teacher_unchanged']
                    if phase!='hard_probe':assert tr['optimizer_steps']==[64]*(1 if result['method'].startswith('first_') else 6)
                    params=np.load(path/(tag+'_parameters.npz'))
                    if result['method'].startswith('first_'):assert all(not np.any(params[f'offset_{i}']) for i in range(1,6))
                rows.append(dict(phase=phase,dataset=meta['dataset'],seed=meta['seed'],T=result['T'],method=result['method'],accuracy_percent=acc,
                    teacher_agreement_percent=100*float((a['logits'].argmax(1)==a['teacher'].argmax(1)).mean()),
                    logit_mse=float(np.square(a['logits']-a['teacher']).mean()),parameters=tr.get('trainable_parameters',tr.get('learned_scalars',0)),
                    calibration_seconds=tr.get('calibration_seconds',0),initial_hard_loss=tr.get('initial_hard_train',{}).get('total',''),
                    final_hard_loss=tr.get('final_hard_train',{}).get('total',''),source=str(path.relative_to(ROOT)/f'{tag}.npz'),source_sha256=sha(path/f'{tag}.npz')))
            for t in meta['T']:
                for m in (['book_message'] if phase in ['book_screen','full_book_screen'] else ['first_message','all_message']):
                    for control in (['book_local','book_shuffled','book_identity','book_nearest'] if phase in ['book_screen','full_book_screen'] else ['all_local','first_local','first_shuffled','first_identity']):
                        x=arrays.get(f'{m}_T{t}');y=arrays.get(f'{control}_T{t}')
                        if x is None or y is None:continue
                        assert np.array_equal(x['labels'],y['labels']) and np.array_equal(x['indices'],y['indices'])
                        d=(x['logits'].argmax(1)==x['labels']).astype(float)-(y['logits'].argmax(1)==y['labels']).astype(float)
                        se=d.std(ddof=1)/np.sqrt(len(d))*100;mean=d.mean()*100
                        paired.append(dict(phase=phase,dataset=meta['dataset'],T=t,method=m,control=control,accuracy_delta_pp=mean,
                            descriptive_paired_normal_low=mean-1.96*se,descriptive_paired_normal_high=mean+1.96*se,graphs=len(d),qualification='exploratory one teacher; not seed-level CI or external confirmation'))
            runs.append(dict(phase=phase,dataset=meta['dataset'],source=str(path.relative_to(ROOT)),seconds=done['total_elapsed_seconds'],rows=done['rows']))
    output=ROOT/'analysis';output.mkdir(exist_ok=True)
    write(output/'endpoints.csv',rows);write(output/'paired.csv',paired);write(output/'run_costs.csv',runs)
    text=['# 阶段1结果（自动生成）','','512训练图、512验证图，教师41；不是多种子结论。硬搜索诊断预算高于64步Adam，不能合并统计。','',
          '|阶段|数据集|T|条件|准确率%|参数|校准秒|初始/最终硬训练损失|','|---|---|---:|---|---:|---:|---:|---|']
    for r in rows:
        text.append(f"|{r['phase']}|{r['dataset']}|{r['T']}|{r['method']}|{r['accuracy_percent']:.4f}|{r['parameters']}|{r['calibration_seconds']:.3f}|{r['initial_hard_loss']} → {r['final_hard_loss']}|")
    (output/'SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed_runs=len(runs),endpoints=len(rows),paired_contrasts=len(paired))))

if __name__=='__main__':main()
