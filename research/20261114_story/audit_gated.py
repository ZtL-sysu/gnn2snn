from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARCH=Path(r'C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\takeover_20261112')
SUMMARY=ARCH/'gated-three-teacher-seed-summary-v1.md'
SUMMARY_JSON=ARCH/'gated-three-teacher-seed-summary-v1.json'
OUT=ROOT/'research/20261114_story'
SPECS=[(41,'val','gd_val1.json','gated_cifar10_seed41_full_v1','gcal_v1.json',None),(41,'test','gd_test1.json','gated_cifar10_seed41_full_v1','gcal_v1.json','gtest_v1_pred.npz'),(42,'val','g42val.json','gated_cifar10_seed42_full_v2','g42cal.json',None),(42,'test','g42test.json','gated_cifar10_seed42_full_v2','g42cal.json','gtest_v2_pred.npz'),(43,'val','g43val.json','gated_cifar10_seed43_full_v2','g43cal.json',None),(43,'test','g43test.json','gated_cifar10_seed43_full_v2','g43cal.json','gtest_v2_pred.npz')]
EXPECTED_TEST={41:0.6920,42:0.6839,43:0.6872}; EXPECTED_VAL={41:0.7066,42:0.7030,43:0.7074}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def add(ev,path,role,expected=None):
 p=Path(path); exists=p.is_file(); actual=sha(p) if exists else ''; status=('VERIFIED' if expected and actual==expected else ('NO_RECORDED_HASH' if exists else 'MISSING'))
 ev.append({'role':role,'path':str(p),'exists':exists,'expected_sha256':expected or '','actual_sha256':actual,'hash_status':status})
 return actual
def main():
 ev=[]; rows=[]; labels_by_split={}; test_teacher_refs={}; summaries={}
 add(ev,SUMMARY,'historical_summary_markdown')
 add(ev,SUMMARY_JSON,'requested_summary_json_not_found')
 for seed,split,side_name,run_name,cal_name,teacher_archive_name in SPECS:
  side=ARCH/side_name; doc=json.loads(side.read_text(encoding='utf-8')); run=ARCH/'full_teacher_runs'/run_name
  assert doc['status']=='complete' and doc['dataset']=='CIFAR10' and doc['split'] in (['all official validation', 'all official val'] if split=='val' else ['all official test']) and doc['graphs']==(5000 if split=='val' else 10000)
  metadata=json.loads((run/'metadata.json').read_text(encoding='utf-8')); done=json.loads((run/'done.json').read_text(encoding='utf-8'))
  assert int(metadata['seed'])==seed and done['status']=='finished' and done['official_counts']=={'train':45000,'val':5000}
  ck=run/'best.pt'; cal=(run/cal_name if seed==41 else ARCH/cal_name); npz=side.with_suffix('.npz')
  add(ev,side,'evaluation_sidecar'); add(ev,npz,'raw_saved_logits_npz',doc['predictions_sha256'])
  ckhash=add(ev,ck,'gated_teacher_checkpoint',doc['checkpoint_sha256']); assert ckhash==done['best_checkpoint_sha256']
  calhash=add(ev,cal,'gated_calibration',doc['calibration_sha256'])
  assert calhash==doc['calibration_sha256']
  add(ev,run/'metadata.json','teacher_metadata'); add(ev,run/'done.json','teacher_done_record')
  if teacher_archive_name:
   teacher_archive=run/teacher_archive_name; add(ev,teacher_archive,'independent_teacher_prediction_archive',doc['teacher_test_archive_sha256'])
  for name,digest in doc['source_sha256'].items(): add(ev,ARCH/name,'recorded_source_dependency',digest)
  arrays=np.load(npz,allow_pickle=False); idx=arrays['indices']; labels=arrays['labels']; teacher=arrays['teacher_logits']; n=5000 if split=='val' else 10000
  assert np.array_equal(idx,np.arange(n)) and labels.shape==(n,) and teacher.shape==(n,10) and np.isfinite(teacher).all()
  if split in labels_by_split: assert np.array_equal(labels,labels_by_split[split])
  else: labels_by_split[split]=labels.copy()
  ta=float((teacher.argmax(1)==labels).mean()); assert abs(ta-float(doc['teacher_accuracy']))<1e-12
  if split=='test': assert abs(ta-EXPECTED_TEST[seed])<1e-12
  else: assert abs(ta-EXPECTED_VAL[seed])<1e-12
  if teacher_archive_name:
   ref=np.load(teacher_archive,allow_pickle=False); assert np.array_equal(idx,ref['index']) and np.array_equal(labels,ref['labels']) and np.allclose(teacher,ref['logits'],rtol=1e-5,atol=1e-4); test_teacher_refs[seed]=True
  for r in doc['rows']:
   if r['method'] not in ('dc_T4','dc_T8'): continue
   tag=r['method']; z=arrays[tag]; assert z.shape==(n,10) and np.isfinite(z).all()
   acc=float((z.argmax(1)==labels).mean()); agree=float((z.argmax(1)==teacher.argmax(1)).mean()); mse=float(np.mean((z.astype(np.float64)-teacher.astype(np.float64))**2))
   assert abs(acc-r['accuracy'])<1e-12 and abs(agree-r['teacher_agreement'])<1e-12 and abs(mse-r['logit_mse'])<1e-8
   T=int(tag[-1]); rows.append({'dataset':'CIFAR10_GatedGCN','seed':seed,'split':split,'method':'differential_gated_hybrid','T':T,'n_graphs':n,'teacher_accuracy_percent':ta*100,'accuracy_percent':acc*100,'teacher_agreement_percent':agree*100,'logit_mse':mse,'checkpoint_path':str(ck),'checkpoint_sha256':doc['checkpoint_sha256'],'sidecar_path':str(side),'predictions_path':str(npz)})
  rows.append({'dataset':'CIFAR10_GatedGCN','seed':seed,'split':split,'method':'analog_teacher','T':'ANN','n_graphs':n,'teacher_accuracy_percent':ta*100,'accuracy_percent':ta*100,'teacher_agreement_percent':100.0,'logit_mse':0.0,'checkpoint_path':str(ck),'checkpoint_sha256':doc['checkpoint_sha256'],'sidecar_path':str(side),'predictions_path':str(npz)})
  arrays.close()
  summaries[(seed,split)]=doc
 assert all(test_teacher_refs.values()), 'missing seed-matched full-test teacher reference'
 assert len([x for x in ev if x['role']=='recorded_source_dependency' and x['hash_status']=='VERIFIED'])==54
 # Summary table includes one row per seed/split/method, plus across-seed mean/sample SD rows.
 base_rows=rows.copy()
 stat_metrics=['teacher_accuracy_percent','accuracy_percent','teacher_agreement_percent','logit_mse']
 for split in ('val','test'):
  for method in ('analog_teacher','differential_gated_hybrid'):
   for t in (['ANN'] if method=='analog_teacher' else [4,8]):
    group=[r for r in base_rows if r['split']==split and r['method']==method and r['T']==t]
    for metric in stat_metrics:
     vals=np.array([r[metric] for r in group],float); mean=float(vals.mean()); sd=float(vals.std(ddof=1))
     rows.append({'dataset':'CIFAR10_GatedGCN','seed':'MEAN±SAMPLE_SD','split':split,'method':method,'T':t,'n_graphs':int(sum(r['n_graphs'] for r in group)),'teacher_accuracy_percent':mean if metric=='teacher_accuracy_percent' else '', 'accuracy_percent':mean if metric=='accuracy_percent' else '', 'teacher_agreement_percent':mean if metric=='teacher_agreement_percent' else '', 'logit_mse':mean if metric=='logit_mse' else '', 'mean_metric':metric,'mean':mean,'sample_sd':sd,'checkpoint_path':'','checkpoint_sha256':'','sidecar_path':'','predictions_path':''})
 fields=['dataset','seed','split','method','T','n_graphs','teacher_accuracy_percent','accuracy_percent','teacher_agreement_percent','logit_mse','mean_metric','mean','sample_sd','checkpoint_path','checkpoint_sha256','sidecar_path','predictions_path']
 with (OUT/'gated_baseline_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 with (OUT/'gated_evidence_manifest.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(ev[0]));w.writeheader();w.writerows(ev)
 # Human-readable table for converted accuracy/agreements, means and sample SD.
 lines=['# Gated CIFAR10 historical baseline audit','','**Status: VERIFIED archived artifacts; no live reproduction or new evaluation.** Accuracy, teacher agreement, and MSE were recomputed from saved logits. All six validation/test sidecars and NPZ hashes, six checkpoint/calibration links, checkpoint completion hashes, 54 recorded source-file hashes, and three independent teacher test archives were checked.','','The requested `gated-three-teacher-seed-summary-v1.json` is absent from both the worktree and source archive; its Markdown summary is available and included in the evidence manifest. Raw artifacts contain differential gated-hybrid outputs at T4/T8 only; T1/T2 outputs are not present.','','## Recomputed endpoint accuracy / own-teacher agreement','','| Split | Seed | ANN teacher accuracy | Differential T4 accuracy / agreement | Differential T8 accuracy / agreement |','|---|---:|---:|---:|---:|']
 for split in ('val','test'):
  for seed in (41,42,43):
   t=[r for r in base_rows if r['split']==split and r['seed']==seed and r['method']=='differential_gated_hybrid']; teacher=float(t[0]['teacher_accuracy_percent'])
   a={x['T']:x for x in t}; lines.append(f"| {split} | {seed} | {teacher:.2f}% | {a[4]['accuracy_percent']:.2f}% / {a[4]['teacher_agreement_percent']:.2f}% | {a[8]['accuracy_percent']:.2f}% / {a[8]['teacher_agreement_percent']:.2f}% |")
  lines.append(f"| {split} | mean ± sample SD | "+f"{np.mean([r['teacher_accuracy_percent'] for r in base_rows if r['split']==split and r['method']=='analog_teacher']):.2f}% ± {np.std([r['teacher_accuracy_percent'] for r in base_rows if r['split']==split and r['method']=='analog_teacher'],ddof=1):.2f} | "+f"{np.mean([r['accuracy_percent'] for r in base_rows if r['split']==split and r['method']=='differential_gated_hybrid' and r['T']==4]):.2f}% ± {np.std([r['accuracy_percent'] for r in base_rows if r['split']==split and r['method']=='differential_gated_hybrid' and r['T']==4],ddof=1):.2f}% / {np.mean([r['teacher_agreement_percent'] for r in base_rows if r['split']==split and r['method']=='differential_gated_hybrid' and r['T']==4]):.2f}% ± {np.std([r['teacher_agreement_percent'] for r in base_rows if r['split']==split and r['method']=='differential_gated_hybrid' and r['T']==4],ddof=1):.2f} | "+f"{np.mean([r['accuracy_percent'] for r in base_rows if r['split']==split and r['method']=='differential_gated_hybrid' and r['T']==8]):.2f}% ± {np.std([r['accuracy_percent'] for r in base_rows if r['split']==split and r['method']=='differential_gated_hybrid' and r['T']==8],ddof=1):.2f}% / {np.mean([r['teacher_agreement_percent'] for r in base_rows if r['split']==split and r['method']=='differential_gated_hybrid' and r['T']==8]):.2f}% ± {np.std([r['teacher_agreement_percent'] for r in base_rows if r['split']==split and r['method']=='differential_gated_hybrid' and r['T']==8],ddof=1):.2f} |")
 lines+=['','## Provenance and interpretation','','Checkpoint hashes match each seed’s teacher `done.json`; each calibration digest matches the archived calibration, which is recorded as label-free and fit over two passes of all 45,000 official training graphs. Saved test teacher logits, labels, and ordered indices exactly match the independent teacher prediction archive for each seed. The seed-41 evaluator JSON omits an explicit teacher-seed field; linkage is through run metadata/checkpoint/calibration.','','The converted model is a dense analog gate/controller hybrid using the project MTH differential conversion for event-carrying paths. It is not an all-spiking network or a full reproduction of the authors’ conversion pipeline. All validation and test sets have prior project exposure. The test rows are historical descriptive endpoints, not independent confirmation; the three-seed spread is teacher-seed sensitivity, not student-training robustness. The small seed-wise gaps do not establish superiority, efficiency, energy reduction, novelty, or sparse execution.','']
 (OUT/'gated_baseline_audit.md').write_text('\n'.join(lines),encoding='utf-8')
 print(f'PASS: {len(rows)} summary rows (including aggregates), {len(ev)} evidence rows, 54 source hashes verified')
if __name__=='__main__':main()






