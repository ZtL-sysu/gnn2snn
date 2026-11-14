from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=Path(r'C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision\takeover_20261112')
SUMMARY=ROOT/'research/visual-graphs/takeover_20261112/fixed-gcn-test-three-seed-summary-v1.json'; OUT=ROOT/'research/20261114_story'
SEEDS=[('MNIST',41,'dc_mn_test1.json'),('MNIST',42,'rep/M42/dc_test_v3.json'),('MNIST',43,'rep/M43/dc_test_v3.json'),('CIFAR10',41,'dc_cf_test1.json'),('CIFAR10',42,'rep/C42/dc_test_v3.json'),('CIFAR10',43,'rep/C43/dc_test_v3.json')]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 summary=json.loads(SUMMARY.read_text(encoding='utf-8')); expected={(r['dataset'],r['seed']):r for r in summary['rows']}; evidence=[]; rows=[]
 def record(p,role,expected_hash=None):
  p=Path(p); actual=sha(p) if p.is_file() else ''; status='VERIFIED' if expected_hash and actual==expected_hash else ('NO_RECORDED_HASH' if not expected_hash else 'MISMATCH'); evidence.append({'role':role,'path':str(p),'exists':p.is_file(),'expected_sha256':expected_hash or '', 'actual_sha256':actual,'hash_status':status})
 for ds,seed,rel in SEEDS:
  side=ARCHIVE/rel; doc=json.loads(side.read_text(encoding='utf-8')); assert doc['status']=='complete' and doc['dataset']==ds and doc['seed']==seed and doc['graphs']==10000
  npz=Path(doc['predictions_file']); ref=Path(doc['reference_predictions']); checkpoint=Path(doc['teacher_run'])/'best.pt'; cal=Path(doc['calibration_path'])
  record(side,'prediction_sidecar',summary['input_sha256'].get(str(side))); record(npz,'prediction_logits_npz',summary['input_sha256'].get(str(npz))); record(ref,'same_checkpoint_teacher_reference',doc.get('reference_predictions_sha256')); record(checkpoint,'teacher_checkpoint',doc.get('checkpoint_sha256')); record(cal,'calibration',doc.get('calibration_sha256'))
  assert doc['input_hashes_start']==doc['input_hashes_end'] and sha(npz)==doc['predictions_sha256']; assert all(x['hash_status']!='MISMATCH' for x in evidence)
  with np.load(npz) as p,np.load(ref) as q:
   assert np.array_equal(p['indices'],np.arange(10000)) and np.array_equal(p['indices'],q['index']) and np.array_equal(p['labels'],q['labels'])
   y=p['labels']; ann=p['teacher_logits']; assert ann.shape==(10000,10) and np.allclose(ann,q['logits'],rtol=1e-5,atol=1e-4)
   r={'dataset':ds,'seed':seed,'teacher_accuracy_percent':float((ann.argmax(1)==y).mean()*100)}
   for t in (1,2,4,8):
    x=p[f'T{t}_logits']; r[f'T{t}_accuracy_percent']=float((x.argmax(1)==y).mean()*100); r[f'T{t}_teacher_agreement_percent']=float((x.argmax(1)==ann.argmax(1)).mean()*100); r[f'T{t}_logit_mse']=float(np.mean((x.astype(np.float64)-ann.astype(np.float64))**2))
   e=expected[(ds,seed)]; assert abs(r['teacher_accuracy_percent']-e['teacher_accuracy_percent'])<1e-8
   for t in (1,2,4,8):
    assert abs(r[f'T{t}_accuracy_percent']-e[f'T{t}']['accuracy_percent'])<1e-8 and abs(r[f'T{t}_teacher_agreement_percent']-e[f'T{t}']['teacher_agreement_percent'])<1e-8 and abs(r[f'T{t}_logit_mse']-e[f'T{t}']['mse'])<1e-10
  r.update({'checkpoint_epoch':doc['checkpoint_epoch'],'checkpoint_path':str(checkpoint),'checkpoint_sha256':doc['checkpoint_sha256'],'teacher_run':doc['teacher_run'],'prediction_path':str(npz),'sidecar_path':str(side),'qualification':doc['qualification']}); rows.append(r)
 with (OUT/'baseline_summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (OUT/'evidence_manifest.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(evidence[0]));w.writeheader();w.writerows(evidence)
 stats=[]
 for ds in ('MNIST','CIFAR10'):
  group=[r for r in rows if r['dataset']==ds]
  for m in ['teacher_accuracy_percent']+[f'T{t}_{s}' for t in (1,2,4,8) for s in ('accuracy_percent','teacher_agreement_percent','logit_mse')]:
   v=np.array([r[m] for r in group],float);stats.append((ds,m,v.mean(),v.std(ddof=1)))
 md=['# Historical fixed-GCN baseline audit','','**Status: VERIFIED archived historical audit; not a live reproduction.** Six saved prediction bundles were re-read and all aggregate metrics independently recomputed from labels/logits. Sidecar, NPZ, reference, checkpoint, and calibration hashes were checked where recorded. No GPU evaluation was run.','','## Scope and caveat','','These are project MTH-ReLU-branch graph adaptations using pinned source units and common full-train histogram calibration; not full DCGS reproduction or evidence of superiority. The complete official test split had prior project exposure. These are descriptive exposed-test fidelity endpoints, not untouched-test estimates. Seeds vary teacher training; each dataset reuses the same 10,000 examples.','','## Recomputed mean ± sample SD across seeds','','| Dataset | Metric | Mean | Sample SD |','|---|---|---:|---:|']
 for ds,m,a,b in stats:md.append(f'| {ds} | {m} | {a:.6g} | {b:.6g} |')
 md+=['','## Per-seed summary','','| Dataset | Seed | Teacher accuracy | T1 acc / agree / MSE | T2 acc / agree / MSE | T4 acc / agree / MSE | T8 acc / agree / MSE |','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  z=[f"{r[f'T{t}_accuracy_percent']:.2f}% / {r[f'T{t}_teacher_agreement_percent']:.2f}% / {r[f'T{t}_logit_mse']:.6g}" for t in (1,2,4,8)];md.append(f"| {r['dataset']} | {r['seed']} | {r['teacher_accuracy_percent']:.2f}% | "+' | '.join(z)+' |')
 md+=['','## Checkpoint provenance','','| Dataset | Seed | Epoch | Checkpoint path | SHA-256 |','|---|---:|---:|---|---|']
 for r in rows:md.append(f"| {r['dataset']} | {r['seed']} | {r['checkpoint_epoch']} | `{r['checkpoint_path']}` | `{r['checkpoint_sha256']}` |")
 md+=['','Sidecars report matching input hashes at run start and end. NPZ metrics match the committed six-row summary within strict floating-point tolerances. The manifest records expected and observed file hashes.','','## Limits and unknowns','','- This verifies archived artifacts and internal provenance records; it does not reproduce teacher training, conversion, or inference from source code.','- Dataset/runtime code was not re-executed. Sidecar semantics are accepted subject to file consistency checks.','- Archived qualification disclaims full DCGS reproduction and an untouched-test claim.','- Gated results are omitted; this audit only establishes the six fixed-GCN baseline rows.','']
 (OUT/'audit_baselines.md').write_text('\n'.join(md),encoding='utf-8');print(f'PASS: {len(rows)} rows, {len(evidence)} evidence records')
if __name__=='__main__':main()
