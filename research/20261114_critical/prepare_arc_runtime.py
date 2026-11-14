"""Import verified real local run records into ARC's documented artifact layout.

No experiment is executed or retroactively credited as an ARC stage12 run.
The imported records retain the real raw-array identity and original scope.
"""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    src=ROOT/'writing/arc_run_v1'
    for name in ['scientific_summary.json','scientific_analysis.md']:shutil.copyfile(src/name,out/name)
    shutil.copytree(src/'stage-16',out/'stage-16')
    imported=out/'stage-12/runs';imported.mkdir(parents=True)
    with (ROOT/'analysis/rescue_endpoints.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    manifest=[]
    for row in rows:
        path=ROOT/row['source'];assert sha(path)==row['sha256']
        a=np.load(path);accuracy=100*float((a['logits'].argmax(1)==a['labels']).mean())
        assert abs(accuracy-float(row['accuracy_percent']))<1e-9
        record=dict(status='complete',mode='imported_real_local',imported_run=True,arc_experiment_execution=False,
                    metrics=dict(accuracy_percent=accuracy,logit_mse=float(np.square(a['logits']-a['teacher']).mean())),
                    dataset=row['dataset'],teacher_seed=41,graphs=512,T=int(row['T']),method=row['method'],
                    source_evidence=str(path.resolve()),source_sha256=sha(path),qualification='one-teacher exploratory validation; no new experiment or held-out confirmation')
        filename=f"{row['phase']}_{row['dataset']}_{row['method']}_T{row['T']}.json"
        (imported/filename).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        manifest.append(dict(file=str((imported/filename).relative_to(out)),sha256=sha(imported/filename),source_sha256=sha(path)))
    (out/'import_manifest.json').write_text(json.dumps(dict(records=manifest,arc_stage12_executed=False),indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(ROOT/'native_arc.py',out/'native_arc_source.py')
    print('Imported verified real endpoint records:',len(manifest))
if __name__=='__main__':main()
