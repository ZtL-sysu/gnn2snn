"""Record a primary-reviewed real gate; never manufactures its substantive report."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
VALIDATOR=Path('C:/Users/xhf/.codex/skills/f-research-v1/scripts/pipeline_state.py')
def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--evidence',nargs='+',required=True);p.add_argument('--executor',required=True);a=p.parse_args()
    state=ROOT/'pipeline-state.json'
    subprocess.run([sys.executable,str(VALIDATOR),str(state),'--before',a.stage],check=True)
    data=json.loads(state.read_text(encoding='utf-8'))
    def entry(name):
        f=ROOT/name;assert f.is_file() and f.stat().st_size
        return dict(path=name,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    s=next(s for s in data['stages'] if s['id']==a.stage);assert s['status'] in ['pending','running']
    s.update(status='passed',contract_version=data['contract_version'],inputs=data['contract'],evidence=[entry(n) for n in a.evidence],gate_verdict='pass',actual_executor=a.executor)
    data['history'].append(dict(action='pass',stage=a.stage,reason='primary accepted substantive report and recorded actual artifacts'))
    state.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,str(VALIDATOR),str(state)],check=True)
    lines=['# Pipeline state','',f"Contract: {data['contract_version']}. Empirical draft acceptance is separate from method efficacy and publication readiness.",'','| Stage | Status | Evidence |','|---|---|---|']
    for s in data['stages']:lines.append('| '+s['id']+' | '+s['status']+' | '+', '.join(e['path'] for e in s['evidence'])+' |')
    (ROOT/'pipeline-state.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
