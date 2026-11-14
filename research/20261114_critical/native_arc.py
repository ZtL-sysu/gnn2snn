"""Context-isolated native-writer bridge for actual local ARC writing stages.

Production waits for a live native assistant response to the emitted request.
Exact request-bound live replies may resume a failed execution, with their original
generation identity preserved; no canned answer or new model call is imputed. Unit-test
fixtures are explicitly marked and never grant a production stage acceptance.
"""
import argparse,copy,hashlib,inspect,json,sys,time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ARC=Path('C:/code/AutoResearchClaw')
SKILL=Path('C:/Users/xhf/.codex/skills/f-research-v1')
sys.path.insert(0,str(ARC));sys.path.insert(0,str(SKILL/'scripts'))
from writing_adapter import response_parts
from researchclaw.llm.client import LLMResponse

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False)

class NativeFileClient:
    def __init__(self,requests,exchange,stage,*,test_exchange=None,resume_exchange=None):
        self.requests=[Path(p) for p in requests];self.exchange=Path(exchange)
        self.exchange.mkdir(parents=True,exist_ok=False)
        self.stage=stage;self.test_exchange=test_exchange;self.traces=[];self.parts=[]
        self.resume_exchange=Path(resume_exchange) if resume_exchange else None
        self.config=SimpleNamespace(primary_model='native-assistant-model-not-exposed')

    def chat(self,messages,**kwargs):
        idx=len(self.traces)
        if idx>=len(self.requests):raise RuntimeError('No approved clean task for additional ARC call')
        source=self.requests[idx];request=json.loads(source.read_text(encoding='utf-8'))
        outgoing=copy.deepcopy(request['messages'])
        if self.stage==17 and self.parts:
            outgoing.append({'role':'user','content':'Approved current draft from previous scientific writing calls:\n\n'+'\n\n'.join(self.parts)})
        stack=any(f.function.startswith('_execute_') and '/researchclaw/' in f.filename.replace('\\','/') for f in inspect.stack())
        name=f'call_{idx+1:02d}';req=self.exchange/(name+'_request.json');res=self.exchange/(name+'_response.json')
        dump(req,{'messages':outgoing})
        record=dict(stage=self.stage,call=idx+1,approved_request_sha256=sha(source),
            emitted_request_sha256=sha(req),arc_call_stack_verified=stack,test_only=self.test_exchange is not None,
            writer='native assistant through live file exchange',model='not exposed by native execution',
            network_api_invoked=False,original_arc_prompt_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest())
        self.traces.append(record)
        if self.resume_exchange is not None:
            prior_req=self.resume_exchange/(name+'_request.json');prior_res=self.resume_exchange/(name+'_response.json')
            receipt=json.loads((self.resume_exchange/(name+'_binding.json')).read_text(encoding='utf-8'))
            if sha(prior_req)!=sha(req) or receipt['request_sha256'].lower()!=sha(req):
                raise SystemExit('Cannot resume: native request/receipt differs')
            answer=json.loads(prior_res.read_text(encoding='utf-8'))
            prior_trace=self.resume_exchange/(name+'_trace.json')
            prior_record=json.loads(prior_trace.read_text(encoding='utf-8')) if prior_trace.exists() else {}
            if prior_record.get('response_sha256') and sha(prior_res)!=prior_record['response_sha256']:
                # A writer edited its response after the live consumer had read it.
                # Recover the exact consumed prose from the NEXT emitted request,
                # never from a guessed reconstruction or an unbound new answer.
                next_req=self.resume_exchange/f'call_{idx+2:02d}_request.json'
                combined=json.loads(next_req.read_text(encoding='utf-8'))['messages'][-1]['content']
                prefix='Approved current draft from previous scientific writing calls:\n\n'+'\n\n'.join(self.parts)+'\n\n'
                if not combined.startswith(prefix):raise SystemExit('Cannot recover consumed manuscript unambiguously')
                answer={'manuscript':combined[len(prefix):],
                        'notes':'Recovery: exact consumed prose extracted from the next original bound request; original notes are not writer context.'}
                record.update(recovered_consumed_prose_from=str(next_req),recovery_source_sha256=sha(next_req),
                              original_trace_response_sha256=prior_record['response_sha256'],
                              late_response_edit_detected=True)
            dump(res,answer)
            record.update(response_origin='resumed previously live native exchange',
                          original_exchange=str(self.resume_exchange),original_response_sha256=sha(prior_res),
                          new_native_generation=False,original_execution_identity=receipt.get('execution_identity'))
        elif self.test_exchange is not None:
            answer=self.test_exchange(outgoing);dump(res,answer)
        else:
            print(f'NATIVE_REQUEST stage={self.stage} call={idx+1} path={req}',flush=True)
            deadline=time.monotonic()+1800
            while not (res.exists() and (self.exchange/(name+'_binding.json')).exists()):
                if time.monotonic()>deadline:raise TimeoutError('Native writer did not answer this live request')
                time.sleep(.25)
            binding=json.loads((self.exchange/(name+'_binding.json')).read_text(encoding='utf-8'))
            if binding['request_sha256'].lower()!=sha(req):raise SystemExit('Response bound to a different request')
            answer=json.loads(res.read_text(encoding='utf-8'))
        try:
            manuscript,notes=response_parts(answer)
        except ValueError as exc:
            record['response_validation_error']=str(exc)
            dump(self.exchange/(name+'_rejection.json'),record)
            if self.test_exchange is not None:raise
            # The official helper retries ordinary exceptions. A malformed live
            # response must stop this execution, not consume the next section task.
            raise SystemExit('Rejected native response; explicit corrected run required') from exc
        record['response_sha256']=sha(res)
        dump(self.exchange/(name+'_trace.json'),record)
        self.parts.append(manuscript)
        return LLMResponse(content=manuscript,model=record['model'],finish_reason='native_response')

def real_local_config(run_dir):
    from researchclaw.config import RCConfig
    import yaml
    config_data=yaml.safe_load((ARC/'config.researchclaw.example.yaml').read_text(encoding='utf-8'))
    config_data['research']['topic']='Critical-Site Sensitivity and Realizable Calibration in GCN-to-SNN Conversion'
    config_data['llm'].update(primary_model='native-assistant',fallback_models=[],api_key='',base_url='http://unused.invalid')
    # ARC names actual subprocess/local experiment mode 'sandbox'. Its example
    # defaults to simulated; our imported raw experiment evidence is real local data.
    config_data.setdefault('experiment',{})['mode']='sandbox'
    return RCConfig.from_dict(config_data,project_root=run_dir,check_paths=False)

def execute_stage(stage,run_dir,client):
    from researchclaw.pipeline.stage_impls import _paper_writing as writing
    from researchclaw.pipeline.stage_impls import _review_publish as review
    run_dir=Path(run_dir).resolve();out=run_dir/f'stage-{stage}';out.mkdir(exist_ok=False)
    config=real_local_config(run_dir)
    # Accepted scientific snapshots take precedence over raw generation outputs.
    accepted=run_dir/'scientific_current_draft.md'
    allowed={'outline.md':run_dir/'stage-16/outline.md','paper_draft.md':accepted if accepted.exists() else run_dir/'stage-17/paper_draft.md',
             'reviews.md':run_dir/'stage-18/reviews.md',
             'paper_revised.md':run_dir/'stage-19/paper_revised.md',
             'references.bib':run_dir/'references.bib',
             'paper_final.md':run_dir/'stage-22/paper_final.md',
             'experiment_summary.json':run_dir/'scientific_summary.json'}
    def safe_read(_root,name):
        path=allowed.get(name)
        return path.read_text(encoding='utf-8') if path and path.is_file() else ''
    analysis=(run_dir/'scientific_analysis.md').read_text(encoding='utf-8') if (run_dir/'scientific_analysis.md').exists() else ''
    functions={16:writing._execute_paper_outline,17:writing._execute_paper_draft,
               18:review._execute_peer_review,19:review._execute_paper_revision,
               20:review._execute_quality_gate,22:review._execute_export_publish,23:review._execute_citation_verify}
    if stage not in functions:raise ValueError(stage)
    # Scoped runtime adapters; neither ARC source nor global configuration changes.
    with ExitStack() as patches:
        for module in [writing,review]:
            for name,value in [('_build_context_preamble',lambda *a,**k:''),('_read_prior_artifact',safe_read),
                               ('_read_best_analysis',lambda *a,**k:analysis),('_get_evolution_overlay',lambda *a,**k:'')]:
                patches.enter_context(patch.object(module,name,value))
        result=functions[stage](out,run_dir,config,None,llm=client)
    report=dict(stage=stage,status=result.status.value,artifacts=list(result.artifacts),
                actual_arc_executor=functions[stage].__module__+'.'+functions[stage].__name__,
                executor_source_sha256=sha(inspect.getfile(functions[stage])),client_source_sha256=sha(__file__),
                production=client is None or client.test_exchange is None,calls=client.traces if client else [],
                mode='actual programmatic ARC execution' if client is None else 'native file exchange')
    dump(out/'native_execution.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--requests',nargs='+',type=Path)
    p.add_argument('--programmatic',action='store_true')
    p.add_argument('--resume-exchange',type=Path)
    args=p.parse_args();exchange=args.run_dir/f'native-exchange-{args.stage}'
    if args.programmatic and args.stage not in [22,23]:p.error('Programmatic mode restricted to export/citation stages')
    if not args.programmatic and not args.requests:p.error('Native execution requires approved tasks')
    client=None if args.programmatic else NativeFileClient(args.requests,exchange,args.stage,resume_exchange=args.resume_exchange)
    report=execute_stage(args.stage,args.run_dir,client);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
