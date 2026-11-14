import json
from pathlib import Path
from native_arc import NativeFileClient,execute_stage,real_local_config

def test_imported_actual_local_data_is_not_marked_simulated(tmp_path):
    assert real_local_config(tmp_path).experiment.mode=='sandbox'

def test_real_arc_outline_boundary_excludes_history(tmp_path):
    canary='PRIVATE_EXECUTION_HISTORY_CANARY_7861'
    (tmp_path/'iteration_context.json').write_text(json.dumps({'iteration':2,'reviews_excerpt':canary}))
    request=tmp_path/'clean.json'
    messages=[{'role':'system','content':'Scientific outline only.'},{'role':'user','content':'Describe a bounded graph conversion study.'}]
    request.write_text(json.dumps({'messages':messages,'provenance':{'internal':canary}}))
    seen=[]
    def native_test_exchange(actual):
        seen.append(actual)
        return {'manuscript':'# Test-only scientific outline\n\n## Introduction\nA bounded question.','notes':'Fixture only; no production writing credit.'}
    client=NativeFileClient([request],tmp_path/'exchange',16,test_exchange=native_test_exchange)
    result=execute_stage(16,tmp_path,client)
    assert result['status']=='done'
    assert seen==[messages]
    assert canary not in json.dumps(seen)
    assert client.traces[0]['arc_call_stack_verified']
    assert client.traces[0]['test_only'] is True

def test_unknown_response_fields_rejected(tmp_path):
    import pytest
    request=tmp_path/'clean.json';request.write_text(json.dumps({'messages':[{'role':'user','content':'x'}]}))
    c=NativeFileClient([request],tmp_path/'exchange',16,test_exchange=lambda _: {'manuscript':'x','notes':'x','extra':'invalid'})
    with pytest.raises(ValueError):c.chat([{'role':'user','content':'historical input'}])
