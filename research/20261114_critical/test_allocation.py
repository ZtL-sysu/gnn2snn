import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'20261114_story'))
import torch
from test_calibration_core import TinyFolded,synthetic_batch
from allocation import scores,select_correction
from objective import message_operator,make_permutation
from allocation_core import AllocationGCN
from calibration_core import CalibratedGCN

def test_message_score_exact_single_node():
    torch.manual_seed(31);m=TinyFolded().eval();b=synthetic_batch()
    e=torch.randn(b.num_nodes,8);c=torch.randn_like(e)
    for policy in ['message','identity','shuffled']:
        perm=make_permutation(b) if policy=='shuffled' else None
        actual=scores(m,b,e,c,policy)
        base=message_operator(m,b,e,perm,policy=='identity')
        for i in range(b.num_nodes):
            correction=torch.zeros_like(e);correction[i]=c[i]
            changed=message_operator(m,b,e+correction,perm,policy=='identity')
            expected=base.square().sum()-changed.square().sum()
            assert torch.allclose(actual[i],expected,atol=1e-5)

def test_budget_and_no_label_dependency():
    torch.manual_seed(31);m=TinyFolded().eval();b=synthetic_batch()
    z=torch.randn(b.num_nodes,8);s=z.relu()+.1;theta=torch.ones(8)
    for policy in ['local','message','identity','shuffled','random']:
        correction,info=select_correction(m,b,z,s,theta,policy)
        assert info['selected_nodes']==2
        assert info['extra_nonzero_events']<=2*8
        assert info['max_opportunity_slots']==2*8
        assert torch.all(((s+correction-z.relu()).square().sum(1)-(s-z.relu()).square().sum(1))<=1e-6)
        without=b.clone();del without.y
        assert torch.equal(correction,select_correction(m,without,z,s,theta,policy)[0])

def test_disabled_allocation_bitwise_and_active_finite():
    torch.manual_seed(31);m=TinyFolded().eval();b=synthetic_batch();th=[[.2]*8]*4+[[.2]*4,[.2]*2]
    old=CalibratedGCN(m,th);new=AllocationGCN(m,th)
    for t in [1,2,4,8]:assert torch.equal(old(b,t)[0],new(b,t)[0])
    new.policy='message'
    assert torch.isfinite(new(b,2)[0]).all()
    assert new.last_allocation['selected_nodes']==2
