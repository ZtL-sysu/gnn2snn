import sys
from pathlib import Path
from unittest.mock import patch
import torch
from torch_geometric.data import Data,Batch
sys.path.insert(0,str(Path(__file__).parent))
from run_prefix_diagnostic import prefix_forward,core
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model


def test_real_residual_gcn_zero_tick2_and_local_prefix_equivalence():
    torch.manual_seed(29)
    model=fold_model(BenchmarkGCN(3,width=16).eval()).eval()
    net=core.CalibratedGCN(model,[torch.ones(w) for w in (16,16,16,16,8,4)]).eval().requires_grad_(False)
    batch=Batch.from_data_list([Data(x=torch.rand(4,1),pos=torch.rand(4,2),edge_index=torch.tensor([[0,1,2,3],[1,2,3,0]]))])
    with patch.object(core,'teacher_sites',side_effect=AssertionError('online teacher called')):
        for policy in ('baseline','all_eligible','deterministic_half'):
            z,trace=prefix_forward(net,batch,2,policy,[41])
            assert trace['second_input_verified_zero']
            assert torch.count_nonzero(trace['first_decode_error_max'])==0
    with torch.no_grad():expected,_=net(batch,2)
    actual,_=prefix_forward(net,batch,2,'baseline',[41])
    assert torch.equal(expected,actual)


def test_identical_integrated_input_and_event_counts_do_not_fix_next_layer_decode():
    from prefix_codec import _SOURCE_MTH_STEP
    theta=torch.ones(1)
    target=torch.tensor([[.8]])
    decoded=[];counts=[]
    for first in (.01,1.):
        current1=torch.tensor([[first]])
        current2=2*(target-current1)
        assert torch.equal(current1+current2/2,target)
        state=core._new_site_state(theta,torch.zeros(1),(1,1),'cpu',torch.float32)
        q1=_SOURCE_MTH_STEP(state,current1,theta,1)
        q2=_SOURCE_MTH_STEP(state,current2,theta,2)
        decoded.append(q1+q2/2)
        counts.append(int((q1!=0).sum()+(q2!=0).sum()))
    assert counts==[2,2]
    assert decoded[0].item()==.5078125 and decoded[1].item()==.75
    # A fixed downstream class boundary distinguishes these histories despite
    # exactly the same integrated input and active event count.
    assert decoded[0].item()<.6<decoded[1].item()
