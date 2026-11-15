import sys
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data,Batch
sys.path.insert(0,str(Path(__file__).parent))
from run_swap_oracle import swap_forward,core
from event_swap import propose_swaps
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model
from summarize_swap_oracle import class_margins,budget_mask


def test_swap_hook_preserves_noop_and_first_site_multisets():
    torch.manual_seed(23)
    model=fold_model(BenchmarkGCN(3,width=16).eval()).eval()
    net=core.CalibratedGCN(model,[torch.ones(w) for w in (16,16,16,16,8,4)]).eval().requires_grad_(False)
    batch=Batch.from_data_list([Data(x=torch.rand(5,1),pos=torch.rand(5,2),
        edge_index=torch.tensor([[0,1,2,3,4],[1,2,3,4,0]]))])
    degree=torch.bincount(batch.edge_index[0],minlength=batch.num_nodes)
    with torch.no_grad():
        for T in (1,2):
            z,trace=swap_forward(net,batch,T)
            assert torch.equal(z,net(batch,T)[0])
            assert trace['events'].dtype==torch.int64
        pairs=propose_swaps(trace['first'],batch.ptr,degree,[17],count=1)[0]
        z,changed=swap_forward(net,batch,2,pairs)
        assert torch.equal(changed['before'].sort(0).values,changed['first'].sort(0).values)
        assert torch.equal(changed['before']!=0,changed['first']!=0)


def test_total_cost_masks_reject_first_site_only_equivalence():
    a={'pairs':np.array([[[-1,-1,-1]],[[0,1,0]],[[0,1,0]]])}
    for T in (1,2):
        a[f'events_T{T}']=np.array([[100],[101],[99]])
        a[f'deliveries_T{T}']=np.array([[300],[300],[299]])
        a[f'amplitude_T{T}']=np.array([[100.],[100.],[99.]])
    margins={2:np.array([[1.],[1.],[1.]])}
    assert budget_mask(a,margins,'exact_events').flatten().tolist()==[True,False,False]
    assert budget_mask(a,margins,'nonincrease').flatten().tolist()==[True,False,True]
    assert budget_mask(a,margins,'first_site_only').flatten().tolist()==[True,True,True]


def test_margin_includes_competitors_outside_teacher_top2():
    z=np.array([[[2.,1.,3.]]])
    assert class_margins(z,np.array([0])).item()==-1.
