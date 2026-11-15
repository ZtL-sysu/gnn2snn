import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.data import Data,Batch
sys.path.insert(0,str(Path(__file__).parent))
from learned_swap_core import learned_forward,core
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model


def fixture():
    torch.manual_seed(37)
    model=fold_model(BenchmarkGCN(3).eval()).eval()
    net=core.CalibratedGCN(model,[torch.ones(w) for w in (146,146,146,146,73,36)]).eval().requires_grad_(False)
    batch=Batch.from_data_list([Data(x=torch.rand(5,1),pos=torch.rand(5,2),
        edge_index=torch.tensor([[0,1,2,3,4],[1,2,3,4,0]]))])
    return net,batch


def test_zero_selector_exact_baseline_and_no_online_teacher():
    net,batch=fixture()
    model={'mean':np.zeros(162),'std':np.ones(162),'coef':np.zeros((163,8)),'alpha':.001}
    calls=[];original=core._mth_step
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    with patch.object(core,'teacher_sites',side_effect=AssertionError('online teacher called')):
        for T in (1,2,4,8):
            with torch.no_grad():expected,_=net(batch,T)
            calls.clear()
            with patch.object(core,'_mth_step',counted):
                actual,trace=learned_forward(net,batch,T,model,'graph',[19])
            assert torch.equal(actual,expected)
            assert len(calls)==6*T  # No additional downstream trial forward.
            assert trace['choice'].tolist()==[0]
            assert trace['events'].dtype==torch.int64


def test_first_site_choice_shared_across_prefixes_when_swapping():
    net,batch=fixture()
    model={'mean':np.zeros(162),'std':np.ones(162),'coef':np.zeros((163,8)),'alpha':.001}
    model['coef'][-1]=np.array([1.,1.,-1.,-1.,-1.,-1.,-1.,-1.])
    _,a=learned_forward(net,batch,1,model,'nongraph',[41])
    _,b=learned_forward(net,batch,2,model,'nongraph',[41])
    assert np.array_equal(a['choice'],b['choice'])
    assert a['choice'].item()>0
    for T in (4,8):
        _,later=learned_forward(net,batch,T,model,'nongraph',[41])
        assert np.array_equal(a['choice'],later['choice'])
