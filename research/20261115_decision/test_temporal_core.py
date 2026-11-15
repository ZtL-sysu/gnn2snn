import sys
from pathlib import Path
from unittest.mock import patch
import torch
from torch_geometric.data import Data, Batch
sys.path.insert(0, str(Path(__file__).parent))
from temporal_core import TemporalController, temporal_cache, temporal_forward, core
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model


def fixture():
    torch.manual_seed(41)
    model = fold_model(BenchmarkGCN(3, width=146).eval()).eval()
    net = core.CalibratedGCN(model, [torch.ones(w) for w in (146,146,146,146,73,36)]).eval().requires_grad_(False)
    graphs = [Data(x=torch.rand(8,1), pos=torch.rand(8,2),
                   edge_index=torch.tensor([[0,1,2,3,4,5,6,7,0,0],[1,2,3,4,5,6,7,0,3,5]])) for _ in range(2)]
    return net, Batch.from_data_list(graphs)


def test_zero_selector_all_modes_exact_baseline_and_no_teacher():
    net,batch = fixture()
    for mode in ('nongraph','graph','shuffled','toppin'):
        c = TemporalController(mode)
        cache = temporal_cache(batch,mode,[4,5])
        with patch.object(core,'teacher_sites',side_effect=AssertionError('online teacher')):
            for T in (1,2,4,8):
                with torch.no_grad(): expected, expected_sites = net(batch,T)
                actual,sites,tr = temporal_forward(net,batch,T,c,cache,[4,5])
                assert torch.equal(actual,expected)
                assert all(torch.equal(a,b) for a,b in zip(sites,expected_sites))
                assert tr['choice_count'].sum()==0
    assert c.weight.numel()==584


def test_forced_selection_prefix_value_and_projected_derivative():
    net,batch=fixture()
    c=TemporalController('nongraph')
    with torch.no_grad(): c.weight[0].fill_(1)
    _,s1,tr1=temporal_forward(net,batch,1,c,None,[4,5])
    _,s2,tr2=temporal_forward(net,batch,2,c,None,[4,5])
    assert tr1['choice_count'].sum()>0
    assert torch.equal(tr1['choice_count'],tr2['choice_count'])
    assert tr2['first_decode_error_max'].max()==0
    g1=torch.autograd.grad(s1[0].sum(),c.weight)[0]
    g2=torch.autograd.grad(s2[0].sum(),c.weight)[0]
    assert torch.count_nonzero(g1)>0
    assert torch.count_nonzero(g2)==0


def test_cache_is_graph_only_and_joint_shuffle_preserves_distribution():
    _,batch=fixture()
    a=temporal_cache(batch,'graph',[4,5])
    b=temporal_cache(batch,'shuffled',[4,5])
    batch.x.fill_(999)
    assert torch.equal(a,temporal_cache(batch,'graph',[4,5]))
    for lo,hi in zip(batch.ptr[:-1],batch.ptr[1:]):
        assert torch.equal(a[lo:hi].sort(0).values,b[lo:hi].sort(0).values)
