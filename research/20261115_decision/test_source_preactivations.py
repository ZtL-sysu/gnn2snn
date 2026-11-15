"""Pre-ReLU source-statistics extraction must preserve the frozen GCN path."""
import sys
from pathlib import Path
import torch
from torch_geometric.data import Data,Batch
sys.path.insert(0,str(Path(__file__).parent))
from run_source_ti import teacher_preactivations,core,source_threshold_network
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model


def test_six_preactivations_match_teacher_sites_on_irregular_graphs():
    torch.manual_seed(41)
    model=fold_model(BenchmarkGCN(3,width=16).eval()).eval()
    graphs=[Data(x=torch.rand(3,1),pos=torch.rand(3,2),
        edge_index=torch.tensor([[0,0,1,2],[1,2,0,1]])),
        Data(x=torch.rand(2,1),pos=torch.rand(2,2),edge_index=torch.tensor([[0,1],[1,0]]))]
    batch=Batch.from_data_list(graphs)
    with torch.no_grad():
        expected,sites=core.teacher_sites(model,batch)
        actual,pre=teacher_preactivations(model,batch)
    assert len(pre)==6
    assert torch.equal(expected,actual)
    assert all(torch.equal(z.relu(),a) for z,a in zip(pre,sites))
    assert any((z<0).any() for z in pre)


def test_source_zero_threshold_semantics_without_epsilon_repair():
    torch.manual_seed(42)
    model=fold_model(BenchmarkGCN(3,width=16).eval()).eval()
    batch=Batch.from_data_list([Data(x=torch.rand(3,1),pos=torch.rand(3,2),
        edge_index=torch.tensor([[0,1,2],[1,2,0]]))])
    thresholds=[torch.zeros(w) for w in (16,16,16,16,8,4)]
    net=source_threshold_network(model,thresholds).eval()
    with torch.no_grad():
        for T in (1,2,4,8):
            logits,sites=net(batch,T)
            assert torch.equal(logits,model.mlp[4].bias[None,:])
            assert all(torch.count_nonzero(s)==0 for s in sites)
    assert all(torch.count_nonzero(t)==0 for t in net.thresholds)
