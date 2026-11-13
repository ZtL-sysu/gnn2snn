"""CPU-only synthetic checks; no official dataset load."""
import copy
import tempfile
from pathlib import Path
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from gated_teacher_model import GatedTeacher
import gated_affine_cal_v1 as cal


def reject(fn, kind=(ValueError,RuntimeError,FileExistsError)):
    try:
        fn()
    except kind:
        return
    raise AssertionError('expected rejection')


def graph():
    return Data(x=torch.randn(3,3),pos=torch.randn(3,2),
                edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                edge_attr=torch.rand(4,1),y=torch.tensor([2]))


def main():
    torch.manual_seed(1326)
    torch.set_num_threads(2)
    model = GatedTeacher(hidden_dim=8,out_dim=8).eval()
    ds = cal.LabelFreeGraphView([graph() for _ in range(3)])
    batch = Batch.from_data_list([ds[0],ds[1]])
    values = cal.capture_inputs(model,batch)
    assert len(values)==14 and 'y' not in batch
    # Independent native layer recurrence, not a second copy of the capture hooks.
    h = model.embedding_h(torch.cat((batch.x,batch.pos),1))
    e = model.embedding_e(batch.edge_attr)
    qh,qe = torch.zeros_like(h),torch.zeros_like(e)
    expected = {}
    for i,layer in enumerate(model.layers):
        expected[f'raw_node_{i}'], expected[f'raw_edge_{i}'] = h.abs(),e.abs()
        if i:
            expected[f'branch_cumsum_node_{i}'] = qh
            expected[f'branch_cumsum_edge_{i}'] = qe
        hn,en = layer(batch.edge_index,h,e)
        qh,qe = qh+(hn-h),qe+(en-e)
        h,e = hn,en
    for s in expected:
        torch.testing.assert_close(values[s],expected[s],rtol=2e-5,atol=2e-6)
    loader = DataLoader(ds,batch_size=2,shuffle=False)
    stats = cal.calibrate_loader(model,loader,'cpu',expected=3,width=8,batch_size=2)
    cal.validate_statistics(stats,width=8,expected=3,batch_size=2)
    assert stats['pass_graph_counts']==[3,3] and stats['pass_batch_counts']==[2,2]
    for i in range(4):
        assert stats['sites'][f'raw_node_{i}']['rows']==9
        assert stats['sites'][f'raw_edge_{i}']['rows']==12
    for mutation in ('counts','theta','scope','rows','nonfinite'):
        broken = copy.deepcopy(stats)
        if mutation=='counts': broken['sites']['raw_node_0']['channel_counts'][0][0]+=1
        elif mutation=='theta': broken['sites']['raw_node_0']['per_channel_thresholds'][0]*=2
        elif mutation=='scope': broken['pass_graph_counts'][1]-=1
        elif mutation=='rows': broken['pass_rows_per_site'][0]['raw_node_0']-=1
        else: broken['sites']['raw_node_0']['maxima'][0]=float('nan')
        reject(lambda:cal.validate_statistics(broken,width=8,expected=3,batch_size=2))
    reject(lambda:cal.calibrate_loader(model,loader,'cpu',expected=4,width=8,batch_size=2))
    with tempfile.TemporaryDirectory(dir=cal.HERE) as tmp:
        source=Path(tmp)/'source'; source.write_text('before')
        hashes={str(source):cal.sha(source)}
        cal.require_unchanged(hashes)
        source.write_text('after')
        reject(lambda:cal.require_unchanged(hashes))
        reject(lambda:cal.output_path(Path(tmp)/'result.json'))
    reject(lambda:cal.output_path(Path(__file__)))
    model.layers[0].bn_node_h.train()
    reject(lambda:cal.capture_inputs(model,batch))
    model.eval()
    labeled = batch.clone(); labeled.y=torch.tensor([1,2])
    reject(lambda:cal.capture_inputs(model,labeled))
    bad=batch.clone(); bad.x[0,0]=float('nan')
    reject(lambda:cal.capture_inputs(model,bad))
    original=model.layers[0].bn_node_h.running_mean.clone()
    model.layers[0].bn_node_h.running_mean[0]=float('nan')
    reject(lambda:cal.capture_inputs(model,batch))
    model.layers[0].bn_node_h.running_mean.copy_(original)
    assert all(not l._forward_pre_hooks and not l.bn_node_h._forward_hooks and
               not l.bn_node_e._forward_hooks for l in model.layers)
    doc={'calibration':stats}
    generic=cal.profile_thresholds(doc,'generic',1)
    first=cal.profile_thresholds(doc,'first_analog',1)
    centered=cal.profile_thresholds(doc,'centered',8)
    assert len(generic)==8 and len(first)==len(centered)==6
    for a,b in zip(generic[2:],first): torch.testing.assert_close(a,b,rtol=0,atol=0)
    reject(lambda:cal.profile_thresholds(doc,'centered',2))
    # Frozen real checkpoint/source validation without opening the graph dataset.
    teacher,info=cal.load_teacher('cpu')
    assert info['metadata']['seed']==41 and not teacher.training
    print('PASS: affine calibration hooks, two-pass histograms, coverage, thresholds, rejects, real teacher binding; no official data')


if __name__=='__main__': main()
