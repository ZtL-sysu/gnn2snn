import sys, json, os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
LEGACY=HERE.parent/'visual-graphs'/'takeover_20261112'
sys.path.insert(0,str(LEGACY))
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model
from calibration_core import CalibratedGCN, _mth_step as core_step, teacher_sites
import calibration_core as cc
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import dcgs_fixed_seed_eval_v2 as ref

def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(20261114); np.random.seed(20261114)
    assets=Path(r'C:\code\phd-snn\auto-paper\20261110-152407\Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs\f-research-v1-run-20261110-172308-superpixel-graph-vision')
    run=assets/'takeover_20261112'/'full_teacher_runs'/'MNIST_seed41_full'
    cal=json.loads((run/'calibration_hist_v1.json').read_text())
    meta=json.loads((run/'metadata.json').read_text())
    state=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    dev=torch.device('cuda')
    model=BenchmarkGCN(meta['input_dim']).to(dev); model.load_state_dict(state['model']); model.eval().requires_grad_(False)
    folded=fold_model(model).eval()
    thresholds=[cal['channel_thresholds'][s] for s in ['gcn_0','gcn_1','gcn_2','gcn_3','mlp_0','mlp_1']]
    net=CalibratedGCN(folded,thresholds).to(dev).eval()
    data=GNNBenchmarkDataset(root=str(assets/'data'/'pyg_bench'),name='MNIST',split='val')
    vi=np.random.default_rng(20261115).permutation(len(data))[:32]
    batch=next(iter(DataLoader(data[vi.tolist()],batch_size=32,shuffle=False))).to(dev)
    del batch.y
    train=GNNBenchmarkDataset(root=str(assets/'data'/'pyg_bench'),name='MNIST',split='train')
    ti=np.random.default_rng(20261114).permutation(len(train))[:64]
    for b in DataLoader(train[ti.tolist()],batch_size=32,shuffle=False):
        b=b.to(dev); del b.y
        with torch.no_grad(): teacher_sites(folded,b)
    original=model(batch)
    teacher_sites(folded,batch)
    MTH,_,_=ref.require_author_source(Path(r'C:\code\gnn2snn-dcgs-reference'))
    all_core=[]
    def core_wrapper(state,current,theta,t):
        out=core_step(state,current,theta,t)
        idx=len(all_core)
        all_core.append((current.detach().clone(),out.detach().clone()))
        return out
    cc._mth_step=core_wrapper
    core_results={}; source_results={}
    source_runs=[]
    orig_fresh=ref.fresh_mth
    def fresh(th,device):
        mod=orig_fresh(MTH,th,device)
        old=mod.forward
        slot={'rows':[]}
        def traced(x):
            y=old(x); slot['rows'].append((x.detach().clone(),y.detach().clone())); return y
        mod.forward=traced
        source_runs.append(slot)
        return mod
    ref.fresh_mth=lambda _,th,device:fresh(th,device)
    for T in (1,2,4,8):
        with torch.no_grad():
            core_results[T]=net(batch,T)[0]
            source_results[T]=ref.dcgs_event_forward(folded,batch,thresholds,T,MTH)[0]
        print('T',T,'maxerr',float((core_results[T]-source_results[T]).abs().max()),'equal',torch.equal(core_results[T],source_results[T]))
    # Run both repeatedly to detect stochastic accumulation.
    print('repeat')
    for T in (4,4,4):
        with torch.no_grad():
            a=net(batch,T)[0]; b=ref.dcgs_event_forward(folded,batch,thresholds,T,MTH)[0]
        print(T,float((a-b).abs().max()),torch.equal(a,b))
    # source log capture, per site active ticks and current/spike comparisons.
    source_runs.clear()
    with torch.no_grad(): ref.dcgs_event_forward(folded,batch,thresholds,4,MTH)
    src1=[x['rows'][:] for x in source_runs]; source_runs.clear()
    core1=[]; all_core.clear()
    with torch.no_grad(): core_step_save=net(batch,4)[0]
    core1=all_core[:]
    print('core after source trace output',float((core_step_save-source_results[4]).abs().max()))
    # Source logs include warmup at index 0 and four active events after.
    for site in range(6):
        for tick in range(1,5):
            ci=(tick-1)*6+site
            sr1=src1[site][tick][0]; ss1=src1[site][tick][1]
            cr=core1[ci][0]; cs=core1[ci][1]
            if not torch.equal(cr,sr1) or not torch.equal(cs,ss1):
                curr=float((cr-sr1).abs().max()); spike=float((cs-ss1).abs().max())
                print('first/mismatch site tick',site+1,tick,'current max',curr,'spike max',spike)
                print('curr differing entries',(cr!=sr1).sum().item(),'spike differing entries',(cs!=ss1).sum().item())
                print('source current range',float(sr1.min()),float(sr1.max()),'core',float(cr.min()),float(cr.max()))
                print('source spike range',float(ss1.min()),float(ss1.max()),'core',float(cs.min()),float(cs.max()))
                print('theta',thresholds[site][:5])
                return
    print('no per-site diff')
if __name__=='__main__': main()
