"""Source-linked threshold-iteration graph adaptation; no new algorithm claim."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_objectives import historical,core,CRITICAL
from run_interventions import measured_forward


@torch.no_grad()
def teacher_preactivations(model,batch):
    raw=torch.cat((batch.x.float(),batch.pos.float()),dim=-1)
    h=model.embedding(raw)
    src,dst=batch.edge_index;n=batch.num_nodes
    out=torch.bincount(src,minlength=n).clamp_min(1).float()
    inc=torch.bincount(dst,minlength=n).clamp_min(1).float()
    weights=out[src].rsqrt()*inc[dst].rsqrt()
    pre=[]
    for conv in model.convs:
        z=conv(h,batch.edge_index,weights);pre.append(z);h=h+z.relu()
    h=global_mean_pool(h,batch.batch)
    z=model.mlp[0](h);pre.append(z);h=z.relu()
    z=model.mlp[2](h);pre.append(z);h=z.relu()
    return model.mlp[4](h),pre


def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def source_threshold_network(model,thresholds):
    """Permit source converter's exact zero thresholds without changing frozen core."""
    values=[torch.as_tensor(t,dtype=torch.float32) for t in thresholds]
    if any(not torch.isfinite(t).all() or (t<0).any() for t in values):
        raise ValueError('source converter output must be finite and nonnegative')
    # The historical calibration constructor requires strictly positive values.
    # Install source-legal zero channels explicitly after constructing its states;
    # _mth_step itself handles zero exactly (no event), as the source neuron does.
    net=core.CalibratedGCN(model,[torch.where(t>0,t,torch.ones_like(t)) for t in values])
    with torch.no_grad():
        for target,value in zip(net.thresholds,values):target.copy_(value.to(target.device))
    return net


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['MNIST','CIFAR10'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(20261115)
    start=time.monotonic()
    meta,_,_,folded,loaded,cal,data_root=historical.load_teacher(args.dataset,41,historical.DEFAULT_ASSETS,
        CRITICAL/'runs/allocation'/f'{args.dataset}41',torch.device('cuda'))
    from source_ti import fit_source_ti
    train=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='train')
    val=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='val')
    pre_batches=[]
    first=None
    for b in DataLoader(train[meta['train_indices']],batch_size=32,shuffle=False):
        b=b.cuda();del b.y
        if first is None:first=b
        a,pre=teacher_preactivations(folded,b)
        with torch.no_grad():expected,sites=core.teacher_sites(folded,b)
        assert torch.equal(a,expected)
        assert all(torch.equal(z.relu(),act) for z,act in zip(pre,sites))
        pre_batches.append([z.cpu() for z in pre])
    batches=list(DataLoader(val[meta['val_indices']],batch_size=32,shuffle=False))
    import dcgs_fixed_seed_eval_v2 as ref
    author=Path('C:/code/gnn2snn-dcgs-reference')
    mth,commit,provenance=ref.require_author_source(author)
    snapshot=args.output/'source';snapshot.mkdir()
    for path in (Path(__file__),HERE/'source_ti.py',HERE/'run_interventions.py',author/'converter/threshold_getter.py'):
        shutil.copyfile(path,snapshot/path.name)
    record=dict(dataset=args.dataset,seed=41,test_loaded=False,labels_used=False,teacher_retrained=False,
        train_indices=meta['train_indices'],val_indices=meta['val_indices'],calibration_graphs=512,
        batch_size=32,levels=8,threshold_scale=1,calibration_horizons=[1,2],evaluated_prefixes=[1,2,4,8],
        qualification='source TI + graph adaptation, source rule executes unmodified; exposed val512; not literal full ImageNet pipeline reproduction',
        checkpoint_sha256=meta['checkpoint_sha256'],calibration_sha256=meta['calibration_sha256'],
        processed_file_sha256=cal['data']['processed_files_sha256'],source_commit=commit,source_provenance=provenance,
        preactivation_acceptance='bitwise logits and all six teacher-site ReLUs over every calibration batch',
        snapshots={p.name:historical.sha(p) for p in snapshot.iterdir()},torch=torch.__version__)
    save(args.output/'metadata.json',record)
    rows=[]
    for H in (1,2):
        fit_start=time.monotonic()
        try:
            thresholds,fit=fit_source_ti(pre_batches,effective_steps=H*(2**8),author_repo=author)
        except Exception as exc:
            save(args.output/f'H{H}_failure.json',dict(error_type=type(exc).__name__,error=str(exc),
                status='source failure retained; no repaired threshold or clipping retry'))
            raise
        fit_seconds=time.monotonic()-fit_start
        save(args.output/f'H{H}_fit.json',fit)
        np.savez_compressed(args.output/f'H{H}_thresholds.npz',**{f'threshold_{i}':v.detach().cpu().numpy() for i,v in enumerate(thresholds)})
        net=source_threshold_network(folded,thresholds).cuda().eval().requires_grad_(False)
        exact={}
        with torch.no_grad():
            for T in (1,2,4,8):
                a,_,_=ref.dcgs_event_forward(folded,first,thresholds,T,mth)
                b,_=net(first,T);exact[str(T)]=bool(torch.equal(a,b));assert exact[str(T)]
        save(args.output/f'H{H}_source_acceptance.json',exact)
        for T in (1,2,4,8):
            parts={k:[] for k in ('logits','teacher','labels','events','amplitude')}
            with torch.no_grad():
                for b in batches:
                    b=b.clone().cuda();labels=b.y.flatten().cpu().numpy();del b.y
                    a,_=core.teacher_sites(folded,b);z,_,trace=measured_forward(net,b,T)
                    values=dict(logits=z.cpu().numpy(),teacher=a.cpu().numpy(),labels=labels,
                        events=trace['events'].cpu().numpy().astype(np.int64),amplitude=trace['amplitude'].cpu().numpy())
                    for key,value in values.items():parts[key].append(value)
            arrays={k:np.concatenate(v) for k,v in parts.items()};arrays['indices']=np.asarray(meta['val_indices'])
            np.savez_compressed(args.output/f'H{H}_T{T}.npz',**arrays)
            row=dict(calibration_horizon=H,T=T,source_intended_operating_point=(H==T),
                accuracy_percent=float(100*np.mean(arrays['logits'].argmax(1)==arrays['labels'])),
                teacher_agreement_percent=float(100*np.mean(arrays['logits'].argmax(1)==arrays['teacher'].argmax(1))),
                events=int(arrays['events'].sum()),amplitude_sum=float(arrays['amplitude'].astype(np.float64).sum()),
                optimizer_updates=0,statistic_plus_TI_seconds=fit_seconds,threshold_storage_bytes=sum(v.numel()*4 for v in thresholds))
            rows.append(row);save(args.output/'results.json',rows);print(json.dumps(row),flush=True)
    save(args.output/'DONE.json',dict(endpoints=len(rows),seconds=time.monotonic()-start))


if __name__=='__main__':main()
