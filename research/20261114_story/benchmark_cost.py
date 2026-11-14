"""Matched synchronized software latency and explicit operation ledger, not energy."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

from run_screen import DEFAULT_ASSETS, LEGACY, sha, write_json, resolve_calibration
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model
import calibration_core as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args()
    if args.repeats<2: raise ValueError('repeats must be >=2')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    m=json.loads((args.run/'metadata.json').read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dev=torch.device('cuda')
    r=args.assets/'takeover_20261112'/'full_teacher_runs'/f"{m['dataset']}_seed{m['seed']}_full"
    if sha(r/'best.pt')!=m['checkpoint_sha256']: raise RuntimeError('checkpoint mismatch')
    calpath=resolve_calibration(args.assets,m['dataset'],m['seed'])
    if sha(calpath)!=m['calibration_sha256']: raise RuntimeError('calibration mismatch')
    if sha(Path(core.__file__))!=m['core_sha256']: raise RuntimeError('core mismatch')
    for filename,digest in m['dependency_sha256'].items():
        if sha(LEGACY/filename)!=digest: raise RuntimeError('dependency mismatch')
    cal=json.loads(calpath.read_text())
    teacher_meta=json.loads((r/'metadata.json').read_text())
    for filename,digest in cal['data']['processed_files_sha256'].items():
        if sha(args.assets/'data'/'pyg_bench'/m['dataset']/'processed'/filename)!=digest:
            raise RuntimeError('data mismatch')
    state=torch.load(r/'best.pt',map_location=dev,weights_only=False)
    if state['epoch']!=cal['teacher_epoch'] or state['epoch']!=m['checkpoint_epoch']:
        raise RuntimeError('epoch mismatch')
    model=BenchmarkGCN(teacher_meta['input_dim']).to(dev)
    model.load_state_dict(state['model'])
    model.eval().requires_grad_(False)
    folded=fold_model(model)
    theta=[cal['channel_thresholds'][s] for s in ['gcn_0','gcn_1','gcn_2','gcn_3','mlp_0','mlp_1']]
    dataset=GNNBenchmarkDataset(root=str(args.assets/'data'/'pyg_bench'),name=m['dataset'],split='val')
    batch=next(iter(DataLoader(dataset[m['val_indices'][:32]],batch_size=32))).to(dev)
    del batch.y
    n=int(batch.num_nodes); e=int(batch.edge_index.size(1)); g=int(batch.num_graphs)
    conditions={'teacher':lambda:model(batch)}
    nets={}
    for t in m['T']:
        for method in ['baseline','node','graph','mix','shuffled','output']:
            param=args.run/f'{method}_T{t}_parameters.npz'
            if method!='baseline' and not param.is_file(): continue
            net=core.CalibratedGCN(folded,theta).to(dev).eval()
            bias=torch.zeros(10,device=dev)
            if param.is_file():
                with np.load(param) as z,torch.no_grad():
                    for i,offset in enumerate(net.offsets):
                        value=z[f'offset_{i}']
                        if value.shape!=tuple(offset.shape) or not np.isfinite(value).all() or np.abs(value).max()>.500001:
                            raise ValueError('invalid offset archive')
                        offset.copy_(torch.as_tensor(value,device=dev))
                    if 'output_bias' in z:
                        if z['output_bias'].shape!=(10,) or not np.isfinite(z['output_bias']).all():
                            raise ValueError('invalid bias archive')
                        bias=torch.as_tensor(z['output_bias'],device=dev)
            tag=f'{method}_T{t}'
            nets[tag]=(net,t)
            conditions[tag]=(lambda net=net,t=t,bias=bias,method=method:
                              net(batch,t)[0]+bias if method=='output' else net(batch,t)[0])
    times={k:[] for k in conditions}
    with torch.inference_mode():
        for f in conditions.values():
            for _ in range(10): f()
        torch.cuda.synchronize()
        order=list(conditions)
        rng=np.random.default_rng(20261114)
        for _ in range(args.repeats):
            for key in rng.permutation(order):
                torch.cuda.synchronize(); start=time.perf_counter()
                result=conditions[key]()
                torch.cuda.synchronize()
                times[key].append((time.perf_counter()-start)*1000)
    rows=[]
    for key,v in times.items():
        a=np.asarray(v)
        rows.append(dict(condition=key,batch_ms_mean=float(a.mean()),batch_ms_median=float(np.median(a)),
                         batch_ms_std=float(a.std(ddof=1)),batch_ms_p10=float(np.quantile(a,.1)),
                         batch_ms_p90=float(np.quantile(a,.9)),per_graph_ms_median=float(np.median(a)/g),
                         raw_batch_ms=v))
    counts={}
    original=core._mth_step
    for tag,(net,t) in nets.items():
        site_rows=[]
        def counted(state,current,theta,tick):
            event=original(state,current,theta,tick)
            site_rows.append(dict(tick=tick,scalars=event.numel(),positive=int((event>0).sum()),
                                  negative=int((event<0).sum()),zero=int((event==0).sum())))
            return event
        core._mth_step=counted
        try:
            with torch.no_grad(): net(batch,t)
        finally: core._mth_step=original
        counts[tag]=site_rows
    # Actual dense PyG execution evaluates every projection and edge every tick.
    # MAC means one multiply-accumulate pair, not FLOPs or hardware joules.
    width=146; d=teacher_meta['input_dim']
    per_tick=dict(embedding_dense_MAC=n*d*width,
                  gcn_dense_projection_MAC=4*n*width*width,
                  gcn_edge_weight_multiplications=4*e*width,
                  gcn_edge_scatter_additions=4*e*width,
                  residual_additions=4*n*width,
                  mean_pool_scatter_additions=n*width,
                  mlp_dense_MAC=g*(146*73+73*36+36*10))
    write_json(args.output,dict(dataset=m['dataset'],seed=m['seed'],graphs=g,nodes=n,edges=e,
                               warmup_per_condition=10,repeats=args.repeats,device=torch.cuda.get_device_name(),
                               torch=torch.__version__,batch_indices=m['val_indices'][:32],latency=rows,
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                               event_counts=counts,dense_operation_ledger_per_tick=per_tick,
                               script_sha256=sha(__file__),input_metadata_sha256=sha(args.run/'metadata.json'),
                               checkpoint_sha256=m['checkpoint_sha256'],calibration_sha256=m['calibration_sha256'],
                               core_sha256=m['core_sha256'],dependency_sha256=m['dependency_sha256'],
                               event_semantics='signed eight-magnitude scalar events; zero events do not skip dense kernels',
                               scope='software core returns decoded six-site diagnostics as well as logits in all conversion conditions; data transfer excluded; fixed cached batch; not optimized deployment latency',
                               omitted_from_partial_dense_ledger=['BN teacher overhead','bias additions','degree normalization','membrane and eight-level comparison/state arithmetic','six-site decoder arithmetic','initialization and allocations'],
                               full_arithmetic_energy_proxy='NOT CLAIMED: partial operation ledger only',
                               hardware_energy='NOT MEASURED'))
    print(json.dumps({'output':str(args.output),'conditions':len(rows)}))


if __name__=='__main__': main()
