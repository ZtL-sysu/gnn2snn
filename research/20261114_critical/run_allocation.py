"""Critical-site structure calibration screen; no test loading. Adapted from frozen run_screen."""
import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '20261114_story'
sys.path.insert(0,str(OLD))
from objective import loss_terms, make_permutation
from codebook_core import CodebookGCN
from codebook_fit import fit_codebook
from allocation_core import AllocationGCN
LEGACY = HERE.parent / 'visual-graphs' / 'takeover_20261112'
sys.path.insert(0, str(LEGACY))
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model
from calibration_core import CalibratedGCN, teacher_sites

DEFAULT_ASSETS = Path('C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision')
SITES = ['gcn_0', 'gcn_1', 'gcn_2', 'gcn_3', 'mlp_0', 'mlp_1']


def resolve_calibration(assets, dataset, seed):
    """Resolve the two historical layouts without changing calibration bytes."""
    root=Path(assets)/'takeover_20261112'
    primary=root/'full_teacher_runs'/f'{dataset}_seed{seed}_full'/'calibration_hist_v1.json'
    if primary.is_file(): return primary
    replica=root/'rep'/f"{'M' if dataset=='MNIST' else 'C'}{seed}"/'cal.json'
    if replica.is_file(): return replica
    raise FileNotFoundError(f'No audited calibration for {dataset} seed {seed}')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def metrics(z, a, y):
    z, a = torch.as_tensor(z), torch.as_tensor(a)
    return dict(accuracy_percent=float((z.argmax(1).numpy()==y).mean()*100),
                teacher_agreement_percent=float((z.argmax(1)==a.argmax(1)).float().mean()*100),
                logit_mse=float((z-a).square().mean()),
                cross_entropy=float(F.cross_entropy(z, torch.as_tensor(y).long())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['MNIST','CIFAR10'], required=True)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    p.add_argument('--author-repo', type=Path, default=Path('C:/code/gnn2snn-dcgs-reference'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda', choices=['cpu','cuda'])
    p.add_argument('--train-count', type=int, default=512)
    p.add_argument('--val-count', type=int, default=512)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--T', nargs='+', type=int, default=[1,2])
    p.add_argument('--methods', nargs='+', default=['random','local','identity','shuffled','message'])
    args=p.parse_args()
    campaign_start=time.monotonic()
    if args.steps<1 or not (0<args.lr<=0.01):
        raise ValueError('invalid optimization budget')
    if any(t not in [1,2,4,8] for t in args.T) or len(set(args.T))!=len(args.T):
        raise ValueError('invalid T')
    if not set(args.methods)<=set(['random','local','identity','shuffled','message']) or len(set(args.methods))!=len(args.methods):
        raise ValueError('invalid methods')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261114)
    np.random.seed(20261114)
    dev=torch.device(args.device)
    runs=args.assets/'takeover_20261112'/'full_teacher_runs'
    run=runs/f'{args.dataset}_seed{args.seed}_full'
    calpath=resolve_calibration(args.assets,args.dataset,args.seed)
    cal=json.loads(calpath.read_text())
    metadata=json.loads((run/'metadata.json').read_text())
    if cal['dataset']!=args.dataset or metadata['seed']!=args.seed:
        raise RuntimeError('calibration/teacher identity mismatch')
    checkpoint=run/'best.pt'
    if sha(checkpoint)!=cal['checkpoint_sha256']:
        raise RuntimeError('checkpoint/calibration mismatch')
    for filename,digest in cal['code_sha256'].items():
        if sha(LEGACY/filename)!=digest:
            raise RuntimeError('calibration dependency mismatch: '+filename)
    data_root=args.assets/'data'/'pyg_bench'
    for filename, digest in cal['data']['processed_files_sha256'].items():
        if sha(data_root/args.dataset/'processed'/filename)!=digest:
            raise RuntimeError('data hash mismatch: '+filename)
    state=torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['epoch']!=cal['teacher_epoch']:
        raise RuntimeError('teacher/calibration epoch mismatch')
    model=BenchmarkGCN(metadata['input_dim']).to(dev)
    model.load_state_dict(state['model']); model.eval()
    model.requires_grad_(False)
    folded=fold_model(model).eval()
    thresholds=[cal['channel_thresholds'][s] for s in SITES]
    net=AllocationGCN(folded, thresholds).to(dev).eval()
    train=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='train')
    val=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='val')
    expected=55000 if args.dataset=='MNIST' else 45000
    if len(train)!=expected or len(val)!=5000:
        raise RuntimeError('official split count mismatch')
    ti=np.random.default_rng(20261114).permutation(len(train))[:args.train_count]
    vi=np.random.default_rng(20261115).permutation(len(val))[:args.val_count]
    if not (0 < len(ti)==args.train_count<=expected and 0 < len(vi)==args.val_count<=5000):
        raise ValueError('invalid sample counts')
    train_batches=[]
    for b in DataLoader(train[ti.tolist()],batch_size=32,shuffle=False):
        b=b.to(dev)
        del b.y
        with torch.no_grad():
            a, sites=teacher_sites(folded,b)
        train_batches.append((b,a,sites))
    val_batches=list(DataLoader(val[vi.tolist()],batch_size=32,shuffle=False))
    import dcgs_fixed_seed_eval_v2 as ref
    mth, source_commit, provenance=ref.require_author_source(args.author_repo)
    first=val_batches[0].to(dev)
    equality={}
    with torch.no_grad():
        original=model(first); folded_logits, _=teacher_sites(folded,first)
        fold_error=float((original-folded_logits).abs().max())
        if not torch.allclose(original,folded_logits,rtol=1e-5,atol=1e-4):
            raise RuntimeError('BN folding mismatch')
        for t in [1,2,4,8]:
            a,_,_=ref.dcgs_event_forward(folded,first,thresholds,t,mth)
            b,_=net(first,t)
            equality[str(t)]=float((a-b).abs().max())
            if not torch.equal(a,b):
                raise RuntimeError(f'hard baseline mismatch T{t}: {equality[str(t)]}')
    meta=dict(dataset=args.dataset,seed=args.seed,T=args.T,steps=args.steps,
              train_indices=ti.tolist(),val_indices=vi.tolist(),batch_size=32,
              optimizer='none; frozen previous offsets',lr=None,offset_bound=0.5,
              checkpoint_sha256=sha(checkpoint),calibration_sha256=sha(calpath),
              calibration_path=str(calpath.resolve()),
              checkpoint_epoch=state['epoch'],source_commit=source_commit,
              source_provenance=provenance,fold_max_abs=fold_error,baseline_exact_max_abs=equality,
              script_sha256=sha(__file__),core_sha256=sha(OLD/'calibration_core.py'),
              dependency_sha256={name:sha(LEGACY/name) for name in
                                  ['full_teacher_gcn.py','fold_teacher_bn.py','dcgs_fixed_seed_eval_v2.py']},
              git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
              torch=torch.__version__,device=str(dev),gpu=torch.cuda.get_device_name() if dev.type=='cuda' else None,
              deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
              CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
              train_graphs_official=len(train),val_graphs_official=len(val),test_loaded=False,
              qualification='historically exposed validation; current campaign does not access test; exploratory screen')
    sync()
    base_path=HERE/'runs'/'screen'/f'{args.dataset}{args.seed}'/'all_local_T1_parameters.npz'
    params=np.load(base_path)
    with torch.no_grad():
        for i,offset in enumerate(net.offsets):offset.copy_(torch.as_tensor(params[f'offset_{i}'],device=dev))
    meta['base_parameters_sha256']=sha(base_path)
    meta['base_parameters_path']=str(base_path.resolve())
    meta['base_parameter_scope']='T1 trained all_local .001; same snapshot deployed for every evaluated prefix'
    shutil.copyfile(base_path,args.output/'input_base_parameters.npz')
    meta['setup_and_acceptance_seconds']=time.monotonic()-campaign_start
    source_dir=args.output/'source'
    source_dir.mkdir()
    snapshot_paths=[Path(__file__),HERE/'objective.py',HERE/'allocation.py',HERE/'allocation_core.py',OLD/'calibration_core.py']+[LEGACY/name for name in meta['dependency_sha256']]
    meta['source_snapshots']={}
    for source in snapshot_paths:
        target=source_dir/source.name
        shutil.copyfile(source,target)
        meta['source_snapshots'][source.name]=sha(target)
    write_json(args.output/'metadata.json',meta)
    rows=[]

    def evaluate(tag,t,net,bias=None):
        parts={k:[] for k in ['logits','teacher','labels','node_mse','graph_mse','margin','margin_shift','flip','nodes','mean_degree','degree_variance']}
        allocation_records=[]
        sync(); start=time.monotonic()
        with torch.no_grad():
            for batch in val_batches:
                batch=batch.clone().to(dev)
                labels=batch.y.view(-1).long().cpu().numpy()
                del batch.y
                a, acts=teacher_sites(folded,batch)
                z, sites=net(batch,t)
                allocation_records.append(net.last_allocation)
                if bias is not None: z=z+bias
                local=[]
                for i,(s,r) in enumerate(zip(sites,acts)):
                    e=(s-r).square().mean(-1,keepdim=True)
                    local.append(global_mean_pool(e,batch.batch).flatten() if i<4 else e.flatten())
                ap=a.argmax(1)
                rival=a.clone(); rival.scatter_(1,ap[:,None],float('-inf'))
                margin=a.gather(1,ap[:,None]).flatten()-rival.max(1).values
                zr=z.clone(); zr.scatter_(1,ap[:,None],float('-inf'))
                zm=z.gather(1,ap[:,None]).flatten()-zr.max(1).values
                degree=torch.bincount(batch.edge_index[0],minlength=batch.num_nodes).float()
                dm=global_mean_pool(degree[:,None],batch.batch).flatten()
                d2=global_mean_pool(degree.square()[:,None],batch.batch).flatten()
                values=dict(logits=z,teacher=a,labels=labels,node_mse=torch.stack(local,1),
                            graph_mse=(z-a).square().mean(1),margin=margin,margin_shift=zm-margin,
                            flip=z.argmax(1)!=ap,nodes=batch.ptr[1:]-batch.ptr[:-1],
                            mean_degree=dm,degree_variance=d2-dm.square())
                for k,v in values.items():
                    parts[k].append(v.detach().cpu().numpy() if torch.is_tensor(v) else v)
        sync(); elapsed=time.monotonic()-start
        arrays={k:np.concatenate(v) for k,v in parts.items()}
        arrays['indices']=vi
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise RuntimeError('non-finite predictions/diagnostics')
        np.savez_compressed(args.output/f'{tag}_T{t}.npz',**arrays)
        row=dict(method=tag,T=t,**metrics(arrays['logits'],arrays['teacher'],arrays['labels']),
                 teacher_accuracy_percent=float((arrays['teacher'].argmax(1)==arrays['labels']).mean()*100),
                 val_graphs=len(vi),diagnostic_eval_seconds=elapsed)
        row['allocation']={k:sum(r.get(k,0) for r in allocation_records) for k in ['selected_nodes','max_opportunity_slots','extra_nonzero_events','extra_edge_deliveries']}
        write_json(args.output/f'{tag}_T{t}_allocation.json',allocation_records)
        rows.append(row); print(json.dumps(row),flush=True)
        write_json(args.output/'results.json',rows)

    for method in ['baseline']+args.methods:
        net.policy=None if method=='baseline' else method
        for t in args.T:evaluate(method,t,net)
    sync()
    if any(sha(path)!=meta['source_snapshots'][path.name] for path in snapshot_paths):
        raise RuntimeError('source changed during run')
    if sha(base_path)!=meta['base_parameters_sha256']:raise RuntimeError('base parameters changed')
    write_json(args.output/'DONE.json',dict(status='complete',rows=len(rows),test_loaded=False,
                                          total_elapsed_seconds=time.monotonic()-campaign_start))


if __name__=='__main__':
    main()
