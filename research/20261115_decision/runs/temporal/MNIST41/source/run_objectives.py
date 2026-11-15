"""Matched fixed-budget task-objective screen, same parameters for all prefixes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

HERE=Path(__file__).resolve().parent
CRITICAL=HERE.parent/'20261114_critical'
sys.path.insert(0,str(CRITICAL))
import analyze_allocation as historical
import calibration_core as core
from decision_objectives import decision_loss
from run_interventions import measured_forward


def local_loss(sites, acts):
    return sum((s-a).square().mean()/a.square().mean().clamp_min(1e-6) for s,a in zip(sites,acts))/6


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['MNIST','CIFAR10'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',default=['local_t1','local_joint','logits','kl','ranking'])
    p.add_argument('--steps',type=int,default=64)
    args=p.parse_args()
    assert 0<args.steps<=64
    assert set(args.methods)<=set(['local_t1','local_joint','logits','kl','ranking'])
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261115)
    campaign=time.monotonic()
    src=CRITICAL/'runs'/'allocation'/f'{args.dataset}41'
    meta,_,_,folded,loaded,cal,data_root=historical.load_teacher(args.dataset,41,historical.DEFAULT_ASSETS,src,torch.device('cuda'))
    initial=core.CalibratedGCN(folded,loaded.thresholds).cuda().eval()
    train=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='train')
    val=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='val')
    train_batches=[]
    for b in DataLoader(train[meta['train_indices']],batch_size=32,shuffle=False):
        b=b.cuda(); del b.y
        with torch.no_grad():
            teacher,acts=core.teacher_sites(folded,b)
        train_batches.append((b,teacher,acts))
    val_batches=list(DataLoader(val[meta['val_indices']],batch_size=32,shuffle=False))
    # Verify all four baseline prefixes against source-linked frozen event path.
    import dcgs_fixed_seed_eval_v2 as ref
    mth,commit,provenance=ref.require_author_source(Path('C:/code/gnn2snn-dcgs-reference'))
    acceptance={}
    with torch.no_grad():
        for T in (1,2,4,8):
            z,_,_=ref.dcgs_event_forward(folded,train_batches[0][0],loaded.thresholds,T,mth)
            actual,_=initial(train_batches[0][0],T)
            acceptance[str(T)]=bool(torch.equal(z,actual))
            assert acceptance[str(T)]
    record=dict(dataset=args.dataset,seed=41,train_indices=meta['train_indices'],val_indices=meta['val_indices'],
        checkpoint_sha256=meta['checkpoint_sha256'],calibration_sha256=meta['calibration_sha256'],
        data_hashes=cal['data']['processed_files_sha256'],test_loaded=False,labels_used_for_training=False,
        qualification='teacher-assisted unlabeled post-training calibration; selection-exposed validation512; no new method claim',
        methods=args.methods,steps=args.steps,lr=.01,batch_size=32,parameters=693,offset_bound=.5,
        offline_teacher=True,online_teacher=False,source_commit=commit,source_provenance=provenance,
        source_exact=acceptance,same_parameters_all_prefixes=True,
        optimization_forward_budget='T1 and T2 each step for every arm; local_t1 T2 zero-weight gradient',
        script_sha256=historical.sha(__file__),objectives_sha256=historical.sha(HERE/'decision_objectives.py'))
    (args.output/'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
    results=[]
    for kind in args.methods:
        net=copy.deepcopy(initial)
        optimizer=torch.optim.Adam(net.offsets,lr=.01)
        history=[]
        torch.cuda.synchronize(); start=time.monotonic()
        for step in range(args.steps):
            b,a,acts=train_batches[step%len(train_batches)]
            optimizer.zero_grad(set_to_none=True)
            z1,s1=net(b,1); z2,s2=net(b,2)
            if kind in ('local_t1','local_joint'):
                l1,l2=local_loss(s1,acts),local_loss(s2,acts)
            else:
                l1,l2=decision_loss(z1,a,kind),decision_loss(z2,a,kind)
            loss=l1+0*l2 if kind=='local_t1' else (l1+l2)/2
            loss.backward()
            grad=float(torch.sqrt(sum(o.grad.square().sum() for o in net.offsets)))
            if not np.isfinite(grad): raise RuntimeError('nonfinite gradient')
            optimizer.step()
            with torch.no_grad():
                for o in net.offsets:o.clamp_(-.5,.5)
            history.append(dict(step=step,loss=float(loss.detach()),T1=float(l1.detach()),T2=float(l2.detach()),gradient_norm=grad))
        torch.cuda.synchronize(); fit_seconds=time.monotonic()-start
        np.savez_compressed(args.output/f'{kind}_parameters.npz',**{f'offset_{i}':o.detach().cpu().numpy() for i,o in enumerate(net.offsets)})
        (args.output/f'{kind}_training.json').write_text(json.dumps(dict(seconds=fit_seconds,history=history),indent=2)+'\n')
        for T in (1,2,4,8):
            parts={k:[] for k in ('logits','teacher','labels','events','amplitude')}
            torch.cuda.synchronize(); evstart=time.monotonic()
            with torch.no_grad():
                for b in val_batches:
                    b=b.clone().cuda(); labels=b.y.flatten().cpu().numpy(); del b.y
                    a,_=core.teacher_sites(folded,b)
                    z,_,trace=measured_forward(net,b,T)
                    for key,value in dict(logits=z.cpu().numpy(),teacher=a.cpu().numpy(),labels=labels,
                        events=trace['events'].cpu().numpy(),amplitude=trace['amplitude'].cpu().numpy()).items(): parts[key].append(value)
            arrays={k:np.concatenate(v) for k,v in parts.items()}
            arrays['indices']=np.asarray(meta['val_indices'])
            np.savez_compressed(args.output/f'{kind}_T{T}.npz',**arrays)
            pred=arrays['logits'].argmax(1); ap=arrays['teacher'].argmax(1)
            row=dict(method=kind,T=T,accuracy_percent=float(100*np.mean(pred==arrays['labels'])),
                teacher_agreement_percent=float(100*np.mean(pred==ap)),events=int(arrays['events'].sum()),
                amplitude_sum=float(arrays['amplitude'].sum()),parameters=693,parameter_bytes=2772,
                calibration_seconds=fit_seconds,evaluation_instrumented_seconds=time.monotonic()-evstart)
            results.append(row); print(json.dumps(row),flush=True)
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'DONE.json').write_text(json.dumps(dict(seconds=time.monotonic()-campaign,endpoints=len(results)))+'\n')


if __name__=='__main__':main()
