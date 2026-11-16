"""Training-only exact local temporal-prefix codeword diagnostic."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_objectives import historical,core,CRITICAL
from intervention import replace_event
from prefix_codec import prefix_alternative


def half_mask(shape,ptr,graph_ids,device):
    mask=np.zeros(shape,dtype=bool)
    for g,(lo,hi) in enumerate(zip(ptr[:-1].cpu().tolist(),ptr[1:].cpu().tolist())):
        seed=int.from_bytes(hashlib.sha256(f'20261119:{graph_ids[g]}'.encode()).digest()[:16],'little')
        mask[lo:hi]=np.random.default_rng(seed).random((hi-lo,shape[1]))<.5
    return torch.as_tensor(mask,device=device)


@torch.no_grad()
def prefix_forward(net,batch,T,policy,graph_ids):
    if policy not in ('baseline','all_eligible','deterministic_half'):raise ValueError(policy)
    original=core._mth_step;calls=0
    n=batch.num_graphs;device=batch.x.device
    degree=torch.bincount(batch.edge_index[0],minlength=batch.num_nodes)
    trace={key:torch.zeros(n,dtype=torch.int64,device=device) for key in ('events','deliveries','first_events')}
    trace['amplitude']=torch.zeros(n,dtype=torch.float64,device=device)
    trace['first_amplitude']=torch.zeros(n,dtype=torch.float64,device=device)
    trace['second_input_verified_zero']=False
    def graph_sum(matrix,dtype=torch.int64):
        value=torch.zeros((n,matrix.shape[1]),dtype=dtype,device=device)
        value.index_add_(0,batch.batch,matrix.to(dtype));return value
    def step(state,current,theta,t):
        nonlocal calls
        site=calls%6;calls+=1
        event=original(state,current,theta,t)
        if site==0 and t==1:
            alternative,eligible,diag=prefix_alternative(state,event,theta)
            selected=eligible if policy=='all_eligible' else (eligible & half_mask(event.shape,batch.ptr,graph_ids,device) if policy=='deterministic_half' else torch.zeros_like(eligible))
            changed=torch.where(selected,alternative,event)
            trace['eligible_by_channel']=graph_sum(eligible)
            trace['occupied_by_channel']=graph_sum(event!=0)
            trace['selected_by_channel']=graph_sum(selected)
            trace['baseline_first_T2_decode']=event+diag['base_q2']/2
            trace['expected_q2']=torch.where(selected,diag['alternative_q2'],diag['base_q2'])
            replace_event(state,event,changed,1);event=changed
        if site==0 and t==2:
            assert torch.count_nonzero(current)==0,'first-site tick2 current assumption failed'
            trace['second_input_verified_zero']=True
            assert torch.equal(event,trace['expected_q2']),'actual source tick2 differs from local probe'
        group=batch.batch if site<4 else torch.arange(n,device=device)
        counts=(event!=0).sum(1)
        trace['events'].index_add_(0,group,counts)
        trace['amplitude'].index_add_(0,group,event.abs().double().sum(1))
        if site<3:trace['deliveries'].index_add_(0,group,counts*degree*(3-site))
        if site==0:
            trace['first_events'].index_add_(0,group,counts)
            trace['first_amplitude'].index_add_(0,group,event.abs().double().sum(1))
        return event
    with patch.object(core,'_mth_step',step):z,sites=net(batch,T)
    assert calls==6*T
    if T>=2:
        assert trace['second_input_verified_zero']
        assert torch.equal(sites[0],trace['baseline_first_T2_decode']),'first-site T2 decode not bitwise preserved'
    trace['first_decode_error_max']=torch.zeros(n,device=device)
    if T>=2:
        error=(sites[0]-trace['baseline_first_T2_decode']).abs().max(1).values
        trace['first_decode_error_max']=torch.stack([error[int(lo):int(hi)].max() for lo,hi in zip(batch.ptr[:-1],batch.ptr[1:])])
    return z,trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['MNIST','CIFAR10'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(20261119)
    start=time.monotonic()
    meta,_,_,folded,loaded,_,data_root=historical.load_teacher(args.dataset,41,historical.DEFAULT_ASSETS,
        CRITICAL/'runs/allocation'/f'{args.dataset}41',torch.device('cuda'))
    net=core.CalibratedGCN(folded,loaded.thresholds).cuda().eval().requires_grad_(False)
    paramfile=HERE/'runs/objectives'/f'{args.dataset}41'/'local_t1_parameters.npz'
    with np.load(paramfile) as params:
        for i,o in enumerate(net.offsets):o.data.copy_(torch.as_tensor(params[f'offset_{i}'],device=o.device))
    ids=meta['train_indices'][:128]
    train=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='train')
    batches=list(DataLoader(train[ids],batch_size=32,shuffle=False))
    results=[]
    for policy in ('baseline','all_eligible','deterministic_half'):
        previous_selected=None
        for T in (1,2):
            offset=0;parts={key:[] for key in ('logits','teacher','labels','events','deliveries','amplitude',
                'first_events','first_amplitude','eligible_by_channel','occupied_by_channel','selected_by_channel','first_decode_error_max')}
            for raw in batches:
                if time.monotonic()-start>300:raise TimeoutError('five-minute diagnostic cap; no restart')
                batch=raw.clone().cuda();labels=batch.y.flatten().cpu().numpy();del batch.y
                with torch.no_grad():teacher,_=core.teacher_sites(folded,batch)
                z,trace=prefix_forward(net,batch,T,policy,ids[offset:offset+batch.num_graphs])
                if policy=='baseline':
                    with torch.no_grad():assert torch.equal(z,net(batch,T)[0])
                values=dict(logits=z.cpu().numpy(),teacher=teacher.cpu().numpy(),labels=labels,
                    **{key:trace[key].cpu().numpy() for key in parts if key not in ('logits','teacher','labels')})
                for key,value in values.items():parts[key].append(value)
                offset+=batch.num_graphs
            a={key:np.concatenate(value) for key,value in parts.items()};a['indices']=np.asarray(ids)
            if T==1:previous_selected=a['selected_by_channel']
            else:assert np.array_equal(previous_selected,a['selected_by_channel'])
            np.savez_compressed(args.output/f'{policy}_T{T}.npz',**a)
            row=dict(policy=policy,T=T,accuracy_percent=float(100*np.mean(a['logits'].argmax(1)==a['labels'])),
                teacher_agreement_percent=float(100*np.mean(a['logits'].argmax(1)==a['teacher'].argmax(1))),
                events=int(a['events'].sum(dtype=np.int64)),amplitude_sum=float(a['amplitude'].sum()),
                first_events=int(a['first_events'].sum(dtype=np.int64)),first_amplitude_sum=float(a['first_amplitude'].sum()),
                eligible=int(a['eligible_by_channel'].sum()),occupied_T1=int(a['occupied_by_channel'].sum()),
                selected=int(a['selected_by_channel'].sum()),first_decode_max_abs=float(a['first_decode_error_max'].max()))
            results.append(row);print(json.dumps(row),flush=True)
    source=args.output/'source';source.mkdir()
    for path in (Path(__file__),HERE/'prefix_codec.py',HERE/'intervention.py'):shutil.copyfile(path,source/path.name)
    record=dict(dataset=args.dataset,teacher_seed=41,training_indices=ids,training_only=True,validation_loaded=False,test_loaded=False,
        optimizer_updates=0,baseline_parameters_sha256=historical.sha(paramfile),checkpoint_sha256=meta['checkpoint_sha256'],
        calibration_sha256=meta['calibration_sha256'],partial_mask_seed=20261119,
        source_sha256={p.name:historical.sha(p) for p in source.iterdir()},
        local_zero_input_probes_per_forward=2,online_teacher=False,
        qualification='local temporal-prefix equivalence diagnostic; no whole-network guarantee or method novelty claim')
    for name,value in [('metadata.json',record),('results.json',results),('DONE.json',dict(endpoints=len(results),seconds=time.monotonic()-start))]:
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
