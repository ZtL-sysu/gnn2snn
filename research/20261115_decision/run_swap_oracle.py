"""Training-only bounded capacity oracle for cost-preserving event reassignment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
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
from torch_geometric.nn import global_mean_pool

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_objectives import historical,core,CRITICAL
from run_interventions import margin
from intervention import replace_event
from objective import message_operator,make_permutation


@torch.no_grad()
def swap_forward(net,batch,T,pairs=None):
    from event_swap import apply_swaps
    original=core._mth_step
    ngraph=batch.num_graphs
    degree=torch.bincount(batch.edge_index[0],minlength=batch.num_nodes)
    trace=dict(call=0,events=torch.zeros(ngraph,dtype=torch.int64,device=batch.x.device),
        amplitude=torch.zeros(ngraph,dtype=torch.float64,device=batch.x.device),
        deliveries=torch.zeros(ngraph,dtype=torch.int64,device=batch.x.device))
    def step(state,current,theta,t):
        site=trace['call']%6;trace['call']+=1
        event=original(state,current,theta,t)
        if site==0 and t==1:
            trace['before']=event.clone();trace['current']=current.clone()
            if pairs is not None:
                after=apply_swaps(event,pairs,batch.ptr,degree)
                replace_event(state,event,after,t);event=after
            trace['first']=event.clone()
            assert torch.equal(event!=0,trace['before']!=0)
            for lo,hi in zip(batch.ptr[:-1].tolist(),batch.ptr[1:].tolist()):
                assert torch.equal(event[lo:hi].sort(dim=0).values,trace['before'][lo:hi].sort(dim=0).values)
        group=batch.batch if site<4 else torch.arange(ngraph,device=event.device)
        counts=(event!=0).sum(1)
        trace['events'].index_add_(0,group,counts)
        trace['amplitude'].index_add_(0,group,event.abs().double().sum(1))
        if site<3:trace['deliveries'].index_add_(0,group,counts*degree*(3-site))
        return event
    with patch.object(core,'_mth_step',step):z,sites=net(batch,T)
    error=trace['first']-trace['current'].relu()
    trace['local_mse']=global_mean_pool(error.square().mean(1,keepdim=True),batch.batch).flatten()
    for key,perm in [('message_mse',None),('shuffled_mse',make_permutation(batch))]:
        message=message_operator(net.model,batch,error,perm)
        trace[key]=global_mean_pool(message.square().mean(1,keepdim=True),batch.batch).flatten()
    return z,trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['MNIST','CIFAR10'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(20261117)
    start=time.monotonic()
    from event_swap import propose_swaps
    meta,_,_,folded,loaded,_,data_root=historical.load_teacher(args.dataset,41,historical.DEFAULT_ASSETS,
        CRITICAL/'runs/allocation'/f'{args.dataset}41',torch.device('cuda'))
    net=core.CalibratedGCN(folded,loaded.thresholds).cuda().eval().requires_grad_(False)
    paramfile=HERE/'runs/objectives'/f'{args.dataset}41'/'local_t1_parameters.npz'
    saved=np.load(paramfile)
    with torch.no_grad():
        for i,o in enumerate(net.offsets):o.copy_(torch.as_tensor(saved[f'offset_{i}'],device=o.device))
    train=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='train')
    ids=meta['train_indices'][:128]
    columns={key:[] for key in ('logits_T1','logits_T2','events_T1','events_T2','amplitude_T1','amplitude_T2',
        'deliveries_T1','deliveries_T2','local_mse','message_mse','shuffled_mse','pairs','teacher','labels')}
    offset=0
    for batch in DataLoader(train[ids],batch_size=32,shuffle=False):
        if time.monotonic()-start>300:
            (args.output/'TIMEOUT.json').write_text(json.dumps({'status':'five-minute cap reached; no restart'})+'\n')
            raise TimeoutError('fixed five-minute diagnostic cap reached')
        batch=batch.cuda();labels=batch.y.flatten().cpu().numpy();del batch.y
        with torch.no_grad():teacher,_=core.teacher_sites(folded,batch)
        z1,tr1=swap_forward(net,batch,1);z2,tr2=swap_forward(net,batch,2)
        with torch.no_grad():
            assert torch.equal(z1,net(batch,1)[0]) and torch.equal(z2,net(batch,2)[0])
        outdegree=torch.bincount(batch.edge_index[0],minlength=batch.num_nodes)
        choices=propose_swaps(tr1['first'].cpu(),batch.ptr.cpu(),outdegree.cpu(),ids[offset:offset+batch.num_graphs],count=8)
        traces={1:[(z1,tr1)],2:[(z2,tr2)]}
        coords=[np.full((batch.num_graphs,3),-1,dtype=np.int64)]
        for pairs in choices:
            coords.append(np.asarray([pair if pair is not None else (-1,-1,-1) for pair in pairs],dtype=np.int64))
            for T in (1,2):traces[T].append(swap_forward(net,batch,T,pairs))
        for T in (1,2):
            columns[f'logits_T{T}'].append(torch.stack([z for z,_ in traces[T]]).cpu().numpy())
            for key in ('events','amplitude','deliveries'):
                columns[f'{key}_T{T}'].append(torch.stack([trace[key] for _,trace in traces[T]]).cpu().numpy())
        for key in ('local_mse','message_mse','shuffled_mse'):
            columns[key].append(torch.stack([trace[key] for _,trace in traces[1]]).cpu().numpy())
        # Save graph-relative node coordinates so a different batch regroup can replay them.
        coordinate=np.stack(coords)
        for g,lo in enumerate(batch.ptr[:-1].cpu().tolist()):
            valid=coordinate[:,g,0]>=0;coordinate[valid,g,0]-=lo;coordinate[valid,g,1]-=lo
        columns['pairs'].append(coordinate)
        columns['teacher'].append(teacher.cpu().numpy());columns['labels'].append(labels)
        offset+=batch.num_graphs
    arrays={k:np.concatenate(v,axis=0 if k in ('teacher','labels') else 1) for k,v in columns.items()}
    arrays['indices']=np.asarray(ids)
    np.savez_compressed(args.output/'candidates.npz',**arrays)
    source=args.output/'source';source.mkdir()
    for file in (Path(__file__),HERE/'event_swap.py',HERE/'intervention.py'):
        shutil.copyfile(file,source/file.name)
    record=dict(dataset=args.dataset,seed=41,training_indices=ids,training_only=True,validation_loaded=False,test_loaded=False,
        baseline_parameters_sha256=historical.sha(paramfile),checkpoint_sha256=meta['checkpoint_sha256'],
        calibration_sha256=meta['calibration_sha256'],candidate_swaps=8,batch_size=32,optimizer_updates=0,
        teacher_access='offline oracle/diagnostic for every selector; not deployable',labels_used_for_selection=False,
        first_site_support_and_channel_multiset_preserved=True,seconds=time.monotonic()-start,
        source_sha256={p.name:historical.sha(p) for p in source.iterdir()},
        candidate_npz_sha256=historical.sha(args.output/'candidates.npz'),
        graph_delivery_metric='residual-branch proxy: source event outdegree * remaining GCN layers; excludes raw embedding/controller/readout')
    (args.output/'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(dataset=args.dataset,graphs=len(ids),candidates_per_graph=9,seconds=record['seconds'])))


if __name__=='__main__':main()
