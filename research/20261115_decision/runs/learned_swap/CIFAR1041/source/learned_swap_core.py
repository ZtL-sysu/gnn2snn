"""Single-pass learned event reassignment: no teacher or downstream lookahead."""
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
from run_objectives import core
from intervention import replace_event
from event_swap import propose_swaps,apply_swaps


@torch.no_grad()
def observe_first(net,batch):
    original=core._mth_step;calls=0;captured={}
    def step(state,current,theta,t):
        nonlocal calls
        site=calls%6;calls+=1
        event=original(state,current,theta,t)
        if site==0 and t==1:
            captured['event']=event.cpu().numpy().copy()
            captured['membrane']=state['mem'].cpu().numpy().copy()
        return event
    with patch.object(core,'_mth_step',step):z,_=net(batch,1)
    return z,captured


@torch.no_grad()
def learned_forward(net,batch,T,model=None,mode='nongraph',graph_ids=None,cache=None,instrument=True):
    from swap_predictor import graph_cache,pair_features,ridge_predict,choose_pairs
    if graph_ids is None:raise ValueError('graph RNG nonces required')
    if len(graph_ids)!=batch.num_graphs:raise ValueError('graph nonce count mismatch')
    original=core._mth_step;calls=0
    degree=torch.bincount(batch.edge_index[0],minlength=batch.num_nodes)
    trace=dict(choice=np.zeros(batch.num_graphs,dtype=np.int64),
        selected_prediction=np.zeros((batch.num_graphs,8)),
        events=torch.zeros(batch.num_graphs,dtype=torch.int64,device=batch.x.device),
        amplitude=torch.zeros(batch.num_graphs,dtype=torch.float64,device=batch.x.device),
        deliveries=torch.zeros(batch.num_graphs,dtype=torch.int64,device=batch.x.device))
    if model is not None and cache is None:
        cache=graph_cache(batch.edge_index.cpu(),batch.ptr.cpu(),graph_ids,mode)
    def step(state,current,theta,t):
        nonlocal calls
        site=calls%6;calls+=1
        event=original(state,current,theta,t)
        if model is not None and site==0 and t==1:
            cpu_event=event.cpu()
            ptr=batch.ptr.cpu()
            proposals=propose_swaps(cpu_event,ptr,torch.as_tensor(cache['outdegree'],dtype=torch.long),graph_ids,count=8)
            array=np.asarray([[p if p is not None else (-1,-1,-1) for p in candidate] for candidate in proposals],dtype=np.int64)
            X,valid=pair_features(cpu_event.numpy(),state['mem'].cpu().numpy(),theta.cpu().numpy(),array,ptr.numpy(),cache,mode)
            pred=ridge_predict(model,X);choice=choose_pairs(pred,valid)
            pairs=[proposals[int(choice[g])-1][g] if choice[g] else None for g in range(batch.num_graphs)]
            after=apply_swaps(event,pairs,batch.ptr,degree)
            if instrument:assert torch.equal(event!=0,after!=0)
            replace_event(state,event,after,t);event=after
            trace['choice']=choice
            for g,c in enumerate(choice):
                if c:trace['selected_prediction'][g]=pred[int(c)-1,g]
        if instrument:
            group=batch.batch if site<4 else torch.arange(batch.num_graphs,device=event.device)
            counts=(event!=0).sum(1)
            trace['events'].index_add_(0,group,counts)
            trace['amplitude'].index_add_(0,group,event.abs().double().sum(1))
            if site<3:trace['deliveries'].index_add_(0,group,counts*degree*(3-site))
        return event
    with patch.object(core,'_mth_step',step):z,_=net(batch,T)
    assert calls==6*T
    return z,trace
