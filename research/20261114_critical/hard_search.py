"""Bounded T1 coordinate diagnostic, not a budget-matched Adam replacement."""
from types import SimpleNamespace
import torch
from objective import message_operator,make_permutation


def candidate_events(current,theta,grid):
    # Use the identical source arithmetic, vectorizing candidate offsets as channels.
    from calibration_core import _new_site_state,_mth_step
    thresholds=theta.expand(len(grid))
    state=_new_site_state(thresholds,theta*grid,(len(current),len(grid)),current.device,current.dtype)
    return _mth_step(state,current[:,None].expand(-1,len(grid)),thresholds,1)


def rank_one_delta(message,changes,column,weights):
    return 2*(changes*(weights*(message@column)[:,None])).sum(0)+(weights*changes.square()).sum(0)*column.square().sum()


def aggregate(batch,x,permutation=None,identity=False):
    if permutation is not None:x=x[permutation]
    if identity:y=x
    else:
        s,d=batch.edge_index;n=batch.num_nodes
        w=torch.bincount(s,minlength=n).clamp_min(1).to(x.dtype)[s].rsqrt()*torch.bincount(d,minlength=n).clamp_min(1).to(x.dtype)[d].rsqrt()
        y=torch.zeros_like(x).index_add(0,d,x[s]*w[:,None])
    return y if permutation is None else y[torch.argsort(permutation)]


@torch.no_grad()
def fit(net,train_batches,mode):
    if not mode.startswith('first_'):raise ValueError('T1 first-site diagnostic only')
    currents=[];targets=[];errors=[];wl=[];wm=[];edges=[];pointers=[0];perms=[];shift=0
    for b,_,acts in train_batches:
        raw=torch.cat([b.x.float(),b.pos.float()],-1)
        s,d=b.edge_index;n=b.num_nodes
        w=torch.bincount(s,minlength=n).clamp_min(1).float()[s].rsqrt()*torch.bincount(d,minlength=n).clamp_min(1).float()[d].rsqrt()
        currents.append(net.model.convs[0](net.model.embedding(raw),b.edge_index,w))
        targets.append(acts[0]);errors.append(net(b,1)[1][0]-acts[0])
        perm=make_permutation(b)
        ref=message_operator(net.model,b,acts[0],perm if mode=='first_shuffled' else None,mode=='first_identity')
        wl.append(torch.full((n,1),1/(len(train_batches)*n*acts[0].shape[1]*float(acts[0].square().mean().clamp_min(1e-6))),device=raw.device))
        wm.append(torch.full((n,1),1/(len(train_batches)*n*ref.shape[1]*float(ref.square().mean().clamp_min(1e-6))),device=raw.device))
        edges.append(b.edge_index+shift);perms.append(perm+shift)
        pointers.extend((b.ptr[1:]+shift).tolist());shift+=n
    current=torch.cat(currents);target=torch.cat(targets);error=torch.cat(errors)
    wl=torch.cat(wl);wm=torch.cat(wm)
    b=SimpleNamespace(edge_index=torch.cat(edges,1),num_nodes=shift)
    perm=torch.cat(perms) if mode=='first_shuffled' else None
    identity=mode=='first_identity'
    projected=message_operator(net.model,b,error,perm,identity)
    grid=torch.linspace(-.5,.5,33,device=current.device)
    selected=[];trace=[]
    for j in range(current.shape[1]):
        candidates=candidate_events(current[:,j],net.thresholds[0][j],grid)-target[:,j,None]
        delta=candidates-error[:,j,None]
        local_delta=(wl*(2*error[:,j,None]*delta+delta.square())).sum(0)
        aggregate_delta=aggregate(b,delta,perm,identity)
        message_delta=rank_one_delta(projected,aggregate_delta,net.model.convs[1].lin.weight[:,j],wm)
        score=local_delta if mode=='first_local' else .5*(local_delta+message_delta)
        # Include the incumbent even if its offset is outside the fixed grid.
        best=int(score.argmin());accepted=float(score[best]) < -1e-10
        if accepted:
            net.offsets[0][j]=grid[best]
            error[:,j]=candidates[:,best]
            projected+=aggregate_delta[:,best,None]*net.model.convs[1].lin.weight[:,j]
        selected.append(accepted)
        trace.append(dict(channel=j,accepted=accepted,objective_delta=float(score[best]) if accepted else 0.))
    return dict(trace=trace,grid_points=33,coordinate_updates=current.shape[1],accepted_updates=sum(selected),
                objective_candidate_evaluations=33*current.shape[1],qualification='diagnostic; exceeds 64 Adam update budget; no validation selection')
