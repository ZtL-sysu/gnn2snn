"""One fixed graph-wise budget of first-tick residual events."""
import math
import torch
from torch_geometric.nn import global_mean_pool
from objective import make_permutation,message_operator


def scores(model,batch,error,correction,policy):
    perm=make_permutation(batch) if policy=='shuffled' else None
    identity=policy=='identity'
    message=message_operator(model,batch,error,perm,identity)
    if perm is not None:message=message[perm]
    if identity:adjoint=message
    else:
        s,d=batch.edge_index;n=batch.num_nodes
        w=torch.bincount(s,minlength=n).clamp_min(1).to(error.dtype)[s].rsqrt()*torch.bincount(d,minlength=n).clamp_min(1).to(error.dtype)[d].rsqrt()
        adjoint=torch.zeros_like(message).index_add(0,s,message[d]*w[:,None])
    if perm is not None:adjoint=adjoint[torch.argsort(perm)]
    if identity:column_norm=torch.ones(batch.num_nodes,device=error.device)
    else:
        column_norm=torch.zeros(batch.num_nodes,device=error.device).index_add(0,s,w.square())
        if torch.unique(s*batch.num_nodes+d).numel()!=s.numel():raise ValueError('parallel edge correction required')
        if perm is not None:column_norm=column_norm[torch.argsort(perm)]
    transformed=torch.nn.functional.linear(correction,model.convs[1].lin.weight)
    return -2*(adjoint*transformed).sum(1)-column_norm*transformed.square().sum(1)


@torch.no_grad()
def select_correction(model,batch,current,spike,theta,policy):
    reference=current.relu();error=spike-reference
    levels=theta[None,:]/(4*2**torch.arange(8,device=theta.device))[:,None]
    levels=torch.cat([torch.zeros_like(levels[:1]),levels.flip(0)],0)
    residual=-error
    distance=(residual.abs()[:,None,:]-levels[None,:,:]).abs()
    choice=distance.argmin(1)
    candidate=levels.gather(0,choice)*residual.sign()
    local=-(2*error*candidate+candidate.square()).sum(1)
    den=global_mean_pool(reference.square().mean(1,keepdim=True),batch.batch).flatten().clamp_min(1e-6)
    if policy in ['message','identity','shuffled']:
        perm=make_permutation(batch) if policy=='shuffled' else None
        ref_message=message_operator(model,batch,reference,perm,policy=='identity')
        mden=global_mean_pool(ref_message.square().mean(1,keepdim=True),batch.batch).flatten().clamp_min(1e-6)
        score=.5*(local/den[batch.batch]+scores(model,batch,error,candidate,policy)/mden[batch.batch])
    else:score=local
    mask=torch.zeros(batch.num_nodes,dtype=torch.bool,device=current.device)
    # Fixed independent per-graph random stream; identical across evaluation repeats.
    gen=torch.Generator().manual_seed(20261117)
    selected=0
    for lo,hi in zip(batch.ptr[:-1].tolist(),batch.ptr[1:].tolist()):
        k=math.ceil((hi-lo)/4);selected+=k
        idx=torch.randperm(hi-lo,generator=gen)[:k].to(current.device) if policy=='random' else torch.argsort(score[lo:hi],descending=True,stable=True)[:k]
        mask[lo+idx]=True
    correction=candidate*mask[:,None]
    return correction,dict(selected_nodes=selected,max_opportunity_slots=selected*current.shape[1],
                          extra_nonzero_events=int((correction!=0).sum()),
                          extra_edge_deliveries=int(((correction!=0).sum(1)*torch.bincount(batch.edge_index[0],minlength=batch.num_nodes)).sum()),
                          first_residual_mse_before=float(error.square().mean()),
                          first_residual_mse_after=float((error+correction).square().mean()))
