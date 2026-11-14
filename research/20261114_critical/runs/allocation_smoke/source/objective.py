"""Critical branch reconstruction and exact next-preactivation perturbation."""
import torch
import torch.nn.functional as F


def make_permutation(batch,seed=20261116):
    # Same batch/graph order yields the same within-graph conjugation for all arms.
    gen=torch.Generator(device='cpu').manual_seed(seed)
    p=torch.arange(batch.num_nodes)
    for lo,hi in zip(batch.ptr[:-1].tolist(),batch.ptr[1:].tolist()):
        q=torch.randperm(hi-lo,generator=gen)
        if hi-lo>1 and torch.equal(q,torch.arange(hi-lo)):
            q=q.roll(1)
        p[lo:hi]=lo+q
    return p.to(batch.x.device)


def message_operator(model,batch,error,permutation=None,identity=False):
    if permutation is not None:
        error=error[permutation]
    transformed=F.linear(error,model.convs[1].lin.weight,None)
    if identity:
        result=transformed
    else:
        source,target=batch.edge_index
        n=batch.num_nodes
        out=torch.bincount(source,minlength=n).clamp_min(1).to(error.dtype)
        inc=torch.bincount(target,minlength=n).clamp_min(1).to(error.dtype)
        weights=out[source].rsqrt()*inc[target].rsqrt()
        result=torch.zeros_like(transformed).index_add(0,target,transformed[source]*weights[:,None])
    if permutation is not None:
        result=result[torch.argsort(permutation)]
    return result


def loss_terms(net,batch,sites,acts,mode,permutation):
    errors=[s-a for s,a in zip(sites,acts)]
    locals_=[e.square().mean()/a.square().mean().clamp_min(1e-6) for e,a in zip(errors,acts)]
    local=sum(locals_)/6 if mode.startswith('all_') else locals_[0]
    perm=permutation if mode=='first_shuffled' else None
    identity=mode=='first_identity'
    msg=message_operator(net.model,batch,errors[0],perm,identity)
    with torch.no_grad():
        reference=message_operator(net.model,batch,acts[0],perm,identity)
    message=msg.square().mean()/reference.square().mean().clamp_min(1e-6)
    total=local if mode in ['first_local','all_local'] else .5*(local+message)
    return dict(local=local,message=message,total=total)
