"""One bounded hard-codebook coordinate sweep including message cross terms."""
from types import SimpleNamespace
import torch
from objective import message_operator,make_permutation
from hard_search import aggregate,rank_one_delta


def hard_values(current,books):
    # books: descending levels x candidate count. Same midpoint comparison as core.
    smaller=torch.cat([books[1:],torch.zeros_like(books[:1])],0)
    positive=(current[:,None,None]>=(books+smaller)[None,:,:]*.5).to(current.dtype)
    transitions=positive.clone();transitions[:,1:]-=positive[:,:-1]
    return (transitions*books[None,:,:]).sum(1)


@torch.no_grad()
def refine(net,train_batches,initial,mode):
    currents=[];targets=[];wl=[];wm=[];edges=[];perms=[];shift=0
    identity=mode=='book_identity'
    for batch,_,acts in train_batches:
        value=acts[0];n,c=value.shape;s,d=batch.edge_index
        ew=torch.bincount(s,minlength=n).clamp_min(1).float()[s].rsqrt()*torch.bincount(d,minlength=n).clamp_min(1).float()[d].rsqrt()
        current=net.model.convs[0](net.model.embedding(torch.cat([batch.x.float(),batch.pos.float()],-1)),batch.edge_index,ew)
        currents.append(current);targets.append(value)
        perm=make_permutation(batch)
        reference=message_operator(net.model,batch,value,perm if mode=='book_shuffled' else None,identity)
        wl.append(torch.full((n,1),1/(len(train_batches)*n*c*float(value.square().mean().clamp_min(1e-6))),device=value.device))
        wm.append(torch.full((n,1),1/(len(train_batches)*n*c*float(reference.square().mean().clamp_min(1e-6))),device=value.device))
        edges.append(batch.edge_index+shift);perms.append(perm+shift);shift+=n
    current=torch.cat(currents);target=torch.cat(targets);wl=torch.cat(wl);wm=torch.cat(wm)
    batch=SimpleNamespace(edge_index=torch.cat(edges,1),num_nodes=shift)
    perm=torch.cat(perms) if mode=='book_shuffled' else None
    book=initial.clone();c=book.shape[1]
    quantized=torch.stack([hard_values(current[:,j],book[:,j,None])[:,0] for j in range(c)],1)
    error=quantized-target
    message=message_operator(net.model,batch,error,perm,identity)
    def objective(e,m):
        local=(wl*e.square()).sum()
        return float(local if mode=='book_local' else .5*(local+(wm*m.square()).sum()))
    initial_objective=objective(error,message);trace=[]
    # One sweep only, ascending channel/level order. Every arm receives same sweep.
    for j in range(c):
        column=net.model.convs[1].lin.weight[:,j]
        for k in range(8):
            assigned=quantized[:,j]==book[k,j]
            v=assigned.float()[:,None]
            av=aggregate(batch,v,perm,identity)
            num=(wl*error[:,j,None]*v).sum()
            den=(wl*v.square()).sum()
            if mode!='book_local':
                num+=(wm*av*(message@column)[:,None]).sum()
                den+=(wm*av.square()).sum()*column.square().sum()
            step=-num/den.clamp_min(1e-12)
            lo=book[k+1,j]+1e-7 if k<7 else torch.tensor(1e-8,device=book.device)
            hi=book[k-1,j]-1e-7 if k>0 else torch.maximum(current[:,j].max(),book[0,j])
            if bool(hi<lo):continue
            proposals=(book[k,j]+step*torch.tensor([1.,.5,.25],device=book.device)).clamp(lo,hi)
            choices=book[:,j,None].expand(-1,3).clone();choices[k]=proposals
            candidate=hard_values(current[:,j],choices)
            delta=candidate-quantized[:,j,None]
            local_delta=(wl*(2*error[:,j,None]*delta+delta.square())).sum(0)
            ad=aggregate(batch,delta,perm,identity)
            md=rank_one_delta(message,ad,column,wm)
            score=local_delta if mode=='book_local' else .5*(local_delta+md)
            best=int(score.argmin());accept=float(score[best]) < -1e-10
            if accept:
                book[k,j]=proposals[best]
                quantized[:,j]=candidate[:,best]
                error[:,j]=quantized[:,j]-target[:,j]
                message+=ad[:,best,None]*column
            trace.append(dict(channel=j,level=k,accepted=accept,objective_delta=float(score[best]) if accept else 0.))
    # Recompute complete operator after rank-one updates; do not trust cached score.
    final_message=message_operator(net.model,batch,error,perm,identity)
    final_objective=objective(error,final_message)
    if final_objective>initial_objective+1e-5:raise RuntimeError('full hard objective increased')
    return book,dict(initial_objective=initial_objective,final_objective=final_objective,
                    coordinate_budget=8*c,candidates_per_coordinate=3,trace=trace,
                    accepted=sum(r['accepted'] for r in trace),optimizer='one hard coordinate sweep with incumbent retention')
