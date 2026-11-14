"""Fixed alphabet-size weighted Lloyd diagnostic; no novelty claim."""
import torch
from objective import message_operator,make_permutation


@torch.no_grad()
def lloyd(values,weights,initial,iterations=16):
    book=initial.sort(dim=0).values
    history=[]
    for iteration in range(iterations+1):
        levels=torch.cat([torch.zeros_like(book[:1]),book],0)
        distance=(values[:,None,:]-levels[None,:,:]).square()
        labels=distance.argmin(1)
        loss=(distance.gather(1,labels[:,None,:]).squeeze(1)*weights).sum()/weights.sum()
        history.append(float(loss))
        if iteration==iterations:break
        updated=book.clone()
        for k in range(1,9):
            mask=labels==k
            denom=(weights*mask).sum(0)
            mean=(weights*mask*values).sum(0)/denom.clamp_min(1e-12)
            updated[k-1]=torch.where(denom>0,mean,book[k-1])
        book=updated.clamp_min(1e-8).sort(dim=0).values
    if any(y>x+1e-5 for x,y in zip(history,history[1:])):raise RuntimeError('Lloyd loss increased')
    return book.flip(0),history


@torch.no_grad()
def fit_codebook(net,train_batches,mode):
    if mode not in ['book_local','book_message','book_shuffled','book_identity']:
        raise ValueError(mode)
    values=[];weights=[]
    for batch,_,acts in train_batches:
        value=acts[0];n,c=value.shape
        local_den=value.square().mean().clamp_min(1e-6)
        weight=torch.ones_like(value)/local_den
        if mode!='book_local':
            identity=mode=='book_identity'
            p=make_permutation(batch) if mode=='book_shuffled' else None
            ref=message_operator(net.model,batch,value,p,identity)
            denom=ref.square().mean().clamp_min(1e-6)
            if identity:degree_sensitivity=torch.ones(n,device=value.device)
            else:
                s,d=batch.edge_index
                ew=torch.bincount(s,minlength=n).clamp_min(1).float()[s].rsqrt()*torch.bincount(d,minlength=n).clamp_min(1).float()[d].rsqrt()
                # Duplicated edges would require summing weights before squaring.
                if torch.unique(s*n+d).numel()!=s.numel():raise ValueError('diagonal proxy assumes no parallel edges')
                degree_sensitivity=torch.zeros(n,device=value.device).index_add(0,s,ew.square())
                if p is not None:degree_sensitivity=degree_sensitivity[torch.argsort(p)]
            channel_sensitivity=net.model.convs[1].lin.weight.square().sum(0)
            weight=.5*(weight+degree_sensitivity[:,None]*channel_sensitivity[None,:]/denom)
        values.append(value)
        weights.append(weight/(len(train_batches)*n*c))
    initial=net.thresholds[0][None,:]/(2**torch.arange(8,device=value.device))[:,None]
    book,history=lloyd(torch.cat(values),torch.cat(weights),initial,iterations=16)
    return book,dict(optimizer='weighted_Lloyd',iterations=16,objective_history=history,
                     levels_per_channel=8,learned_scalars=int(book.numel()),
                     qualification='fixed alphabet size, higher storage than membrane offsets; message weight is diagonal proxy only, omits cross terms')
