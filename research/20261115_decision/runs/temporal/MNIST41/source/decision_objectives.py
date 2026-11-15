"""Ordinary teacher-assisted objectives: controls, not novel algorithms."""
import torch
import torch.nn.functional as F


def decision_loss(z, teacher, kind):
    scale=teacher.detach().std(dim=1,unbiased=False).clamp_min(1.)[:,None]
    if kind=='logits':
        return ((z-teacher)/scale).square().mean()
    if kind=='kl':
        return F.kl_div(F.log_softmax(z/scale,dim=1), F.softmax(teacher/scale,dim=1),reduction='batchmean')
    if kind=='ranking':
        classes=teacher.topk(2,dim=1).indices
        gap=z.gather(1,classes[:,:1])-z.gather(1,classes[:,1:])
        return F.softplus(1.-gap/scale).mean()
    raise ValueError(kind)
