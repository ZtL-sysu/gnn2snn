import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'20261114_story'))
from test_calibration_core import TinyFolded, synthetic_batch
from calibration_core import CalibratedGCN,teacher_sites
from objective import message_operator,make_permutation,loss_terms

def test_actual_preactivation_perturbation():
    torch.manual_seed(9)
    model=TinyFolded().eval(); b=synthetic_batch()
    h=torch.randn(b.num_nodes,8); e=torch.randn_like(h)
    s,d=b.edge_index
    w=torch.bincount(s,minlength=b.num_nodes).float().clamp_min(1)[s].rsqrt()*torch.bincount(d,minlength=b.num_nodes).float().clamp_min(1)[d].rsqrt()
    actual=model.convs[1](h+e,b.edge_index,w)-model.convs[1](h,b.edge_index,w)
    assert torch.allclose(actual,message_operator(model,b,e),atol=1e-6)

def test_structure_control_and_gradient():
    torch.manual_seed(9); b=synthetic_batch(); model=TinyFolded().eval()
    p=make_permutation(b)
    assert torch.equal(b.batch,b.batch[p])
    assert not torch.equal(p,torch.arange(b.num_nodes))
    e=torch.randn(b.num_nodes,8,requires_grad=True)
    # Conjugation preserves the graph operator's singular values and changes alignment.
    actual=message_operator(model,b,e,permutation=p)
    expected=message_operator(model,b,e[p])[torch.argsort(p)]
    assert torch.equal(actual,expected)
    actual.square().mean().backward()
    assert torch.isfinite(e.grad).all() and e.grad.abs().sum()>0

def test_loss_forward_and_frozen_teacher():
    torch.manual_seed(9); b=synthetic_batch(); model=TinyFolded().eval()
    net=CalibratedGCN(model,[[.23]*8]*4+[[.25]*4,[.27]*2])
    with torch.no_grad(): _,acts=teacher_sites(model,b)
    z,sites=net(b,1); before=z.detach().clone()
    for mode in ['first_local','first_message','first_shuffled','first_identity','all_local','all_message']:
        parts=loss_terms(net,b,sites,acts,mode,make_permutation(b))
        assert torch.isfinite(parts['total'])
    loss_terms(net,b,sites,acts,'first_message',make_permutation(b))['total'].backward()
    assert net.offsets[0].grad is not None
    assert torch.equal(before,net(b,1)[0])
    assert all(p.grad is None for p in model.parameters())
