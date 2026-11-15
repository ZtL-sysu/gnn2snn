import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from decision_objectives import decision_loss


def test_logits_and_kl_zero_at_teacher_and_finite_gradient():
    a=torch.tensor([[3.,1.,-1.],[0.,1.,2.]])
    for name in ('logits','kl'):
        z=a.clone().requires_grad_()
        loss=decision_loss(z,a,name)
        assert abs(float(loss))<1e-6
        loss.backward()
        assert torch.isfinite(z.grad).all()


def test_ranking_penalizes_reversing_teacher_order():
    a=torch.tensor([[3.,1.,-1.]])
    good=decision_loss(a,a,'ranking')
    bad=decision_loss(a.flip(1),a,'ranking')
    assert good<bad
