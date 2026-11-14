import torch
from hard_search import candidate_events,rank_one_delta
from calibration_core import _new_site_state,_mth_step

def test_candidates_exact_source_arithmetic():
    torch.manual_seed(2); current=torch.randn(50); grid=torch.linspace(-.5,.5,33); theta=torch.tensor(.7)
    actual=candidate_events(current,theta,grid)
    for j,o in enumerate(grid):
        state=_new_site_state(theta[None],(theta*o)[None],(50,1),'cpu',torch.float32)
        expected=_mth_step(state,current[:,None],theta[None],1)[:,0]
        assert torch.equal(actual[:,j],expected)

def test_rank_one_exact_objective_change():
    torch.manual_seed(2)
    message=torch.randn(13,7); changes=torch.randn(13,5); w=torch.randn(7)
    weights=torch.rand(13,1)
    actual=rank_one_delta(message,changes,w,weights)
    expected=torch.stack([(((message+changes[:,j,None]*w).square()-message.square())*weights).sum() for j in range(5)])
    assert torch.allclose(actual,expected,rtol=1e-5,atol=1e-4)
