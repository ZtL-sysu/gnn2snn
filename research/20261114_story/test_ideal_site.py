"""CPU-only tests for the ideal-site oracle hook; never loads real datasets."""
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research'/'visual-graphs'/'takeover_20261112'))
sys.path.insert(0,str(HERE))
from test_calibration_core import TinyFolded, synthetic_batch
import calibration_core as core
from calibration_core import CalibratedGCN, teacher_sites
from ideal_site_diagnostic import idealized_site_mask, forward_with_mask


def make_core():
    torch.manual_seed(20261114)
    model=TinyFolded().eval()
    theta=[[.23]*8 for _ in range(4)]+[[.25]*4,[.27]*2]
    net=CalibratedGCN(model,theta).eval()
    return model,net


def test_empty_mask_is_bitwise_baseline_and_restores_hook():
    model,net=make_core(); batch=synthetic_batch(); original=core._mth_step
    with torch.no_grad(): expected,_=net(batch,2)
    with idealized_site_mask(()) as stats:
        with torch.no_grad(): actual,_=net(batch,2)
    assert core._mth_step is original
    assert torch.equal(actual,expected)
    assert stats['calls']==12 and stats['site_calls']==[2]*6 and stats['selected_calls']==0


def test_all_six_ideal_sites_recover_teacher_at_all_t():
    model,net=make_core(); batch=synthetic_batch()
    with torch.no_grad(): teacher,_=teacher_sites(model,batch)
    for T in (1,2,4,8):
        actual,_,stats=forward_with_mask(net,batch,T,range(6))
        assert torch.allclose(actual,teacher,rtol=1e-5,atol=1e-5),(T,float((actual-teacher).abs().max()))
        assert stats['calls']==6*T and stats['selected_calls']==6*T


def test_label_independence_and_exception_restoration():
    model,net=make_core(); batch=synthetic_batch(); original=core._mth_step
    stripped=batch.clone(); del stripped.y
    with idealized_site_mask((0,4)):
        with torch.no_grad(): a,_=net(batch,2)
    with idealized_site_mask((0,4)):
        with torch.no_grad(): b,_=net(stripped,2)
    assert torch.equal(a,b)
    try:
        with idealized_site_mask((1,)):
            raise RuntimeError('injected failure')
    except RuntimeError:
        pass
    assert core._mth_step is original


def test_selected_site_uses_causal_relu_difference():
    _,net=make_core(); batch=synthetic_batch()
    with idealized_site_mask((0,)) as stats:
        with torch.no_grad(): net(batch,1)
    assert stats['selected_calls']==1
    assert stats['site_calls']==[1]*6
    assert core._mth_step.__name__=='_mth_step'

if __name__=='__main__':
    import pytest
    raise SystemExit(pytest.main([__file__,'-q']))
