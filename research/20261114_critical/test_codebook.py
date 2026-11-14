import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'20261114_story'))
import torch
from test_calibration_core import TinyFolded,synthetic_batch
from calibration_core import CalibratedGCN
from codebook_core import CodebookGCN
from codebook_fit import lloyd

def test_original_baseline_bitwise():
    torch.manual_seed(13);b=synthetic_batch();m=TinyFolded().eval();th=[[.2]*8]*4+[[.2]*4,[.2]*2]
    a=CalibratedGCN(m,th);q=CodebookGCN(m,th)
    for t in [1,2,4,8]:assert torch.equal(a(b,t)[0],q(b,t)[0])

def test_eight_level_nearest_first_site():
    torch.manual_seed(13);b=synthetic_batch();m=TinyFolded().eval();th=[[.2]*8]*4+[[.2]*4,[.2]*2]
    book=torch.tensor([.8,.7,.6,.5,.4,.3,.2,.1])[:,None].expand(-1,8)
    q=CodebookGCN(m,th,first_book=book)
    raw=torch.cat([b.x,b.pos],-1);s,d=b.edge_index;n=b.num_nodes
    w=torch.bincount(s,minlength=n).float().clamp_min(1)[s].rsqrt()*torch.bincount(d,minlength=n).float().clamp_min(1)[d].rsqrt()
    current=m.convs[0](m.embedding(raw),b.edge_index,w)
    levels=torch.cat([torch.zeros(1,8),book.flip(0)],0)
    nearest=(current[:,None,:]-levels[None,:,:]).abs().argmin(1)
    expected=levels.gather(0,nearest)
    assert torch.equal(q(b,1)[1][0],expected)

def test_weighted_lloyd_monotone_fixed_cardinality():
    torch.manual_seed(13);x=torch.rand(200,8)*2;weights=torch.rand_like(x)+.1
    book,history=lloyd(x,weights,torch.linspace(.1,1.8,8)[:,None].expand(-1,8).clone(),iterations=16)
    assert book.shape==(8,8) and (book>=0).all()
    assert all(y<=x+1e-6 for x,y in zip(history,history[1:]))
