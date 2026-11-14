import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'20261114_story'))
import torch
from test_calibration_core import TinyFolded,synthetic_batch
from calibration_core import teacher_sites
from codebook_core import CodebookGCN
from full_book import refine,hard_values

def test_hard_channel_values_exact():
    torch.manual_seed(15);b=synthetic_batch();m=TinyFolded().eval();th=[[.2]*8]*4+[[.2]*4,[.2]*2]
    book=torch.linspace(.7,.02,8)[:,None].expand(-1,8).clone()
    q=CodebookGCN(m,th,first_book=book)
    raw=torch.cat([b.x,b.pos],-1);s,d=b.edge_index;n=b.num_nodes
    ew=torch.bincount(s,minlength=n).clamp_min(1).float()[s].rsqrt()*torch.bincount(d,minlength=n).clamp_min(1).float()[d].rsqrt()
    current=m.convs[0](m.embedding(raw),b.edge_index,ew)
    expected=q(b,1)[1][0]
    for j in range(8):assert torch.equal(hard_values(current[:,j],book[:,j,None])[:,0],expected[:,j])

def test_full_message_refinement_monotone():
    torch.manual_seed(15);b=synthetic_batch();m=TinyFolded().eval();th=[[.2]*8]*4+[[.2]*4,[.2]*2]
    with torch.no_grad():a,acts=teacher_sites(m,b)
    q=CodebookGCN(m,th)
    book=torch.linspace(.7,.02,8)[:,None].expand(-1,8).clone()
    for mode in ['book_local','book_message','book_shuffled','book_identity']:
        result,info=refine(q,[(b,a,acts)],book,mode)
        assert result.shape==book.shape
        assert info['final_objective']<=info['initial_objective']+1e-5
