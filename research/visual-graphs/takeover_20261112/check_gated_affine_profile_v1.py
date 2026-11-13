"""CPU synthetic profile acceptance, no official dataset loaded."""
import copy
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gated_teacher_model import GatedTeacher
import gated_affine_profile_v1 as P


def reject(fn):
    try: fn()
    except (ValueError,RuntimeError,FileExistsError): return
    raise AssertionError('expected rejection')


def main():
    torch.manual_seed(1413);torch.set_num_threads(2)
    model=GatedTeacher(hidden_dim=8,out_dim=8).eval()
    data=[Data(x=torch.randn(3,3),pos=torch.randn(3,2),
               edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
               edge_attr=torch.rand(4,1),y=torch.tensor([1])) for _ in range(3)]
    clean=P.C.LabelFreeGraphView(data)
    relu,line,_=P.A.require_author_sources()
    for mode in P.A.MODES:
        thresholds=[torch.ones(8)*2 for _ in P.A.site_specs(mode,model)]
        loader=DataLoader(clean,batch_size=2,shuffle=False)
        result=P.profile_loader(model,loader,thresholds,mode,relu,line,'cpu',expected=3,width=8,batch_size=2)
        units=8 if mode=='generic' else 6
        assert result['actual_source_warmup_calls']==units*2
        assert result['actual_active_calls']==units*4*2
        assert result['diagnostic_head_calls']==6 and result['normal_endpoint_head_calls']==2
        # Independently accumulate separate endpoint forwards and NumPy metrics.
        for t in (1,2,4):
            teacher_arrays=[];pred_arrays=[]
            counts={s:P.empty_stats() for s,_,_ in P.A.site_specs(mode,model)}
            for batch in loader:
                with torch.no_grad():
                    teacher_arrays.append(model(batch).numpy())
                    endpoint=P.A.affine_mode_forward(model,batch,thresholds,t,mode,relu,line)
                    pred_arrays.append(endpoint['logits'].numpy())
                for name,part in zip(endpoint['site_names'],endpoint['site_stats']):
                    P.A._sum_stats(counts[name],part)
            ref=np.concatenate(teacher_arrays).astype(np.float64)
            pred=np.concatenate(pred_arrays).astype(np.float64)
            row=result['profiles'][str(t)]
            assert row['sites']==counts
            assert row['teacher_class_agreement']==np.mean(pred.argmax(1)==ref.argmax(1))
            assert abs(row['logit_mse']-np.mean((pred-ref)**2))<1e-10
            logp=ref-np.logaddexp.reduce(ref,axis=1,keepdims=True)
            logq=pred-np.logaddexp.reduce(pred,axis=1,keepdims=True)
            kl=np.mean(np.sum(np.exp(logp)*(logp-logq),axis=1))
            assert abs(row['teacher_to_candidate_kl']-kl)<1e-10
            assert row['nodes']==9 and row['directed_edges']==12 and row['final_batch_graphs']==1
            assert row['prospective_projection_adds']==8*sum(
                (4 if s.endswith('.node') else 1)*v['nonzero_scalar_events'] for s,v in counts.items())
        reject(lambda:P.profile_loader(model,loader,thresholds,mode,relu,line,'cpu',expected=4,width=8,batch_size=2))
    logits=torch.tensor([[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]])
    assert P.metric_sums(logits,logits)=={'agree':1,'squared_error':0.,'kl_sum':0.,'graphs':1}
    reject(lambda:P.metric_sums(logits[:,:9],logits))
    bad=logits.clone();bad[0,0]=float('nan');reject(lambda:P.metric_sums(bad,logits))
    example=P.A._site_stats(torch.tensor([[0.,0.],[1.,0.],[0.,-1.]]),torch.ones(2))
    P.validate_stats(example,3,2,1)
    for key in ('zero','nonzero_rows','nonzero_scalar_events'):
        st=copy.deepcopy(example);st[key]+=10;reject(lambda:P.validate_stats(st,3,2,1))
    st=copy.deepcopy(example);st['signed_threshold_level_events']['invalid']=1
    reject(lambda:P.validate_stats(st,3,2,1))
    reject(lambda:P.profile_loader(model,DataLoader(data,batch_size=2),thresholds,'centered',relu,line,'cpu',expected=3,width=8,batch_size=2))
    print('PASS: all modes/source prefixes, independent metrics, row/count/coverage/label guards; CPU synthetic only')


if __name__=='__main__': main()
