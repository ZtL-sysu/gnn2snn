"""CPU/GPU diagnostic for replacing selected quantizers with ideal causal ReLU steps.

This is an oracle diagnostic, not a trainable model or deployable spiking method.
The core is never edited; the process-local quantizer hook is restored on exit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
HERE=Path(__file__).resolve().parent
LEGACY=HERE.parent/'visual-graphs'/'takeover_20261112'
DEFAULT_ASSETS=Path('C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision')
SITES=('gcn_0','gcn_1','gcn_2','gcn_3','mlp_0','mlp_1')
MASKS={'baseline':(), 'site0':(0,), 'site1':(1,), 'site2':(2,), 'site3':(3,), 'site4':(4,), 'site5':(5,), 'nodes0_3':(0,1,2,3), 'head4_5':(4,5), 'all6':(0,1,2,3,4,5)}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

@contextmanager
def idealized_site_mask(site_indices):
    """Temporarily replace selected site calls by causal ReLU-difference steps."""
    import calibration_core as core
    selected=frozenset(int(i) for i in site_indices)
    if any(i not in range(6) for i in selected): raise ValueError('site indices must be in [0,5]')
    original=core._mth_step
    stats={'calls':0,'selected_calls':0,'site_calls':[0]*6}
    def hooked(state,current,theta,tick):
        site=stats['calls']%6; stats['calls']+=1; stats['site_calls'][site]+=1
        if site not in selected:return original(state,current,theta,tick)
        stats['selected_calls']+=1
        u=state.get('ideal_input')
        a_prev=state.get('ideal_output')
        if u is None:u=torch_zeros_like(current)
        if a_prev is None:a_prev=torch_zeros_like(current)
        u=u+current/float(tick)
        a=u.clamp_min(0)
        event=float(tick)*(a-a_prev)
        state['ideal_input']=u
        state['ideal_output']=a
        return event
    core._mth_step=hooked
    try:
        yield stats
    finally:
        core._mth_step=original

def torch_zeros_like(x):
    # Keep torch a lazy import so file-level inspection/precheck stays lightweight.
    import torch
    return torch.zeros_like(x)

def forward_with_mask(net,batch,T,site_indices):
    """One no-gradient call with zero offsets; returns logits, sites, hook counts."""
    import torch
    if any(bool((p.detach()!=0).any()) for p in net.offsets):
        raise ValueError('ideal-site diagnostic requires all six offsets to be zero')
    with idealized_site_mask(site_indices) as stats:
        with torch.no_grad(): logits,decoded=net(batch,T)
    if stats['calls']!=6*T or any(c!=T for c in stats['site_calls']):
        raise RuntimeError(f'unexpected site call sequence: {stats}')
    return logits,decoded,stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='completed 5,000-graph validation confirmation directory')
    p.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    # File/hash checks only; this validates no dataset and does not access test examples.
    from evaluate_frozen_test import validate_inputs
    prepared=validate_inputs(args.run,args.assets)
    if not set((1,2)).issubset(prepared['t_values']):raise ValueError('confirmation must contain T1 and T2')
    output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_geometric.datasets import GNNBenchmarkDataset
    from torch_geometric.loader import DataLoader
    sys.path.insert(0,str(LEGACY)); sys.path.insert(0,str(HERE))
    from full_teacher_gcn import BenchmarkGCN
    from fold_teacher_bn import fold_model
    import calibration_core as core
    from calibration_core import CalibratedGCN,teacher_sites
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.manual_seed(20261114); np.random.seed(20261114)
    device=torch.device(args.device)
    m=prepared['meta']; dataset=prepared['dataset']; seed=prepared['seed']; teacher_run=prepared['teacher_run']
    batch_size=int(m.get('batch_size',0))
    if batch_size<1:raise ValueError('confirmation batch_size is missing or invalid')
    state=torch.load(prepared['checkpoint'],map_location='cpu',weights_only=False)
    model=BenchmarkGCN(int(json.loads((teacher_run/'metadata.json').read_text())['input_dim'])).to(device)
    model.load_state_dict(state['model']);model.eval().requires_grad_(False)
    folded=fold_model(model).to(device).eval().requires_grad_(False)
    cal=prepared['cal']; theta=[cal['channel_thresholds'][name] for name in SITES]
    net=CalibratedGCN(folded,theta).to(device).eval()
    with torch.no_grad():
        for off in net.offsets:off.zero_()
    val=GNNBenchmarkDataset(root=str(prepared['assets']/'data'/'pyg_bench'),name=dataset,split='val')
    if len(val)!=5000:raise RuntimeError(f'expected official full validation (5000), got {len(val)}')
    val_indices=np.asarray(m['val_indices'],dtype=np.int64)
    if len(val_indices)!=5000 or len(np.unique(val_indices))!=5000 or val_indices.min()<0 or val_indices.max()>=5000:raise ValueError('invalid full-validation index permutation')
    loader=DataLoader(val[val_indices.tolist()],batch_size=batch_size,shuffle=False)
    expected=[(name,T,sites) for T in (1,2) for name,sites in MASKS.items()]
    collected={(name,T):{k:[] for k in ('logits','teacher','labels','indices','margin','margin_shift')} for name,T,_ in expected}
    t0=time.monotonic(); original_step=core._mth_step; all6_teacher_max_abs=0.0
    with torch.no_grad():
        for bno,batch in enumerate(loader,1):
            start=(bno-1)*batch_size; labels=batch.y.view(-1).long().cpu().numpy(); idx=val_indices[start:start+len(labels)].copy()
            batch=batch.to(device); del batch.y
            teacher, _=teacher_sites(folded,batch)
            teacher_np=teacher.detach().cpu().numpy(); teacher_pred=teacher.argmax(1)
            rival=teacher.clone(); rival.scatter_(1,teacher_pred[:,None],float('-inf'))
            teacher_margin=(teacher.gather(1,teacher_pred[:,None]).flatten()-rival.max(1).values)
            for name,T,sites in expected:
                logits,_,stats=forward_with_mask(net,batch,T,sites)
                if name=='baseline' and not torch.equal(logits,core_baseline_reference(net,batch,T,original_step)):
                    raise RuntimeError(f'empty mask changed baseline bits at T{T}')
                if name=='all6':
                    max_abs=float((logits-teacher).abs().max())
                    all6_teacher_max_abs=max(all6_teacher_max_abs,max_abs)
                    if not torch.allclose(logits,teacher,rtol=1e-5,atol=1e-4):
                        raise RuntimeError(f'all-six oracle failed folded-teacher recovery at T{T}: {max_abs}')
                lp=logits.argmax(1); zr=logits.clone(); zr.scatter_(1,teacher_pred[:,None],float('-inf'))
                margin_shift=(logits.gather(1,teacher_pred[:,None]).flatten()-zr.max(1).values)-teacher_margin
                d=collected[(name,T)];d['logits'].append(logits.cpu().numpy());d['teacher'].append(teacher_np);d['labels'].append(labels);d['indices'].append(idx)
                d['margin'].append(teacher_margin.cpu().numpy());d['margin_shift'].append(margin_shift.cpu().numpy())
            if bno%10==0:print(f'Processed {min(bno*batch_size,5000)}/5000 official validation graphs',flush=True)
    if core._mth_step is not original_step:raise RuntimeError('quantizer hook was not restored')
    results=[]
    for name,T,sites in expected:
        d={k:np.concatenate(v) for k,v in collected[(name,T)].items()}
        if len(d['indices'])!=5000 or not np.array_equal(d['indices'],val_indices):raise RuntimeError('validation ordering/count mismatch')
        z=d['logits'];a=d['teacher'];y=d['labels']
        if not all(np.isfinite(v).all() for v in d.values()):raise RuntimeError(f'nonfinite diagnostic output: {name} T{T}')
        if name=='baseline':
            with np.load(prepared['run']/f'baseline_T{T}.npz',allow_pickle=False) as frozen:
                for key,actual in [('indices',d['indices']),('labels',y),('teacher',a),('logits',z)]:
                    if not np.array_equal(actual,frozen[key]):
                        raise RuntimeError(f'baseline {key} differs bitwise from frozen confirmation at T{T}')
        row={'condition':name,'ideal_sites':[SITES[i] for i in sites],'T':T,'graphs':5000,
             'accuracy_percent':float((z.argmax(1)==y).mean()*100),'teacher_accuracy_percent':float((a.argmax(1)==y).mean()*100),
             'teacher_agreement_percent':float((z.argmax(1)==a.argmax(1)).mean()*100),
             'logit_mse':float(np.mean((z.astype(np.float64)-a.astype(np.float64))**2)),
             'teacher_margin_mean':float(d['margin'].mean()),'margin_shift_mean':float(d['margin_shift'].mean())}
        results.append(row)
        np.savez_compressed(output/f'{name}_T{T}.npz',indices=d['indices'],labels=y,teacher_logits=a,logits=z,teacher_margin=d['margin'],margin_shift=d['margin_shift'])
        print(json.dumps(row),flush=True)
    if sha(HERE/'calibration_core.py')!=sha(prepared['run']/'source'/'calibration_core.py'):raise RuntimeError('scientific core snapshot changed during diagnostic')
    source=output/'source';source.mkdir()
    source_hashes={}
    for name,digest in m['source_snapshots'].items():
        src=prepared['run']/'source'/name;dst=source/name;shutil.copy2(src,dst)
        if sha(dst)!=digest:raise RuntimeError(f'source snapshot copy mismatch: {name}')
        source_hashes[name]=digest
    shutil.copy2(Path(__file__),source/'ideal_site_diagnostic.py');source_hashes['ideal_site_diagnostic.py']=sha(source/'ideal_site_diagnostic.py')
    meta={'status':'complete','dataset':dataset,'seed':seed,'split':'full official validation; historically exposed',
          'graphs':5000,'batch_size':batch_size,'indices':val_indices.tolist(),'T':[1,2],'conditions':list(MASKS),
          'checkpoint_sha256':sha(prepared['checkpoint']),'calibration_sha256':sha(prepared['calibration']),
          'confirmation_metadata_sha256':sha(prepared['run']/'metadata.json'),'source_sha256':source_hashes,
          'parameter_sha256':prepared['param_hashes'],
          'all6_max_abs_vs_folded_teacher':all6_teacher_max_abs,
          'core_sha256':sha(HERE/'calibration_core.py'),'original_quantizer_restored':core._mth_step is original_step,
          'zero_offsets':True,'optimization_or_training':False,'device':str(device),'torch':torch.__version__,
          'elapsed_sec':time.monotonic()-t0,
          'qualification':'continuous-valued causal ReLU oracle diagnostic, not a deployable SNN or method; grouped restorations affect different numbers of sites and effects are not additive'}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'DONE.json').write_text(json.dumps({'status':'complete','rows':len(results),'test_loaded':False,'optimization_or_training':False},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','output':str(output),'conditions':len(results),'seconds':meta['elapsed_sec']}))

def core_baseline_reference(net,batch,T,original_step):
    """Reference empty-hook output with exact original function, under no_grad."""
    import calibration_core as core
    saved=core._mth_step
    try:
        core._mth_step=original_step
        return net(batch,T)[0]
    finally:
        core._mth_step=saved

if __name__=='__main__':main()
