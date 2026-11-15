"""Fixed six-arm finite-event reassignment experiment; no test split or tuning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_objectives import historical,core,CRITICAL
from learned_swap_core import observe_first,learned_forward
from event_swap import propose_swaps
from summarize_swap_oracle import class_margins

ARMS=['nongraph_local','graph_local','nongraph_decision','graph_decision','shuffled_decision','toppin_decision']
COSTS=[('events',1),('events',2),('amplitude',1),('amplitude',2),('deliveries',1),('deliveries',2)]


def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    from swap_predictor import graph_cache,pair_features,ridge_fit,ridge_predict,choose_pairs
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['MNIST','CIFAR10'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(20261118)
    meta,_,_,folded,loaded,_,data_root=historical.load_teacher(args.dataset,41,historical.DEFAULT_ASSETS,
        CRITICAL/'runs/allocation'/f'{args.dataset}41',torch.device('cuda'))
    net=core.CalibratedGCN(folded,loaded.thresholds).cuda().eval().requires_grad_(False)
    paramfile=HERE/'runs/objectives'/f'{args.dataset}41'/'local_t1_parameters.npz'
    with np.load(paramfile) as params:
        for i,o in enumerate(net.offsets):o.data.copy_(torch.as_tensor(params[f'offset_{i}'],device=o.device))
    bankroot=HERE/'runs/swap_oracle'/f'{args.dataset}41'
    bank=dict(np.load(bankroot/'candidates.npz'))
    bankmeta=json.loads((bankroot/'metadata.json').read_text())
    assert historical.sha(paramfile)==bankmeta['baseline_parameters_sha256']
    assert bankmeta['checkpoint_sha256']==meta['checkpoint_sha256']
    trainids=bank['indices'].tolist();assert trainids==meta['train_indices'][:128]
    train=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='train')
    val=GNNBenchmarkDataset(root=str(data_root),name=args.dataset,split='val')
    features={mode:[] for mode in ('nongraph','graph','shuffled','toppin')};valids=[]
    offset=0;cache_fit_seconds={mode:0. for mode in features}
    for batch in DataLoader(train[trainids],batch_size=32,shuffle=False):
        batch=batch.cuda();del batch.y
        z,observed=observe_first(net,batch);n=batch.num_graphs
        assert np.array_equal(z.cpu().numpy(),bank['logits_T1'][0,offset:offset+n])
        degree=torch.bincount(batch.edge_index[0],minlength=batch.num_nodes).cpu()
        proposals=propose_swaps(torch.as_tensor(observed['event']),batch.ptr.cpu(),degree,trainids[offset:offset+n],count=8)
        pairs=np.asarray([[p if p is not None else (-1,-1,-1) for p in candidate] for candidate in proposals],dtype=np.int64)
        relative=pairs.copy()
        for g,lo in enumerate(batch.ptr[:-1].cpu().tolist()):
            valid=relative[:,g,0]>=0;relative[valid,g,0]-=lo;relative[valid,g,1]-=lo
        assert np.array_equal(relative,bank['pairs'][1:,offset:offset+n]),'candidate bank replay mismatch'
        for mode in features:
            then=time.perf_counter();cache=graph_cache(batch.edge_index.cpu(),batch.ptr.cpu(),trainids[offset:offset+n],mode)
            X,valid=pair_features(observed['event'],observed['membrane'],net.thresholds[0].cpu().numpy(),pairs,batch.ptr.cpu().numpy(),cache,mode)
            cache_fit_seconds[mode]+=time.perf_counter()-then;features[mode].append(X)
            if mode=='nongraph':valids.append(valid)
        offset+=n
    features={mode:np.concatenate(parts,axis=1) for mode,parts in features.items()}
    valid=np.concatenate(valids,axis=1)
    teacher_class=bank['teacher'].argmax(1)
    decision=np.stack([class_margins(bank[f'logits_T{T}'],teacher_class)[1:]-class_margins(bank[f'logits_T{T}'],teacher_class)[:1] for T in (1,2)],axis=-1)
    local=-(bank['local_mse'][1:]-bank['local_mse'][:1]);local=np.stack([local,local],axis=-1)
    cost=np.stack([bank[f'{key}_T{T}'][1:]-bank[f'{key}_T{T}'][:1] for key,T in COSTS],axis=-1)
    models={};fit_audit=[]
    for arm in ARMS:
        mode,objective=arm.split('_');X=features[mode];target=np.concatenate([decision if objective=='decision' else local,cost],axis=-1)
        then=time.perf_counter();model=ridge_fit(X[valid],target[valid],alpha=.001);models[arm]=model
        pred=ridge_predict(model,X)
        np.savez_compressed(args.output/f'{arm}_model.npz',**model)
        np.savez_compressed(args.output/f'{arm}_calibration.npz',features=X,valid=valid,targets=target,predictions=pred,indices=bank['indices'])
        fit_audit.append(dict(arm=arm,examples=int(valid.sum()),features=X.shape[-1],output_heads=target.shape[-1],
            fitted_coefficients=int(model['coef'].size),stored_numeric_bytes=sum(np.asarray(v).nbytes for v in model.values()),
            fit_seconds=time.perf_counter()-then,feature_build_seconds=cache_fit_seconds[mode],
            training_output_mse=np.square(pred[valid]-target[valid]).mean(0).tolist()))
    save(args.output/'fit_audit.json',fit_audit)
    batches=list(DataLoader(val[meta['val_indices']],batch_size=32,shuffle=False))
    source=args.output/'source';source.mkdir()
    for path in (Path(__file__),HERE/'learned_swap_core.py',HERE/'swap_predictor.py',HERE/'event_swap.py',HERE/'intervention.py'):
        shutil.copyfile(path,source/path.name)
    record=dict(dataset=args.dataset,teacher_seed=41,proposal_seed=20261117,shuffle_seed=20261118,
        train_indices=trainids,val_indices=meta['val_indices'],test_loaded=False,labels_used_for_fit=False,
        source_sha256={p.name:historical.sha(p) for p in source.iterdir()},bank_sha256=historical.sha(bankroot/'candidates.npz'),
        baseline_parameters_sha256=historical.sha(paramfile),checkpoint_sha256=meta['checkpoint_sha256'],
        calibration_sha256=meta['calibration_sha256'],ridge_alpha=.001,arms=ARMS,graph_count_calibration=128,
        baseline_prior_calibration_graphs=512,extra_candidate_bank=8,prediction_heads=['T1score','T2score']+[f'{key}_T{T}' for key,T in COSTS],
        online_teacher=False,online_downstream_lookahead=False,online_full_passes=1,
        source_replay='all128 hard baseline logits and all eight coordinate pairs exact',
        qualification='one bounded prototype, selection-exposed validation512; no independent confirmation or novelty claim')
    save(args.output/'metadata.json',record)
    results=[];choices_by_T={}
    for arm in ['baseline']+ARMS:
        model=models.get(arm);mode=arm.split('_')[0] if arm!='baseline' else 'nongraph'
        for T in (1,2):
            parts={k:[] for k in ('logits','teacher','labels','events','amplitude','deliveries','choice','selected_prediction')}
            offset=0;cache_seconds=0.;cache_bytes=0
            for raw in batches:
                if time.monotonic()-start>600:
                    save(args.output/'TIMEOUT.json',dict(status='ten-minute cap, no restart'));raise TimeoutError('prototype cap reached')
                b=raw.clone().cuda();labels=b.y.flatten().cpu().numpy();del b.y
                ids=meta['val_indices'][offset:offset+b.num_graphs]
                then=time.perf_counter();cache=graph_cache(raw.edge_index,raw.ptr,ids,mode) if model is not None else None
                cache_seconds+=time.perf_counter()-then
                if cache is not None:cache_bytes+=sum(v.nbytes for v in cache.values() if isinstance(v,np.ndarray))
                with torch.no_grad():a,_=core.teacher_sites(folded,b)
                z,trace=learned_forward(net,b,T,model,mode,ids,cache)
                values=dict(logits=z.cpu().numpy(),teacher=a.cpu().numpy(),labels=labels,
                    **{key:trace[key].cpu().numpy() for key in ('events','amplitude','deliveries')},
                    choice=trace['choice'],selected_prediction=trace['selected_prediction'])
                for key,value in values.items():parts[key].append(value)
                offset+=b.num_graphs
            arrays={k:np.concatenate(v) for k,v in parts.items()};arrays['indices']=np.asarray(meta['val_indices'])
            np.savez_compressed(args.output/f'{arm}_T{T}.npz',**arrays)
            if T==1:choices_by_T[arm]=arrays['choice']
            else:assert np.array_equal(choices_by_T[arm],arrays['choice']),'prefix-dependent selector choice'
            pred=arrays['logits'].argmax(1)
            row=dict(arm=arm,T=T,accuracy_percent=float(100*np.mean(pred==arrays['labels'])),
                teacher_agreement_percent=float(100*np.mean(pred==arrays['teacher'].argmax(1))),
                events=int(arrays['events'].sum(dtype=np.int64)),amplitude_sum=float(arrays['amplitude'].sum()),
                branch_delivery_proxy=int(arrays['deliveries'].sum(dtype=np.int64)),selected_swap_graphs=int((arrays['choice']!=0).sum()),
                cache_build_seconds=cache_seconds,cache_bytes_over512_graphs=cache_bytes)
            results.append(row);save(args.output/'results.json',results);print(json.dumps(row),flush=True)
    # Timing does not call teacher, compute diagnostic errors, or precompute state-dependent choices.
    raw=batches[0];b=raw.clone().cuda();del b.y;ids=meta['val_indices'][:b.num_graphs]
    configs=[('baseline','plain')]+[(arm,temperature) for arm in ARMS for temperature in ('warm','cold')]
    cache={arm:graph_cache(raw.edge_index,raw.ptr,ids,arm.split('_')[0]) for arm in ARMS}
    samples={f'{arm}/{temp}':[] for arm,temp in configs}
    schedule=[(arm,temp) for arm,temp in configs for _ in range(7)];random.Random(20261118).shuffle(schedule)
    counts={k:0 for k in samples}
    for arm,temp in schedule:
        torch.cuda.synchronize();then=time.perf_counter()
        if arm=='baseline':
            with torch.no_grad():net(b,1)
        else:learned_forward(net,b,1,models[arm],arm.split('_')[0],ids,cache[arm] if temp=='warm' else None,instrument=False)
        torch.cuda.synchronize();ms=(time.perf_counter()-then)*1000
        key=f'{arm}/{temp}';counts[key]+=1
        if counts[key]>2:samples[key].append(ms)
    save(args.output/'timing.json',dict(batch_graphs=b.num_graphs,T=1,warmups=2,repeats=5,teacher_in_timed_block=False,
        state_dependent_proposals_and_prediction_in_timed_block=True,diagnostic_counters_in_timed_block=False,
        metric='synchronized dense PyTorch+NumPy host wall, includes Python/controller/CPU transfers; not hardware energy',
        results={key:dict(raw_ms=times,median_ms_per_graph=float(np.median(times)/b.num_graphs)) for key,times in samples.items()}))
    save(args.output/'DONE.json',dict(endpoints=len(results),seconds=time.monotonic()-start))


if __name__=='__main__':main()
