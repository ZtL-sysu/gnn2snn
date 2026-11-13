"""Frozen train-only affine-base fidelity/event profiles; dense reference backend."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import gated_affine_cal_v1 as C
import gated_affine_base_v1 as A

PREFIXES = (1,2,4)
DEPENDENCIES = ('gated_affine_profile_v1.py','check_gated_affine_profile_v1.py',
                'gated_affine_base_v1.py','check_gated_affine_base_v1.py')


def empty_stats():
    return {'positive':0,'negative':0,'zero':0,'nonzero_scalar_events':0,'nonzero_rows':0,
            'sum_abs_graded_amplitude':0.,'signed_threshold_level_events':{}}


def validate_stats(st,rows,width,prefix):
    for k in ('positive','negative','zero','nonzero_scalar_events','nonzero_rows'):
        if type(st[k]) is not int or st[k]<0: raise RuntimeError('invalid count')
    if (st['positive']+st['negative']+st['zero'] != rows*width*prefix or
            st['positive']+st['negative'] != st['nonzero_scalar_events'] or
            sum(st['signed_threshold_level_events'].values()) != st['nonzero_scalar_events'] or
            not math.isfinite(st['sum_abs_graded_amplitude']) or st['sum_abs_graded_amplitude']<0 or
            not 0<=st['nonzero_rows']<=rows*prefix or
            not st['nonzero_rows']<=st['nonzero_scalar_events']<=st['nonzero_rows']*width):
        raise RuntimeError('event partition/row mismatch')
    for k,v in st['signed_threshold_level_events'].items():
        if k not in {f'{s}2^-{i}' for s in ('+','-') for i in range(8)} or type(v) is not int or v<0:
            raise RuntimeError('invalid signed level partition')


def metric_sums(logits,teacher):
    if logits.shape != teacher.shape or logits.ndim!=2 or logits.shape[1]!=10:
        raise RuntimeError('logit shape mismatch')
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(teacher).all()):
        raise RuntimeError('nonfinite logits')
    # Float64 for stable external reporting, never used in candidate state.
    pred, ref = logits.double(),teacher.double()
    logp, logq = F.log_softmax(ref,1),F.log_softmax(pred,1)
    return {'agree':int((pred.argmax(1)==ref.argmax(1)).sum()),
            'squared_error':float((pred-ref).square().sum()),
            'kl_sum':float((logp.exp()*(logp-logq)).sum()),'graphs':int(pred.shape[0])}


@torch.no_grad()
def profile_loader(model,loader,thresholds,mode,unit_relu,unit_line,device,
                   expected=45000,width=70,batch_size=128):
    specs=A.site_specs(mode,model)
    sites=[s[0] for s in specs]
    stats={str(t):{s:empty_stats() for s in sites} for t in PREFIXES}
    metrics={str(t):{'agree':0,'squared_error':0.,'kl_sum':0.,'graphs':0} for t in PREFIXES}
    graphs=batches=nodes=edges=last=warmups=active=heads=0
    for batch in loader:
        batch=batch.to(device)
        C.validate_eval(model,batch)
        ref=model(batch)
        result=A.affine_mode_forward(model,batch,thresholds,4,mode,unit_relu,unit_line,
                                     capture_prefix_stats=True,capture_prefix_logits=True)
        if (result['site_names']!=sites or result['source_warmup_calls']!=len(sites) or
                result['active_unit_calls']!=len(sites)*4 or result['head_calls']!=3 or
                result['source_ticks']!=[4]*len(sites) or
                set(result['prefix_site_stats'])!={'1','2','4'} or
                set(result['prefix_logits'])!={'1','2','4'}):
            raise RuntimeError('source call/prefix accounting mismatch')
        last=int(batch.num_graphs); graphs+=last; batches+=1
        nodes+=int(batch.num_nodes); edges+=int(batch.num_edges)
        warmups+=result['source_warmup_calls'];active+=result['active_unit_calls'];heads+=result['head_calls']
        for t in PREFIXES:
            key=str(t)
            parts=result['prefix_site_stats'][key]
            if len(parts)!=len(sites): raise RuntimeError('incomplete site stats')
            for s,part in zip(sites,parts):
                rows=int(batch.num_nodes if s.endswith('.node') else batch.num_edges)
                validate_stats(part,rows,width,t)
                A._sum_stats(stats[key][s],part)
            sums=metric_sums(result['prefix_logits'][key],ref)
            for k,v in sums.items(): metrics[key][k]+=v
        if batches%50==0:
            print(json.dumps({'progress':'affine_profile','mode':mode,'graphs':graphs,'batches':batches}),flush=True)
    if (graphs!=expected or batches!=math.ceil(expected/batch_size) or
            last!=(expected%batch_size or batch_size)):
        raise RuntimeError('incomplete profile coverage')
    profiles={}
    for t in PREFIXES:
        key=str(t); st=stats[key]; m=metrics[key]
        if m['graphs']!=graphs: raise RuntimeError('metric coverage mismatch')
        for s in sites:
            validate_stats(st[s],nodes if s.endswith('.node') else edges,width,t)
        # One coded stream feeding A/B/D/E (nodes) or C (edges), without branch fanout.
        projection_adds=width*sum((4 if s.endswith('.node') else 1)*st[s]['nonzero_scalar_events'] for s in sites)
        profiles[key]={'prefix_T':t,'graphs':graphs,'batches':batches,'nodes':nodes,
                       'directed_edges':edges,'final_batch_graphs':last,'sites':st,
                       'teacher_class_agreement':m['agree']/graphs,
                       'logit_mse':m['squared_error']/(graphs*10),'teacher_to_candidate_kl':m['kl_sum']/graphs,
                       'metric_sums':m,'prospective_projection_adds':projection_adds,
                       'prefix_logical_active_calls':len(sites)*t*batches}
    return {'profiles':profiles,'source_units_per_batch':len(sites),
            'actual_source_warmup_calls':warmups,'actual_active_calls':active,
            'diagnostic_head_calls':heads,'normal_endpoint_head_calls':batches,
            'source_head_scope':'three diagnostic prefix heads; normal endpoint head once per batch'}


def run(output,calibration,mode,scale,author_repo,device):
    C.output_path(output)
    if mode not in A.MODES or scale not in (1,8): raise ValueError('unsupported mode/scale')
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    dev=torch.device(device)
    model,teacher=C.load_teacher(dev)
    doc,calhash=C.validate_calibration_document(calibration,teacher)
    thresholds=C.profile_thresholds(doc,mode,scale)
    relu,line,author=A.require_author_sources(author_repo)
    bound=copy.deepcopy(doc['input_hashes_start'])
    bound.update({str((C.HERE/n).resolve()):C.sha(C.HERE/n) for n in DEPENDENCIES})
    bound[str(Path(calibration).resolve())]=calhash
    for item in author['source_files'].values():
        file=Path(author['author_repo'])/item['path'];bound[str(file.resolve())]=C.sha(file)
    data=GNNBenchmarkDataset(root=str(C.DATA_ROOT),name='CIFAR10',split='train')
    if len(data)!=45000: raise RuntimeError('full official train required')
    loader=DataLoader(C.LabelFreeGraphView(data),batch_size=128,shuffle=False,num_workers=0)
    timer=time.perf_counter()
    result=profile_loader(model,loader,thresholds,mode,relu,line,dev)
    end=C.require_unchanged(bound)
    A.validate_source_unchanged(author_repo,author)
    result.update({'status':'complete','dataset':'CIFAR10','seed':41,'mode':mode,'threshold_scale':scale,
                   'scope':'all_45000_official_train_graphs_label_free_fixed_order',
                   'executed_T':4,'prefix_time_grid':[1,2,4], 'teacher':teacher,
                   'threshold_vectors':[v.tolist() for v in thresholds],
                   'calibration':str(Path(calibration).resolve()),'calibration_sha256':calhash,
                   'author_source':author,'input_hashes_start':bound,'input_hashes_end':end,
                   'elapsed_seconds':time.perf_counter()-timer,
                   'runtime':{'device':str(dev),'torch':torch.__version__,'python':sys.version,
                              'deterministic':torch.are_deterministic_algorithms_enabled(),
                              'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                              'labels_passed':False,'validation_or_test_opened':False},
                   'qualification':'dense reference; same analog MLP head; graded signed events; prospective ADD count is not energy or efficient execution'})
    C.write_exclusive(output,result)
    print(json.dumps({'status':'complete','mode':mode,'scale':scale,'output':str(output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--mode',choices=A.MODES,required=True);p.add_argument('--scale',type=int,choices=(1,8),required=True)
    p.add_argument('--author-repo',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    a=p.parse_args();run(a.output,a.calibration,a.mode,a.scale,a.author_repo,a.device)
