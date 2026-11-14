"""One-time frozen full-test evaluation; no training, loss optimization, or model selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Must be set before torch import for deterministic CUDA matrix operations.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'visual-graphs' / 'takeover_20261112'
DEFAULT_ASSETS = Path('C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision')
METHODS = ('baseline', 'node', 'graph', 'mix', 'output', 'shuffled')
TS = (1, 2, 4, 8)
WIDTHS = (146, 146, 146, 146, 73, 36)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_inputs(run: Path, assets: Path):
    """Validate only run/checkpoint/calibration/source files; never load a dataset."""
    import numpy as np
    # Keep asset-layout resolution identical to the frozen screen runner.
    sys.path.insert(0, str(HERE))
    from run_screen import resolve_calibration

    run = run.resolve(); assets = assets.resolve()
    required = [run/'metadata.json', run/'results.json', run/'DONE.json', run/'source']
    for p in required:
        if not p.exists(): raise FileNotFoundError(p)
    meta = load_json(run/'metadata.json'); results = load_json(run/'results.json'); done = load_json(run/'DONE.json')
    if done.get('status') != 'complete': raise ValueError('confirmation run is not marked complete')
    if done.get('test_loaded') is not False or meta.get('test_loaded') is not False:
        raise ValueError('confirmation run already loaded test data')
    if meta.get('val_graphs_official') != 5000 or len(meta.get('val_indices', [])) != 5000:
        raise ValueError('input must be full 5,000-graph official validation confirmation')
    dataset = meta.get('dataset'); seed = int(meta['seed'])
    if dataset not in ('MNIST', 'CIFAR10'): raise ValueError('unsupported dataset')
    expected_train = 55000 if dataset == 'MNIST' else 45000
    if meta.get('train_graphs_official') != expected_train: raise ValueError('official train split count mismatch')
    t_values = tuple(int(t) for t in meta.get('T', []))
    if not t_values or len(set(t_values)) != len(t_values) or any(t not in TS for t in t_values):
        raise ValueError('invalid T list in confirmation metadata')
    if not isinstance(results, list) or done.get('rows') != len(results): raise ValueError('DONE/result row count mismatch')
    result_keys = [(r.get('method'), int(r.get('T', -1))) for r in results]
    if len(set(result_keys)) != len(result_keys): raise ValueError('duplicate condition rows in results.json')
    expected_keys = {(m, t) for m in METHODS for t in t_values}
    if set(result_keys) != expected_keys: raise ValueError(f'confirmation conditions mismatch: expected {sorted(expected_keys)}, got {sorted(result_keys)}')
    if any(int(r.get('val_graphs', -1)) != 5000 for r in results): raise ValueError('confirmation results must cover all 5,000 validation graphs')

    # Confirm the exact code snapshots used for the full validation run.
    snapshots = meta.get('source_snapshots')
    if not isinstance(snapshots, dict) or not snapshots: raise ValueError('source snapshot manifest missing')
    required_snapshots = {'run_screen.py', 'calibration_core.py', 'full_teacher_gcn.py', 'fold_teacher_bn.py', 'dcgs_fixed_seed_eval_v2.py'}
    if not required_snapshots.issubset(snapshots): raise ValueError('incomplete frozen source snapshot set')
    if snapshots['run_screen.py'] != meta.get('script_sha256'): raise ValueError('run_screen snapshot/hash mismatch')
    for name, digest in snapshots.items():
        p = run/'source'/name
        if not p.is_file() or sha(p) != digest: raise ValueError(f'source snapshot mismatch: {p}')
        # run_screen.py is orchestration and asset-path resolution only; its
        # archived bytes remain verified above, but later orchestration edits
        # must not invalidate a frozen calibration. Mathematical code below
        # must still match the exact snapshot used by that calibration.
        if name == 'run_screen.py':
            if digest != meta.get('script_sha256'): raise ValueError('archived run_screen snapshot/hash mismatch')
            continue
        active = (HERE/name) if name == 'calibration_core.py' else (LEGACY/name)
        if not active.is_file() or sha(active) != digest: raise ValueError(f'active scientific source differs from frozen snapshot: {active}')
    if sha(HERE/'calibration_core.py') != meta.get('core_sha256'):
        raise ValueError('calibration_core.py differs from confirmation metadata')
    baseline_gate = meta.get('baseline_exact_max_abs', {})
    if any(float(baseline_gate.get(str(t), float('inf'))) != 0.0 for t in TS):
        raise ValueError('confirmation lacks exact T1/2/4/8 hard-baseline equivalence')
    for name, digest in meta.get('dependency_sha256', {}).items():
        if name not in snapshots or snapshots[name] != digest: raise ValueError(f'dependency provenance mismatch: {name}')

    teacher_run = assets/'takeover_20261112'/'full_teacher_runs'/f'{dataset}_seed{seed}_full'
    checkpoint = teacher_run/'best.pt'; calpath = resolve_calibration(assets,dataset,seed)
    recorded_calibration_path = meta.get('calibration_path')
    if recorded_calibration_path and Path(recorded_calibration_path).resolve() != calpath.resolve():
        raise ValueError('resolved calibration path differs from frozen confirmation metadata')
    cal = load_json(calpath)
    if sha(checkpoint) != meta.get('checkpoint_sha256') or sha(checkpoint) != cal.get('checkpoint_sha256'):
        raise ValueError('checkpoint hash mismatch')
    if sha(calpath) != meta.get('calibration_sha256'): raise ValueError('calibration hash mismatch')
    if int(meta.get('checkpoint_epoch', -1)) != int(cal.get('teacher_epoch', -2)): raise ValueError('checkpoint/calibration epoch mismatch')
    if int(cal.get('teacher_epoch', -1)) != int(meta.get('checkpoint_epoch', -2)): raise ValueError('run/checkpoint epoch mismatch')
    data_root = assets/'data'/'pyg_bench'/dataset/'processed'
    data_hashes = cal.get('data', {}).get('processed_files_sha256', {})
    if not data_hashes: raise ValueError('calibration processed-data hash manifest missing')
    for name, digest in data_hashes.items():
        p = data_root/name
        if not p.is_file() or sha(p) != digest: raise ValueError(f'processed dataset hash mismatch: {p}')
    if 'test_data.pt' not in data_hashes: raise ValueError('test split hash is not recorded')

    # Validate every saved condition and its six frozen offsets without loading model/data.
    params = {}; param_hashes = {}
    for method, t in sorted(expected_keys):
        result_file = run/f'{method}_T{t}.npz'
        if not result_file.is_file(): raise FileNotFoundError(result_file)
        if method == 'baseline': continue
        path = run/f'{method}_T{t}_parameters.npz'
        if not path.is_file(): raise FileNotFoundError(path)
        param_hashes[f'{method}_T{t}'] = sha(path)
        with np.load(path, allow_pickle=False) as bundle:
            keys = {f'offset_{i}' for i in range(6)}
            if not keys.issubset(bundle.files): raise ValueError(f'{path}: missing one of six offset arrays')
            for i, width in enumerate(WIDTHS):
                value = bundle[f'offset_{i}']
                if value.shape != (width,) or not np.isfinite(value).all() or np.any(np.abs(value) > .5 + 1e-7):
                    raise ValueError(f'{path}: invalid offset_{i} shape, value, or bound')
            if method == 'output':
                if 'output_bias' not in bundle.files or bundle['output_bias'].shape != (10,) or not np.isfinite(bundle['output_bias']).all():
                    raise ValueError(f'{path}: invalid output bias')
            elif 'output_bias' in bundle.files:
                raise ValueError(f'{path}: unexpected output bias for {method}')
    teacher_json = teacher_run/'teacher_full_test.json'; teacher_npz = teacher_run/'teacher_full_test_predictions.npz'
    teacher_doc = load_json(teacher_json)
    if teacher_doc.get('checkpoint_sha256') != meta['checkpoint_sha256'] or teacher_doc.get('graphs') != 10000:
        raise ValueError('archived teacher test reference metadata mismatch')
    if sha(teacher_npz) != teacher_doc.get('prediction_artifact_sha256'):
        raise ValueError('archived teacher test reference hash mismatch')
    ref_hash = sha(teacher_npz)
    if not teacher_npz.is_file(): raise FileNotFoundError(teacher_npz)
    return dict(run=run, assets=assets, meta=meta, results=results, done=done,
                teacher_run=teacher_run, checkpoint=checkpoint, calibration=calpath, cal=cal,
                param_hashes=param_hashes, teacher_test_json=teacher_json,
                teacher_test_npz=teacher_npz, teacher_test_sha256=ref_hash,
                data_hashes=data_hashes, t_values=t_values, dataset=dataset, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='completed full-validation confirmation directory')
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--output', type=Path, help='new output directory; required outside --precheck')
    parser.add_argument('--precheck', action='store_true', help='validate hashes/parameters and compile only; never load data')
    args = parser.parse_args()
    prepared = validate_inputs(args.run, args.assets)
    # Parse/compile source in memory; no bytecode file and no test data access.
    compile(Path(__file__).read_text(encoding='utf-8'), str(Path(__file__)), 'exec')
    if args.precheck:
        print(json.dumps({'status':'precheck_pass','test_loaded':False,'dataset':prepared['dataset'],
                          'seed':prepared['seed'],'T':prepared['t_values'],
                          'conditions':len(prepared['results']),'parameter_archives':len(prepared['param_hashes'])}))
        return
    if args.output is None: parser.error('--output is required unless --precheck is used')
    output=args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    # Publish a self-contained copy of the actual fitted inputs, not only hashes.
    input_dir=output/'input_confirmation'; input_dir.mkdir()
    for name in ('metadata.json','DONE.json','results.json'):
        shutil.copy2(prepared['run']/name,input_dir/name)
    for condition,digest in prepared['param_hashes'].items():
        name=f'{condition}_parameters.npz'
        shutil.copy2(prepared['run']/name,output/name)
        if sha(output/name)!=digest: raise RuntimeError('parameter copy mismatch: '+name)

    # Load the implementations whose hashes were checked above.
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_geometric.datasets import GNNBenchmarkDataset
    from torch_geometric.loader import DataLoader
    sys.path.insert(0, str(LEGACY))
    from full_teacher_gcn import BenchmarkGCN
    from fold_teacher_bn import fold_model
    from calibration_core import CalibratedGCN, teacher_sites

    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261114); np.random.seed(20261114)
    device=torch.device('cuda')
    m=prepared['meta']; dataset=prepared['dataset']; teacher_run=prepared['teacher_run']
    state=torch.load(prepared['checkpoint'],map_location='cpu',weights_only=False)
    if int(state['epoch']) != int(m['checkpoint_epoch']): raise RuntimeError('loaded checkpoint epoch mismatch')
    model=BenchmarkGCN(int(load_json(teacher_run/'metadata.json')['input_dim'])).to(device)
    model.load_state_dict(state['model']); model.eval().requires_grad_(False)
    folded=fold_model(model).to(device).eval().requires_grad_(False)
    thresholds=[prepared['cal']['channel_thresholds'][s] for s in ('gcn_0','gcn_1','gcn_2','gcn_3','mlp_0','mlp_1')]
    nets={}
    for method in METHODS:
        for t in prepared['t_values']:
            net=CalibratedGCN(folded,thresholds).to(device).eval()
            bias=None
            if method!='baseline':
                path=output/f'{method}_T{t}_parameters.npz'
                with np.load(path,allow_pickle=False) as bundle, torch.no_grad():
                    for i,offset in enumerate(net.offsets): offset.copy_(torch.as_tensor(bundle[f'offset_{i}'],device=device,dtype=offset.dtype))
                    if method=='output': bias=torch.as_tensor(bundle['output_bias'],device=device,dtype=torch.float32)
            elif any(torch.count_nonzero(p).item() for p in net.offsets): raise RuntimeError('baseline offsets are not zero')
            nets[(method,t)]=(net,bias)

    # Copy and re-hash every frozen implementation snapshot used by confirmation.
    out_source=output/'source'; out_source.mkdir()
    source_hashes={}
    for name,expected in m['source_snapshots'].items():
        src=prepared['run']/'source'/name; dst=out_source/name
        shutil.copy2(src,dst)
        if sha(dst)!=expected: raise RuntimeError(f'copied source snapshot hash mismatch: {name}')
        source_hashes[name]=expected
    evaluator_copy=out_source/'evaluate_frozen_test.py'; shutil.copy2(Path(__file__),evaluator_copy)
    source_hashes['evaluate_frozen_test.py']=sha(evaluator_copy)

    data_root=prepared['assets']/'data'/'pyg_bench'
    test=GNNBenchmarkDataset(root=str(data_root),name=dataset,split='test')
    if len(test)!=10000: raise RuntimeError(f'expected 10,000 official test graphs, found {len(test)}')
    ref=np.load(prepared['teacher_test_npz'],allow_pickle=False)
    ref_idx=ref['index']; ref_y=ref['labels']; ref_logits=ref['logits']
    if not (np.array_equal(ref_idx,np.arange(10000)) and ref_y.shape==(10000,) and ref_logits.shape==(10000,10)):
        raise RuntimeError('teacher test reference has unexpected index/label/logit shape')
    method_parts={(method,t):{'logits':[]} for method in METHODS for t in prepared['t_values']}
    teacher_parts=[]; folded_parts=[]; labels_parts=[]; index_parts=[]; start=time.monotonic()
    loader=DataLoader(test,batch_size=128,shuffle=False)
    with torch.no_grad():
        for batch_no,batch in enumerate(loader,1):
            start_idx=(batch_no-1)*128
            labels=batch.y.view(-1).long().cpu().numpy()
            indices=np.arange(start_idx,start_idx+len(labels),dtype=np.int64)
            batch=batch.to(device); del batch.y
            teacher_logits=model(batch)
            folded_logits,_=teacher_sites(folded,batch)
            teacher_parts.append(teacher_logits.detach().cpu().numpy()); folded_parts.append(folded_logits.detach().cpu().numpy())
            labels_parts.append(labels); index_parts.append(indices)
            for (method,t),(net,bias) in nets.items():
                logits,_=net(batch,t)
                if bias is not None: logits=logits+bias
                method_parts[(method,t)]['logits'].append(logits.detach().cpu().numpy())
            if batch_no%10==0: print(f'Processed {min(batch_no*128,10000)}/10000 official test graphs',flush=True)
    teacher_logits=np.concatenate(teacher_parts); folded_logits=np.concatenate(folded_parts)
    labels=np.concatenate(labels_parts); indices=np.concatenate(index_parts)
    if not (np.array_equal(indices,ref_idx) and np.array_equal(labels,ref_y)):
        raise RuntimeError('current official test ordering/labels differ from archived teacher reference')
    teacher_archive_max=float(np.max(np.abs(teacher_logits-ref_logits)))
    if not np.allclose(teacher_logits,ref_logits,rtol=1e-5,atol=1e-4): raise RuntimeError('current original teacher logits differ from archived test reference')
    fold_max=float(np.max(np.abs(folded_logits-teacher_logits)))
    if not np.allclose(folded_logits,teacher_logits,rtol=1e-5,atol=1e-4): raise RuntimeError(f'folded teacher mismatch: max abs {fold_max}')
    teacher_pred=teacher_logits.argmax(1); fold_pred=folded_logits.argmax(1)
    fold_pred_diffs=int(np.count_nonzero(teacher_pred!=fold_pred))
    # Revalidate the frozen inputs before publishing any endpoint as complete.
    end_inputs=validate_inputs(prepared['run'],prepared['assets'])
    if end_inputs['param_hashes']!=prepared['param_hashes']:
        raise RuntimeError('calibration parameters changed during test evaluation')
    for name in ('metadata.json','DONE.json','results.json'):
        if sha(input_dir/name)!=sha(prepared['run']/name):
            raise RuntimeError('confirmation input changed during evaluation: '+name)
    if sha(Path(__file__))!=source_hashes['evaluate_frozen_test.py']:
        raise RuntimeError('test evaluator changed during evaluation')
    results=[]
    for method in METHODS:
        for t in prepared['t_values']:
            z=np.concatenate(method_parts[(method,t)]['logits'])
            if z.shape!=(10000,10) or not np.isfinite(z).all(): raise RuntimeError(f'invalid predictions: {method} T{t}')
            y=torch.as_tensor(labels,dtype=torch.long)
            zt=torch.as_tensor(z,dtype=torch.float32); at=torch.as_tensor(teacher_logits,dtype=torch.float32)
            row={'method':method,'T':t,'accuracy_percent':float((zt.argmax(1)==y).float().mean()*100),
                 'teacher_accuracy_percent':float((at.argmax(1)==y).float().mean()*100),
                 'teacher_agreement_percent':float((zt.argmax(1)==at.argmax(1)).float().mean()*100),
                 'logit_mse':float((zt-at).square().mean()),'cross_entropy':float(F.cross_entropy(zt,y)),
                 'graphs':10000,'prediction_differences_vs_archived_teacher':int(np.count_nonzero(z.argmax(1)!=ref_logits.argmax(1)))}
            results.append(row)
            np.savez_compressed(output/f'{method}_T{t}.npz',indices=indices,labels=labels,teacher_logits=teacher_logits,converted_logits=z)
            print(json.dumps(row),flush=True)
    metadata={'status':'complete','dataset':dataset,'seed':prepared['seed'],'split':'full official test, historically exposed',
              'graphs':10000,'batch_size':128,'T':list(prepared['t_values']),'methods':list(METHODS),
              'checkpoint_sha256':sha(prepared['checkpoint']),'calibration_sha256':sha(prepared['calibration']),
              'teacher_reference_sha256':prepared['teacher_test_sha256'],'confirmation_metadata_sha256':sha(prepared['run']/'metadata.json'),
              'confirmation_DONE_sha256':sha(prepared['run']/'DONE.json'),'confirmation_results_sha256':sha(prepared['run']/'results.json'),
              'calibration_config':{key:m[key] for key in ('steps','optimizer','lr','batch_size','offset_bound')},
              'parameter_sha256':prepared['param_hashes'],'core_sha256':sha(HERE/'calibration_core.py'),
              'dependency_sha256':{name:sha(LEGACY/name) for name in m['dependency_sha256']},'source_snapshots':source_hashes,
              'processed_data_sha256':prepared['data_hashes'],'device':torch.cuda.get_device_name(device),
              'torch':torch.__version__,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
              'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'test_loaded':True,'training_or_optimization':False,
              'teacher_fold_max_abs_difference':fold_max,'teacher_fold_prediction_differences':fold_pred_diffs,
              'teacher_original_max_abs_difference_vs_archive':teacher_archive_max,
              'teacher_original_prediction_differences_vs_archive':int(np.count_nonzero(teacher_pred!=ref_logits.argmax(1))),
              'qualification':'frozen full-test endpoint; test and validation data historically exposed; descriptive only'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'DONE.json').write_text(json.dumps({'status':'complete','rows':len(results),'test_loaded':True,'training_or_optimization':False,'total_elapsed_seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','output':str(output),'seconds':time.monotonic()-start,'rows':len(results)}))

if __name__=='__main__': main()
