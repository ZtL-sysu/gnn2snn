"""Frozen-teacher per-channel percentile baseline; no test loading or parameter fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '20261114_story'
sys.path.insert(0, str(OLD))
LEGACY = HERE.parent / 'visual-graphs' / 'takeover_20261112'
sys.path.insert(0, str(LEGACY))
from full_teacher_gcn import BenchmarkGCN
from fold_teacher_bn import fold_model
from calibration_core import teacher_sites
from percentile_core import PercentileGCN, channelwise_quantile_thresholds

DEFAULT_ASSETS = Path('C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision')
SITES = ['gcn_0', 'gcn_1', 'gcn_2', 'gcn_3', 'mlp_0', 'mlp_1']
CONDITIONS = {'author_cfg': (4.0, 4), 'matched8': (1.0, 8)}


def resolve_calibration(assets, dataset, seed):
    root = Path(assets) / 'takeover_20261112'
    primary = root / 'full_teacher_runs' / f'{dataset}_seed{seed}_full' / 'calibration_hist_v1.json'
    if primary.is_file():
        return primary
    replica = root / 'rep' / f"{'M' if dataset == 'MNIST' else 'C'}{seed}" / 'cal.json'
    if replica.is_file():
        return replica
    raise FileNotFoundError(f'No audited calibration for {dataset} seed {seed}')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def metrics(logits, teacher, labels):
    logits, teacher = torch.as_tensor(logits), torch.as_tensor(teacher)
    labels = torch.as_tensor(labels).long()
    return dict(accuracy_percent=float((logits.argmax(1) == labels).float().mean() * 100),
                teacher_agreement_percent=float((logits.argmax(1) == teacher.argmax(1)).float().mean() * 100),
                logit_mse=float((logits - teacher).square().mean()),
                cross_entropy=float(F.cross_entropy(logits, labels)))


def evaluation_batch(batch,device):
    batch=batch.clone().to(device)
    labels=batch.y.view(-1).long().cpu()
    del batch.y
    return batch,labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], required=True)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    p.add_argument('--author-repo', type=Path, default=Path('C:/code/gnn2snn-dcgs-reference'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--train-count', type=int, default=512)
    p.add_argument('--val-count', type=int, default=512)
    p.add_argument('--T', nargs='+', type=int, default=[1, 2])
    args = p.parse_args()
    if args.train_count < 1 or args.val_count < 1 or any(t not in (1, 2, 4, 8) for t in args.T):
        raise ValueError('invalid graph counts or T')
    if len(set(args.T)) != len(args.T):
        raise ValueError('duplicate T')
    campaign_start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261114)
    np.random.seed(20261114)
    dev = torch.device(args.device)
    runs = args.assets / 'takeover_20261112' / 'full_teacher_runs'
    run = runs / f'{args.dataset}_seed{args.seed}_full'
    calpath = resolve_calibration(args.assets, args.dataset, args.seed)
    cal = json.loads(calpath.read_text(encoding='utf-8'))
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    if cal['dataset'] != args.dataset or metadata['seed'] != args.seed:
        raise RuntimeError('calibration/teacher identity mismatch')
    checkpoint = run / 'best.pt'
    if sha(checkpoint) != cal['checkpoint_sha256']:
        raise RuntimeError('checkpoint/calibration mismatch')
    for filename, digest in cal['code_sha256'].items():
        if sha(LEGACY / filename) != digest:
            raise RuntimeError('calibration dependency mismatch: ' + filename)
    data_root = args.assets / 'data' / 'pyg_bench'
    for filename, digest in cal['data']['processed_files_sha256'].items():
        if sha(data_root / args.dataset / 'processed' / filename) != digest:
            raise RuntimeError('data hash mismatch: ' + filename)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['epoch'] != cal['teacher_epoch']:
        raise RuntimeError('teacher/calibration epoch mismatch')
    model = BenchmarkGCN(metadata['input_dim']).to(dev)
    model.load_state_dict(state['model'])
    model.eval().requires_grad_(False)
    folded = fold_model(model).eval().requires_grad_(False)
    train = GNNBenchmarkDataset(root=str(data_root), name=args.dataset, split='train')
    val = GNNBenchmarkDataset(root=str(data_root), name=args.dataset, split='val')
    expected = 55000 if args.dataset == 'MNIST' else 45000
    if len(train) != expected or len(val) != 5000:
        raise RuntimeError('official split count mismatch')
    ti = np.random.default_rng(20261114).permutation(len(train))[:args.train_count]
    vi = np.random.default_rng(20261115).permutation(len(val))[:args.val_count]
    if len(ti) != args.train_count or len(vi) != args.val_count or args.train_count > expected or args.val_count > 5000:
        raise ValueError('requested subset exceeds official split')
    collection_start = time.monotonic()
    collected = [[] for _ in range(6)]
    for batch in DataLoader(train[ti.tolist()], batch_size=32, shuffle=False):
        batch = batch.to(dev)
        del batch.y
        with torch.no_grad():
            _, activations = teacher_sites(folded, batch)
        for site, activation in zip(collected, activations):
            site.append(activation.detach().cpu())
    site_values = [torch.cat(site, dim=0) for site in collected]
    quantiles = channelwise_quantile_thresholds(site_values, quantile=0.999, floor=1e-6)
    collection_seconds = time.monotonic() - collection_start
    val_batches = list(DataLoader(val[vi.tolist()], batch_size=32, shuffle=False))

    import dcgs_fixed_seed_eval_v2 as ref
    mth, source_commit, provenance = ref.require_author_source(args.author_repo)
    first = val_batches[0].to(dev)
    with torch.no_grad():
        original = model(first)
        folded_logits, _ = teacher_sites(folded, first)
    fold_error = float((original - folded_logits).abs().max())
    if not torch.allclose(original, folded_logits, rtol=1e-5, atol=1e-4):
        raise RuntimeError('BN folding mismatch')

    snapshot_sources = [Path(__file__), HERE / 'percentile_core.py', OLD / 'calibration_core.py',
                        LEGACY / 'full_teacher_gcn.py', LEGACY / 'fold_teacher_bn.py',
                        LEGACY / 'dcgs_fixed_seed_eval_v2.py',
                        Path(args.author_repo) / 'neurons' / 'relu_neuron.py']
    source_dir = args.output / 'source'
    source_dir.mkdir()
    source_snapshots = {}
    for i, source in enumerate(snapshot_sources):
        dest = source_dir / f'{i:02d}_{source.name}'
        shutil.copyfile(source, dest)
        source_snapshots[str(source.resolve())] = sha(dest)

    thresholds_by_condition = {}
    equality = {}
    for condition, (scale, levels) in CONDITIONS.items():
        thresholds = [q * scale for q in quantiles]
        candidate = PercentileGCN(folded, thresholds, num_levels=levels).to(dev).eval()
        thresholds_by_condition[condition] = thresholds
        if levels == 8:
            with torch.no_grad():
                source_logits, _, _ = ref.dcgs_event_forward(folded, first, thresholds, 1, mth)
                local_logits, _ = candidate(first, 1)
            equality[condition] = float((source_logits - local_logits).abs().max())
            if not torch.equal(source_logits, local_logits):
                raise RuntimeError(f'author MTH_diff exact match failed: {equality[condition]}')
        else:
            equality[condition] = 'per-step exact equality tested against author MTH_diff(num_thresholds=4) in test_percentile.py'

    meta = dict(dataset=args.dataset, seed=args.seed, T=args.T,
                train_indices=ti.tolist(), val_indices=vi.tolist(), batch_size=32,
                quantile=0.999, quantile_floor=1e-6,
                conditions={name: dict(theta_scale=spec[0], num_thresholds=spec[1])
                            for name, spec in CONDITIONS.items()},
                trainable_parameters=0, optimizer='none', parameter_updates=0,
                checkpoint_sha256=sha(checkpoint), calibration_sha256=sha(calpath),
                calibration_path=str(calpath.resolve()), checkpoint_epoch=state['epoch'],
                source_commit=source_commit, source_provenance=provenance,
                fold_max_abs=fold_error, author_equality=equality,
                script_sha256=sha(__file__), core_sha256=sha(HERE / 'percentile_core.py'),
                dependency_sha256={name: sha(LEGACY / name) for name in
                                   ['full_teacher_gcn.py', 'fold_teacher_bn.py', 'dcgs_fixed_seed_eval_v2.py']},
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
                torch=torch.__version__, device=str(dev), gpu=torch.cuda.get_device_name() if dev.type == 'cuda' else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                train_graphs_official=len(train), val_graphs_official=len(val), test_loaded=False,
                qualification='GCN adaptation of author percentile/MTH_diff baseline; validation subset; exploratory screen',
                calibration_seconds=collection_seconds, teacher_frozen=True, teacher_retrained=False,
                source_snapshots=source_snapshots)
    write_json(args.output / 'metadata.json', meta)
    np.savez_compressed(args.output / 'percentile_thresholds.npz',
                        **{f'quantile_{i}': q.numpy() for i, q in enumerate(quantiles)},
                        **{f'{condition}_site_{i}': theta.cpu().numpy()
                           for condition, values in thresholds_by_condition.items()
                           for i, theta in enumerate(values)})

    rows = []
    for condition, thresholds in thresholds_by_condition.items():
        levels = CONDITIONS[condition][1]
        candidate = PercentileGCN(folded, thresholds, num_levels=levels).to(dev).eval()
        for ticks in args.T:
            parts = {key: [] for key in ('logits', 'teacher', 'labels', 'site_mse', 'indices')}
            offset = 0
            sync()
            eval_start = time.monotonic()
            with torch.no_grad():
                for batch in val_batches:
                    batch,labels = evaluation_batch(batch,dev)
                    teacher_logits, target_sites = teacher_sites(folded, batch)
                    logits, sites = candidate(batch, ticks)
                    site_errors = []
                    for i, (actual, target) in enumerate(zip(sites, target_sites)):
                        error = (actual - target).square().mean(-1, keepdim=True)
                        site_errors.append(global_mean_pool(error, batch.batch).flatten() if i < 4 else error.flatten())
                    count = labels.numel()
                    parts['logits'].append(logits.cpu())
                    parts['teacher'].append(teacher_logits.cpu())
                    parts['labels'].append(labels)
                    parts['site_mse'].append(torch.stack(site_errors, dim=1).cpu())
                    parts['indices'].append(torch.as_tensor(vi[offset:offset + count]))
                    offset += count
            sync()
            arrays = {key: torch.cat(values).numpy() for key, values in parts.items()}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise RuntimeError('non-finite validation output')
            stem = f'{condition}_T{ticks}'
            np.savez_compressed(args.output / f'{stem}.npz', **arrays)
            row = dict(condition=condition, T=ticks, num_thresholds=levels,
                       **metrics(arrays['logits'], arrays['teacher'], arrays['labels']),
                       teacher_accuracy_percent=float((arrays['teacher'].argmax(1) == arrays['labels']).mean() * 100),
                       val_graphs=len(vi), diagnostic_eval_seconds=time.monotonic() - eval_start)
            rows.append(row)
            print(json.dumps(row), flush=True)
            write_json(args.output / 'results.json', rows)

    for source in snapshot_sources:
        key = str(source.resolve())
        snapshot = next((source_dir / name for name in sorted(p.name for p in source_dir.iterdir())
                         if key.endswith(name[3:])), None)
        if snapshot is None or sha(source) != sha(snapshot):
            raise RuntimeError(f'source changed during run: {source}')
    write_json(args.output / 'DONE.json', dict(status='complete', rows=len(rows), test_loaded=False,
               parameter_updates=0, calibration_seconds=collection_seconds,
               total_elapsed_seconds=time.monotonic() - campaign_start))


if __name__ == '__main__':
    main()
