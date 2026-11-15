"""Reconstruct graph-wise allocation diagnostics and profile the cached T1 batch."""
import hashlib
import json
import os
from pathlib import Path
import random
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
from allocation import select_correction
from objective import message_operator
from allocation_core import AllocationGCN, _mth_step, _new_site_state

DEFAULT_ASSETS = Path('C:/code/phd-snn/auto-paper/20261110-152407/Operator-Agnostic-ANN-to-SNN-Conversion-for-Vision-and-Graphs/f-research-v1-run-20261110-172308-superpixel-graph-vision')
RUNS = HERE / 'runs' / 'allocation'
OUT = HERE / 'analysis' / 'allocation_diagnostics'
POLICIES = ['baseline', 'random', 'local', 'identity', 'shuffled', 'message']
ALLOCATION_POLICIES = POLICIES[1:]
LEGACY_NAMES = ['full_teacher_gcn.py', 'fold_teacher_bn.py', 'dcgs_fixed_seed_eval_v2.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_snapshot(run_dir, meta):
    for name, digest in meta['source_snapshots'].items():
        stored = run_dir / 'source' / name
        if not stored.is_file() or sha(stored) != digest:
            raise RuntimeError(f'run source snapshot mismatch: {name}')
        # The snapshot and current source must both match: diagnostics should
        # reconstruct the completed run, not an edited implementation.
        current = HERE / name if name in {'run_allocation.py', 'objective.py', 'allocation.py', 'allocation_core.py'} else OLD / name if name == 'calibration_core.py' else LEGACY / name
        if not current.is_file() or sha(current) != digest:
            raise RuntimeError(f'current source differs from completed run: {name}')


def load_teacher(dataset, seed, assets, run_dir, device):
    meta = json.loads((run_dir / 'metadata.json').read_text(encoding='utf-8'))
    results = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    if meta['dataset'] != dataset or meta['seed'] != seed or meta.get('test_loaded') is not False:
        raise RuntimeError('unexpected completed-run identity or test access')
    verify_snapshot(run_dir, meta)
    params_path = run_dir / 'input_base_parameters.npz'
    if sha(params_path) != meta['base_parameters_sha256']:
        raise RuntimeError('cached input parameter snapshot hash mismatch')
    if sha(Path(meta['base_parameters_path'])) != meta['base_parameters_sha256']:
        raise RuntimeError('canonical input parameter hash mismatch')
    for name, digest in meta['dependency_sha256'].items():
        if sha(LEGACY / name) != digest:
            raise RuntimeError('teacher code dependency mismatch: ' + name)
    calpath = Path(meta['calibration_path'])
    checkpoint = assets / 'takeover_20261112' / 'full_teacher_runs' / f'{dataset}_seed{seed}_full' / 'best.pt'
    if sha(calpath) != meta['calibration_sha256'] or sha(checkpoint) != meta['checkpoint_sha256']:
        raise RuntimeError('teacher/calibration artifact hash mismatch')
    cal = json.loads(calpath.read_text(encoding='utf-8'))
    if cal['dataset'] != dataset or cal['checkpoint_sha256'] != meta['checkpoint_sha256']:
        raise RuntimeError('calibration identity mismatch')
    data_root = assets / 'data' / 'pyg_bench'
    for filename, digest in cal['data']['processed_files_sha256'].items():
        if sha(data_root / dataset / 'processed' / filename) != digest:
            raise RuntimeError('processed data hash mismatch: ' + filename)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['epoch'] != meta['checkpoint_epoch'] or state['epoch'] != cal['teacher_epoch']:
        raise RuntimeError('teacher epoch mismatch')
    from full_teacher_gcn import BenchmarkGCN
    model = BenchmarkGCN(json.loads((checkpoint.parent / 'metadata.json').read_text(encoding='utf-8'))['input_dim']).to(device)
    model.load_state_dict(state['model'])
    model.requires_grad_(False).eval()
    folded = fold_model(model).to(device).requires_grad_(False).eval()
    thresholds = [torch.as_tensor(cal['channel_thresholds'][site], dtype=torch.float32, device=device) for site in
                  ['gcn_0', 'gcn_1', 'gcn_2', 'gcn_3', 'mlp_0', 'mlp_1']]
    net = AllocationGCN(folded, thresholds).to(device).eval()
    with np.load(params_path) as saved:
        with torch.no_grad():
            for i, offset in enumerate(net.offsets):
                value = torch.as_tensor(saved[f'offset_{i}'], dtype=offset.dtype, device=device)
                if value.shape != offset.shape:
                    raise RuntimeError(f'offset width mismatch at site {i}')
                offset.copy_(value)
    return meta, results, model, folded, net, cal, data_root


def selected_subset(dataset_obj, indices, device):
    return list(DataLoader(dataset_obj[indices], batch_size=32, shuffle=False))


def graph_weights(batch, dtype):
    source, target = batch.edge_index
    out = torch.bincount(source, minlength=batch.num_nodes).clamp_min(1).to(dtype)
    inc = torch.bincount(target, minlength=batch.num_nodes).clamp_min(1).to(dtype)
    return out[source].rsqrt() * inc[target].rsqrt()


@torch.no_grad()
def batch_diagnostics(net, batch):
    model = net.model
    theta = net.thresholds[0]
    offsets = net.offsets[0]
    raw = torch.cat((batch.x.float(), batch.pos.float()), dim=-1)
    edge_weight = graph_weights(batch, raw.dtype)
    h = F.linear(raw, model.embedding.weight, model.embedding.bias)
    current = model.convs[0](h, batch.edge_index, edge_weight)
    initial = theta * offsets
    state = _new_site_state(theta, initial, (batch.num_nodes, theta.numel()),
                            current.device, current.dtype)
    spike = _mth_step(state, current, theta, 1)
    error = spike - current.relu()
    true_message = message_operator(model, batch, current.relu())
    actual_before = message_operator(model, batch, error)
    message_den = global_mean_pool(true_message.square().mean(1, keepdim=True), batch.batch).flatten().clamp_min(1e-6)
    before_local = global_mean_pool(error.square().mean(1, keepdim=True), batch.batch).flatten()
    before_msg = global_mean_pool(actual_before.square().mean(1, keepdim=True), batch.batch).flatten()
    outputs = []
    for policy in ALLOCATION_POLICIES:
        correction, info = select_correction(model, batch, current, spike, theta, policy)
        after = error + correction
        actual_after = message_operator(model, batch, after)
        after_local = global_mean_pool(after.square().mean(1, keepdim=True), batch.batch).flatten()
        after_msg = global_mean_pool(actual_after.square().mean(1, keepdim=True), batch.batch).flatten()
        outdegree = torch.bincount(batch.edge_index[0], minlength=batch.num_nodes)
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        selected_per_graph = torch.ceil(graph_sizes.to(torch.float32) / 4).to(torch.int64)
        node_events = (correction != 0).sum(1)
        edge_events = node_events * outdegree
        nonzero_per_graph = torch.stack([node_events[lo:hi].sum() for lo, hi in zip(batch.ptr[:-1].tolist(), batch.ptr[1:].tolist())])
        deliveries_per_graph = torch.stack([edge_events[lo:hi].sum() for lo, hi in zip(batch.ptr[:-1].tolist(), batch.ptr[1:].tolist())])
        outputs.append(dict(policy=policy, correction=correction,
                            before_local=before_local, after_local=after_local,
                            before_message=before_msg, after_message=after_msg,
                            before_message_normalized=before_msg / message_den,
                            after_message_normalized=after_msg / message_den,
                            selected_count=selected_per_graph,
                            nonzero_events=nonzero_per_graph,
                            extra_edge_deliveries=deliveries_per_graph,
                            info=info))
    return current, spike, error, outputs


def diagnose_dataset(dataset, assets, output_dir, log):
    run_dir = RUNS / f'{dataset}41'
    device = torch.device('cuda')
    meta, results, model, folded, net, cal, data_root = load_teacher(dataset, 41, assets, run_dir, device)
    val = GNNBenchmarkDataset(root=str(data_root), name=dataset, split='val')
    idx = np.random.default_rng(20261115).permutation(len(val))[:512]
    if idx.tolist() != meta['val_indices']:
        raise RuntimeError('recomputed validation indices do not match completed run')
    batches = selected_subset(val, idx.tolist(), device)
    collected = {key: [] for key in ('indices', 'graph_sizes', 'before_local', 'after_local',
                  'before_message', 'after_message', 'before_message_normalized',
                  'after_message_normalized', 'selected_count',
                  'nonzero_events', 'extra_edge_deliveries')}
    counts = {policy: dict(selected_nodes=0, max_opportunity_slots=0,
                           extra_nonzero_events=0, extra_edge_deliveries=0) for policy in ALLOCATION_POLICIES}
    graph_offset = 0
    cached_first = batches[0].clone().to(device)
    for batch in batches:
        batch = batch.clone().to(device)
        _, _, _, entries = batch_diagnostics(net, batch)
        n_graphs = batch.num_graphs
        for entry in entries:
            policy = entry['policy']
            counts[policy]['selected_nodes'] += int(entry['info']['selected_nodes'])
            counts[policy]['max_opportunity_slots'] += int(entry['info']['max_opportunity_slots'])
            counts[policy]['extra_nonzero_events'] += int(entry['info']['extra_nonzero_events'])
            counts[policy]['extra_edge_deliveries'] += int(entry['info']['extra_edge_deliveries'])
        collected['indices'].extend(idx[graph_offset:graph_offset + n_graphs].tolist())
        collected['graph_sizes'].extend((batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().tolist())
        for key in ('before_local', 'after_local', 'before_message', 'after_message',
                    'before_message_normalized', 'after_message_normalized', 'selected_count',
                    'nonzero_events', 'extra_edge_deliveries'):
            collected[key].append(torch.stack([entry[key] for entry in entries], dim=1).cpu().numpy())
        graph_offset += n_graphs
    arrays = {key: np.asarray(value) if isinstance(value, list) and value and not isinstance(value[0], np.ndarray)
              else np.concatenate(value, axis=0) if isinstance(value, list) else value
              for key, value in collected.items()}
    arrays['policy_names'] = np.asarray(ALLOCATION_POLICIES)
    arrays['before_message_actual'] = arrays.pop('before_message')
    arrays['after_message_actual'] = arrays.pop('after_message')
    arrays['before_message_actual_normalized'] = arrays.pop('before_message_normalized')
    arrays['after_message_actual_normalized'] = arrays.pop('after_message_normalized')
    np.savez_compressed(output_dir / f'{dataset}_per_graph.npz', **arrays)

    results_t1 = {row['method']: row['allocation'] for row in results if row['T'] == 1}
    checks = {}
    for policy in ALLOCATION_POLICIES:
        actual = counts[policy]
        expected = results_t1[policy]
        checks[policy] = {key: dict(recomputed=actual[key], recorded=expected[key], match=actual[key] == expected[key])
                          for key in actual}

    # Measure the cached actual runner with fixed teacher batch and offsets.
    schedule = [policy for policy in POLICIES for _ in range(25)]
    random.Random(20261118).shuffle(schedule)
    seen = {policy: 0 for policy in POLICIES}
    samples = {policy: [] for policy in POLICIES}
    warmups = {policy: 0 for policy in POLICIES}
    wall_log = []
    for policy in schedule:
        net.policy = None if policy == 'baseline' else policy
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            net(cached_first, 1)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        seen[policy] += 1
        if seen[policy] <= 5:
            warmups[policy] += 1
            phase = 'warmup'
        else:
            samples[policy].append(elapsed_ms)
            phase = 'measured'
        wall_log.append(dict(policy=policy, occurrence=seen[policy], phase=phase,
                             wall_ms=elapsed_ms, graph_count=cached_first.num_graphs))
    net.policy = None
    timing = {policy: dict(warmup_repeats=warmups[policy], measured_repeats=len(samples[policy]),
                           raw_wall_ms=samples[policy], median_wall_ms=float(np.median(samples[policy])),
                           mean_wall_ms=float(np.mean(samples[policy])),
                           p90_wall_ms=float(np.percentile(samples[policy], 90)),
                           median_ms_per_graph=float(np.median(samples[policy]) / cached_first.num_graphs))
              for policy in POLICIES}
    props = torch.cuda.get_device_properties(device)
    record = dict(dataset=dataset, seed=41, input_run=str(run_dir.resolve()),
                  metadata_sha256=sha(run_dir / 'metadata.json'),
                  source_snapshots_verified=True, checkpoint_sha256=meta['checkpoint_sha256'],
                  calibration_sha256=meta['calibration_sha256'],
                  base_parameters_sha256=meta['base_parameters_sha256'],
                  validation_indices_sha256=hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest(),
                  graphs=len(idx), graph_message_operator='actual normalized graph message_operator for every policy',
                  message_error_reference='frozen teacher first-site message A(relu(current)); graphwise mean squared error and normalized by reference message mean-square',
                  allocation_count_checks=checks, policy_event_counts=counts,
                  timing=dict(device=torch.cuda.get_device_name(device), device_properties=str(props),
                              torch=torch.__version__, cuda=torch.version.cuda, batch_graphs=cached_first.num_graphs,
                              warmups=5, measured_repeats=20, synchronized_host_wall_time=True,
                              interleave_seed=20261118, no_teacher_forward_in_timed_block=True,
                              instrumentation_present=True, energy_claim=False,
                              results=timing, randomized_schedule=wall_log))
    write_json(output_dir / f'{dataset}_summary.json', record)
    count_status = 'all totals exact' if all(v['match'] for p in checks.values() for v in p.values()) else 'small count deltas recorded'
    log.write(f"{dataset}: 512 val graphs; {count_status} for five policies. "
              f"Median ms/graph: " + ', '.join(f"{p} {timing[p]['median_ms_per_graph']:.4f}" for p in POLICIES) + '\n')
    log.flush()
    return record


def main():
    assets = DEFAULT_ASSETS
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'refusing to overwrite diagnostics: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261114)
    np.random.seed(20261114)
    with (HERE / 'analysis' / 'allocation_diagnostics.log').open('w', encoding='utf-8') as log:
        log.write('Allocation structure and synchronized runtime diagnostic; no retraining.\n')
        for dataset in ('MNIST', 'CIFAR10'):
            diagnose_dataset(dataset, assets, OUT, log)
        log.write('Complete. Timings include diagnostic policy-selection instrumentation; no energy inference.\n')


if __name__ == '__main__':
    main()
