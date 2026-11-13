"""Full-training affine-input statistics; no candidate or held-out execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from activation_histogram import ActivationHistogram
from gated_frozen_gate_calibrate import (LabelFreeGraphView, validate_teacher_run,
                                         build_teacher, resolve_teacher_source)

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / 'data' / 'pyg_bench'
TEACHER_DIR = HERE / 'full_teacher_runs' / 'gated_cifar10_seed41_full_v1'
PROTOCOL = HERE / 'affine-base-fulltrain-protocol-v1.md'
SITES = tuple(k for i in range(4) for k in (f'raw_node_{i}', f'raw_edge_{i}')) + tuple(
    k for i in range(1, 4) for k in (f'branch_cumsum_node_{i}', f'branch_cumsum_edge_{i}'))
DEPENDENCIES = ('gated_affine_cal_v1.py', 'check_gated_affine_cal_v1.py',
                'activation_histogram.py', 'gated_teacher_model.py',
                'gated_frozen_gate_calibrate.py', 'affine-base-fulltrain-protocol-v1.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def output_path(path):
    path = Path(path).resolve()
    if path.parent != HERE.resolve():
        raise ValueError('output must be directly inside takeover directory')
    if path.exists():
        raise FileExistsError('refusing existing/partial output')
    return path


def write_exclusive(path, document):
    path = output_path(path)
    payload = json.dumps(document, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as f:
        f.write(payload)


def validate_eval(model, batch):
    if any(m.training for m in model.modules()):
        raise ValueError('all model submodules must be eval')
    if 'y' in batch and batch.y is not None:
        raise ValueError('labels prohibited')
    for key in ('x', 'pos', 'edge_attr'):
        v = getattr(batch, key, None)
        if v is not None and not bool(torch.isfinite(v).all()):
            raise ValueError('nonfinite input ' + key)
    if not all(l.residual and l.use_batch_norm for l in model.layers):
        raise ValueError('identity residual and BN required')


@torch.no_grad()
def capture_inputs(model, batch):
    """Observe native teacher once, accumulate branches without state subtraction."""
    validate_eval(model, batch)
    values, calls, handles = {}, {}, []
    cumulative = {'node': None, 'edge': None}

    def put(key, value):
        calls[key] = calls.get(key, 0) + 1
        if calls[key] != 1:
            raise RuntimeError('duplicate site capture ' + key)
        if value.ndim != 2 or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise RuntimeError('invalid nonnegative captured input ' + key)
        values[key] = value.detach()

    def pre(index, args):
        _, h, e = args
        for kind, v in (('node', h), ('edge', e)):
            put(f'raw_{kind}_{index}', v.abs())
            if index:
                if cumulative[kind] is None:
                    raise RuntimeError('missing earlier branch')
                put(f'branch_cumsum_{kind}_{index}', cumulative[kind])

    def post(kind, out):
        branch = torch.relu(out.detach())
        if not bool(torch.isfinite(branch).all()):
            raise RuntimeError('nonfinite BN branch')
        cumulative[kind] = branch if cumulative[kind] is None else cumulative[kind] + branch

    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_pre_hook(lambda _m, args, j=i: pre(j, args)))
        handles.append(layer.bn_node_h.register_forward_hook(lambda _m, _a, out: post('node', out)))
        handles.append(layer.bn_node_e.register_forward_hook(lambda _m, _a, out: post('edge', out)))
    try:
        logits = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    if set(values) != set(SITES) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError('incomplete captures or nonfinite teacher logits')
    return values


@torch.no_grad()
def calibrate_loader(model, loader, device, expected=45000, width=70, batch_size=128):
    maxima, hists = {}, {}
    graph_counts, batch_counts, row_counts, totals = [], [], [], []
    for pass_index in range(2):
        graphs = batches = nodes = edges = 0
        rows = {s: 0 for s in SITES}
        last = 0
        if pass_index:
            hists = {s: ActivationHistogram(maxima[s], bins=512) for s in SITES}
        for batch in loader:
            batch = batch.to(device)
            values = capture_inputs(model, batch)
            last = int(batch.num_graphs)
            graphs += last
            batches += 1
            nodes += int(batch.num_nodes)
            edges += int(batch.num_edges)
            for s, value in values.items():
                nrows = int(batch.num_nodes if '_node_' in s else batch.num_edges)
                if value.shape != (nrows, width):
                    raise RuntimeError('captured shape mismatch ' + s)
                rows[s] += nrows
                if pass_index == 0:
                    mx = value.amax(0)
                    maxima[s] = mx if s not in maxima else torch.maximum(maxima[s], mx)
                else:
                    hists[s].update(value)
            if batches % 50 == 0:
                print(json.dumps({'progress': 'affine_cal', 'pass': pass_index+1,
                                  'graphs': graphs, 'batches': batches}), flush=True)
        if (graphs != expected or batches != math.ceil(expected/batch_size) or
                last != (expected % batch_size or batch_size)):
            raise RuntimeError('incomplete calibration coverage or partial batch')
        graph_counts.append(graphs)
        batch_counts.append(batches)
        row_counts.append(rows)
        totals.append({'nodes': nodes, 'directed_edges': edges, 'final_batch_graphs': last})
    if row_counts[0] != row_counts[1] or totals[0] != totals[1]:
        raise RuntimeError('pass coverage mismatch')
    sites, thresholds = {}, {}
    for s in SITES:
        hist = hists[s]
        th, glob = hist.thresholds(.999, floor=1e-6)
        if hist.rows != row_counts[0][s] or not bool((hist.channel.sum(1) == hist.rows).all()):
            raise RuntimeError('histogram rows mismatch')
        thresholds[s] = th.cpu().tolist()
        sites[s] = {'rows': hist.rows, 'channels': width, 'maxima': maxima[s].cpu().tolist(),
                    'channel_counts': hist.channel.cpu().tolist(),
                    'global_counts': hist.global_counts.cpu().tolist(),
                    'per_channel_thresholds': thresholds[s], 'global_threshold': float(glob)}
    return {'passes': 2, 'pass_graph_counts': graph_counts, 'pass_batch_counts': batch_counts,
            'pass_rows_per_site': row_counts, 'pass_totals': totals,
            'bins': 512, 'quantile': .999, 'inactive_channel_floor': 1e-6,
            'activation_rows_per_site': row_counts[0], 'sites': sites, 'channel_thresholds': thresholds}


def load_teacher(device):
    run, meta, done, checkpoint, digest = validate_teacher_run(TEACHER_DIR)
    if meta.get('seed') != 41:
        raise RuntimeError('expected seed41 teacher')
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get('metadata') != meta:
        raise RuntimeError('checkpoint metadata mismatch')
    model = build_teacher(state, device)
    if (len(model.layers), model.embedding_h.in_features, model.embedding_h.out_features,
            model.edge_dim) != (4, 5, 70, 1):
        raise RuntimeError('wrong architecture')
    return model, {'run': str(run.resolve()), 'checkpoint_sha256': digest,
                   'metadata': meta, 'done': done, 'epoch': int(state['epoch'])}


def input_hashes():
    paths = [HERE/n for n in DEPENDENCIES]
    paths += [TEACHER_DIR/n for n in ('metadata.json', 'done.json', 'best.pt')]
    meta = json.loads((TEACHER_DIR/'metadata.json').read_text(encoding='utf-8'))
    paths += [resolve_teacher_source(HERE, n) for n in meta['provenance']['source_sha256']]
    paths += [DATA_ROOT/'CIFAR10'/'processed'/n for n in ('pre_filter.pt', 'pre_transform.pt', 'train_data.pt')]
    return {str(p.resolve()): sha(p) for p in paths}


def require_unchanged(start):
    current = {p: sha(p) for p in start}
    if current != start:
        raise RuntimeError('bound input source/data/checkpoint changed')
    return current


def validate_statistics(cal, width=70, expected=45000, batch_size=128):
    if (cal.get('passes') != 2 or cal.get('bins') != 512 or cal.get('quantile') != .999 or
            cal.get('inactive_channel_floor') != 1e-6 or cal.get('pass_graph_counts') != [expected]*2 or
            cal.get('pass_batch_counts') != [math.ceil(expected/batch_size)]*2):
        raise RuntimeError('incomplete or incompatible calibration')
    if set(cal['sites']) != set(SITES) or set(cal['channel_thresholds']) != set(SITES):
        raise RuntimeError('wrong site set')
    totals = cal['pass_totals']
    if len(totals) != 2 or totals[0] != totals[1] or totals[0]['final_batch_graphs'] != (expected % batch_size or batch_size):
        raise RuntimeError('pass totals mismatch')
    for s in SITES:
        v = cal['sites'][s]
        rows = totals[0]['nodes' if '_node_' in s else 'directed_edges']
        if v['rows'] != rows or v['channels'] != width or rows < 1:
            raise RuntimeError('site row/width mismatch')
        if cal['pass_rows_per_site'] != [cal['activation_rows_per_site']]*2 or cal['activation_rows_per_site'][s] != rows:
            raise RuntimeError('pass row mismatch')
        counts = torch.as_tensor(v['channel_counts'])
        mx = torch.tensor(v['maxima'], dtype=torch.float64)
        th = torch.tensor(v['per_channel_thresholds'], dtype=torch.float64)
        if (counts.shape != (width,512) or counts.dtype != torch.int64 or bool((counts<0).any()) or
                not bool((counts.sum(1)==rows).all()) or mx.shape != (width,) or th.shape != (width,) or
                not bool(torch.isfinite(mx).all()) or bool((mx<0).any()) or
                not bool(torch.isfinite(th).all()) or bool((th<=0).any())):
            raise RuntimeError('invalid histogram/maxima/thresholds')
        target = math.ceil(rows*.999)
        hit = (counts.cumsum(1)>=target).to(torch.int64).argmax(1)
        wanted = ((hit+1)*mx/512).clamp(min=1e-6)
        if not torch.allclose(wanted, th, rtol=2e-7, atol=1e-10) or v['per_channel_thresholds'] != cal['channel_thresholds'][s]:
            raise RuntimeError('threshold quantile mismatch')
        glob = v['global_counts']
        if len(glob) != 512 or any(type(x) is not int or x<0 for x in glob) or sum(glob) != rows*width:
            raise RuntimeError('invalid global histogram')


def validate_calibration_document(path, teacher_info=None):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if (doc.get('status') != 'complete' or doc.get('dataset') != 'CIFAR10' or doc.get('seed') != 41 or
            doc.get('scope') != 'all_45000_official_train_graphs_label_free_two_pass'):
        raise RuntimeError('wrong calibration scope/teacher or partial result')
    if teacher_info is not None and doc.get('teacher') != teacher_info:
        raise RuntimeError('wrong calibration teacher')
    if not doc.get('input_hashes_start') or doc.get('input_hashes_end') != doc['input_hashes_start']:
        raise RuntimeError('calibration bound input mismatch')
    require_unchanged(doc['input_hashes_start'])
    validate_statistics(doc['calibration'])
    return doc, sha(path)


def profile_thresholds(doc, mode, scale):
    if mode not in ('generic','first_analog','centered') or scale not in (1,8):
        raise ValueError('unsupported mode/scale')
    family = 'branch_cumsum' if mode == 'centered' else 'raw'
    return [torch.tensor(doc['calibration']['channel_thresholds'][f'{family}_{kind}_{i}'])*scale
            for i in range(0 if mode=='generic' else 1,4) for kind in ('node','edge')]


def run(output, device):
    output_path(output)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    dev = torch.device(device)
    model, teacher = load_teacher(dev)
    start = input_hashes()
    data = GNNBenchmarkDataset(root=str(DATA_ROOT), name='CIFAR10', split='train')
    if len(data) != 45000:
        raise RuntimeError('full official train required')
    loader = DataLoader(LabelFreeGraphView(data), batch_size=128, shuffle=False, num_workers=0)
    timer = time.perf_counter()
    cal = calibrate_loader(model, loader, dev)
    validate_statistics(cal)
    end = require_unchanged(start)
    doc = {'status':'complete','dataset':'CIFAR10','seed':41,
           'scope':'all_45000_official_train_graphs_label_free_two_pass', 'teacher':teacher,
           'calibration':cal, 'input_hashes_start':start, 'input_hashes_end':end,
           'elapsed_seconds':time.perf_counter()-timer,
           'quantity_definition':{'raw':'abs teacher layer input',
                                  'branch_cumsum':'sum earlier post-BN ReLU branches; not subtraction'},
           'runtime':{'device':str(dev),'torch':torch.__version__, 'batch_size':128,
                      'deterministic':torch.are_deterministic_algorithms_enabled(),
                      'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                      'labels_passed':False,'validation_or_test_opened':False}}
    write_exclusive(output, doc)
    print(json.dumps({'status':'complete','output':str(output),'graphs':45000}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    run(args.output,args.device)
