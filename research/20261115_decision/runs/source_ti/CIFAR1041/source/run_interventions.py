"""Bounded causal hard-event diagnostic; no fitted deployable method or test split."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

HERE = Path(__file__).resolve().parent
CRITICAL = HERE.parent / '20261114_critical'
sys.path.insert(0, str(CRITICAL))
import analyze_allocation as historical
import calibration_core as core
from objective import message_operator
from intervention import adjacent_amplitude, replace_event


def margin(z, cls):
    other = z.clone()
    other[torch.arange(len(z), device=z.device), cls] = -torch.inf
    return z.gather(1, cls[:, None]).flatten() - other.max(1).values


@torch.no_grad()
def measured_forward(net, batch, T, direction=0, selector='random'):
    """Alter one nonzero first-site event/graph; keep full state consistent."""
    original_step = core._mth_step
    trace = dict(call=0, events=torch.zeros(batch.num_graphs, device=batch.x.device),
                 amplitude=torch.zeros(batch.num_graphs, device=batch.x.device))

    def step(state, current, theta, t):
        site = trace['call'] % 6
        trace['call'] += 1
        spike = original_step(state, current, theta, t)
        if site == 0 and t == 1:
            trace['current'] = current.clone()
            trace['original'] = spike.clone()
            if direction:
                changed = spike.clone()
                adjacent = adjacent_amplitude(spike, theta, direction)
                movable = (spike != 0) & (adjacent != spike)
                coords = []
                for g, (lo, hi) in enumerate(zip(batch.ptr[:-1].tolist(), batch.ptr[1:].tolist())):
                    candidates = movable[lo:hi].flatten().nonzero().flatten()
                    if len(candidates):
                        if selector == 'residual':
                            scores = (spike[lo:hi] - current[lo:hi].relu()).abs().flatten()[candidates]
                            q = int(candidates[scores.argmax()])
                        else:
                            # Deterministic graph-coordinate choice, independent of labels/logits.
                            q = int(candidates[(g * 104729 + 41) % len(candidates)])
                        row, channel = lo + q // spike.shape[1], q % spike.shape[1]
                        changed[row, channel] = adjacent[row, channel]
                        coords.append([g, row-lo, channel])
                replace_event(state, spike, changed, t)
                spike = changed
                trace['coordinates'] = coords
            trace['first_event'] = spike.clone()
            trace['first_nonzero_preserved'] = bool(torch.equal(spike != 0, trace['original'] != 0))
        group = batch.batch if site < 4 else torch.arange(batch.num_graphs, device=spike.device)
        trace['events'].index_add_(0, group, (spike != 0).sum(1).float())
        trace['amplitude'].index_add_(0, group, spike.abs().sum(1))
        return spike

    with patch.object(core, '_mth_step', step):
        logits, sites = net(batch, T)
    error = trace['first_event'] - trace['current'].relu()
    msg = message_operator(net.model, batch, error)
    trace['message_mse'] = global_mean_pool(msg.square().mean(1, keepdim=True), batch.batch).flatten()
    return logits, sites, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['MNIST', 'CIFAR10'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=512)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261115)
    start = time.monotonic()
    run_dir = CRITICAL/'runs'/'allocation'/f'{args.dataset}41'
    meta, _, _, folded, loaded, _, data_root = historical.load_teacher(
        args.dataset, 41, historical.DEFAULT_ASSETS, run_dir, torch.device('cuda'))
    net = core.CalibratedGCN(folded, loaded.thresholds).cuda().eval()
    for dst, src in zip(net.offsets, loaded.offsets):
        dst.data.copy_(src.data)
    net.requires_grad_(False)
    val = GNNBenchmarkDataset(root=str(data_root), name=args.dataset, split='val')
    ids = meta['val_indices'][:args.count]
    rows, equality = [], []
    offset = 0
    for batch in DataLoader(val[ids], batch_size=32, shuffle=False):
        batch = batch.to('cuda')
        labels = batch.y.flatten().clone()
        del batch.y
        teacher, _ = core.teacher_sites(folded, batch)
        cls = teacher.argmax(1)
        base = {}
        for T in (1, 2):
            z, sites, trace = measured_forward(net, batch, T)
            exact, exact_sites = net(batch, T)
            assert torch.equal(z, exact)
            assert all(torch.equal(a,b) for a,b in zip(sites, exact_sites))
            archived = np.load(run_dir/f'baseline_T{T}.npz')
            assert np.allclose(z.cpu().numpy(), archived['logits'][offset:offset+len(z)], atol=1e-5, rtol=1e-5)
            base[T] = (z, trace)
            equality.append(dict(T=T, batch_offset=offset, exact_unpatched=True))
        for selector in ('deterministic_coordinate', 'residual'):
            for direction in (-1, 1):
                for T in (1, 2):
                    z, _, trace = measured_forward(net, batch, T, direction, selector)
                    bz, bt = base[T]
                    assert trace['first_nonzero_preserved']
                    coords={c[0]:c[1:] for c in trace.get('coordinates',[])}
                    for g in range(len(z)):
                        rows.append(dict(dataset=args.dataset, index=ids[offset+g], T=T,
                            actual_first_event_changed=g in coords,
                            changed_node=coords.get(g,[-1,-1])[0],changed_channel=coords.get(g,[-1,-1])[1],
                            selector=selector, direction=direction, teacher_class=int(cls[g]), label=int(labels[g]),
                            teacher_margin=float(margin(teacher,cls)[g]), baseline_margin=float(margin(bz,cls)[g]),
                            changed_margin=float(margin(z,cls)[g]), baseline_pred=int(bz[g].argmax()),
                            changed_pred=int(z[g].argmax()), baseline_message_mse=float(bt['message_mse'][g]),
                            changed_message_mse=float(trace['message_mse'][g]), baseline_events=int(bt['events'][g]),
                            changed_events=int(trace['events'][g]), baseline_amplitude=float(bt['amplitude'][g]),
                            changed_amplitude=float(trace['amplitude'][g])))
        offset += len(labels)
    with (args.output/'graphs.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    summary=[]
    for T in (1,2):
        r=[x for x in rows if x['T']==T]
        improved=[x for x in r if x['changed_message_mse'] < x['baseline_message_mse']-1e-12]
        summary.append(dict(T=T, interventions=len(r), message_improved=len(improved),
            message_improved_margin_worse=sum(x['changed_margin'] < x['baseline_margin']-1e-7 for x in improved),
            message_improved_prediction_changed=sum(x['baseline_pred']!=x['changed_pred'] for x in improved),
            message_improved_agreement_lost=sum(x['baseline_pred']==x['teacher_class'] and x['changed_pred']!=x['teacher_class'] for x in improved),
            message_improved_agreement_gained=sum(x['baseline_pred']!=x['teacher_class'] and x['changed_pred']==x['teacher_class'] for x in improved),
            nonzero_total_event_delta_min=min(x['changed_events']-x['baseline_events'] for x in r),
            nonzero_total_event_delta_max=max(x['changed_events']-x['baseline_events'] for x in r)))
    torch.cuda.synchronize()
    record=dict(dataset=args.dataset, seed=41, graph_count=len(ids), val_indices=ids,
        qualification='selection-exposed validation; repeated interventions within graph, not independent samples; diagnostics only',
        test_loaded=False, first_site_nonzero_events_preserved=True,
        total_event_equality_claim=False, teacher_access='offline diagnostic only; not a deployable policy',
        base_parameters='historical all_local T1 lr.001 shared at T1/T2', source_acceptance=equality,
        checkpoint_sha256=meta['checkpoint_sha256'], calibration_sha256=meta['calibration_sha256'],
        script_sha256=historical.sha(__file__), intervention_sha256=historical.sha(HERE/'intervention.py'),
        seconds=time.monotonic()-start, summary=summary)
    (args.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(seconds=record['seconds'], summary=summary)))


if __name__=='__main__':
    main()
