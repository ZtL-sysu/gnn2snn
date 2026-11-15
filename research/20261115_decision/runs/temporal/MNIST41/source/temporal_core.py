"""Causal first-site two-word selector; frozen original source dynamics."""
import numpy as np
import torch
from unittest.mock import patch
from run_objectives import core
from prefix_codec import prefix_bidirectional
from intervention import replace_event
from swap_predictor import graph_cache


class TemporalController(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ('nongraph', 'graph', 'shuffled', 'toppin'):
            raise ValueError(mode)
        self.mode = mode
        self.weight = torch.nn.Parameter(torch.zeros(4, 146))

    def forward(self, state, theta, cache):
        rho = (state['mem'].detach() / theta).clamp(-2, 2)
        if self.mode == 'nongraph':
            c0, c1 = rho.square(), rho.abs()
        else:
            if cache is None or cache.shape != (rho.shape[0], 2):
                raise ValueError('missing graph descriptor cache')
            c0, c1 = cache[:, :1], cache[:, 1:]
        w = self.weight
        return w[0] + w[1]*rho + w[2]*c0 + w[3]*c1


def temporal_cache(batch, mode, graph_ids):
    if mode == 'nongraph':
        return None
    d = graph_cache(batch.edge_index.detach().cpu(), batch.ptr.detach().cpu(), graph_ids, mode)
    keys = {'graph': ('mass', 'mass2'), 'shuffled': ('shuffled_mass', 'shuffled_mass2'),
            'toppin': ('indegree', 'mean_reciprocal_in_neighbor_indegree')}[mode]
    a = np.column_stack([d[k] for k in keys]).astype(np.float64)
    for lo, hi in zip(d['ptr'][:-1], d['ptr'][1:]):
        part = a[lo:hi]
        a[lo:hi] = (part-part.mean(0)) / np.maximum(part.std(0), 1e-6)
    return torch.as_tensor(a, dtype=batch.x.dtype, device=batch.x.device)


def temporal_forward(net, batch, T, controller=None, cache=None, graph_ids=None, instrument=True):
    if T < 1:
        raise ValueError('T must be positive')
    if controller is not None and controller.mode != 'nongraph' and cache is None:
        cache = temporal_cache(batch, controller.mode, graph_ids)
    original = core._mth_step
    calls = 0
    n, device = batch.num_graphs, batch.x.device
    trace = {k: torch.zeros(n, device=device, dtype=torch.int64)
             for k in ('events', 'deliveries', 'choice_count')}
    trace['amplitude'] = torch.zeros(n, device=device, dtype=torch.float64)
    trace['first_decode_error_max'] = torch.zeros(n, device=device)
    degree = torch.bincount(batch.edge_index[0], minlength=batch.num_nodes) if instrument else None
    first_chosen = expected_decode = expected_q2 = None

    def step(state, current, theta, t):
        nonlocal calls, first_chosen, expected_decode, expected_q2
        site = calls % 6
        calls += 1
        event = original(state, current, theta, t)
        if controller is not None and site == 0 and t == 1:
            alternative, eligible, diag = prefix_bidirectional(state, event, theta)
            score = controller(state, theta, cache)
            hard = (score > 0) & eligible
            # Parenthesized zero-valued surrogate preserves exact hard alphabet.
            soft = torch.sigmoid(score)
            gate = hard.to(event.dtype) + (soft-soft.detach())*eligible
            changed = event + (alternative-event).detach()*gate
            expected_decode = (event+diag['base_q2']/2).detach()
            expected_q2 = torch.where(hard, diag['selected_q2'], diag['base_q2'])
            replace_event(state, event, changed, 1)
            event = first_chosen = changed
            if instrument:
                trace['choice_count'].index_add_(0, batch.batch, hard.sum(1))
        if controller is not None and site == 0 and t == 2:
            if instrument:
                assert torch.count_nonzero(current) == 0, 'first-site next input is not zero'
                assert torch.equal(event, expected_q2), 'source/probe disagreement'
            # Project the surrogate onto the exactly preserved local prefix.
            # Forward values remain the natural source event; no event is forced.
            projected = event.detach() - 2*(first_chosen-first_chosen.detach())
            replace_event(state, event, projected, 2)
            event = projected
        if instrument:
            group = batch.batch if site < 4 else torch.arange(n, device=device)
            counts = (event.detach() != 0).sum(1)
            trace['events'].index_add_(0, group, counts)
            trace['amplitude'].index_add_(0, group, event.detach().abs().double().sum(1))
            if site < 3:
                trace['deliveries'].index_add_(0, group, counts*degree*(3-site))
        return event

    with patch.object(core, '_mth_step', step):
        z, sites = net(batch, T)
    assert calls == 6*T
    if controller is not None and T == 2 and instrument:
        errors = (sites[0].detach()-expected_decode).abs().max(1).values
        assert torch.count_nonzero(errors) == 0, 'first local T2 prefix changed'
        trace['first_decode_error_max'] = torch.stack([errors[lo:hi].max() for lo, hi in
            zip(batch.ptr[:-1].tolist(), batch.ptr[1:].tolist())])
    return z, sites, trace
