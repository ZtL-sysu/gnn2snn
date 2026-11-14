"""Experimental fork of frozen calibration_core: optional nearest eight-level first-site alphabet.
No baseline source modified; same differential recurrence, signed event semantics.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class CodebookGCN(nn.Module):
    """Pinned MTH_diff recurrence with six trainable channelwise initial states.

    ``folded_model`` is held frozen in eval mode. ``thresholds`` are six
    positive channel vectors for four node branches and two graph MLP sites.
    The returned decoded site values use the author's cumulative ``sum(e_t/t)``
    decoder, matching the classifier event accumulation.
    """

    def __init__(self, folded_model: nn.Module, thresholds: Sequence[torch.Tensor], first_book=None):
        super().__init__()
        if len(thresholds) != 6:
            raise ValueError("exactly six site threshold vectors are required")
        self.model = folded_model
        self.model.requires_grad_(False)
        self.model.eval()
        widths = [conv.out_channels for conv in self.model.convs]
        widths.extend([self.model.mlp[0].out_features, self.model.mlp[2].out_features])
        if len(self.model.convs) != 4:
            raise ValueError("folded model must have four graph-convolution sites")
        self.offsets = nn.ParameterList()
        self._threshold_names = []
        for i, (theta, width) in enumerate(zip(thresholds, widths)):
            value = torch.as_tensor(theta, dtype=torch.float32).detach().reshape(-1).clone()
            if value.numel() != width:
                raise ValueError(f"site {i} threshold width {value.numel()} != {width}")
            if not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError(f"site {i} thresholds must be finite and positive")
            name = f"threshold_{i}"
            self.register_buffer(name, value)
            self._threshold_names.append(name)
            self.offsets.append(nn.Parameter(torch.zeros(width, dtype=torch.float32)))

        book=torch.empty(0) if first_book is None else torch.as_tensor(first_book,dtype=torch.float32).detach().clone()
        if book.numel() and (book.shape!=(8,widths[0]) or not torch.isfinite(book).all() or (book<0).any() or (book[1:]>book[:-1]).any()):
            raise ValueError('first codebook must contain eight descending nonnegative channel levels')
        self.register_buffer('first_book',book)

    @property
    def thresholds(self):
        return [getattr(self, name) for name in self._threshold_names]

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, batch, T: int):
        if T not in (1, 2, 4, 8):
            raise ValueError("T must be one of 1, 2, 4, 8")
        if self.model.training:
            self.model.eval()
        device = batch.x.device
        dtype = torch.float32
        theta = [v.to(device=device, dtype=dtype) for v in self.thresholds]
        initial = [th * off.to(device=device, dtype=dtype)
                   for th, off in zip(theta, self.offsets)]
        raw = torch.cat([batch.x.float(), batch.pos.float()], dim=-1)
        n = raw.size(0)
        source, target = batch.edge_index
        deg_out = torch.bincount(source, minlength=n).clamp(min=1).to(dtype)
        deg_in = torch.bincount(target, minlength=n).clamp(min=1).to(dtype)
        edge_weight = deg_out[source].rsqrt() * deg_in[target].rsqrt()
        graph_count = int(batch.num_graphs)
        states = [_new_site_state(th, init, shape, device, dtype)
                  for th, init, shape in zip(theta[:4], initial[:4],
                                             [(n, w) for w in [t.numel() for t in theta[:4]]])]
        states.extend(_new_site_state(theta[i], initial[i], (graph_count, theta[i].numel()),
                                      device, dtype) for i in (4, 5))
        if self.first_book.numel():
            levels=self.first_book.to(device=device,dtype=dtype)
            states[0]['threshold_steps']=levels[:,None,:]
            smaller=torch.cat([levels[1:],torch.zeros_like(levels[:1])],0)
            states[0]['decision_steps']=((levels+smaller)*.5)[:,None,:]
        decoded = [None] * 6
        logits = None
        model = self.model

        for t in range(1, T + 1):
            first = t == 1
            raw_event = raw if first else torch.zeros_like(raw)
            h = F.linear(raw_event, model.embedding.weight,
                         model.embedding.bias if first else None)
            for i, conv in enumerate(model.convs):
                z = conv(h, batch.edge_index, edge_weight)
                if not first and conv.bias is not None:
                    z = z - conv.bias
                spike = _mth_step(states[i], z, theta[i], t)
                decoded[i] = spike / float(t) if decoded[i] is None else decoded[i] + spike / float(t)
                h = h + spike

            pooled = global_mean_pool(h, batch.batch)
            z = F.linear(pooled, model.mlp[0].weight,
                         model.mlp[0].bias if first else None)
            h = _mth_step(states[4], z, theta[4], t)
            decoded[4] = h / float(t) if decoded[4] is None else decoded[4] + h / float(t)

            z = F.linear(h, model.mlp[2].weight,
                         model.mlp[2].bias if first else None)
            h = _mth_step(states[5], z, theta[5], t)
            decoded[5] = h / float(t) if decoded[5] is None else decoded[5] + h / float(t)

            event_logits = F.linear(h, model.mlp[4].weight,
                                    model.mlp[4].bias if first else None)
            contribution = event_logits / float(t)
            logits = contribution if logits is None else logits + contribution
        return logits, decoded


def _new_site_state(theta, initial_mem, shape, device, dtype):
    return {
        "mem": initial_mem.expand(shape).clone(),
        "cumulative_out": torch.zeros(shape, device=device, dtype=dtype),
        "exp_in": torch.zeros(shape, device=device, dtype=dtype),
        "exp_out": torch.zeros(shape, device=device, dtype=dtype),
        "threshold_steps": theta.reshape(1, 1, -1) / (2 ** torch.arange(8, device=device, dtype=dtype)).reshape(8, 1, 1),
    }


def _mth_step(state, current, theta, t):
    """One active MTH_diff call; ``t`` is its one-based active tick index."""
    mem = state["mem"] + current + state["exp_in"] - state["exp_out"]
    cumulative = state["cumulative_out"] + state["exp_out"]
    thresholds = state["threshold_steps"]
    decisions=state.get('decision_steps',thresholds*.75)
    positive = (mem.unsqueeze(0) >= decisions).to(mem.dtype)
    negative = ((mem.unsqueeze(0) <= -decisions).to(mem.dtype)
                * (cumulative.unsqueeze(0) >= thresholds).to(mem.dtype))
    pos_true = positive.clone()
    neg_true = negative.clone()
    pos_true[1:] -= positive[:-1]
    neg_true[1:] -= negative[:-1]
    hard = (thresholds * pos_true).sum(dim=0) - (thresholds * neg_true).sum(dim=0)
    # Preserve exact hard forward values while supplying the requested ReLU
    # surrogate derivative to the offset parameters.
    soft = F.relu(mem)
    spike = hard + (soft - soft.detach())
    mem = mem - spike
    cumulative = cumulative + spike
    state["mem"] = mem
    state["cumulative_out"] = cumulative
    state["exp_in"] = state["exp_in"] + current / float(t)
    state["exp_out"] = state["exp_out"] + spike / float(t)
    return spike


def teacher_sites(folded_model: nn.Module, batch):
    """Return frozen-teacher logits and six target ReLU activations."""
    folded_model.eval()
    raw = torch.cat([batch.x.float(), batch.pos.float()], dim=-1)
    h = folded_model.embedding(raw)
    source, target = batch.edge_index
    n = h.shape[0]
    deg_out = torch.bincount(source, minlength=n).clamp(min=1).to(h.dtype)
    deg_in = torch.bincount(target, minlength=n).clamp(min=1).to(h.dtype)
    edge_weight = deg_out[source].rsqrt() * deg_in[target].rsqrt()
    sites = []
    for conv in folded_model.convs:
        branch = F.relu(conv(h, batch.edge_index, edge_weight))
        sites.append(branch)
        h = h + branch
    h = global_mean_pool(h, batch.batch)
    h = F.relu(folded_model.mlp[0](h))
    sites.append(h)
    h = F.relu(folded_model.mlp[2](h))
    sites.append(h)
    logits = folded_model.mlp[4](h)
    return logits, sites
