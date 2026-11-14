"""Fixed per-channel percentile thresholds with the pinned MTH_diff recurrence."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


def channelwise_quantile_thresholds(site_activations, quantile=0.999, floor=1e-6):
    """Compute six independent channelwise quantiles over concatenated examples."""
    if len(site_activations) != 6:
        raise ValueError("exactly six post-ReLU site activation tensors are required")
    if not (0.0 < quantile < 1.0) or floor <= 0:
        raise ValueError("quantile must be in (0,1), floor must be positive")
    thresholds = []
    for site in site_activations:
        values = torch.as_tensor(site, dtype=torch.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("each site tensor must have shape [observations, channels]")
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("post-ReLU activations must be finite and nonnegative")
        thresholds.append(torch.quantile(values, quantile, dim=0).clamp_min(floor).detach())
    return thresholds


def _new_state(theta, shape, num_levels, device, dtype):
    steps = theta.reshape(1, 1, -1) / (2 ** torch.arange(num_levels, device=device, dtype=dtype)).reshape(num_levels, 1, 1)
    return dict(mem=torch.zeros(shape, device=device, dtype=dtype),
                cumulative_out=torch.zeros(shape, device=device, dtype=dtype),
                exp_in=torch.zeros(shape, device=device, dtype=dtype),
                exp_out=torch.zeros(shape, device=device, dtype=dtype),
                threshold_steps=steps)


def source_mth_diff_step(current, theta, state, tick, num_levels):
    """One author-source MTH_diff step; the dict also exposes state for tests."""
    if tick < 1 or num_levels not in (4, 8):
        raise ValueError("tick must be positive and num_levels must be 4 or 8")
    dtype, device = current.dtype, current.device
    theta = torch.as_tensor(theta, device=device, dtype=dtype)
    if state is None:
        state = _new_state(theta, current.shape, num_levels, device, dtype)
    mem = state["mem"] + current + state["exp_in"] - state["exp_out"]
    cumulative = state["cumulative_out"] + state["exp_out"]
    thresholds = state["threshold_steps"]
    positive = (mem.unsqueeze(0) >= thresholds * 0.75).to(dtype)
    negative = ((mem.unsqueeze(0) <= -thresholds * 0.75).to(dtype)
                * (cumulative.unsqueeze(0) >= thresholds).to(dtype))
    pos_true, neg_true = positive.clone(), negative.clone()
    pos_true[1:] -= positive[:-1]
    neg_true[1:] -= negative[:-1]
    spike = (thresholds * pos_true).sum(dim=0) - (thresholds * neg_true).sum(dim=0)
    mem = mem - spike
    cumulative = cumulative + spike
    state["mem"], state["cumulative_out"] = mem, cumulative
    state["exp_in"] = state["exp_in"] + current / float(tick)
    state["exp_out"] = state["exp_out"] + spike / float(tick)
    state["_reference_mem"] = mem
    state["_reference_cumulative"] = cumulative
    return spike, state


class PercentileGCN(nn.Module):
    """Frozen GCN with fixed per-site thresholds and no trainable parameters."""

    def __init__(self, folded_model: nn.Module, thresholds: Sequence[torch.Tensor], num_levels=8):
        super().__init__()
        if len(thresholds) != 6 or num_levels not in (4, 8):
            raise ValueError("six site thresholds and 4 or 8 levels are required")
        self.model = folded_model
        self.model.requires_grad_(False)
        self.model.eval()
        widths = [conv.out_channels for conv in self.model.convs]
        widths.extend([self.model.mlp[0].out_features, self.model.mlp[2].out_features])
        if len(self.model.convs) != 4:
            raise ValueError("folded model must have four graph-convolution sites")
        self.num_levels = int(num_levels)
        self._threshold_names = []
        for i, (theta, width) in enumerate(zip(thresholds, widths)):
            value = torch.as_tensor(theta, dtype=torch.float32).detach().reshape(-1).clone()
            if value.numel() != width or not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError(f"site {i} thresholds must be finite positive channel vectors")
            name = f"threshold_{i}"
            self.register_buffer(name, value)
            self._threshold_names.append(name)

    @property
    def thresholds(self):
        return [getattr(self, name) for name in self._threshold_names]

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, batch, T: int):
        if T not in (1, 2, 4, 8):
            raise ValueError("T must be one of 1, 2, 4, 8")
        device, dtype = batch.x.device, torch.float32
        theta = [v.to(device=device, dtype=dtype) for v in self.thresholds]
        raw = torch.cat([batch.x.float(), batch.pos.float()], dim=-1)
        n = raw.size(0)
        source, target = batch.edge_index
        deg_out = torch.bincount(source, minlength=n).clamp(min=1).to(dtype)
        deg_in = torch.bincount(target, minlength=n).clamp(min=1).to(dtype)
        edge_weight = deg_out[source].rsqrt() * deg_in[target].rsqrt()
        count = int(batch.num_graphs)
        shapes = [(n, th.numel()) for th in theta[:4]] + [(count, theta[i].numel()) for i in (4, 5)]
        states = [_new_state(th, shape, self.num_levels, device, dtype)
                  for th, shape in zip(theta, shapes)]
        decoded = [None] * 6
        logits = None
        model = self.model
        for tick in range(1, T + 1):
            first = tick == 1
            raw_event = raw if first else torch.zeros_like(raw)
            h = F.linear(raw_event, model.embedding.weight,
                         model.embedding.bias if first else None)
            for i, conv in enumerate(model.convs):
                z = conv(h, batch.edge_index, edge_weight)
                if not first and conv.bias is not None:
                    z = z - conv.bias
                spike, states[i] = source_mth_diff_step(z, theta[i], states[i], tick, self.num_levels)
                decoded[i] = spike / float(tick) if decoded[i] is None else decoded[i] + spike / float(tick)
                h = h + spike
            pooled = global_mean_pool(h, batch.batch)
            z = F.linear(pooled, model.mlp[0].weight,
                         model.mlp[0].bias if first else None)
            h, states[4] = source_mth_diff_step(z, theta[4], states[4], tick, self.num_levels)
            decoded[4] = h / float(tick) if decoded[4] is None else decoded[4] + h / float(tick)
            z = F.linear(h, model.mlp[2].weight,
                         model.mlp[2].bias if first else None)
            h, states[5] = source_mth_diff_step(z, theta[5], states[5], tick, self.num_levels)
            decoded[5] = h / float(tick) if decoded[5] is None else decoded[5] + h / float(tick)
            event_logits = F.linear(h, model.mlp[4].weight,
                                    model.mlp[4].bias if first else None)
            contribution = event_logits / float(tick)
            logits = contribution if logits is None else logits + contribution
        return logits, decoded
