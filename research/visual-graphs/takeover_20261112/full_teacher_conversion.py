"""Staged rate/count conversion for the full residual GCN.

This is a numerical staged model: each layer waits for the prior layer's rate,
so it is not a claim of online parallel latency, measured energy, or a faithful
multilayer event-driven implementation. Six explicit thresholds cover four GCN
residual branches and two MLP activations. See conversion-staging-note.md.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


def graph_weights(edge_index, n, dtype):
    source, target = edge_index
    outdegree = torch.bincount(source, minlength=n).clamp(min=1).to(dtype)
    indegree = torch.bincount(target, minlength=n).clamp(min=1).to(dtype)
    return outdegree[source].rsqrt() * indegree[target].rsqrt()


def reference_activations(folded_model, batch):
    """Per-layer ReLU targets of a BN-folded teacher, in its actual order."""
    h = folded_model.embedding(torch.cat([batch.x.float(), batch.pos.float()], -1))
    weights = graph_weights(batch.edge_index, h.shape[0], h.dtype)
    activations = []
    for conv in folded_model.convs:
        branch = F.relu(conv(h, batch.edge_index, weights))
        activations.append(branch)
        h = h + branch
    h = global_mean_pool(h, batch.batch)
    for layer in (folded_model.mlp[0], folded_model.mlp[2]):
        h = F.relu(layer(h))
        activations.append(h)
    return activations


def count_quantize(current, theta, T):
    theta = torch.as_tensor(theta, device=current.device, dtype=current.dtype)
    if T < 1 or not bool(torch.isfinite(theta).all()) or bool((theta <= 0).any()):
        raise ValueError("T and theta must be positive")
    counts = torch.round(torch.clamp(current, min=0) * (T / theta)).clamp(0, T)
    return counts * (theta / T), counts


def balanced_step(counts, T, t):
    if t < 1 or t > T:
        raise ValueError("step outside 1..T")
    return torch.round(t * counts / T) - torch.round((t - 1) * counts / T)


@torch.no_grad()
def staged_forward(folded_model, batch, thresholds, T):
    if folded_model.training or len(thresholds) != 6:
        raise ValueError("Expected eval teacher and six thresholds")
    h = folded_model.embedding(torch.cat([batch.x.float(), batch.pos.float()], -1))
    weights = graph_weights(batch.edge_index, h.shape[0], h.dtype)
    spike_counts = []
    for i, conv in enumerate(folded_model.convs):
        branch, counts = count_quantize(conv(h, batch.edge_index, weights), thresholds[i], T)
        h = h + branch
        spike_counts.append(int(counts.sum().item()))
    h = global_mean_pool(h, batch.batch)
    for i, layer in enumerate((folded_model.mlp[0], folded_model.mlp[2]), start=4):
        h, counts = count_quantize(layer(h), thresholds[i], T)
        spike_counts.append(int(counts.sum().item()))
    return folded_model.mlp[4](h), spike_counts
