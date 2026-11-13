"""Causal mixed ANN-edge-controller / hard-IF-node simulator for GatedTeacher.

Unlike frozen-gate replay, every tick recomputes gates from this simulator's
current prefix-rate node states. Edge operations are dense ANN calculations;
only node and MLP ReLU sites use hard integrate-and-fire counts.
"""
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


def _input_embeddings(model, batch):
    if getattr(batch, "edge_attr", None) is None:
        raise ValueError("GatedTeacher requires edge_attr")
    x = batch.x.float()
    if x.shape[-1] != model.embedding_h.in_features:
        pos = getattr(batch, "pos", None)
        if pos is None:
            raise ValueError("Missing positions needed by teacher")
        x = torch.cat((x, pos.float()), dim=-1)
    if x.shape[-1] != model.embedding_h.in_features:
        raise ValueError("Node feature width does not match teacher")
    return (model.embedding_h(x),
            model.embedding_e(batch.edge_attr.float().reshape(-1, model.edge_dim)))


def _threshold(value, width, ref, name):
    theta = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    if theta.ndim != 1 or theta.numel() != width:
        raise ValueError(f"{name} must have one threshold per channel ({width})")
    if not bool(torch.isfinite(theta).all()) or bool((theta <= 0).any()):
        raise ValueError(f"{name} thresholds must be finite and positive")
    return theta.reshape(1, width)


def _pool(model, h, batch):
    pool = {"mean": global_mean_pool, "sum": global_add_pool,
            "max": global_max_pool}[model.readout]
    return pool(h, batch.batch)


@torch.no_grad()
def causal_hybrid_forward(model, batch, thresholds: Sequence, T: int,
                          record_trace: bool = False, init_fraction: float = 0.0):
    """Run a causal T-tick hybrid simulation; never calls the ANN teacher forward.

    Per-tick order is layer 0..L-1 then MLP. Each layer consumes the previous
    layer's current-tick prefix-rate decode; membranes/counts persist across
    ticks, while edge states are recomputed from edge embeddings at each tick
    and passed only across depth. BatchNorm must be frozen in eval mode.
    """
    if model.training or T < 1 or len(thresholds) != len(model.layers) + 2:
        raise ValueError("Expected eval model, one threshold vector per activation site, and T>=1")
    input_h, input_e = _input_embeddings(model, batch)
    node_theta = [_threshold(thresholds[i], layer.out_channels, input_h, f"node_{i}")
                  for i, layer in enumerate(model.layers)]
    fc_layers = model.MLP_layer.FC_layers
    mlp_theta = [_threshold(thresholds[len(model.layers) + i], fc.out_features,
                             input_h, f"mlp_{i}") for i, fc in enumerate(fc_layers[:-1])]

    node_mem = [input_h.new_zeros((input_h.shape[0], layer.out_channels)) + init_fraction * theta for layer, theta in zip(model.layers, node_theta)]
    node_counts = [torch.zeros_like(mem) for mem in node_mem]
    mlp_mem = [input_h.new_zeros((batch.num_graphs, fc.out_features)) + init_fraction * theta for fc, theta in zip(fc_layers[:-1], mlp_theta)]
    mlp_counts = [torch.zeros_like(mem) for mem in mlp_mem]
    site_spikes_by_tick = [[] for _ in thresholds]
    tick_logits, rate_history = [], []
    per_layer_work = [
        {"gate_scalar_evaluations": 0, "sigmoid_elements": 0,
         "target_normalization_edge_channel_elements": 0,
         "gate_controller_linear_macs": 0,
         "edge_message_gate_multiply_elements": 0,
         "edge_state_update_elements": 0}
        for _ in model.layers
    ]

    for t in range(1, T + 1):
        h = input_h
        e = input_e
        tick_rates = []
        for i, layer in enumerate(model.layers):
            src, dst = batch.edge_index
            if layer.use_batch_norm and e.shape[0] == 0:
                raise ValueError("BatchNorm on edge states requires at least one edge")
            d_node = layer.D(h)
            e_node = layer.E(h)
            z = layer.C(e) + d_node[src] + e_node[dst]
            gates = torch.sigmoid(z)
            b = layer.B(h)
            numerator = h.new_zeros((h.shape[0], layer.out_channels))
            denominator = h.new_zeros((h.shape[0], layer.out_channels))
            if dst.numel():
                numerator.index_add_(0, dst, gates * b[src])
                denominator.index_add_(0, dst, gates)
            current = layer.A(h) + numerator / (denominator + layer.eps)
            if layer.use_batch_norm:
                current = layer.bn_node_h(current)
                edge_update = layer.bn_node_e(z)
            else:
                edge_update = z
            edge_update = F.relu(edge_update)
            if layer.residual:
                edge_update = e + edge_update
            e = F.dropout(edge_update, layer.dropout, training=False)

            v = node_mem[i] + current
            spike = (v >= node_theta[i]).to(dtype=current.dtype)
            node_mem[i] = v - spike * node_theta[i]
            node_counts[i] = node_counts[i] + spike
            branch_rate = node_counts[i] * (node_theta[i] / t)
            h = h + branch_rate if layer.residual else branch_rate
            tick_rates.append(h.detach().clone() if record_trace else h)
            site_spikes_by_tick[i].append(int(spike.sum().item()))

            work = per_layer_work[i]
            node_count, edge_count = h.shape[0], src.numel()
            edge_channels = edge_count * layer.out_channels
            work["gate_scalar_evaluations"] += edge_channels
            work["sigmoid_elements"] += edge_channels
            work["target_normalization_edge_channel_elements"] += edge_channels
            work["gate_controller_linear_macs"] += (
                edge_count * layer.in_channels * layer.out_channels
                + 2 * node_count * layer.in_channels * layer.out_channels
            )
            work["edge_message_gate_multiply_elements"] += edge_channels
            work["edge_state_update_elements"] += edge_channels

        h = _pool(model, h, batch)
        for j, fc in enumerate(fc_layers[:-1]):
            current = fc(h)
            v = mlp_mem[j] + current
            spike = (v >= mlp_theta[j]).to(dtype=current.dtype)
            mlp_mem[j] = v - spike * mlp_theta[j]
            mlp_counts[j] = mlp_counts[j] + spike
            h = mlp_counts[j] * (mlp_theta[j] / t)
            site = len(model.layers) + j
            site_spikes_by_tick[site].append(int(spike.sum().item()))
        logits = fc_layers[-1](h)
        tick_logits.append(logits)
        if record_trace:
            rate_history.append(tick_rates)

    avg_logits = torch.stack(tick_logits, dim=0).mean(dim=0)
    site_spikes = [sum(tick_values) for tick_values in site_spikes_by_tick]
    total_gate_work = {
        key: int(sum(layer[key] for layer in per_layer_work))
        for key in per_layer_work[0]
    } if per_layer_work else {}
    diagnostics = {
        "T": int(T), "site_spikes": site_spikes,
        "site_spikes_by_tick": site_spikes_by_tick,
        "total_spikes": int(sum(site_spikes)),
        "gate_work_by_layer": per_layer_work,
        "gate_work": total_gate_work,
        "tick_logits": torch.stack(tick_logits, dim=0).detach(),
        "final_node_branch_rates": [counts * (theta / T) for counts, theta in zip(node_counts, node_theta)],
        "final_mlp_rates": [counts * (theta / T) for counts, theta in zip(mlp_counts, mlp_theta)],
    }
    if record_trace:
        diagnostics["node_rates_by_tick"] = rate_history
    return avg_logits, diagnostics
