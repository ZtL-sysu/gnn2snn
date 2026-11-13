"""Teacher-conditioned frozen-gate prototype for GatedTeacher.

The controller runs one complete, label-free ANN forward pass per graph batch
and stores each layer's edge gates. The node path can then be replayed exactly
or with staged channel-threshold counts. This is a numerical hybrid reference,
not a causal/event-driven SNN implementation.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


@dataclass
class FrozenGateController:
    gates: list
    edge_states: list


def _inputs(model, batch):
    if getattr(batch, "edge_attr", None) is None:
        raise ValueError("GatedTeacher requires edge_attr")
    x = batch.x.float()
    if x.shape[-1] != model.embedding_h.in_features:
        pos = getattr(batch, "pos", None)
        if pos is None:
            raise ValueError("Missing positions needed by the teacher")
        x = torch.cat([x, pos.float()], dim=-1)
    if x.shape[-1] != model.embedding_h.in_features:
        raise ValueError("Node feature width does not match teacher")
    e = batch.edge_attr.float().reshape(-1, model.edge_dim)
    return model.embedding_h(x), model.embedding_e(e)


def _layer_parts(layer, edge_index, h, e):
    src, dst = edge_index
    z = layer.C(e) + layer.D(h)[src] + layer.E(h)[dst]
    gates = torch.sigmoid(z)
    b = layer.B(h)
    numerator = h.new_zeros((h.shape[0], layer.out_channels))
    denominator = h.new_zeros((h.shape[0], layer.out_channels))
    if dst.numel():
        numerator.index_add_(0, dst, gates * b[src])
        denominator.index_add_(0, dst, gates)
    return layer.A(h) + numerator / (denominator + layer.eps), z, gates


@torch.no_grad()
def extract_teacher_gates(model, batch):
    """Run the eval teacher once; return gates and edge states without labels."""
    if model.training:
        raise ValueError("Teacher must be in eval mode")
    h, e = _inputs(model, batch)
    gates_by_layer, edge_states = [], []
    for layer in model.layers:
        node_pre, edge_pre, gates = _layer_parts(layer, batch.edge_index, h, e)
        node_act = layer.bn_node_h(node_pre) if layer.use_batch_norm else node_pre
        edge_act = layer.bn_node_e(edge_pre) if layer.use_batch_norm else edge_pre
        node_act, edge_act = F.relu(node_act), F.relu(edge_act)
        if layer.residual:
            node_act, edge_act = h + node_act, e + edge_act
        h = F.dropout(node_act, layer.dropout, training=False)
        e = F.dropout(edge_act, layer.dropout, training=False)
        gates_by_layer.append(gates.detach().clone())
        edge_states.append(e.detach().clone())
    return FrozenGateController(gates_by_layer, edge_states)


def target_normalize_gates(gates, dst, num_nodes, eps=1e-6):
    """Normalize E x C gates over incoming edges independently per target/channel."""
    if gates.ndim != 2 or dst.ndim != 1 or gates.shape[0] != dst.numel():
        raise ValueError("Expected gates [E,C] and dst [E]")
    denominator = gates.new_zeros((num_nodes, gates.shape[1]))
    if dst.numel():
        denominator.index_add_(0, dst, gates)
    return gates / (denominator[dst] + eps)


def _quantize_activation(x, theta, T):
    theta = torch.as_tensor(theta, dtype=x.dtype, device=x.device)
    if theta.ndim != 1 or theta.numel() != x.shape[-1]:
        raise ValueError("Each site needs one threshold per output channel")
    if T < 1 or not bool(torch.isfinite(theta).all()) or bool((theta <= 0).any()):
        raise ValueError("T and thresholds must be finite and positive")
    shape = (1,) * (x.ndim - 1) + (theta.numel(),)
    theta = theta.reshape(shape)
    counts = torch.round(torch.clamp(x, min=0) * (T / theta)).clamp(0, T)
    return counts * (theta / T), counts


def _pool(model, h, batch):
    pool = {"mean": global_mean_pool, "sum": global_add_pool,
            "max": global_max_pool}[model.readout]
    return pool(h, batch.batch)


@torch.no_grad()
def frozen_gate_forward(model, batch, controller: FrozenGateController,
                        thresholds: Optional[Sequence] = None, T: Optional[int] = None):
    """Replay fixed gates exactly, or quantize six ReLU sites in staged order.

    The real-valued input embedding and residual bypasses are shared. Edge
    states/gates are supplied by the ANN controller; node BN, residuals, and
    MLP biases remain in their original eval-mode positions. If thresholds are
    supplied, sites are four node branches followed by two MLP ReLUs. The final
    classifier is analog. Returns `(logits, diagnostics)`.
    """
    if model.training:
        raise ValueError("Teacher must be in eval mode")
    quantized = thresholds is not None
    if quantized and (len(thresholds) != len(model.layers) + 2 or T is None):
        raise ValueError("Expected one threshold vector per activation site and T")
    if len(controller.gates) != len(model.layers):
        raise ValueError("Controller layer count does not match teacher")
    h, _ = _inputs(model, batch)
    sites = []
    for i, (layer, gates) in enumerate(zip(model.layers, controller.gates)):
        if gates.shape != (batch.edge_index.shape[1], layer.out_channels):
            raise ValueError(f"Gate shape mismatch at layer {i}")
        src, dst = batch.edge_index
        b = layer.B(h)
        numerator = h.new_zeros((h.shape[0], layer.out_channels))
        denominator = h.new_zeros((h.shape[0], layer.out_channels))
        if dst.numel():
            numerator.index_add_(0, dst, gates * b[src])
            denominator.index_add_(0, dst, gates)
        node_pre = layer.A(h) + numerator / (denominator + layer.eps)
        node_act = layer.bn_node_h(node_pre) if layer.use_batch_norm else node_pre
        node_act = F.relu(node_act)
        if quantized:
            node_act, counts = _quantize_activation(node_act, thresholds[i], T)
            sites.append({"site": i, "counts": counts.detach(),
                          "spikes": int(counts.sum().item())})
        if layer.residual:
            node_act = h + node_act
        h = F.dropout(node_act, layer.dropout, training=False)

    h = _pool(model, h, batch)
    fc_layers = model.MLP_layer.FC_layers
    for i, fc in enumerate(fc_layers[:-1], start=len(model.layers)):
        h = F.relu(fc(h))
        if quantized:
            h, counts = _quantize_activation(h, thresholds[i], T)
            sites.append({"site": i, "counts": counts.detach(),
                          "spikes": int(counts.sum().item())})
    logits = fc_layers[-1](h)
    return logits, {"sites": sites, "quantized": quantized, "T": T}


@torch.no_grad()
def teacher_conditioned_forward(model, batch, thresholds=None, T=None):
    """Convenience API: label-free full teacher controller then fixed-gate replay."""
    controller = extract_teacher_gates(model, batch)
    logits, diagnostics = frozen_gate_forward(model, batch, controller, thresholds, T)
    return logits, controller, diagnostics
