"""Source-equivalent PyG adaptation of the public ResGatedGCN CIFAR teacher.

Edges use PyG's source-to-target convention. For every directed edge u->v,
the layer computes C(e_uv) + D(h_u) + E(h_v), then target-wise normalizes
sigmoid gates before reducing B(h_u). This module adapts operators only; it
does not claim reproduction of the original DGL training/data pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, batch_norm=True, residual=True, eps=1e-6):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = float(dropout)
        self.use_batch_norm = bool(batch_norm)
        self.residual = bool(residual and input_dim == output_dim)
        self.eps = float(eps)
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, edge_index, h, e):
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if h.ndim != 2 or e.ndim != 2 or e.shape[0] != edge_index.shape[1]:
            raise ValueError("Expected node features [N,F] and one edge feature row per edge")
        if h.shape[1] != self.in_channels or e.shape[1] != self.in_channels:
            raise ValueError("Node and edge embeddings must match layer input width")
        h_in, e_in = h, e
        src, dst = edge_index
        a = self.A(h)
        b = self.B(h)
        # DGL apply_edges(u_add_v): source receives D, target receives E.
        edge_logits = self.C(e) + self.D(h)[src] + self.E(h)[dst]
        gates = torch.sigmoid(edge_logits)
        weighted_messages = gates * b[src]
        numerator = h.new_zeros((h.shape[0], self.out_channels))
        denominator = h.new_zeros((h.shape[0], self.out_channels))
        if dst.numel():
            numerator.index_add_(0, dst, weighted_messages)
            denominator.index_add_(0, dst, gates)
        h = a + numerator / (denominator + self.eps)
        e = edge_logits
        if self.use_batch_norm:
            h = self.bn_node_h(h)
            if e.shape[0] == 0:
                raise ValueError("BatchNorm on edge states requires at least one edge")
            e = self.bn_node_e(e)
        h = F.relu(h)
        e = F.relu(e)
        if self.residual:
            h = h_in + h
            e = e_in + e
        return F.dropout(h, self.dropout, self.training), F.dropout(e, self.dropout, self.training)


class GatedTeacher(nn.Module):
    """CIFAR10 config from public_baselines/gated_cifar.json, parameterized by dimensions."""
    def __init__(self, node_dim=5, edge_dim=1, hidden_dim=70, out_dim=70, n_classes=10,
                 layers=4, dropout=0.0, batch_norm=True, residual=True, readout="mean"):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        if readout not in {"mean", "sum", "max"}:
            readout = "mean"  # matches public GatedGCNNet fallback
        self.readout = readout
        self.edge_dim = edge_dim
        self.embedding_h = nn.Linear(node_dim, hidden_dim)
        self.embedding_e = nn.Linear(edge_dim, hidden_dim)
        dims = [hidden_dim] * (layers - 1) + [out_dim]
        previous = hidden_dim
        self.layers = nn.ModuleList()
        for width in dims:
            self.layers.append(GatedGCNLayer(previous, width, dropout, batch_norm, residual))
            previous = width
        self.MLP_layer = MLPReadout(out_dim, n_classes, hidden_layers=2)

    def forward(self, batch, return_states=False):
        if getattr(batch, "edge_attr", None) is None:
            raise ValueError("GatedTeacher requires the public baseline's scalar edge_attr values")
        node_features = batch.x.float()
        if node_features.shape[-1] != self.embedding_h.in_features:
            pos = getattr(batch, "pos", None)
            if pos is None:
                raise ValueError("Missing positions needed for benchmark node features")
            node_features = torch.cat([node_features, pos.float()], dim=-1)
        if node_features.shape[-1] != self.embedding_h.in_features:
            raise ValueError("Node feature and position width does not match teacher")
        h = self.embedding_h(node_features)
        e = self.embedding_e(batch.edge_attr.float().reshape(-1, self.edge_dim))
        edge_index = batch.edge_index
        for layer in self.layers:
            h, e = layer(edge_index, h, e)
        pool = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}[self.readout]
        graph_h = pool(h, batch.batch)
        logits = self.MLP_layer(graph_h)
        if return_states:
            return logits, h, e
        return logits


class MLPReadout(nn.Module):
    """Public benchmark MLPReadout: 2 ReLU hidden layers with floor-halved widths."""
    def __init__(self, input_dim, output_dim, hidden_layers=2):
        super().__init__()
        dims = [input_dim // (2 ** i) for i in range(hidden_layers + 1)]
        self.FC_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(hidden_layers)]
            + [nn.Linear(dims[-1], output_dim, bias=True)]
        )
        self.L = hidden_layers

    def forward(self, x):
        for layer in self.FC_layers[:-1]:
            x = F.relu(layer(x))
        return self.FC_layers[-1](x)


def cifar10_parameter_count():
    model = GatedTeacher(node_dim=5, edge_dim=1, hidden_dim=70, out_dim=70,
                         n_classes=10, layers=4, dropout=0.0, batch_norm=True,
                         residual=True, readout="mean")
    return sum(p.numel() for p in model.parameters())
