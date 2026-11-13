"""Oracle-free GatedGCN diagnostic with analog edges and DCGS node units.

This is a project adaptation of established differential coding. Dense gate
calculations run every tick; graded signed source outputs are not binary IF.
"""
import torch
import torch.nn.functional as F

from gated_causal_hybrid import _input_embeddings, _pool, _threshold
from dcgs_fixed_residual_gcn_event_forward_v1 import fresh_mth, _event_counts, _accumulate_stats


@torch.no_grad()
def differential_hybrid_forward(model, batch, thresholds, T, MTH_diff,
                                exact_relu=False, capture_trace=False):
    if model.training or T < 1 or len(thresholds) != len(model.layers) + 2:
        raise ValueError("eval model, positive T, and one threshold/site required")
    if getattr(batch, "y", None) is not None:
        raise ValueError("remove labels before conversion")
    input_h, input_e = _input_embeddings(model, batch)
    fc = model.MLP_layer.FC_layers
    widths = [layer.out_channels for layer in model.layers] + [x.out_features for x in fc[:-1]]
    theta = [_threshold(value, width, input_h, str(i)).reshape(-1)
             for i, (value, width) in enumerate(zip(thresholds, widths))]
    shapes = [(input_h.shape[0], w) for w in widths[:len(model.layers)]]
    shapes += [(batch.num_graphs, w) for w in widths[len(model.layers):]]
    rates = [input_h.new_zeros(shape) for shape in shapes]
    previous_current = [torch.zeros_like(x) for x in rates]
    previous_relu = [torch.zeros_like(x) for x in rates]
    modules = [] if exact_relu else [fresh_mth(MTH_diff, th, input_h.device) for th in theta]
    for module, rate in zip(modules, rates):
        warmup = module(torch.zeros_like(rate))
        if bool(warmup.any()) or int(module.T) != 0:
            raise RuntimeError("source zero-latch semantics changed")
    stats = [{"positive": 0, "negative": 0, "zero": 0, "nonzero_scalar_events": 0,
              "sum_abs_graded_amplitude": 0.0, "signed_threshold_level_events": {}}
             for _ in rates]
    traces, gate_traces = [], []
    src, dst = batch.edge_index

    def activate(current, site, tick):
        event_current = tick * (current - previous_current[site])
        previous_current[site] = current.clone()
        if exact_relu:
            target = F.relu(current)
            event = tick * (target - previous_relu[site])
            previous_relu[site] = target.clone()
        else:
            event = modules[site](event_current)
            if int(modules[site].T) != tick:
                raise RuntimeError("source active tick desynchronized")
            _accumulate_stats(stats[site], _event_counts(event, theta[site], event.shape[-1]))
        rates[site] = rates[site] + event / tick
        return rates[site]

    for tick in range(1, T + 1):
        h, e = input_h, input_e
        gate_tick = []
        for i, layer in enumerate(model.layers):
            z = layer.C(e) + layer.D(h)[src] + layer.E(h)[dst]
            gates = torch.sigmoid(z)
            b = layer.B(h)
            numerator = h.new_zeros((h.shape[0], layer.out_channels))
            denominator = torch.zeros_like(numerator)
            numerator.index_add_(0, dst, gates * b[src])
            denominator.index_add_(0, dst, gates)
            current = layer.A(h) + numerator / (denominator + layer.eps)
            edge_update = z
            if layer.use_batch_norm:
                current = layer.bn_node_h(current)
                edge_update = layer.bn_node_e(edge_update)
            if capture_trace:
                gate_tick.append({"h_input": h.clone(), "e_input": e.clone(),
                                  "gates": gates.clone(), "denominator": denominator.clone(),
                                  "current": current.clone()})
            edge_update = F.relu(edge_update)
            e = e + edge_update if layer.residual else edge_update
            branch = activate(current, i, tick)
            h = h + branch if layer.residual else branch
        h = _pool(model, h, batch)
        for j, linear in enumerate(fc[:-1], len(model.layers)):
            h = activate(linear(h), j, tick)
        logits = fc[-1](h)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite converted logits")
        if capture_trace:
            traces.append(logits.clone())
            gate_traces.append(gate_tick)
    return logits, {
        "T": T, "source_warmup_calls": len(modules),
        "active_unit_calls": len(modules) * T, "sites": stats,
        "analog_gate_evaluations": len(model.layers) * T,
        "analog_gate_edge_channel_elements": T * sum(dst.numel() * x.out_channels for x in model.layers),
        "tick_logits": traces,
        "gate_traces": gate_traces,
        "scope": "analog edge controller each tick; source signed graded node events",
    }


def synthetic_acceptance(MTH_diff):
    from torch_geometric.data import Data, Batch
    from gated_teacher_model import GatedTeacher
    torch.manual_seed(773)
    torch.set_num_threads(4)
    graphs = [Data(x=torch.randn(3, 3), edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]),
                   edge_attr=torch.randn(4, 1), y=torch.tensor([i])) for i in (0, 1)]
    batch = Batch.from_data_list(graphs)
    model = GatedTeacher(node_dim=3, edge_dim=1, hidden_dim=4, out_dim=4, layers=4).eval()
    for layer in model.layers:
        for bn in (layer.bn_node_h, layer.bn_node_e):
            bn.running_mean.copy_(torch.randn_like(bn.running_mean))
            bn.running_var.copy_(torch.rand_like(bn.running_var) + .2)
    with torch.no_grad():
        expected = model(batch)
    del batch.y
    # Conversion must not rely on the teacher forward to obtain hidden states.
    model.forward = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("teacher oracle"))
    thresholds = [torch.ones(4)] * 4 + [torch.ones(2), torch.ones(1)]
    _, exact = differential_hybrid_forward(model, batch, thresholds, 8, MTH_diff,
                                          exact_relu=True, capture_trace=True)
    for logits in exact["tick_logits"]:
        torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)
    actual, diag = differential_hybrid_forward(model, batch, thresholds, 8, MTH_diff)
    assert diag["source_warmup_calls"] == 6 and diag["active_unit_calls"] == 48
    assert torch.isfinite(actual).all()
    values = torch.tensor([.8, .8, .8, 0., -.3, .7], dtype=torch.float64)
    previous = 0.
    reconstructed = 0.
    for tick, current in enumerate(values, 1):
        event = tick * (current - previous)
        reconstructed += event / tick
        torch.testing.assert_close(torch.as_tensor(reconstructed), current)
        previous = current
    return {"exact_relu_teacher_ticks": 8, "oracle_forward_disabled": True,
            "label_free": True, "signed_abrupt_current_encoding": True,
            "source_warmup_calls": 6, "active_unit_calls": 48}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from dcgs_fixed_residual_gcn_event_forward_v1 import require_author_source
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", required=True, type=Path)
    args = parser.parse_args()
    unit, _, _ = require_author_source(args.author_repo)
    print(json.dumps(synthetic_acceptance(unit)))
