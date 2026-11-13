"""Matched control prototypes for residual-prefix validation (synthetic use first)."""
from pathlib import Path
import copy
import torch
import torch.nn.functional as F

from gated_causal_hybrid import _input_embeddings, _pool
from gated_affine_base_v1 import (SourcePrefixEncoder, _site_stats, _sum_stats,
                                  _raw_embeddings, _graph_layer, _analog_head,
                                  validate_batch_model_thresholds)


@torch.no_grad()
def dc_adapt_analog_head_forward(model, batch, thresholds, T, MTH_diff,
                                 capture_trace=False, source_factory=None):
    """Four graph ReLU sites use pinned MTH_diff; the original analog MLP runs once."""
    if model.training or any(m.training for m in model.modules()) or T not in (1, 2, 4, 8):
        raise ValueError("eval model and T in {1,2,4,8} required")
    if getattr(batch, "y", None) is not None:
        raise ValueError("remove labels before conversion")
    if len(thresholds) != len(model.layers):
        raise ValueError("one calibrated threshold vector per gated graph branch required")
    h, e = _input_embeddings(model, batch)
    base_h, base_e = h, e
    enc, stats, names = [], [], []
    for i, layer in enumerate(model.layers):
        th = torch.as_tensor(thresholds[i], dtype=h.dtype, device=h.device).reshape(-1)
        if th.numel() != layer.out_channels or not bool(torch.isfinite(th).all()) or bool((th <= 0).any()):
            raise ValueError(f"invalid graph-branch threshold at layer {i}")
        unit = (MTH_diff(T=0, thresh=th, step_mode="s", num_thresholds=8).to(h.device)
                if source_factory is None else source_factory(h, th, MTH_diff, f"layer{i}.node"))
        zero = unit(torch.zeros((h.shape[0], layer.out_channels), device=h.device, dtype=h.dtype))
        if bool(zero.any()) or int(unit.T) != 0:
            raise RuntimeError("MTH_diff zero latch changed")
        enc.append({"node": unit, "previous": torch.zeros((h.shape[0], layer.out_channels), device=h.device, dtype=h.dtype),
                    "decoded": torch.zeros((h.shape[0], layer.out_channels), device=h.device, dtype=h.dtype), "th": th})
        stats.append({"positive": 0, "negative": 0, "zero": 0, "nonzero_scalar_events": 0,
                      "nonzero_rows": 0, "sum_abs_graded_amplitude": 0.0,
                      "signed_threshold_level_events": {}})
        names.append(f"layer{i}.node")
    src, dst = batch.edge_index.long()
    previous = [torch.zeros_like(h) for _ in model.layers]
    traces, tick_logits = [], []
    for tick in range(1, T + 1):
        h, e = base_h, base_e
        layer_trace = []
        for i, layer in enumerate(model.layers):
            z = layer.C(e) + layer.D(h)[src] + layer.E(h)[dst]
            gates = torch.sigmoid(z)
            b = layer.B(h)
            num = h.new_zeros((h.shape[0], layer.out_channels)); den = torch.zeros_like(num)
            num.index_add_(0, dst, gates * b[src]); den.index_add_(0, dst, gates)
            cur = layer.A(h) + num / (den + layer.eps)
            eu = z
            if layer.use_batch_norm:
                cur, eu = layer.bn_node_h(cur), layer.bn_node_e(eu)
            # Analog edge path and raw edge residual are unchanged.
            e = e + F.relu(eu) if layer.residual else F.relu(eu)
            unit = enc[i]
            source_input = tick * (cur - unit["previous"])
            unit["previous"] = cur.clone()
            event = unit["node"](source_input)
            if int(unit["node"].T) != tick:
                raise RuntimeError("source active tick desynchronized")
            unit["decoded"] = unit["decoded"] + event / tick
            _sum_stats(stats[i], _site_stats(event, unit["th"]))
            h = h + unit["decoded"] if layer.residual else unit["decoded"]
            if capture_trace:
                layer_trace.append({"layer": i, "gates": gates.clone(), "node_current": cur.clone(),
                                    "node_decoded": unit["decoded"].clone(), "edge_state": e.clone()})
        graph_h = _pool(model, h, batch)
        if capture_trace or tick == T:
            logits = _analog_head(model, graph_h)
            tick_logits.append(logits)
        if capture_trace: traces.append(layer_trace)
    return {"mode": "dc_adapt_analog_head", "T": T, "logits": tick_logits[-1],
            "tick_logits": tick_logits if capture_trace else [], "head_calls": len(tick_logits),
            "source_warmup_calls": len(enc), "active_unit_calls": len(enc) * T,
            "source_ticks": [int(x["node"].T) for x in enc], "site_names": names,
            "site_stats": stats, "prefix_site_stats": {}, "prefix_logits": {},
            "trace": traces, "analog_head": "same GatedTeacher MLPReadout; final head once",
            "scope": "dense analog edges/gates/raw residual; four pinned MTH_diff graph branches; no hidden oracle"}


@torch.no_grad()
def per_consumer_forward(model, batch, thresholds, T, MTH_diff, MTH_diff_line,
                         capture_trace=False, encoder_factory=None):
    """Generic raw-coordinate control with independent A/B/D/E and C encoder states."""
    specs = validate_batch_model_thresholds(model, batch, thresholds, T, "generic")
    base_h, base_e = _raw_embeddings(model, batch)
    # Preserve each layer's calibrated threshold vectors; state is independent.
    node_class, edge_class = MTH_diff_line, MTH_diff_line
    def new(ref, th, cls, name):
        return SourcePrefixEncoder(ref, th, cls, name) if encoder_factory is None else encoder_factory(ref, th, cls, name)
    enc, theta_by_site = {}, {}
    for i, layer in enumerate(model.layers):
        theta_node = torch.as_tensor(thresholds[2*i], dtype=base_h.dtype, device=base_h.device)
        theta_edge = torch.as_tensor(thresholds[2*i+1], dtype=base_e.dtype, device=base_e.device)
        for projection in "ABDE":
            name = f"layer{i}.{projection}"
            enc[name] = new(base_h, theta_node, node_class, name)
            theta_by_site[name] = theta_node
        name = f"layer{i}.C"
        enc[name] = new(base_e, theta_edge, edge_class, name)
        theta_by_site[name] = theta_edge
    stats = {name: {"positive": 0, "negative": 0, "zero": 0, "nonzero_scalar_events": 0,
                    "nonzero_rows": 0, "sum_abs_graded_amplitude": 0.0,
                    "signed_threshold_level_events": {}} for name in enc}
    src, dst = batch.edge_index.long()
    tick_logits, traces = [], []
    for tick in range(1, T + 1):
        raw_h, raw_e = base_h, base_e
        tick_trace = []
        for i, layer in enumerate(model.layers):
            h_inputs, e_inputs, events = {}, {}, {}
            for p in "ABDE":
                name = f"layer{i}.{p}"; h_inputs[p], events[name] = enc[name].step(raw_h, tick)
            name = f"layer{i}.C"; e_inputs["C"], events[name] = enc[name].step(raw_e, tick)
            for name, event in events.items(): _sum_stats(stats[name], _site_stats(event, theta_by_site[name]))
            z = layer.C(e_inputs["C"]) + layer.D(h_inputs["D"])[src] + layer.E(h_inputs["E"])[dst]
            gates = torch.sigmoid(z); b = layer.B(h_inputs["B"])
            num = raw_h.new_zeros((raw_h.shape[0], layer.out_channels)); den = torch.zeros_like(num)
            num.index_add_(0, dst, gates * b[src]); den.index_add_(0, dst, gates)
            node_cur = layer.A(h_inputs["A"]) + num / (den + layer.eps); edge_cur = z
            if layer.use_batch_norm:
                node_cur, edge_cur = layer.bn_node_h(node_cur), layer.bn_node_e(edge_cur)
            node_branch, edge_branch = F.relu(node_cur), F.relu(edge_cur)
            raw_h = raw_h + node_branch if layer.residual else node_branch
            raw_e = raw_e + edge_branch if layer.residual else edge_branch
            if capture_trace: tick_trace.append({"layer": i, "projection_inputs": {k: v.clone() for k,v in h_inputs.items()},
                                                "edge_C_input": e_inputs["C"].clone(), "gates": gates.clone()})
        if capture_trace or tick == T:
            logits = _analog_head(model, _pool(model, raw_h, batch)); tick_logits.append(logits)
        if capture_trace: traces.append(tick_trace)
    ordered = list(enc)
    return {"mode": "per_consumer", "T": T, "logits": tick_logits[-1], "tick_logits": tick_logits if capture_trace else [],
            "head_calls": T if capture_trace else 1,
            "source_warmup_calls": len(enc), "active_unit_calls": len(enc)*T,
            "source_ticks": [int(getattr(enc[n], "T", getattr(getattr(enc[n], "module", None), "T", -1))) for n in ordered],
            "site_names": ordered, "site_stats": [stats[n] for n in ordered], "prefix_site_stats": {}, "prefix_logits": {},
            "trace": traces, "encoder_kind": "five independent signed MTH_diff_line sources per layer",
            "analog_head": "same GatedTeacher MLPReadout; final head once", "scope": "each decoded source feeds its named projection"}


def affine_mode_forward(model, batch, thresholds, T, mode, MTH_diff, MTH_diff_line,
                        **kwargs):
    """Compatibility dispatcher for evaluation integration."""
    if mode == "dc_adapt_analog_head":
        return dc_adapt_analog_head_forward(model, batch, thresholds, T, MTH_diff,
                                            capture_trace=kwargs.get("capture_trace", False))
    if mode == "per_consumer":
        return per_consumer_forward(model, batch, thresholds, T, MTH_diff, MTH_diff_line,
                                    capture_trace=kwargs.get("capture_trace", False),
                                    encoder_factory=kwargs.get("encoder_factory"))
    if mode == "cse_only":
        from gated_affine_base_v1 import affine_mode_forward as base_forward
        out = base_forward(model, batch, thresholds, T, "generic", MTH_diff, MTH_diff_line, **kwargs)
        out["mode"] = "cse_only"
        out["trajectory_alias"] = "generic; ordinary shared-generic alias, mathematically identical"
        return out
    from gated_affine_base_v1 import affine_mode_forward as base_forward
    return base_forward(model, batch, thresholds, T, mode, MTH_diff, MTH_diff_line, **kwargs)
