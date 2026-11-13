"""Synthetic safeguards for matched controls; no official data are loaded."""
import argparse
import os
import sys
from pathlib import Path
import torch
from torch_geometric.data import Batch, Data

GNN2SNN_ROOT = Path(os.environ.get("GNN2SNN_ROOT", Path(__file__).resolve().parents[4])).resolve()
TAKEOVER = GNN2SNN_ROOT / "research" / "visual-graphs" / "takeover_20261112"
RUN_CODE = Path(__file__).resolve().parent
for p in (str(TAKEOVER), str(RUN_CODE)):
    if p not in sys.path: sys.path.insert(0, p)

import matched_controls as controls
from gated_affine_base_v1 import affine_mode_forward as generic_forward
from gated_teacher_model import GatedTeacher
from dcgs_fixed_residual_gcn_event_forward_v1 import require_author_source
from gated_affine_base_v1 import require_author_sources


class CountingEncoder:
    def __init__(self, ref, threshold, cls, site):
        from gated_affine_base_v1 import SourcePrefixEncoder
        self.inner = SourcePrefixEncoder(ref, threshold, cls, site)
        self.site = site
        SOURCE_THRESHOLDS[site] = torch.as_tensor(threshold).detach().cpu().clone()
        CALLS.append(site)
    def step(self, raw, tick): return self.inner.step(raw, tick)
    @property
    def T(self): return int(self.inner.module.T)
    @property
    def active_calls(self): return self.inner.active_calls
    @property
    def warmup_calls(self): return self.inner.warmup_calls


class IdentityReLUSource:
    """Synthetic oracle source: encode exact positive prefix deltas, preserving source clock."""
    def __init__(self, reference, threshold, cls, site):
        self.T = 0; self.initialized = False; self.raw = torch.zeros_like(reference); self.previous = torch.zeros_like(reference)
    def __call__(self, value):
        if not self.initialized:
            assert not bool(value.any())
            self.initialized = True
            return torch.zeros_like(value)
        tick = self.T + 1
        self.raw = self.raw + value / tick
        target = torch.relu(self.raw)
        event = tick * (target - self.previous)
        self.previous = target.clone(); self.T = tick
        return event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-repo", type=Path,
                        default=Path(os.environ.get("DCGS_REPO", "")),
                        required=not bool(os.environ.get('DCGS_REPO')))
    args = parser.parse_args()
    MTH, MTH_line, provenance = require_author_sources(args.author_repo)
    # Independently bind the exact MTH_diff source contract used by DCGS.
    assert require_author_source(args.author_repo)[0] is MTH
    torch.manual_seed(513); torch.set_num_threads(1)
    graphs = [Data(x=torch.randn(4, 3), edge_index=torch.tensor([[0,1,2,3,0],[1,2,3,0,2]]),
                   edge_attr=torch.randn(5,1), y=torch.tensor([i])) for i in range(2)]
    batch = Batch.from_data_list(graphs)
    model = GatedTeacher(node_dim=3, edge_dim=1, hidden_dim=4, out_dim=4, layers=2).eval()
    with torch.no_grad(): expected = model(batch)
    # Restore ground truth is already saved; conversion calls must not enter teacher.forward.
    model.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("teacher hidden oracle called"))
    unlabeled = batch.clone(); del unlabeled.y
    generic_th = [torch.tensor([.2,.3,.4,.5]), torch.tensor([.25,.35,.45,.55]),
                  torch.tensor([.8,.9,1.0,1.1]), torch.tensor([.65,.75,.85,.95])]
    global CALLS, SOURCE_THRESHOLDS; CALLS=[]; SOURCE_THRESHOLDS={}
    head_counts = {"per": 0, "dc": 0}
    hook_per = model.MLP_layer.register_forward_hook(lambda *a: head_counts.__setitem__("per", head_counts["per"] + 1))
    per = controls.per_consumer_forward(model, unlabeled, generic_th, 4, MTH, MTH_line,
                                        capture_trace=True, encoder_factory=CountingEncoder)
    hook_per.remove()
    shared = generic_forward(model, unlabeled, generic_th, 4, "generic", MTH, MTH_line,
                             capture_trace=True)
    torch.testing.assert_close(per["logits"], shared["logits"], rtol=0, atol=0)
    assert len(CALLS) == 2 * 5 and len(set(CALLS)) == len(CALLS)
    for layer in range(2):
        for projection in "ABDE":
            assert torch.equal(SOURCE_THRESHOLDS[f"layer{layer}.{projection}"], generic_th[2*layer])
        assert torch.equal(SOURCE_THRESHOLDS[f"layer{layer}.C"], generic_th[2*layer+1])
    assert per["source_warmup_calls"] == 10 and per["active_unit_calls"] == 40
    assert per["head_calls"] == head_counts["per"] == 4
    hook_per_default = model.MLP_layer.register_forward_hook(lambda *a: head_counts.__setitem__("per_default", head_counts.get("per_default", 0) + 1))
    per_default = controls.per_consumer_forward(model, unlabeled, generic_th, 4, MTH, MTH_line)
    hook_per_default.remove()
    assert per_default["head_calls"] == head_counts["per_default"] == 1
    # Every independent decoder is consumed by its named projection and represents the shared trajectory.
    for t in range(4):
        for layer in range(2):
            tr = per["trace"][t][layer]
            assert set(tr["projection_inputs"]) == set("ABDE")
            for value in tr["projection_inputs"].values():
                torch.testing.assert_close(value, shared["trace"][t]["layers"][layer]["consumer_node_input"], rtol=0, atol=0)
            torch.testing.assert_close(tr["edge_C_input"], shared["trace"][t]["layers"][layer]["consumer_edge_input"], rtol=0, atol=0)
            # Adjacent raw residual input is exactly the previous layer's raw output.
            if layer > 0:
                torch.testing.assert_close(shared["trace"][t]["layers"][layer]["raw_node_input"],
                                           shared["trace"][t]["layers"][layer-1]["raw_node_output"], rtol=0, atol=0)
                torch.testing.assert_close(shared["trace"][t]["layers"][layer]["raw_edge_input"],
                                           shared["trace"][t]["layers"][layer-1]["raw_edge_output"], rtol=0, atol=0)
    assert len(per["tick_logits"]) == 4 and per["head_calls"] == 4
    # No-label guard and positive finite threshold guard.
    try: controls.per_consumer_forward(model, batch, generic_th, 4, MTH, MTH_line)
    except ValueError as e: assert "unlabeled" in str(e)
    else: raise AssertionError("label guard failed")
    bad = list(generic_th); bad[0] = torch.zeros(4)
    try: controls.per_consumer_forward(model, unlabeled, bad, 4, MTH, MTH_line)
    except ValueError: pass
    else: raise AssertionError("threshold guard failed")
    hook_dc = model.MLP_layer.register_forward_hook(lambda *a: head_counts.__setitem__("dc", head_counts["dc"] + 1))
    dc = controls.dc_adapt_analog_head_forward(model, unlabeled,
                                               [torch.full((4,), .4) for _ in range(2)], 4, MTH,
                                               capture_trace=True, source_factory=IdentityReLUSource)
    hook_dc.remove()
    assert dc["source_warmup_calls"] == 2 and dc["active_unit_calls"] == 8
    assert dc["head_calls"] == head_counts["dc"] == 4 and torch.isfinite(dc["logits"]).all()
    for step_logits in dc["tick_logits"]:
        torch.testing.assert_close(step_logits, expected, rtol=1e-5, atol=1e-5)
    hook_dc_default = model.MLP_layer.register_forward_hook(lambda *a: head_counts.__setitem__("dc_default", head_counts.get("dc_default", 0) + 1))
    dc_default = controls.dc_adapt_analog_head_forward(model, unlabeled,
                                                       [torch.full((4,), .4) for _ in range(2)], 4, MTH)
    hook_dc_default.remove()
    assert dc_default["head_calls"] == head_counts["dc_default"] == 1
    print({"status":"PASS", "pinned_commit":provenance["commit"],
           "per_consumer_source_calls":len(CALLS), "per_consumer_active_calls":40,
           "per_consumer_equals_generic_bitwise":True, "dc_adapt_sites":2,
           "identity_source_restores_teacher_each_tick":True, "teacher_forward_disabled":True,
           "actual_per_consumer_head_calls_capture/default":[head_counts["per"], head_counts["per_default"]],
           "actual_dc_head_calls_capture/default":[head_counts["dc"], head_counts["dc_default"]]})


if __name__ == "__main__": main()
