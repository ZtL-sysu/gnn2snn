"""CPU acceptance checks for calibrated GCN event conversion core."""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[2]
TAKEOVER = ROOT / "research" / "visual-graphs" / "takeover_20261112"
AUTH = Path(r"C:\code\gnn2snn-dcgs-reference")
sys.path.insert(0, str(TAKEOVER))
sys.path.insert(0, str(AUTH))

from neurons.relu_neuron import MTH_diff
import dcgs_fixed_seed_eval_v2 as dcgs
from calibration_core import CalibratedGCN, teacher_sites


def synthetic_batch():
    g1 = Data(x=torch.tensor([[.2, .4], [.8, -.1], [.3, .7]]),
              pos=torch.tensor([[0., 0.], [.2, .1], [.4, .2]]),
              edge_index=torch.tensor([[0, 1, 1, 2, 2, 0],
                                       [1, 0, 2, 1, 0, 2]], dtype=torch.long),
              y=torch.tensor([2]))
    g2 = Data(x=torch.tensor([[.1, -.2], [.6, .5]]),
              pos=torch.tensor([[.1, .3], [.5, .2]]),
              edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
              y=torch.tensor([1]))
    return Batch.from_data_list([g1, g2])


class TinyFolded(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(4, 8)
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList([GCNConv(8, 8, normalize=False, add_self_loops=False)
                                    for _ in range(4)])
        self.mlp = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2),
                                 nn.ReLU(), nn.Linear(2, 3))

    def forward(self, batch):
        logits, _ = teacher_sites(self, batch)
        return logits


def main():
    torch.manual_seed(20261114)
    batch = synthetic_batch()
    folded = TinyFolded().eval()
    theta = [[.23] * 8 for _ in range(4)] + [[.25] * 4, [.27] * 2]
    core = CalibratedGCN(folded, theta)
    frozen = all(not p.requires_grad for p in core.model.parameters())
    assert frozen and not core.model.training
    assert sum(p.numel() for p in core.offsets) == 38
    checks = {"model_frozen_eval": frozen, "tiny_offset_count": 38,
              "zero_offset_max_abs_error": {}, "label_independent": {},
              "offset_grad_finite_nonzero": None}

    stripped = Batch.from_data_list([Data(x=d.x, pos=d.pos, edge_index=d.edge_index)
                                     for d in batch.to_data_list()])
    for T in (1, 2, 4, 8):
        with torch.no_grad():
            expected = dcgs.dcgs_event_forward(folded, batch, theta, T, MTH_diff)[0]
            actual, sites = core(batch, T)
            no_y, _ = core(stripped, T)
        err = float((expected - actual).abs().max())
        checks["zero_offset_max_abs_error"][str(T)] = err
        checks["label_independent"][str(T)] = bool(torch.equal(actual, no_y))
        assert err < 1e-6, (T, err)
        assert checks["label_independent"][str(T)]
        assert len(sites) == 6 and all(torch.isfinite(x).all() for x in sites)

    train_core = CalibratedGCN(folded, theta)
    with torch.no_grad():
        for offset in train_core.offsets:
            offset.fill_(0.5)
    logits, decoded = train_core(batch, 2)
    loss = logits.square().mean() + sum(site.square().mean() for site in decoded)
    loss.backward()
    grads = [p.grad for p in train_core.offsets]
    checks["offset_grad_finite_nonzero"] = all(
        g is not None and torch.isfinite(g).all() for g in grads) and any(
        bool((g != 0).any()) for g in grads)
    assert checks["offset_grad_finite_nonzero"]

    full = TinyFolded()
    full.embedding = nn.Linear(4, 146)
    from torch_geometric.nn import GCNConv
    full.convs = nn.ModuleList([GCNConv(146, 146, normalize=False, add_self_loops=False)
                                for _ in range(4)])
    full.mlp = nn.Sequential(nn.Linear(146, 73), nn.ReLU(), nn.Linear(73, 36),
                             nn.ReLU(), nn.Linear(36, 10))
    full_core = CalibratedGCN(full, [[.2] * 146 for _ in range(4)] + [[.2] * 73, [.2] * 36])
    checks["publication_offset_count"] = sum(p.numel() for p in full_core.offsets)
    assert checks["publication_offset_count"] == 693
    out = Path(__file__).with_name("core_checks.json")
    out.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
