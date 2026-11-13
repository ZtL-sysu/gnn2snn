"""CPU-only synthetic checks for shared-threshold profile controls."""
import os
import subprocess
import sys

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gated_teacher_model import GatedTeacher
import gated_affine_shared_profile_v1 as S


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("expected invalid shared thresholds to be rejected")


def test_wrong_cublas_environment():
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":bad"
    proc = subprocess.run([sys.executable, str(S.HERE / "gated_affine_shared_profile_v1.py"), "--help"],
                          env=env, capture_output=True, text=True, timeout=30)
    if proc.returncode == 0 or "CUBLAS_WORKSPACE_CONFIG must be :4096:8" not in (proc.stdout + proc.stderr):
        raise AssertionError("driver did not reject invalid CUBLAS config before imports")


def main():
    torch.manual_seed(20261113)
    torch.set_num_threads(2)
    raw = [torch.tensor([float(i + 1), 2.0, 3.0]) for i in range(6)]
    residual = [torch.tensor([0.5, float(i + 3), 1.0]) for i in range(6)]
    merged = S.merge_threshold_families(raw, residual)
    for i, got in enumerate(merged):
        assert torch.equal(got, torch.maximum(raw[i], residual[i]))
    rejects(lambda: S.merge_threshold_families(raw[:5], residual))
    rejects(lambda: S.merge_threshold_families(raw, residual[:5]))
    bad = list(raw)
    bad[2] = torch.tensor([1.0, float("nan"), 2.0])
    rejects(lambda: S.merge_threshold_families(bad, residual))

    model = GatedTeacher(hidden_dim=8, out_dim=8).eval()
    graphs = [Data(x=torch.randn(3, 3), pos=torch.randn(3, 2),
                   edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                   edge_attr=torch.rand(4, 1), y=torch.tensor([i])) for i in range(3)]
    clean = S.C.LabelFreeGraphView(graphs)
    relu, line, _ = S.A.require_author_sources()
    thresholds = [torch.full((8,), 1.5 + .1 * i) for i in range(6)]
    for mode in S.MODES:
        loader = DataLoader(clean, batch_size=2, shuffle=False)
        got = S.shared_threshold_profile_loader(model, loader, thresholds, mode, relu, line,
                                                "cpu", expected=3, width=8, batch_size=2)
        if got["actual_source_warmup_calls"] != 12 or got["actual_active_calls"] != 48:
            raise AssertionError("shared control source call counts differ")
        if got["diagnostic_head_calls"] != 6 or got["normal_endpoint_head_calls"] != 2:
            raise AssertionError("prefix/endpoint head call counts differ")
        if set(got["profiles"]) != {"1", "2", "4"}:
            raise AssertionError("missing prefix results")
        for t in (1, 2, 4):
            p = got["profiles"][str(t)]
            if p["graphs"] != 3 or p["nodes"] != 9 or p["directed_edges"] != 12:
                raise AssertionError("synthetic coverage totals differ")
            sites = p["sites"]
            expected_sites = {f"layer{i}.{kind}" for i in range(1, 4) for kind in ("node", "edge")}
            if set(sites) != expected_sites:
                raise AssertionError("shared mode site inventory differs")
    test_wrong_cublas_environment()
    audit, records = S.prior_audit_gate(None)
    if audit["status"] != "complete" or len(records) != 6:
        raise AssertionError("current six-profile screen is not bound/authorized")
    print("PASS: shared per-channel max, finite/shape guards, both mode synthetic profiles, "
          "pre-Torch CUBLAS rejection, and current six-profile authorization; no official data")


if __name__ == "__main__":
    main()
