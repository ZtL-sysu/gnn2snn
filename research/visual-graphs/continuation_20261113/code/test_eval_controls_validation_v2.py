"""Synthetic-only tests for the E1 control evaluator. No official data are loaded."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
ROOT = Path(os.environ.get("GNN2SNN_ROOT", CODE_DIR.parents[3])).resolve()
sys.path.insert(0, str(ROOT / "research" / "visual-graphs" / "takeover_20261112"))

import eval_controls_validation_v2 as target  # noqa: E402
import matched_controls as controls  # noqa: E402
from gated_affine_cal_v1 import profile_thresholds  # noqa: E402


def _affine_doc():
    ct = {f"raw_{kind}_{i}": [0.5 + i + (0.1 if kind == "edge" else 0.0)] * 70
          for i in range(4) for kind in ("node", "edge")}
    return {"calibration": {"channel_thresholds": ct}}


def _six_doc():
    ct = {f"gated_{i}": [1.0 + i] * 70 for i in range(4)}
    ct.update({f"mlp_{i}": [0.001] * 8 for i in range(2)})
    return {"calibration": {"channel_thresholds": ct, "full_train_graph_count": 45000}}


def test_threshold_binding_is_frozen_and_shape_correct():
    affine, six = _affine_doc(), _six_doc()
    dc = target.control_thresholds("dc_adapt_analog_head", affine, six, sys.modules["gated_affine_cal_v1"],
                                   torch.device("cpu"), torch.float32)
    assert len(dc) == 4
    for i, vector in enumerate(dc):
        assert vector.numel() == 70 and torch.all(vector == 1.0 + i)
    for mode in ("per_consumer", "cse_only"):
        vectors = target.control_thresholds(mode, affine, six, sys.modules["gated_affine_cal_v1"],
                                            torch.device("cpu"), torch.float32)
        assert len(vectors) == 8
        expected = profile_thresholds(affine, "generic", 1)
        assert all(torch.equal(a, b) for a, b in zip(vectors, expected))
    with pytest.raises(ValueError):
        target.control_thresholds("unlisted_mode", affine, six, sys.modules["gated_affine_cal_v1"],
                                  torch.device("cpu"), torch.float32)


def test_dc_threshold_rejects_wrong_hidden_width():
    six = _six_doc()
    six["calibration"]["channel_thresholds"]["gated_2"] = [1.0] * 3
    with pytest.raises(RuntimeError, match="hidden width"):
        target.control_thresholds("dc_adapt_analog_head", _affine_doc(), six,
                                  sys.modules["gated_affine_cal_v1"], torch.device("cpu"), torch.float32)


def test_evaluator_constants_match_frozen_protocol():
    assert target.TIMESTEPS == (1, 2, 4, 8)
    assert target.EXPECTED_VAL == 5000 and target.BATCH_SIZE == 128
    assert target.MODES == ("dc_adapt_analog_head", "per_consumer", "cse_only")
    assert target.SIX_SITES == ("gated_0", "gated_1", "gated_2", "gated_3", "mlp_0", "mlp_1")


@pytest.mark.skipif(not os.environ.get("DCGS_REPO"), reason="pinned DCGS checkout not configured")
def test_cse_only_is_bitwise_generic_alias():
    from torch_geometric.data import Batch, Data
    from gated_teacher_model import GatedTeacher
    from gated_affine_base_v1 import affine_mode_forward, require_author_sources

    MTH, MTH_line, _ = require_author_sources(Path(os.environ["DCGS_REPO"]))
    torch.manual_seed(17)
    batch = Batch.from_data_list([
        Data(x=torch.randn(4, 3), edge_index=torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]]),
             edge_attr=torch.randn(5, 1)) for _ in range(2)])
    model = GatedTeacher(node_dim=3, edge_dim=1, hidden_dim=4, out_dim=4, layers=2).eval()
    model.forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("teacher oracle used"))
    thresholds = [torch.full((4,), 0.3 + 0.1 * i) for i in range(4)]
    alias = controls.affine_mode_forward(model, batch, thresholds, 4, "cse_only", MTH, MTH_line)
    generic = affine_mode_forward(model, batch, thresholds, 4, "generic", MTH, MTH_line)
    torch.testing.assert_close(alias["logits"], generic["logits"], rtol=0, atol=0)
    assert alias["trajectory_alias"].startswith("generic")
