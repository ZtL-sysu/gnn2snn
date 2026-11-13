import torch
from torch_geometric.data import Batch, Data

import gated_frozen_gate_calibrate as calibration_module
from gated_frozen_gate_calibrate import (
    LabelFreeGraphView, calibrate_loader, resolve_teacher_source,
)
from gated_frozen_gate_validate import evaluate_loader
from gated_teacher_model import GatedTeacher


def _graphs(with_labels):
    rows = []
    for i, n in enumerate((3, 4)):
        source = torch.tensor([0, 2, 1, 0, 3, 2])
        target = torch.tensor([1, 1, 0, 2, 1, 3])
        edge_index = torch.stack((source, target))
        values = dict(
            x=torch.rand(n, 3), pos=torch.rand(n, 2), edge_index=edge_index,
            edge_attr=torch.rand(edge_index.shape[1], 1),
        )
        # Ensure all endpoints are valid for n=3 as well.
        if n == 3:
            values["edge_index"] = edge_index[:, :4]
            values["edge_attr"] = values["edge_attr"][:4]
        if with_labels:
            values["y"] = torch.tensor([i % 3])
        rows.append(Data(**values))
    return rows


def test_label_free_two_pass_calibration_covers_six_sites():
    torch.manual_seed(31)
    model = GatedTeacher(node_dim=5, edge_dim=1, hidden_dim=4, out_dim=4,
                         n_classes=3, layers=4, batch_norm=True, residual=True).eval()
    source = _graphs(with_labels=True)
    view = LabelFreeGraphView(source)
    assert all(view[i].y is None for i in range(len(view)))
    batch = Batch.from_data_list([view[0], view[1]])
    result = calibrate_loader(model, [batch], expected_graphs=2)
    assert result["passes"] == 2
    assert result["pass1_graphs"] == result["pass2_graphs"] == 2
    assert set(result["channel_thresholds"]) == {
        "gated_0", "gated_1", "gated_2", "gated_3", "mlp_0", "mlp_1"
    }
    expected_widths = (4, 4, 4, 4, 2, 1)
    for name, width in zip(result["channel_thresholds"], expected_widths):
        thresholds = result["channel_thresholds"][name]
        assert len(thresholds) == width
        assert all(value > 0 for value in thresholds)
        assert result["sites"][name]["rows"] > 0


def test_validation_aggregation_tracks_oracle_and_staged_metrics():
    torch.manual_seed(37)
    model = GatedTeacher(node_dim=5, edge_dim=1, hidden_dim=4, out_dim=4,
                         n_classes=3, layers=4, batch_norm=True, residual=True).eval()
    batch = Batch.from_data_list(_graphs(with_labels=True))
    unlabeled_batch = Batch.from_data_list(LabelFreeGraphView(_graphs(with_labels=True)))
    calibration_result = calibrate_loader(model, [unlabeled_batch], expected_graphs=2)
    calibration = {"calibration": calibration_result}
    result = evaluate_loader(model, [batch], calibration, torch.device("cpu"), expected_graphs=2)
    assert result["teacher_logits"].shape == (2, 3)
    assert result["exact_oracle_logits"].shape == (2, 3)
    assert result["labels"].shape == (2,)
    assert result["exact_oracle_max_abs_logit_error"] < 1e-5
    for t in (8, 16, 32):
        assert result["staged_logits"][t].shape == (2, 3)
        assert len(result["site_spikes"][t]) == 6
        assert all(count >= 0 for count in result["site_spikes"][t])


def test_metadata_basename_resolves_public_baseline_directory(tmp_path):
    baseline_dir = tmp_path / "public_baselines"
    baseline_dir.mkdir()
    source = baseline_dir / "gated_net.py"
    source.write_text("# synthetic source", encoding="utf-8")
    assert resolve_teacher_source(tmp_path, "gated_net.py") == source.resolve()

    root_source = tmp_path / "gated_teacher_model.py"
    root_source.write_text("# synthetic root source", encoding="utf-8")
    assert resolve_teacher_source(tmp_path, "gated_teacher_model.py") == root_source.resolve()


def test_calibration_moves_each_pass_batch_to_model_device(monkeypatch):
    class TrackingBatch:
        num_graphs = 1

        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(torch.device(device))
            return self

    batch = TrackingBatch()
    def fake_reference_activations(_model, _batch):
        return [torch.tensor([[0.25, 0.75]]) for _ in range(6)]

    monkeypatch.setattr(calibration_module, "reference_activations", fake_reference_activations)
    model = torch.nn.Linear(1, 1).cpu()
    result = calibrate_loader(model, [batch], expected_graphs=1, device="cpu")
    assert batch.devices == [torch.device("cpu"), torch.device("cpu")]
    assert result["pass1_graphs"] == result["pass2_graphs"] == 1
